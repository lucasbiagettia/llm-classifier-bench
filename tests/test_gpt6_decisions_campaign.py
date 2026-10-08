"""Offline replay auditing and integration with both existing campaign factories."""
import csv
from dataclasses import asdict
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from llm_classifier_bench.class_definitions import ClassDefinitionProfile
from llm_classifier_bench.core import ClassDefinition
from llm_classifier_bench.measurement import MeasurementConfig
from test_openai_decisions import Response, transport  # shared mocked HTTP fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module("run_gpt6_decisions")
    monkeypatch.setattr("socket.create_connection", lambda *a, **kw: pytest.fail("No network allowed"))
    return module


@pytest.fixture
def history(replay, tmp_path):
    shared = replay.shared
    classes = tuple(ClassDefinition(f"class{i:02d}", f"Frozen definition {i}") for i in range(20))
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = {"files": {}}
    for split, count in (("train", 130), ("test", 40)):
        path = source_dir / f"{split}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["text", "category"])
            writer.writerows((f"{split} unique {c.name} {i}", c.name) for c in classes for i in range(count))
        source["files"][path.name] = {"sha256": shared.sha256(path)}
    definitions = tmp_path / "definitions.json"
    ClassDefinitionProfile(dataset="banking77", profile="fixture", classes=classes,
                           review_status="approved").write_json(definitions)
    manifest = {"dataset": "banking77", "seed": 42, "class_counts": list(replay.COUNTS),
                "test_examples_per_class": 40, "source": source,
                "definitions": {"path": str(definitions), "sha256": shared.sha256(definitions)},
                "label_sets": {str(k): [c.name for c in classes[:k]] for k in replay.COUNTS}}
    source_rows = shared.load_source(source_dir, manifest)
    results = {"conditions": []}
    for k in replay.COUNTS:
        bundle = shared.build_bundle(source_rows, classes[:k])
        for method in ("openai", "jev"):
            evidence = tmp_path / "history" / f"{method}-k{k}"
            config = {"dataset": {"classes": [asdict(c) for c in bundle.classes]},
                      "measurement": asdict(MeasurementConfig()),
                      "matched_budget": {"examples": 0, "unit": "per_class", "seed": 42, "validation_examples": 0},
                      "split": {"validation_fraction": .2, "seed": 42, "strategy": "deterministic_stratified_by_label"}}
            shared.write_json(evidence / "runs/new/config.json", config)
            (evidence / "predictions.jsonl").write_text("".join(json.dumps(
                {"sample_id": e.sample_id, "input": e.text, "gold_label": e.label}) + "\n" for e in bundle.test))
            results["conditions"].append({"method": method, "k": k, "fit_per_class": 0,
                                          "status": "completed", "evidence": str(evidence)})
    shared.write_json(tmp_path / "manifest.json", manifest)
    shared.write_json(tmp_path / "results.json", results)
    return SimpleNamespace(reference_manifest=tmp_path / "manifest.json", reference_results=tmp_path / "results.json",
        source_dir=source_dir, pricing=ROOT / "pricing/decisions_2026-10-07.json", decisions_model=None,
        decisions_timeout_s=None, output_root=tmp_path / "new-campaigns", execute=False)


def test_matrix_audit_and_planning_are_offline(replay, history, transport, monkeypatch):
    plan = replay.prepare_plan(history)
    assert [(c["k"], c["budget"], c["test_examples"]) for c in plan["cells"]] == [(5, 0, 200), (10, 0, 400), (15, 0, 600), (20, 0, 800)]
    assert plan["maximum_predictions"] == 2000
    assert all(row["comparable_conditions_checked"] == 2 for row in plan["audit"]["conditions"])
    root = replay.run_campaign(history, plan)
    rows = replay.shared.read_json(root / "summary.json")
    assert len(rows) == 4 and all(row["status"] == "dry_run" for row in rows)
    assert len(list((root / "definitions").glob("*.json"))) == 4
    assert not transport.calls
    assert all(not (p / "predictions.jsonl").exists() for p in (root / "runs").iterdir())


@pytest.mark.parametrize("target", ["source", "definitions", "test_predictions", "test_labels", "matrix"])
def test_changed_reference_cannot_reach_execution(replay, history, transport, target):
    shared = replay.shared
    manifest = shared.read_json(history.reference_manifest)
    if target == "source":
        (history.source_dir / "test.csv").write_text("changed")
    elif target == "definitions":
        Path(manifest["definitions"]["path"]).write_text("changed")
    elif target in ("test_predictions", "test_labels"):
        path = Path(shared.read_json(history.reference_results)["conditions"][0]["evidence"]) / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if target == "test_predictions":
            rows.pop()
        else:
            rows[0]["gold_label"] = "changed"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    else:
        manifest["class_counts"] = [5, 10, 20, 25]
        shared.write_json(history.reference_manifest, manifest)
    with pytest.raises(ValueError):
        replay.prepare_plan(history)
    assert not transport.calls and not history.output_root.exists()


def test_failed_execution_stops_after_one_cell_and_keeps_summary(replay, history, transport, monkeypatch):
    plan = replay.prepare_plan(history)
    history.execute = True
    transport.outcomes = [Response({}, 401)]
    with pytest.raises(RuntimeError, match="HTTP 401"):
        replay.run_campaign(history, plan)
    assert len(transport.calls) == 1
    root, = history.output_root.iterdir()
    rows = replay.shared.read_json(root / "summary.json")
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert rows[0]["run_id"] and rows[0]["run_dir"]
    assert (Path(rows[0]["run_dir"]) / "status.json").is_file()
    assert len(list((root / "runs").iterdir())) == 1


def test_paid_execution_uses_the_exact_audited_cohort(replay, history, transport):
    plan = replay.prepare_plan(history)
    # Synthetic transport only: one cell is sufficient to verify the real runner boundary.
    plan["cells"] = plan["cells"][:1]
    history.execute = True

    def post(endpoint, **kwargs):
        payload = json.loads(kwargs["data"])
        choices = payload["questions"][0]["choices"]
        names = [c["value"] for c in choices]
        transport.calls.append((endpoint, kwargs))
        return Response({"model": "gpt-6-luna", "usage": {"input_tokens": 100},
                         "answers": [{"type": "choice", "name": "classification", "choice": names[0],
                                      "confidence": .5, "probabilities": [
                                          {"value": n, "probability": 1 if n == names[0] else 0} for n in names]}]})
    transport.post = post
    root = replay.run_campaign(history, plan)
    row, = replay.shared.read_json(root / "summary.json")
    assert row["status"] == "completed" and row["test_examples"] == 200
    assert row["total_labeled_examples_used"] == 0 and row["validation_examples_used"] == 0
    assert len(transport.calls) == 200
    assert row["multiclass_log_loss"] is not None and row["multiclass_brier_score"] is not None
    assert row["total_cost_usd"] == pytest.approx(.002)
    config = replay.shared.read_json(Path(row["run_dir"]) / "config.json")
    assert config["dataset"]["test_sample_ids"] == plan["conditions"]["5"]["test_sample_ids"]


@pytest.mark.parametrize("script", ["run_banking77_scaling_benchmark", "run_banking77_scaling_benchmark_v2"])
def test_existing_campaign_entry_points_offer_decisions(script, replay, tmp_path, monkeypatch, transport):
    campaign = importlib.import_module(script)
    probe = importlib.import_module("probe_tfidf_classifier")
    bundle = probe.fixture_bundle()
    definitions = tmp_path / "definitions.json"
    ClassDefinitionProfile(dataset=bundle.name, profile="offline", classes=bundle.classes,
                           review_status="approved").write_json(definitions)
    monkeypatch.setattr(campaign, "get_dataset", lambda _: campaign.StaticDataset(bundle))
    monkeypatch.setattr(campaign, "utc_campaign_id", lambda: "decisions-fixture")
    argv = [script, "--classifiers", "openai-decisions", "--class-counts", "2", "--seeds", "17",
            "--train-per-class", "4", "--test-per-class", "1", "--validation-fraction", ".25",
            "--definitions", str(definitions), "--output-root", str(tmp_path), "--dry-run",
            "--matched-budgets", "0", "1", "--budget-unit", "per_class"]
    if script.endswith("_v2"):
        argv += ["--min-train-per-class", "4", "--min-test-per-class", "1", "--strict-support"]
    monkeypatch.setattr(sys, "argv", argv)
    campaign.main()
    rows = json.loads((tmp_path / "decisions-fixture/summary.json").read_text())
    assert [r["status"] for r in rows] == ["dry_run", "unsupported"]
    assert not transport.calls


def simulated_choices(transport, refused_texts):
    def post(endpoint, **kwargs):
        payload = json.loads(kwargs["data"])
        transport.calls.append((endpoint, kwargs))
        names = [c["value"] for c in payload["questions"][0]["choices"]]
        answer = ({"type": "refusal", "name": "classification"} if payload["input"] in refused_texts else
                  {"type": "choice", "name": "classification", "choice": names[0], "confidence": .5,
                   "probabilities": [{"value": n, "probability": 1 if n == names[0] else 0} for n in names]})
        return Response({"model": "gpt-6-luna", "usage": {"input_tokens": 100}, "answers": [answer]})
    transport.post = post


def one_cell(replay, history):
    plan = replay.prepare_plan(history)
    plan["cells"] = plan["cells"][:1]
    source = replay.shared.load_source(history.source_dir, plan)
    bundle = replay.shared.build_bundle(source, [ClassDefinition(**c) for c in plan["conditions"]["5"]["classes"]])
    history.execute = True
    return plan, bundle


def test_refusals_continue_without_losing_examples_or_selectively_scoring(replay, history, transport):
    plan, bundle = one_cell(replay, history)
    refused = {bundle.test[1].text, bundle.test[-1].text}
    simulated_choices(transport, refused)
    root = replay.run_campaign(history, plan)
    row, = replay.shared.read_json(root / "summary.json")
    assert row["status"] == "completed_with_refusals"
    assert row["successful_examples"] == 198 and row["refused_examples"] == 2 and row["pending_examples"] == 0
    assert len(transport.calls) == 200
    assert [json.loads(c[1]["data"])["input"] for c in transport.calls] == [e.text for e in bundle.test]
    assert row["accuracy"] is None and row["multiclass_log_loss"] is None
    assert row["total_cost_usd"] == pytest.approx(.002)  # includes both refused attempts
    directory = Path(row["evidence"])
    outcomes = [json.loads(line) for line in (directory / "outcomes.jsonl").read_text().splitlines()]
    assert [r["sample_id"] for r in outcomes] == [e.sample_id for e in bundle.test]
    failures = [r for r in outcomes if r["status"] == "refused"]
    assert len(failures) == 2 and all(r["probabilities"] is None and r["predicted_label"] is None for r in failures)
    from llm_classifier_bench.metrics.evaluator import evaluate_jsonl
    with pytest.raises(ValueError, match="refusals/pending"):
        evaluate_jsonl(directory / "predictions.jsonl")
    assert len(list((root / "runs").iterdir())) == 2  # no empty segment after terminal refusal
    assert all(not (Path(r) / "metrics.json").exists() for r in row["segment_run_dirs"])
    original = {p: replay.shared.sha256(p) for p in (root / "runs").rglob("*") if p.is_file()}
    history.resume = root
    replay.run_campaign(history, plan)
    assert len(transport.calls) == 200  # idempotent, even for refused samples
    assert all(replay.shared.sha256(p) == digest for p, digest in original.items())


def test_resume_legacy_stopped_campaign_skips_saved_predictions_and_refusal(replay, history, transport, monkeypatch):
    plan, bundle = one_cell(replay, history)
    simulated_choices(transport, {bundle.test[1].text})
    original_exception = replay.OpenAIDecisionsRefusalError
    # Simulate the old campaign behavior: let the runner's refusal abort the campaign.
    monkeypatch.setattr(replay, "OpenAIDecisionsRefusalError", type("NeverRaised", (ValueError,), {}))
    with pytest.raises(original_exception):
        replay.run_campaign(history, plan)
    root, = history.output_root.iterdir()
    assert len(transport.calls) == 2
    original = {p: replay.shared.sha256(p) for p in (root / "runs").rglob("*") if p.is_file()}
    monkeypatch.setattr(replay, "OpenAIDecisionsRefusalError", original_exception)
    history.resume = root
    history.execute = False
    replay.run_campaign(history, plan)
    assert len(transport.calls) == 2  # resume planning is offline
    history.execute = True
    replay.run_campaign(history, plan)
    assert len(transport.calls) == 200
    assert all(replay.shared.sha256(p) == digest for p, digest in original.items())
    row, = replay.shared.read_json(root / "summary.json")
    assert row["status"] == "completed_with_refusals" and row["refused_examples"] == 1
    assert row["total_cost_usd"] == pytest.approx(.002)
    # A changed evidence file cannot be accepted by another resume.
    path = next((root / "runs").glob("*/predictions.jsonl"))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="evidence changed"):
        replay.run_campaign(history, plan)
    assert len(transport.calls) == 200


def test_resume_rejects_changed_model_before_network(replay, history, transport):
    plan, bundle = one_cell(replay, history)
    simulated_choices(transport, set())
    root = replay.run_campaign(history, plan)
    history.resume = root
    plan["decisions"]["requested_model"] = "different-model"
    with pytest.raises(ValueError, match="configuration changed: decisions"):
        replay.run_campaign(history, plan)
    assert len(transport.calls) == 200


@pytest.mark.parametrize("failed_index", [0, 2])
def test_resume_503_retries_only_pending_and_preserves_failed_attempts(replay, history, transport, failed_index):
    plan, bundle = one_cell(replay, history)
    refused = {bundle.test[0].text} if failed_index else set()
    simulated_choices(transport, refused)
    successful_post = transport.post
    failures_left = 2

    def post(endpoint, **kwargs):
        nonlocal failures_left
        if json.loads(kwargs["data"])["input"] == bundle.test[failed_index].text and failures_left:
            failures_left -= 1
            transport.calls.append((endpoint, kwargs))
            return Response({}, 503)
        return successful_post(endpoint, **kwargs)

    transport.post = post
    with pytest.raises(RuntimeError, match="HTTP 503"):
        replay.run_campaign(history, plan)
    root, = history.output_root.iterdir()
    history.resume = root
    original = {p: replay.shared.sha256(p) for p in (root / "runs").rglob("*") if p.is_file()}
    history.execute = False
    replay.run_campaign(history, plan)
    assert len(transport.calls) == failed_index + 1
    history.execute = True
    with pytest.raises(RuntimeError, match="HTTP 503"):
        replay.run_campaign(history, plan)
    assert len(transport.calls) == failed_index + 2
    assert all(replay.shared.sha256(p) == digest for p, digest in original.items())
    original = {p: replay.shared.sha256(p) for p in (root / "runs").rglob("*") if p.is_file()}
    replay.run_campaign(history, plan)
    actual = [json.loads(c[1]["data"])["input"] for c in transport.calls]
    expected = [e.text for e in bundle.test]
    expected[failed_index:failed_index] = [bundle.test[failed_index].text] * 2
    assert actual == expected
    assert all(replay.shared.sha256(p) == digest for p, digest in original.items())
    row, = replay.shared.read_json(root / "summary.json")
    assert row["successful_examples"] == 200 - len(refused)
    assert row["refused_examples"] == len(refused) and row["pending_examples"] == 0
    assert row["status"] == ("completed_with_refusals" if refused else "completed")
    assert row["total_cost_usd"] is None  # 503 usage was not reported; never invent zero cost.
    report = replay.shared.read_json(Path(row["evidence"]) / "cost_report.json")
    assert report["known_cost_subtotal_usd"] == pytest.approx(.002)
    assert set(map(str, (root / "runs").iterdir())) == set(report["source_runs"])
    replay.run_campaign(history, plan)
    assert len(transport.calls) == 202  # Completed resume is idempotent.


@pytest.mark.parametrize("damage", ["status", "sample_id", "http_status", "response_received", "extra_attempt"])
def test_resume_503_requires_matching_failure_evidence(replay, history, transport, damage):
    plan, bundle = one_cell(replay, history)
    transport.outcomes = [Response({}, 503)]
    with pytest.raises(RuntimeError, match="HTTP 503"):
        replay.run_campaign(history, plan)
    root, = history.output_root.iterdir()
    run, = (root / "runs").iterdir()
    if damage == "status":
        status = replay.shared.read_json(run / "status.json")
        status["error_message"] = "OpenAI Decisions HTTP 401"
        replay.shared.write_json(run / "status.json", status)
    else:
        path = run / "usage.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        attempt = next(e for e in events if e["kind"] == "api_attempt")
        if damage == "extra_attempt":
            events.append(dict(attempt))
        else:
            attempt[damage] = {"sample_id": bundle.test[1].sample_id, "http_status": 401,
                               "response_received": False}[damage]
        path.write_text("".join(json.dumps(e) + "\n" for e in events))
    history.resume = root
    with pytest.raises(ValueError):
        replay.run_campaign(history, plan)
    assert len(transport.calls) == 1


def test_refusal_does_not_stop_later_class_counts(replay, history, transport):
    plan = replay.prepare_plan(history)
    plan["cells"] = plan["cells"][:2]
    source = replay.shared.load_source(history.source_dir, plan)
    bundle = replay.shared.build_bundle(source, [ClassDefinition(**c) for c in plan["conditions"]["5"]["classes"]])
    simulated_choices(transport, {bundle.test[1].text})
    history.execute = True
    root = replay.run_campaign(history, plan)
    rows = replay.shared.read_json(root / "summary.json")
    assert [r["class_count"] for r in rows] == [5, 10]
    assert [r["status"] for r in rows] == ["completed_with_refusals", "completed_with_refusals"]
    assert [r["successful_examples"] for r in rows] == [199, 399]
    assert len(transport.calls) == 600  # shared inputs are still evaluated once per distinct K
    history.resume = root
    replay.run_campaign(history, plan)
    assert len(transport.calls) == 600


def test_all_refused_has_no_fabricated_scores_or_division_by_zero(replay, history, transport):
    plan, bundle = one_cell(replay, history)
    bundle = replay.replace(bundle, test=bundle.test[:2])
    refusals = {e.sample_id: {"sample_id": e.sample_id, "input": e.text, "gold_label": e.label,
                            "status": "refused", "predicted_label": None, "probabilities": None} for e in bundle.test}
    state = {"pending": (), "predictions": {}, "refusals": refusals, "runs": [], "evidence_sha256": {},
             "costs": [{"total_cost_usd": .00002, "known_cost_subtotal_usd": .00002}]}
    directory, coverage, metrics = replay.write_cell_artifacts(history.output_root, plan["cells"][0], bundle, state)
    assert coverage["status"] == "completed_with_refusals" and coverage["refused_examples"] == 2
    assert not metrics["accuracy"]["available"] and not metrics["cost_per_1000_usd"]["available"]
    assert metrics["total_cost_usd"]["value"] == .00002
    assert (directory / "predictions.jsonl").read_text() == ""
