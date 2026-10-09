"""Exact historical replay and both general campaign entry points, without APIs."""
import importlib
import json
from pathlib import Path
import sys

import pytest

from llm_classifier_bench.class_definitions import ClassDefinitionProfile
from test_gpt6_decisions_campaign import history, replay
from test_perplexity_decisions import CARD, Response, transport


@pytest.fixture
def perplexity_replay(replay, history):
    module = importlib.import_module("run_perplexity_decisions")
    history.pricing = CARD
    history.perplexity_model = None
    history.perplexity_timeout_s = None
    return module


def test_four_cell_plan_matches_reference_without_network(perplexity_replay, history, transport):
    plan = perplexity_replay.prepare_plan(history)
    assert [c["method"] for c in plan["cells"]] == ["perplexity-decisions"] * 4
    assert plan["maximum_predictions"] == 2000
    assert plan["decisions"]["requested_model"] == "pplx-decider-v1.1-27b"
    root = perplexity_replay.run_campaign(history, plan)
    rows = perplexity_replay.shared.read_json(root / "summary.json")
    assert [r["status"] for r in rows] == ["dry_run"] * 4
    for row in rows:
        config = perplexity_replay.shared.read_json(Path(row["run_dir"]) / "config.json")
        assert config["dataset"]["test_sample_ids"] == plan["conditions"][str(row["class_count"])]["test_sample_ids"]
    assert not transport.calls


def test_execution_uses_exact_audited_inputs_and_prices_native_usage(perplexity_replay, history, transport):
    plan = perplexity_replay.prepare_plan(history)
    plan["cells"] = plan["cells"][:1]
    history.execute = True

    def post(endpoint, **kwargs):
        payload = json.loads(kwargs["data"])
        names = list(payload["questions"]["classification"]["criteria"])
        transport.calls.append((endpoint, kwargs))
        return Response({"model": "pplx-decider-v1.1-27b", "usage": {"input_tokens": 100},
                         "answers": {"classification": {"type": "choice", "choice": names[0],
                             "confidence": .5, "probabilities": {n: int(n == names[0]) for n in names}}}})
    transport.post = post
    root = perplexity_replay.run_campaign(history, plan)
    row, = perplexity_replay.shared.read_json(root / "summary.json")
    assert row["status"] == "completed" and row["test_examples"] == 200
    assert row["total_labeled_examples_used"] == 0
    assert row["multiclass_log_loss"] is not None and row["multiclass_brier_score"] is not None
    assert row["total_cost_usd"] == pytest.approx(.0004)
    source = perplexity_replay.shared.load_source(history.source_dir, plan)
    classes = [perplexity_replay.shared.ClassDefinition(**c) for c in plan["conditions"]["5"]["classes"]]
    bundle = perplexity_replay.shared.build_bundle(source, classes)
    assert [json.loads(c[1]["data"])["state"] for c in transport.calls] == [e.text for e in bundle.test]
    assert len(transport.calls) == 200


def test_failures_stop_campaign_and_keep_summary(perplexity_replay, history, transport):
    plan = perplexity_replay.prepare_plan(history)
    history.execute = True
    transport.outcomes = [Response({}, 429)]
    with pytest.raises(RuntimeError, match="HTTP 429"):
        perplexity_replay.run_campaign(history, plan)
    assert len(transport.calls) == 1
    root, = history.output_root.iterdir()
    row, = perplexity_replay.shared.read_json(root / "summary.json")
    assert row["status"] == "failed" and (Path(row["run_dir"]) / "usage.jsonl").exists()
    assert len(list((root / "runs").iterdir())) == 1


def test_changed_reference_is_rejected_before_execution(perplexity_replay, history, transport):
    plan = perplexity_replay.prepare_plan(history)
    (history.source_dir / "test.csv").write_text("changed")
    history.execute = True
    with pytest.raises(ValueError, match="Audited input changed"):
        perplexity_replay.run_campaign(history, plan)
    assert not transport.calls and not history.output_root.exists()


@pytest.mark.parametrize("script", ["run_banking77_scaling_benchmark", "run_banking77_scaling_benchmark_v2"])
def test_general_campaigns_support_perplexity_and_reject_nonzero_budgets(script, perplexity_replay,
                                                                         tmp_path, monkeypatch, transport):
    campaign = importlib.import_module(script)
    bundle = importlib.import_module("probe_tfidf_classifier").fixture_bundle()
    definitions = tmp_path / "campaign_definitions.json"
    ClassDefinitionProfile(dataset=bundle.name, profile="offline", classes=bundle.classes,
                           review_status="approved").write_json(definitions)
    monkeypatch.setattr(campaign, "get_dataset", lambda _: campaign.StaticDataset(bundle))
    monkeypatch.setattr(campaign, "utc_campaign_id", lambda: "perplexity-fixture")
    argv = [script, "--classifiers", "perplexity-decisions", "--class-counts", "2", "--seeds", "17",
            "--train-per-class", "4", "--test-per-class", "1", "--validation-fraction", ".25",
            "--definitions", str(definitions), "--output-root", str(tmp_path), "--dry-run",
            "--matched-budgets", "0", "1", "--budget-unit", "per_class"]
    if script.endswith("_v2"):
        argv += ["--min-train-per-class", "4", "--min-test-per-class", "1", "--strict-support"]
    monkeypatch.setattr(sys, "argv", argv)
    campaign.main()
    rows = json.loads((tmp_path / "perplexity-fixture/summary.json").read_text())
    assert [r["status"] for r in rows] == ["dry_run", "unsupported"]
    manifest = json.loads((tmp_path / "perplexity-fixture/campaign.json").read_text())
    assert manifest["perplexity_decisions"]["endpoint"] == "https://api.perplexity.ai/v1/decisions"
    assert not transport.calls
