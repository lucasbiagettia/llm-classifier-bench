"""Offline Jev contract, paid-attempt accounting, and both campaign integrations."""
import copy
import importlib
import json
from pathlib import Path
import sys

import pytest
import requests

from llm_classifier_bench.budgets import MatchedBudgetConfig
from llm_classifier_bench.classifiers import JevClassifier
from llm_classifier_bench.classifiers.jev import JevAPIError
from llm_classifier_bench.core import ClassificationInput, ClassDefinition
from llm_classifier_bench.costs import load_pricing, price_entry
from llm_classifier_bench.measurement import MeasurementConfig
from llm_classifier_bench.metrics.evaluator import evaluate_jsonl, results_as_dict
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark
from test_runner import FakeDataset, build_bundle

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "pricing/inference_2026-09-25.json"
FIXTURE = json.loads((ROOT / "tests/fixtures/jev/choice.json").read_text())


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status
        self.headers = {"x-request-id": "fixture-request"}

    def json(self):
        return self.body


class Session:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.calls = []

    def mount(self, *args):
        pass

    def post(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        result = self.outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def transport(monkeypatch):
    session = Session()
    monkeypatch.setenv("JEV_TOKEN", "offline-secret")
    monkeypatch.setattr("llm_classifier_bench.classifiers.jev.requests.Session", lambda: session)
    monkeypatch.setattr("llm_classifier_bench.classifiers.jev.sleep", lambda _: None)
    return session


def prepared(transport, body=None, **kwargs):
    transport.outcomes.append(Response(copy.deepcopy(body if body is not None else FIXTURE["response"])))
    classifier = JevClassifier(**kwargs)
    classifier.prepare(build_bundle().classes)
    classifier.fit(build_bundle().train)
    return classifier


def test_request_frozen_definitions_native_confidence_and_order(transport):
    classifier = prepared(transport)
    prediction = classifier.predict([ClassificationInput("test-1", FIXTURE["request"]["state"], {"gold": "secret-label"})])[0]
    endpoint, request = transport.calls[0]
    assert json.loads(request["data"]) == FIXTURE["request"]
    assert request["allow_redirects"] is False and request["timeout"] == 30
    assert prediction.sample_id == "test-1" and prediction.predicted_label == "Sports"
    assert list(prediction.probabilities) == ["World", "Sports"]
    assert prediction.confidence == .75
    assert prediction.raw_response["provider_confidence"] == .42
    assert prediction.request_id == "fixture-request"
    assert classifier.fitted_metadata()["total_labeled_examples_used"] == 0
    assert "offline-secret" not in json.dumps(prediction.raw_response)


@pytest.mark.parametrize("mutation", [
    lambda b: b.pop("answers"),
    lambda b: b.pop("model"),
    lambda b: b["answers"]["classification"].update(type="score"),
    lambda b: b["answers"]["classification"].pop("probabilities"),
    lambda b: b["answers"]["classification"].update(choice="Other"),
    lambda b: b["answers"]["classification"].update(choice="World"),
    lambda b: b["answers"]["classification"].update(probabilities={"Sports": 1}),
    lambda b: b["answers"]["classification"].update(probabilities={"World": .2, "Sports": .7}),
    lambda b: b["answers"]["classification"]["probabilities"].update(World=float("nan")),
    lambda b: b["answers"]["classification"]["probabilities"].update(World=-.1),
    lambda b: b["answers"]["classification"]["probabilities"].update(World=True),
    lambda b: b["answers"]["classification"]["probabilities"].update(World="0.25"),
    lambda b: b["answers"]["classification"].update(confidence=float("inf")),
])
def test_invalid_outputs_fail_but_keep_usage(transport, mutation):
    body = copy.deepcopy(FIXTURE["response"])
    mutation(body)
    classifier = prepared(transport, body)
    entries = []
    classifier.set_usage_sink(entries.append)
    with pytest.raises(ValueError):
        classifier.predict(build_bundle().inputs()[:1])
    assert len(entries) == 1 and entries[0]["status"] == "failed"
    assert entries[0]["usage"]["input_tokens"] == 300


def test_exact_tie_keeps_provider_choice_even_when_second_in_class_order(transport, tmp_path):
    body = copy.deepcopy(FIXTURE["response"])
    body["answers"]["classification"]["probabilities"] = {"World": .5, "Sports": .5}
    transport.outcomes = [Response(body), Response(body)]
    result = run_benchmark(FakeDataset(build_bundle()), JevClassifier(), BenchmarkRunConfig(output_root=tmp_path))
    assert json.loads(result.status_path.read_text())["status"] == "completed"
    assert results_as_dict(evaluate_jsonl(result.predictions_path))["top_label_ece"]["available"]


def test_missing_native_confidence_and_usage_are_explicitly_unavailable(transport):
    body = copy.deepcopy(FIXTURE["response"])
    body.pop("usage")
    body["answers"]["classification"].pop("confidence")
    classifier = prepared(transport, body)
    entries = []
    classifier.set_usage_sink(entries.append)
    p = classifier.predict(build_bundle().inputs()[:1])[0]
    assert p.confidence == .75 and p.raw_response["provider_confidence"] is None
    entry = dict(entries[0], event_id="e", phase="evaluation")
    assert price_entry(entry, load_pricing(CARD), None)["cost_usd"] is None


def test_warmup_retry_and_evaluation_costs_replay_without_network(transport, tmp_path):
    body = FIXTURE["response"]
    transport.outcomes = [Response(body, 529), Response(body), Response(body), Response(body)]
    result = run_benchmark(FakeDataset(build_bundle()), JevClassifier(max_retries=1, max_requests=4),
                           BenchmarkRunConfig(output_root=tmp_path, pricing_path=CARD,
                                              measurement=MeasurementConfig(warmup_examples=1)))
    usage = [json.loads(line) for line in (result.run_dir / "usage.jsonl").read_text().splitlines()]
    attempts = [e for e in usage if e["kind"] == "api_attempt"]
    assert len(attempts) == 4 and attempts[0]["status"] == "failed"
    assert [e["phase"] for e in attempts] == ["warmup", "warmup", "evaluation", "evaluation"]
    costs = json.loads((result.run_dir / "cost_report.json").read_text())
    assert costs["total_cost_usd"] == pytest.approx(1200 * .042 / 1e6)
    assert costs["cost_per_1000_successful_examples_usd"] == pytest.approx(.0252)
    assert costs["coverage_complete"] is True
    metrics = results_as_dict(evaluate_jsonl(result.predictions_path))
    for key in ("accuracy", "macro_f1", "top_label_ece", "adaptive_ece", "multiclass_log_loss", "multiclass_brier_score"):
        assert metrics[key]["available"]
    assert "offline-secret" not in "".join(p.read_text() for p in result.run_dir.glob("*.json*"))


def test_failed_second_call_preserves_first_and_unknown_charge(transport, tmp_path):
    transport.outcomes = [Response(FIXTURE["response"]), requests.Timeout("secret transport details")]
    with pytest.raises(JevAPIError, match="Jev transport failure"):
        run_benchmark(FakeDataset(build_bundle()), JevClassifier(),
                      BenchmarkRunConfig(output_root=tmp_path, run_id="failure", pricing_path=CARD))
    root = tmp_path / "failure"
    assert len((root / "predictions.jsonl").read_text().splitlines()) == 1
    assert json.loads((root / "status.json").read_text())["status"] == "failed"
    assert json.loads((root / "cost_report.json").read_text())["total_cost_usd"] is None
    assert "secret transport details" not in "".join(p.read_text() for p in root.glob("*.json*"))


@pytest.mark.parametrize("status", [301, 401, 422])
def test_nonretryable_http_errors(transport, status):
    transport.outcomes = [Response({}, status)]
    c = JevClassifier(max_retries=2)
    c.prepare(build_bundle().classes)
    with pytest.raises(JevAPIError):
        c.predict(build_bundle().inputs()[:1])
    assert len(transport.calls) == 1


def test_request_cap_counts_retries(transport):
    transport.outcomes = [Response({}, 529)]
    c = JevClassifier(max_requests=1, max_retries=3)
    c.prepare(build_bundle().classes)
    with pytest.raises(RuntimeError, match="cap exhausted"):
        c.predict(build_bundle().inputs()[:1])
    assert len(transport.calls) == 1


@pytest.mark.parametrize("kwargs", [{"context_window_tokens": 100}, {"max_requests": 1}])
def test_preflight_rejects_before_network(transport, tmp_path, kwargs):
    result = run_benchmark(FakeDataset(build_bundle()), JevClassifier(**kwargs),
                           BenchmarkRunConfig(output_root=tmp_path))
    assert json.loads(result.status_path.read_text())["status"] == "unsupported"
    assert not transport.calls


def test_nonzero_matched_budget_is_unsupported_without_api_calls(transport, tmp_path):
    result = run_benchmark(FakeDataset(build_bundle()), JevClassifier(), BenchmarkRunConfig(
        output_root=tmp_path, matched_budget=MatchedBudgetConfig(1, "per_class")))
    status = json.loads(result.status_path.read_text())
    assert status["status"] == "unsupported" and "zero_shot_only" in status["error_message"]
    assert not transport.calls


def test_zero_budget_and_dry_run_need_no_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("JEV_TOKEN", raising=False)
    result = run_benchmark(FakeDataset(build_bundle()), JevClassifier(), BenchmarkRunConfig(
        output_root=tmp_path, matched_budget=MatchedBudgetConfig(0, "total"), dry_run=True))
    assert json.loads(result.status_path.read_text())["status"] == "dry_run"


def test_limits_and_validation_budget_are_explicit():
    c = JevClassifier()
    with pytest.raises(ValueError):
        c.prepare([ClassDefinition("x", "a"), ClassDefinition("x", "b")])
    plan = c.preflight(build_bundle().classes, build_bundle().inputs(),
                       matched_budget=MatchedBudgetConfig(0, "total", validation_examples=1))
    assert not plan["supported"]
    assert plan["limits"]["choice_options"] == 255


@pytest.mark.parametrize("script", ["run_banking77_scaling_benchmark", "run_banking77_scaling_benchmark_v2"])
@pytest.mark.parametrize("regime", ["matched", "full", "dry"])
def test_campaigns_include_jev_metrics_and_unsupported_cells(script, regime, tmp_path, monkeypatch, transport):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    campaign = importlib.import_module(script)
    probe = importlib.import_module("probe_tfidf_classifier")
    bundle = probe.fixture_bundle()
    from llm_classifier_bench.class_definitions import ClassDefinitionProfile
    definitions = tmp_path / "definitions.json"
    ClassDefinitionProfile(dataset=bundle.name, profile="offline_fixture", classes=bundle.classes,
                           review_status="approved").write_json(definitions)
    monkeypatch.setattr(campaign, "get_dataset", lambda _: campaign.StaticDataset(bundle))
    monkeypatch.setattr(campaign, "utc_campaign_id", lambda: "jev-fixture")
    # Dynamic simulated answers preserve exactly the classes selected by the campaign.
    def post(endpoint, **kwargs):
        transport.calls.append((endpoint, kwargs))
        names = list(json.loads(kwargs["data"])["questions"]["classification"]["criteria"])
        return Response({"model": "jev-1.13.0", "usage": {"input_tokens": 300, "output_tokens": 28},
                         "answers": {"classification": {"type": "choice", "choice": names[1],
                           "confidence": .42, "probabilities": {names[1]: .75, names[0]: .25}}}})
    transport.post = post
    args = [script, "--classifiers", "jev", "--class-counts", "2", "--seeds", "17",
            "--train-per-class", "4", "--test-per-class", "1", "--validation-fraction", ".25",
            "--definitions", str(definitions), "--output-root", str(tmp_path),
            "--pricing", str(CARD), "--jev-max-requests", "2"]
    if regime == "matched":
        args += ["--matched-budgets", "0", "1", "--budget-unit", "per_class"]
    elif regime == "dry":
        args += ["--dry-run"]
    if script.endswith("_v2"):
        args += ["--min-train-per-class", "4", "--min-test-per-class", "1", "--strict-support"]
    monkeypatch.setattr(sys, "argv", args)
    campaign.main()
    root = tmp_path / "jev-fixture"
    rows = json.loads((root / "summary.json").read_text())
    if regime == "dry":
        assert [r["status"] for r in rows] == ["dry_run"]
        assert not transport.calls
        return
    assert [r["status"] for r in rows] == (["completed", "unsupported"] if regime == "matched" else ["completed"])
    row = rows[0]
    assert row["training_examples_used"] == row["validation_examples_used"] == 0
    for key in ("accuracy", "macro_f1", "top_label_ece", "adaptive_ece", "multiclass_log_loss",
                "multiclass_brier_score", "mean_latency_ms", "cost_per_1000_usd"):
        assert row[key] is not None
    assert json.loads((root / "campaign.json").read_text())["jev"]["max_requests"] == 2
    assert len(transport.calls) == 2


def test_label_limit_is_saved_as_unsupported(transport, tmp_path):
    from llm_classifier_bench.datasets import DatasetBundle
    bundle = build_bundle()
    many = DatasetBundle(bundle.name, bundle.classes + tuple(ClassDefinition(f"extra-{i}", "extra class") for i in range(254)),
                         bundle.train, bundle.test)
    result = run_benchmark(FakeDataset(many), JevClassifier(), BenchmarkRunConfig(output_root=tmp_path))
    assert json.loads(result.status_path.read_text())["status"] == "unsupported"
    plan = json.loads((result.run_dir / "inference_plan.json").read_text())
    assert not plan["supported"] and "255" in plan["reason"]
    assert not transport.calls


def test_rounded_probabilities_are_not_renormalized(transport):
    body = copy.deepcopy(FIXTURE["response"])
    body["answers"]["classification"]["probabilities"] = {"World": .25, "Sports": .74995}
    classifier = prepared(transport, body)
    prediction = classifier.predict(build_bundle().inputs()[:1])[0]
    assert prediction.probabilities == {"World": .25, "Sports": .74995}


@pytest.mark.parametrize("changes", [
    {"model": "jev-unknown"}, {"endpoint": "https://example.invalid/v1/systemone"},
    {"usage": {"input_tokens": True}}, {"usage": {"input_tokens": -1}},
    {"usage": {"output_tokens": 2}},
])
def test_unpriced_or_invalid_jev_usage_is_not_free(changes):
    entry = {"event_id": "e", "phase": "evaluation", "provider": "jev", "kind": "api_attempt",
             "endpoint": "https://api.typesafe.ai/v1/systemone", "model": "jev-1.13.0",
             "usage": {"input_tokens": 100, "output_tokens": 10}, **changes}
    assert price_entry(entry, load_pricing(CARD), None)["cost_usd"] is None


def test_alias_records_resolved_version_and_prices_it(transport):
    classifier = prepared(transport, model="jev-latest")
    entries = []
    classifier.set_usage_sink(entries.append)
    prediction = classifier.predict(build_bundle().inputs()[:1])[0]
    assert prediction.model == "jev-1.13.0"
    assert prediction.raw_response["requested_model"] == "jev-latest"
    assert price_entry(dict(entries[0], event_id="e", phase="evaluation"), load_pricing(CARD), None)["cost_usd"] == pytest.approx(.0000126)


def test_jev_rate_does_not_guess_between_account_tiers():
    card = load_pricing(CARD)
    alternative = copy.deepcopy(card["api_rates"][-1])
    alternative.update(service_tiers=["custom"], input_usd_per_million="0.01")
    card["api_rates"].append(alternative)
    entry = {"event_id": "e", "phase": "evaluation", "provider": "jev", "kind": "api_attempt",
             "endpoint": "https://api.typesafe.ai/v1/systemone", "model": "jev-1.13.0",
             "usage": {"input_tokens": 100, "output_tokens": 10}}
    assert price_entry(entry, card, None)["cost_usd"] is None
