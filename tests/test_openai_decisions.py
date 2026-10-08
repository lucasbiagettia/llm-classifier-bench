"""Offline Decisions contract and integration with existing runner/metrics/costs."""
import copy
import json
from pathlib import Path

import pytest
import requests

from llm_classifier_bench.budgets import MatchedBudgetConfig
from llm_classifier_bench.classifiers import Classifier, OpenAIDecisionsClassifier
from llm_classifier_bench.classifiers.openai_decisions import OpenAIDecisionsAPIError, OpenAIDecisionsRefusalError
from llm_classifier_bench.core import ClassificationInput
from llm_classifier_bench.costs import load_pricing, price_entry
from llm_classifier_bench.measurement import MeasurementConfig
from llm_classifier_bench.metrics.evaluator import evaluate_jsonl, results_as_dict
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark
from test_runner import FakeDataset, build_bundle

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "pricing/decisions_2026-10-07.json"
BODY = {"model": "gpt-6-luna", "answers": [{"name": "classification", "type": "choice",
        "choice": "Sports", "confidence": .42,
        "probabilities": [{"value": "Sports", "probability": .75}, {"value": "World", "probability": .25}]}],
        "usage": {"input_tokens": 300, "input_tokens_details": {"cached_tokens": 120, "cache_write_tokens": 30},
                  "output_tokens": 0, "total_tokens": 300}}


class Response:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status
        self.headers = {"x-request-id": "fixture-request"}

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


class Session:
    def __init__(self):
        self.calls, self.outcomes = [], []

    def mount(self, *args, **kwargs):
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
    monkeypatch.setenv("OPENAI_API_KEY", "offline-secret")
    monkeypatch.delenv("OPENAI_DECISIONS_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_DECISIONS_TIMEOUT_S", raising=False)
    monkeypatch.setattr("llm_classifier_bench.classifiers.openai_decisions.load_dotenv", lambda **kw: None)
    monkeypatch.setattr("llm_classifier_bench.classifiers.openai_decisions.requests.Session", lambda: session)
    return session


def prepared(transport, body=None, **kwargs):
    transport.outcomes.append(Response(copy.deepcopy(BODY if body is None else body)))
    classifier = OpenAIDecisionsClassifier(**kwargs)
    classifier.prepare(build_bundle().classes)
    classifier.fit(build_bundle().train)
    return classifier


def test_choice_request_and_normalization_use_only_input_and_frozen_definitions(transport):
    classifier = prepared(transport)
    p = classifier.predict([ClassificationInput("s", "Text to classify", {"gold_label": "secret-target"})])[0]
    assert isinstance(classifier, Classifier)
    endpoint, call = transport.calls[0]
    body = json.loads(call["data"])
    assert endpoint == "https://api.openai.com/v1/decisions"
    assert body["model"] == "gpt-6-luna" and body["input"] == "Text to classify"
    question, = body["questions"]
    assert question["type"] == "choice" and question["name"] == "classification"
    assert question["choices"] == [{"value": c.name, "description": c.description} for c in build_bundle().classes]
    assert set(body) == {"input", "model", "questions"}
    assert call["timeout"] == 120 and call["allow_redirects"] is False
    assert p.sample_id == "s" and p.predicted_label == "Sports"
    assert p.probabilities == {"World": .25, "Sports": .75}
    assert list(p.probabilities) == ["World", "Sports"]
    assert p.confidence == .75 and p.raw_response["provider_confidence"] == .42
    assert p.request_id == "fixture-request" and p.model == "gpt-6-luna"
    assert classifier.fitted_metadata()["total_labeled_examples_used"] == 0
    assert "offline-secret" not in json.dumps(p.raw_response)
    assert "secret-target" not in call["data"].decode()


@pytest.mark.parametrize("mutation", [
    lambda b: b.pop("model"), lambda b: b.update(model=""), lambda b: b.pop("answers"),
    lambda b: b.update(answers={}), lambda b: b["answers"].append(b["answers"][0]),
    lambda b: b["answers"][0].update(type="refusal"), lambda b: b["answers"][0].update(type="score"),
    lambda b: b["answers"][0].update(name="another_question"),
    lambda b: b["answers"][0].update(choice="Unknown"), lambda b: b["answers"][0].update(choice="World"),
    lambda b: b["answers"][0].pop("probabilities"),
    lambda b: b["answers"][0].update(probabilities={"World": .25, "Sports": .75}),
    lambda b: b["answers"][0]["probabilities"].pop(),
    lambda b: b["answers"][0]["probabilities"][0].update(value="World"),
    lambda b: b["answers"][0]["probabilities"][0].update(value=True),
    lambda b: b["answers"][0]["probabilities"][0].update(value="unknown"),
    lambda b: b["answers"][0]["probabilities"][0].update(probability=.1),
    lambda b: b["answers"][0]["probabilities"][0].update(probability=float("nan")),
    lambda b: b["answers"][0]["probabilities"][0].update(probability=float("inf")),
    lambda b: b["answers"][0]["probabilities"][0].update(probability=-.1),
    lambda b: b["answers"][0]["probabilities"][0].update(probability=1.1),
    lambda b: b["answers"][0]["probabilities"][0].update(probability="0.75"),
    lambda b: b["answers"][0]["probabilities"][0].update(probability=True),
    lambda b: b["answers"][0].update(confidence=float("inf")),
])
def test_invalid_answer_keeps_billed_usage(transport, mutation):
    body = copy.deepcopy(BODY)
    mutation(body)
    classifier = prepared(transport, body)
    entries = []
    classifier.set_usage_sink(entries.append)
    with pytest.raises(ValueError):
        classifier.predict(build_bundle().inputs()[:1])
    assert len(entries) == 1 and entries[0]["status"] == "failed"
    assert entries[0]["usage"]["input_tokens"] == 300
    expected_cost = .00003 if body.get("model") == "gpt-6-luna" else None
    assert price_entry(dict(entries[0], event_id="e", phase="evaluation"), load_pricing(CARD), None)["cost_usd"] == expected_cost


def test_ties_rounding_and_native_confidence_are_preserved(transport):
    body = copy.deepcopy(BODY)
    body["answers"][0]["probabilities"] = [{"value": "World", "probability": .5}, {"value": "Sports", "probability": .5}]
    classifier = prepared(transport, body)
    p = classifier.predict(build_bundle().inputs()[:1])[0]
    assert p.predicted_label == "Sports" and p.raw_response["tied_maximum_labels"] == ["World", "Sports"]
    body["answers"][0]["probabilities"] = [{"value": "Sports", "probability": .74995}, {"value": "World", "probability": .25}]
    body["answers"][0].pop("confidence")
    classifier = prepared(transport, body)
    p = classifier.predict(build_bundle().inputs()[:1])[0]
    assert p.confidence == .74995 and p.probabilities["World"] == .25
    assert p.raw_response["provider_confidence"] is None


def test_runner_metrics_and_input_only_cost_replay(transport, tmp_path):
    transport.outcomes = [Response(BODY) for _ in range(3)]
    result = run_benchmark(FakeDataset(build_bundle()), OpenAIDecisionsClassifier(), BenchmarkRunConfig(
        output_root=tmp_path, pricing_path=CARD, measurement=MeasurementConfig(warmup_examples=1)))
    metrics = results_as_dict(evaluate_jsonl(result.predictions_path))
    for key in ("accuracy", "macro_f1", "top_label_ece", "adaptive_ece", "multiclass_log_loss",
                "multiclass_brier_score", "mean_latency_ms", "latency_p50_ms", "total_cost_usd", "cost_per_1000_usd"):
        assert metrics[key]["available"]
    assert metrics["total_cost_usd"]["value"] == pytest.approx(.00009)
    costs = json.loads((result.run_dir / "cost_report.json").read_text())
    assert costs["coverage_complete"] and costs["total_cost_usd"] == pytest.approx(.00009)
    assert costs["cost_per_1000_successful_examples_usd"] == pytest.approx(.045)
    assert "offline-secret" not in "".join(p.read_text() for p in result.run_dir.glob("*.json*"))


def test_refusal_is_distinct_from_http_failure_and_has_sample_request_evidence(transport, tmp_path):
    refusal = copy.deepcopy(BODY)
    refusal["answers"] = [{"type": "refusal", "name": "classification"}]
    transport.outcomes = [Response(BODY), Response(refusal)]
    with pytest.raises(OpenAIDecisionsRefusalError) as caught:
        run_benchmark(FakeDataset(build_bundle()), OpenAIDecisionsClassifier(), BenchmarkRunConfig(
            output_root=tmp_path, run_id="refused", pricing_path=CARD))
    assert caught.value.sample_id == "test-2" and caught.value.request_id == "fixture-request"
    assert "not an HTTP rate-limit or overload error" in str(caught.value)
    assert len(transport.calls) == 2
    root = tmp_path / "refused"
    attempts = [json.loads(line) for line in (root / "usage.jsonl").read_text().splitlines()
                if json.loads(line)["kind"] == "api_attempt"]
    assert [e["decision_answer_type"] for e in attempts] == ["choice", "refusal"]
    assert attempts[-1]["http_status"] == 200 and attempts[-1]["status"] == "failed"
    assert attempts[-1]["error_type"] == "OpenAIDecisionsRefusalError"
    status = json.loads((root / "status.json").read_text())
    assert status["error_type"] == "OpenAIDecisionsRefusalError"
    assert "test-2" in status["error_message"] and "fixture-request" in status["error_message"]
    assert len((root / "predictions.jsonl").read_text().splitlines()) == 1
    costs = json.loads((root / "cost_report.json").read_text())
    assert costs["total_cost_usd"] == pytest.approx(.00006)
    assert not (root / "metrics.json").exists()


@pytest.mark.parametrize("outcome", [Response({}, 429), Response({}, 401), Response({}, 302),
                                     Response(ValueError("bad JSON")), requests.Timeout("secret details")])
def test_errors_are_not_retried_and_preserve_prior_predictions(transport, tmp_path, outcome):
    transport.outcomes = [Response(BODY), outcome]
    with pytest.raises((ValueError, OpenAIDecisionsAPIError)):
        run_benchmark(FakeDataset(build_bundle()), OpenAIDecisionsClassifier(), BenchmarkRunConfig(
            output_root=tmp_path, run_id="failed", pricing_path=CARD))
    assert len(transport.calls) == 2
    root = tmp_path / "failed"
    assert len((root / "predictions.jsonl").read_text().splitlines()) == 1
    assert json.loads((root / "status.json").read_text())["status"] == "failed"
    assert not (root / "metrics.json").exists()
    assert "secret details" not in "".join(p.read_text() for p in root.glob("*.json*"))


@pytest.mark.parametrize("budget", [MatchedBudgetConfig(1, "per_class"), MatchedBudgetConfig(0, "total", validation_examples=1)])
def test_nonzero_matched_labels_are_unsupported_before_network(transport, tmp_path, budget):
    result = run_benchmark(FakeDataset(build_bundle()), OpenAIDecisionsClassifier(), BenchmarkRunConfig(
        output_root=tmp_path, matched_budget=budget))
    assert json.loads(result.status_path.read_text())["status"] == "unsupported"
    assert not transport.calls


def test_zero_budget_dry_run_and_configuration_need_no_api_key(transport, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("OPENAI_DECISIONS_MODEL", "operator-model")
    monkeypatch.setenv("OPENAI_DECISIONS_TIMEOUT_S", "45")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1/")
    classifier = OpenAIDecisionsClassifier()
    assert classifier.model == "operator-model" and classifier.timeout_s == 45
    assert classifier.endpoint == "https://example.invalid/v1/decisions"
    result = run_benchmark(FakeDataset(build_bundle()), classifier, BenchmarkRunConfig(
        output_root=tmp_path, dry_run=True, matched_budget=MatchedBudgetConfig(0, "per_class")))
    assert json.loads(result.status_path.read_text())["status"] == "dry_run"
    assert not transport.calls


@pytest.mark.parametrize("changes", [{"model": "unpriced-snapshot"}, {"endpoint": "https://example.invalid/v1/decisions"},
                                     {"usage": None}, {"usage": {"input_tokens": True}},
                                     {"usage": {"input_tokens": -1}}])
def test_unknown_prices_and_usage_stay_unavailable(changes):
    entry = {"event_id": "e", "phase": "evaluation", "kind": "api_attempt", "provider": "openai",
             "billing_basis": "all_input_tokens", "model": "gpt-6-luna",
             "endpoint": "https://api.openai.com/v1/decisions", "usage": BODY["usage"], **changes}
    assert price_entry(entry, load_pricing(CARD), None)["cost_usd"] is None


@pytest.mark.parametrize("kwargs", [{"timeout_s": 0}, {"timeout_s": True}, {"timeout_s": float("nan")},
                                    {"model": ""}, {"base_url": "http://example.com"},
                                    {"base_url": "https://user:secret@example.com/v1"}])
def test_invalid_configuration_rejected(transport, kwargs):
    with pytest.raises(ValueError):
        OpenAIDecisionsClassifier(**kwargs)
