"""Offline API contract, probability validation and runner cost accounting."""
import copy
import json
from pathlib import Path

import pytest
import requests

from llm_classifier_bench.budgets import MatchedBudgetConfig, UnsupportedConfiguration
from llm_classifier_bench.classifiers import Classifier, PerplexityDecisionsClassifier
from llm_classifier_bench.classifiers.perplexity_decisions import PerplexityDecisionsAPIError
from llm_classifier_bench.core import ClassificationInput, ClassDefinition
from llm_classifier_bench.costs import load_pricing, price_entry
from llm_classifier_bench.measurement import MeasurementConfig
from llm_classifier_bench.metrics.evaluator import evaluate_jsonl, results_as_dict
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark
from test_runner import FakeDataset, build_bundle

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "pricing/perplexity_decisions_2026-10-09.json"
BODY = {"model": "pplx-decider-v1.1-27b", "answers": {"classification": {
    "type": "choice", "choice": "Sports", "confidence": .42,
    "probabilities": {"Sports": .75, "World": .25}}},
    "usage": {"input_tokens": 300, "output_tokens": 1}}


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
        self.calls, self.outcomes, self.mounts = [], [], []

    def mount(self, *args, **kwargs):
        self.mounts.append((args, kwargs))

    def post(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        result = self.outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def transport(monkeypatch):
    session = Session()
    monkeypatch.setenv("PERPLEXITY_API_KEY", "offline-secret")
    for key in ("PERPLEXITY_DECISIONS_MODEL", "PERPLEXITY_BASE_URL", "PERPLEXITY_DECISIONS_TIMEOUT_S"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("llm_classifier_bench.classifiers.perplexity_decisions.load_dotenv", lambda **kw: None)
    monkeypatch.setattr("llm_classifier_bench.classifiers.perplexity_decisions.requests.Session", lambda: session)
    monkeypatch.setattr("socket.create_connection", lambda *a, **kw: pytest.fail("No network allowed"))
    return session


def prepared(transport, body=None, **kwargs):
    transport.outcomes.append(Response(copy.deepcopy(BODY if body is None else body)))
    classifier = PerplexityDecisionsClassifier(**kwargs)
    classifier.prepare(build_bundle().classes)
    classifier.fit(build_bundle().train)
    return classifier


def test_native_contract_preserves_text_rubric_and_separate_confidence(transport, monkeypatch):
    monkeypatch.setenv("OPENAI_ORG_ID", "do-not-forward")
    classifier = prepared(transport)
    prediction, = classifier.predict([ClassificationInput("s", "  Text á\n", {"gold_label": "secret-target"})])
    assert isinstance(classifier, Classifier)
    endpoint, call = transport.calls[0]
    payload = json.loads(call["data"])
    assert endpoint == "https://api.perplexity.ai/v1/decisions"
    assert set(payload) == {"state", "model", "questions"}
    assert payload["state"] == "  Text á\n" and payload["model"] == "pplx-decider-v1.1-27b"
    assert payload["questions"]["classification"]["type"] == "choice"
    assert payload["questions"]["classification"]["criteria"] == {c.name: c.description for c in build_bundle().classes}
    assert set(call["headers"]) == {"Authorization", "Content-Type"}
    assert call["headers"]["Authorization"] == "Bearer offline-secret"
    assert call["timeout"] == 30 and call["allow_redirects"] is False
    assert transport.mounts[0][0][1].max_retries.total == 0
    assert prediction.probabilities == {"World": .25, "Sports": .75}
    assert list(prediction.probabilities) == ["World", "Sports"]
    assert prediction.confidence == .75 and prediction.raw_response["provider_confidence"] == .42
    assert prediction.request_id == "fixture-request" and prediction.sample_id == "s"
    assert classifier.fitted_metadata()["total_labeled_examples_used"] == 0
    assert "secret-target" not in call["data"].decode()


@pytest.mark.parametrize("mutation", [
    lambda b: b.pop("model"), lambda b: b.update(model=""), lambda b: b.update(answers=[]),
    lambda b: b["answers"].update(extra={}), lambda b: b["answers"].update(classification=None),
    lambda b: b["answers"]["classification"].update(type="noul"),
    lambda b: b["answers"]["classification"].update(choice="unknown"),
    lambda b: b["answers"]["classification"].update(choice="World"),
    lambda b: b["answers"]["classification"].update(probabilities=[]),
    lambda b: b["answers"]["classification"]["probabilities"].pop("World"),
    lambda b: b["answers"]["classification"]["probabilities"].update(unknown=0),
    lambda b: b["answers"]["classification"]["probabilities"].update(Sports=.1),
    lambda b: b["answers"]["classification"]["probabilities"].update(Sports=float("nan")),
    lambda b: b["answers"]["classification"]["probabilities"].update(Sports=float("inf")),
    lambda b: b["answers"]["classification"]["probabilities"].update(Sports=-.1),
    lambda b: b["answers"]["classification"]["probabilities"].update(Sports=1.1),
    lambda b: b["answers"]["classification"]["probabilities"].update(Sports=True),
    lambda b: b["answers"]["classification"]["probabilities"].update(Sports=".75"),
    lambda b: b["answers"]["classification"].update(confidence=float("inf")),
])
def test_invalid_answers_retain_usage_without_inventing_predictions(transport, mutation):
    body = copy.deepcopy(BODY)
    mutation(body)
    classifier = prepared(transport, body)
    ledger = []
    classifier.set_usage_sink(ledger.append)
    with pytest.raises(ValueError):
        classifier.predict(build_bundle().inputs()[:1])
    assert len(ledger) == 1 and ledger[0]["status"] == "failed"
    assert ledger[0]["usage"]["input_tokens"] == 300
    cost = price_entry(dict(ledger[0], event_id="e", phase="evaluation"), load_pricing(CARD), None)
    assert cost["cost_usd"] == (pytest.approx(.000006) if body.get("model") else None)


def test_ties_rounding_and_missing_native_confidence(transport):
    body = copy.deepcopy(BODY)
    answer = body["answers"]["classification"]
    answer["probabilities"] = {"World": .5, "Sports": .5}
    answer.pop("confidence")
    prediction, = prepared(transport, body).predict(build_bundle().inputs()[:1])
    assert prediction.predicted_label == "Sports" and prediction.confidence == .5
    assert prediction.raw_response["tied_maximum_labels"] == ["World", "Sports"]
    assert prediction.raw_response["provider_confidence"] is None
    answer["probabilities"] = {"World": .25, "Sports": .74995}
    prediction, = prepared(transport, body).predict(build_bundle().inputs()[:1])
    assert sum(prediction.probabilities.values()) != 1  # preserve rounded provider values


@pytest.mark.parametrize("outcome", [Response({}, 401), Response({}, 429),
    Response(ValueError("<html>private text</html>"), 504), requests.Timeout("private text")])
def test_transport_failures_are_sanitized_and_never_retried(transport, outcome):
    classifier = prepared(transport)
    transport.outcomes = [outcome]
    ledger = []
    classifier.set_usage_sink(ledger.append)
    with pytest.raises(PerplexityDecisionsAPIError) as error:
        classifier.predict(build_bundle().inputs()[:1])
    assert "private text" not in str(error.value)
    assert len(transport.calls) == len(ledger) == 1 and ledger[0]["status"] == "failed"


def test_preflight_limits_and_zero_shot_budget_are_local(transport):
    classifier = PerplexityDecisionsClassifier()
    classes = build_bundle().classes
    for budget in (MatchedBudgetConfig(1, "per_class"), MatchedBudgetConfig(0, "per_class", validation_examples=1)):
        assert not classifier.preflight(classes, [], matched_budget=budget)["supported"]
    too_many = [ClassDefinition(str(i), "description") for i in range(256)]
    assert not classifier.preflight(too_many, [])["supported"]
    with pytest.raises(UnsupportedConfiguration):
        classifier.prepare(too_many)
    assert not classifier.preflight(classes, [ClassificationInput("large", "x" * 262144)])["supported"]
    with pytest.raises(ValueError):
        classifier.prepare((classes[0], classes[0]))
    assert not transport.calls


@pytest.mark.parametrize("kwargs", [{"timeout_s": 0}, {"timeout_s": True}, {"timeout_s": float("nan")},
    {"model": ""}, {"classifier_name": None}, {"base_url": "http://example.com"},
    {"base_url": "https://user:password@example.com/v1"}, {"base_url": "https://example.com/v1?q=1"}])
def test_invalid_configuration(transport, kwargs):
    with pytest.raises(ValueError):
        PerplexityDecisionsClassifier(**kwargs)


def test_environment_and_explicit_configuration(transport, monkeypatch):
    monkeypatch.setenv("PERPLEXITY_DECISIONS_MODEL", "pplx-decider-v1-27b")
    monkeypatch.setenv("PERPLEXITY_DECISIONS_TIMEOUT_S", "45")
    classifier = PerplexityDecisionsClassifier()
    assert classifier.model == "pplx-decider-v1-27b" and classifier.timeout_s == 45
    assert PerplexityDecisionsClassifier(model="override", timeout_s=10).model == "override"
    monkeypatch.delenv("PERPLEXITY_API_KEY")
    with pytest.raises(ValueError, match="PERPLEXITY_API_KEY"):
        PerplexityDecisionsClassifier(session=transport).prepare(build_bundle().classes)


def test_runner_metrics_usage_cost_and_failed_call_accounting(transport, tmp_path):
    transport.outcomes = [Response(BODY) for _ in range(3)]
    result = run_benchmark(FakeDataset(build_bundle()), PerplexityDecisionsClassifier(), BenchmarkRunConfig(
        output_root=tmp_path, pricing_path=CARD, measurement=MeasurementConfig(warmup_examples=1)))
    metrics = results_as_dict(evaluate_jsonl(result.predictions_path))
    for key in ("accuracy", "macro_f1", "top_label_ece", "adaptive_ece", "multiclass_log_loss",
                "multiclass_brier_score", "mean_latency_ms", "total_cost_usd", "cost_per_1000_usd"):
        assert metrics[key]["available"]
    assert metrics["total_cost_usd"]["value"] == pytest.approx(.000018)
    assert "offline-secret" not in "".join(p.read_text() for p in result.run_dir.glob("*.json*"))
    bad_body = copy.deepcopy(BODY)
    bad_body["answers"]["classification"]["choice"] = "unknown"
    transport.outcomes = [Response(BODY), Response(bad_body)]
    with pytest.raises(ValueError):
        run_benchmark(FakeDataset(build_bundle()), PerplexityDecisionsClassifier(), BenchmarkRunConfig(
            output_root=tmp_path, run_id="failed", pricing_path=CARD))
    assert len((tmp_path / "failed/predictions.jsonl").read_text().splitlines()) == 1
    cost = json.loads((tmp_path / "failed/cost_report.json").read_text())
    assert cost["total_cost_usd"] == pytest.approx(.000012)


@pytest.mark.parametrize("model, endpoint, usage", [("unknown", "https://api.perplexity.ai/v1/decisions", {"input_tokens": 300}),
    (BODY["model"], "https://example.com/decisions", {"input_tokens": 300}),
    (BODY["model"], "https://api.perplexity.ai/v1/decisions", {})])
def test_unknown_billing_stays_unavailable(model, endpoint, usage):
    entry = dict(event_id="e", phase="evaluation", kind="api_attempt", provider="perplexity",
                 model=model, endpoint=endpoint, billing_basis="all_input_tokens", usage=usage)
    assert price_entry(entry, load_pricing(CARD), None)["cost_usd"] is None
