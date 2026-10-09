"""Zero-shot choice classification using the Perplexity Decisions HTTP API."""
from __future__ import annotations

import hashlib
import json
import math
import os
from time import perf_counter
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

from llm_classifier_bench.budgets import UnsupportedConfiguration
from llm_classifier_bench.classifiers.base import Prediction
from llm_classifier_bench.core import PROBABILITY_SUM_TOLERANCE
from llm_classifier_bench.costs import capture_api_usage, capture_response

DEFAULT_DECISIONS_MODEL = "pplx-decider-v1.1-27b"
DEFAULT_DECISIONS_BASE_URL = "https://api.perplexity.ai/v1"
DECISIONS_SOURCE = "https://docs.perplexity.ai/docs/decisions/quickstart"


class PerplexityDecisionsAPIError(RuntimeError):
    """Sanitized transport failure; never copy provider bodies into artifacts."""

    def __init__(self, status_code=None):
        self.status_code = status_code
        super().__init__("Perplexity Decisions transport failure" if status_code is None
                         else f"Perplexity Decisions HTTP {status_code}")


class PerplexityDecisionsClassifier:
    supervision_regime = "zero_shot"
    training_examples_used = 0
    validation_examples_used = 0
    context_examples_used = 0
    total_labeled_examples_used = 0

    def __init__(self, *, model=None, api_key=None, base_url=None, timeout_s=None,
                 session=None, classifier_name="perplexity-decisions-zero-shot"):
        load_dotenv(override=False)
        self.model = model if model is not None else os.getenv("PERPLEXITY_DECISIONS_MODEL", DEFAULT_DECISIONS_MODEL)
        base_url = base_url if base_url is not None else os.getenv("PERPLEXITY_BASE_URL", DEFAULT_DECISIONS_BASE_URL)
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Perplexity Decisions base_url must be an HTTPS URL without credentials, query or fragment")
        self.endpoint = base_url.rstrip("/") + "/decisions"
        timeout_s = timeout_s if timeout_s is not None else float(os.getenv("PERPLEXITY_DECISIONS_TIMEOUT_S", "30"))
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be nonempty")
        if not isinstance(classifier_name, str) or not classifier_name.strip():
            raise ValueError("classifier_name must be nonempty")
        self.timeout_s = timeout_s
        self._name = classifier_name
        self._api_key = api_key
        self._session = session
        self._injected_session = session is not None
        self._classes = ()
        self._usage_sink = None

    @property
    def name(self):
        return self._name

    def _check_classes(self, classes):
        if len(classes) < 2 or len({c.name for c in classes}) != len(classes):
            raise ValueError("At least two uniquely named classes are required")
        if len(classes) > 255:
            raise UnsupportedConfiguration("perplexity_choice_limit: at most 255 labels")

    def preflight(self, classes, inputs, *, matched_budget=None):
        reason = None
        try:
            self._check_classes(classes)
        except UnsupportedConfiguration as exc:
            reason = str(exc)
        if matched_budget is not None and (matched_budget.examples or matched_budget.validation_examples):
            reason = "perplexity_zero_shot_only: nonzero matched fit or validation budgets are unsupported"
        largest = max((len(self._encoded(self._payload(classes, e.text))) for e in inputs), default=0)
        if largest > 32 * 1024 * 1024:
            reason = "perplexity_body_limit: request exceeds 32 MiB"
        elif largest + 1024 >= 262144:
            reason = "perplexity_input_limit: conservative UTF-8 byte estimate plus 1024 framing allowance"
        return {"supported": reason is None, "reason": reason, "requested_model": self.model,
                "endpoint": self.endpoint, "question_type": "choice", "question_count": 1,
                "class_count": len(classes), "planned_requests": len(inputs),
                "max_request_bytes": largest, "estimate_policy": "utf8_bytes_plus_1024_v1"}

    def plan_fit(self, classes, examples, *, validation_examples=()):
        self._check_classes(classes)
        return {"live_supported": True, "blocker": None, "planned_remote_operations": [],
                "selection": {"selected_examples": [], "requested_total": 0, "actual_total": 0,
                              "covered_class_count": 0, "class_count": len(classes)},
                **self.fitted_metadata()}

    def prepare(self, classes):
        frozen = tuple(classes)
        self._check_classes(frozen)
        self._api_key = self._api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self._api_key or not self._api_key.strip() or self._api_key.strip() in {"replace_me", "your_api_key_here"}:
            raise ValueError("PERPLEXITY_API_KEY is required for PerplexityDecisionsClassifier")
        if self._session is None:
            self._session = requests.Session()
            self._session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
        self._classes = frozen

    def fit(self, examples, *, validation_examples=()):
        if not self._classes:
            raise RuntimeError("Call prepare before fit")

    def fitted_metadata(self):
        return {"supervision_regime": "zero_shot", "supported_supervision_modes": ["zero_shot"],
                "training_examples_used": 0, "validation_examples_used": 0,
                "context_examples_used": 0, "total_labeled_examples_used": 0,
                "selected_sample_ids": [], "requested_model": self.model, "parameter_training": False}

    def preparation_metadata(self):
        return {"backend": "hosted_api", "remote_preparation_performed": False}

    def inference_metadata(self):
        return {"backend": "hosted_api", "batch_execution": "sequential_single_example",
                "endpoint": self.endpoint, "requested_model": self.model,
                "timeout_s": self.timeout_s, "transport_max_retries": 0,
                "transport_attempts_complete": not self._injected_session, "server_hardware": None,
                "question_type": "choice", "question_count": 1,
                "confidence_semantics": "probability_of_predicted_label; native confidence stored separately",
                "tie_policy": "preserve_provider_choice_if_exact_maximum",
                "limits": {"choice_options": 255, "input_tokens_exclusive": 262144,
                           "request_bytes": 32 * 1024 * 1024},
                "api_contract_source": DECISIONS_SOURCE}

    def set_usage_sink(self, sink):
        self._usage_sink = sink

    def _payload(self, classes, text):
        return {"model": self.model, "state": text, "questions": {"classification": {
            "type": "choice",
            "instructions": "Classify the input text into exactly one of the allowed labels. "
                            "Use the class descriptions as the decision rubric. Do not invent labels.",
            "criteria": {c.name: c.description for c in classes},
        }}}

    @staticmethod
    def _encoded(payload):
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")

    def predict(self, examples):
        if not self._classes or self._session is None:
            raise RuntimeError("Call prepare before predict")
        plan = self.preflight(self._classes, examples)
        if not plan["supported"]:
            raise UnsupportedConfiguration(plan["reason"])
        return [self._predict_one(e) for e in examples]

    def _predict_one(self, example):
        encoded = self._encoded(self._payload(self._classes, example.text))
        digest = hashlib.sha256(encoded).hexdigest()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}
        started = perf_counter()
        with capture_api_usage(self._usage_sink, provider="perplexity", sample_id=example.sample_id,
                               requested_model=self.model,
                               transport_attempts_complete=not self._injected_session) as usage:
            usage.update(endpoint=self.endpoint, billing_basis="all_input_tokens",
                         request_sha256=digest, request_bytes=len(encoded), attempt=1)
            try:
                response = self._session.post(self.endpoint, data=encoded, headers=headers,
                                              timeout=self.timeout_s, allow_redirects=False)
            except requests.RequestException:
                raise PerplexityDecisionsAPIError() from None
            usage.update(response_received=True, http_status=response.status_code,
                         request_id=response.headers.get("x-request-id"))
            try:
                body = response.json()
            except ValueError:
                body = None
            capture_response(usage, body)
            if not 200 <= response.status_code < 300:
                raise PerplexityDecisionsAPIError(response.status_code)
            return self._normalize(body, example.sample_id, (perf_counter() - started) * 1000,
                                   usage["request_id"], digest)

    def _normalize(self, body, sample_id, latency_ms, request_id, digest):
        if not isinstance(body, dict) or not isinstance(body.get("model"), str) or not body["model"].strip():
            raise ValueError("Perplexity Decisions response requires a resolved model")
        answers = body.get("answers")
        if not isinstance(answers, dict) or set(answers) != {"classification"}:
            raise ValueError("Perplexity Decisions requires exactly one named classification answer")
        answer = answers["classification"]
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("Perplexity Decisions requires a choice answer")
        names = [c.name for c in self._classes]
        label = answer.get("choice")
        if not isinstance(label, str) or label not in names:
            raise ValueError("Perplexity Decisions choice must be a configured class")
        entries = answer.get("probabilities")
        if not isinstance(entries, dict) or set(entries) != set(names):
            raise ValueError("Perplexity Decisions probabilities must contain exactly the configured labels")
        probabilities = {name: _probability(entries[name]) for name in names}
        if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=PROBABILITY_SUM_TOLERANCE):
            raise ValueError("Perplexity Decisions probabilities do not sum to one")
        if probabilities[label] != max(probabilities.values()):
            raise ValueError("Perplexity Decisions choice is not a maximum-probability class")
        native = _probability(answer["confidence"]) if answer.get("confidence") is not None else None
        return Prediction(sample_id, label, probabilities[label], probabilities, latency_ms,
                          model=body["model"], request_id=request_id,
                          raw_response={"requested_model": self.model, "resolved_model": body["model"],
                                        "provider_choice": label, "provider_confidence": native,
                                        "provider_confidence_unavailable_reason": "not reported" if native is None else None,
                                        "request_sha256": digest,
                                        "tied_maximum_labels": [n for n in names if probabilities[n] == probabilities[label]]})


def _probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Perplexity Decisions probability/confidence must be finite and in [0,1]")
    return float(value)
