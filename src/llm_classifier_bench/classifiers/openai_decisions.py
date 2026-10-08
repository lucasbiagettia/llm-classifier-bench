"""Zero-shot classification via OpenAI's typed Decisions choice primitive.

Uses the documented HTTP contract with requests, like the Jev adapter, so the
existing OpenAI generative baseline does not require an SDK migration.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from time import perf_counter
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

from llm_classifier_bench.classifiers.base import Prediction
from llm_classifier_bench.costs import capture_api_usage, capture_response
from llm_classifier_bench.core import PROBABILITY_SUM_TOLERANCE

DEFAULT_DECISIONS_MODEL = "gpt-6-luna"
DEFAULT_DECISIONS_BASE_URL = "https://api.openai.com/v1"
DECISIONS_SOURCE = "https://developers.openai.com/api/docs/guides/decisions"


class OpenAIDecisionsAPIError(RuntimeError):
    """Sanitized failure: provider error bodies may contain user input."""

    def __init__(self, status_code=None):
        self.status_code = status_code
        super().__init__("OpenAI Decisions transport failure" if status_code is None
                         else f"OpenAI Decisions HTTP {status_code}")


class OpenAIDecisionsRefusalError(ValueError):
    """A successful API response declined to provide a label/distribution."""

    def __init__(self, sample_id, request_id):
        self.sample_id = sample_id
        self.request_id = request_id
        super().__init__(
            f"OpenAI Decisions refused classification for sample {sample_id} "
            f"(request_id={request_id or 'unavailable'}). The API returned a refusal "
            "without a label or probabilities; this is not an HTTP rate-limit or "
            "overload error. No automatic retry was made. Usage and earlier "
            "predictions are retained."
        )


class OpenAIDecisionsClassifier:
    supervision_regime = "zero_shot"
    training_examples_used = 0
    validation_examples_used = 0
    context_examples_used = 0
    total_labeled_examples_used = 0

    def __init__(self, *, model=None, api_key=None, base_url=None, timeout_s=None,
                 session=None, classifier_name="openai-decisions-zero-shot"):
        load_dotenv(override=False)
        self.model = model if model is not None else os.getenv("OPENAI_DECISIONS_MODEL", DEFAULT_DECISIONS_MODEL)
        base_url = base_url if base_url is not None else os.getenv("OPENAI_BASE_URL", DEFAULT_DECISIONS_BASE_URL)
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("OpenAI Decisions base_url must be an HTTPS URL without credentials, query or fragment")
        self.endpoint = base_url.rstrip("/") + "/decisions"
        timeout_s = timeout_s if timeout_s is not None else float(os.getenv("OPENAI_DECISIONS_TIMEOUT_S", "120"))
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        if not isinstance(self.model, str) or not self.model.strip() or not classifier_name.strip():
            raise ValueError("model and classifier_name must be nonempty")
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

    def preflight(self, classes, inputs, *, matched_budget=None):
        reason = None
        if matched_budget is not None and (matched_budget.examples or matched_budget.validation_examples):
            reason = "zero_shot_only: Decisions does not consume matched training or validation labels"
        return {"supported": reason is None, "reason": reason, "requested_model": self.model,
                "endpoint": self.endpoint, "question_type": "choice", "question_count": 1,
                "class_count": len(classes), "planned_requests": len(inputs)}

    def prepare(self, classes):
        frozen = tuple(classes)
        if len(frozen) < 2 or len({c.name for c in frozen}) != len(frozen):
            raise ValueError("At least two uniquely named classes are required")
        self._classes = frozen
        self._api_key = self._api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key and self._session is None:
            raise ValueError("OPENAI_API_KEY is required for OpenAIDecisionsClassifier")
        if self._session is None:
            self._session = requests.Session()
            self._session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))

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
                "api_contract_source": DECISIONS_SOURCE}

    def set_usage_sink(self, sink):
        self._usage_sink = sink

    def _payload(self, example):
        return {"model": self.model, "input": example.text, "questions": [{
            "type": "choice", "name": "classification",
            "instructions": "Classify the input text into exactly one of the allowed labels. "
                            "Use the class descriptions as the decision rubric. Do not invent labels.",
            "choices": [{"value": c.name, "description": c.description} for c in self._classes],
        }]}

    def predict(self, examples):
        if not self._classes or self._session is None:
            raise RuntimeError("Call prepare before predict")
        return [self._predict_one(e) for e in examples]

    def _predict_one(self, example):
        encoded = json.dumps(self._payload(example), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        for variable, header in (("OPENAI_ORG_ID", "OpenAI-Organization"), ("OPENAI_PROJECT_ID", "OpenAI-Project")):
            if os.getenv(variable):
                headers[header] = os.environ[variable]
        started = perf_counter()
        with capture_api_usage(self._usage_sink, provider="openai", sample_id=example.sample_id,
                               requested_model=self.model,
                               transport_attempts_complete=not self._injected_session) as usage:
            usage.update(endpoint=self.endpoint, billing_basis="all_input_tokens",
                         request_sha256=digest, request_bytes=len(encoded), attempt=1)
            try:
                response = self._session.post(self.endpoint, data=encoded, headers=headers,
                                              timeout=self.timeout_s, allow_redirects=False)
            except requests.RequestException:
                raise OpenAIDecisionsAPIError() from None
            usage.update(response_received=True, http_status=response.status_code,
                         request_id=response.headers.get("x-request-id"))
            try:
                body = response.json()
            except ValueError:
                body = None
            capture_response(usage, body)
            if not 200 <= response.status_code < 300:
                raise OpenAIDecisionsAPIError(response.status_code)
            answers = body.get("answers") if isinstance(body, dict) else None
            answer_type = (answers[0].get("type") if isinstance(answers, list)
                           and len(answers) == 1 and isinstance(answers[0], dict) else None)
            # Preserve the response kind without copying provider/input text.
            usage["decision_answer_type"] = answer_type if answer_type in ("choice", "refusal", "predicate", "score") else None
            return self._normalize(body, example.sample_id, (perf_counter() - started) * 1000,
                                   usage["request_id"], digest)

    def _normalize(self, body, sample_id, latency_ms, request_id, digest):
        if not isinstance(body, dict) or not isinstance(body.get("model"), str) or not body["model"].strip():
            raise ValueError("OpenAI Decisions response requires a resolved model")
        answers = body.get("answers")
        if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], dict):
            raise ValueError("OpenAI Decisions requires exactly one classification answer")
        answer = answers[0]
        if answer.get("type") == "refusal":
            raise OpenAIDecisionsRefusalError(sample_id, request_id)
        if answer.get("type") != "choice" or answer.get("name") != "classification":
            raise ValueError("OpenAI Decisions requires the named classification choice answer")
        names = [c.name for c in self._classes]
        label = answer.get("choice")
        if not isinstance(label, str) or label not in names:
            raise ValueError("OpenAI Decisions choice must be a configured class")
        entries = answer.get("probabilities")
        if not isinstance(entries, list) or len(entries) != len(names):
            raise ValueError("OpenAI Decisions probabilities must contain exactly the configured labels")
        probabilities = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("value"), str):
                raise ValueError("OpenAI Decisions probability values must be string labels")
            name = entry["value"]
            if name not in names or name in probabilities:
                raise ValueError("OpenAI Decisions probabilities contain duplicate or unknown labels")
            probabilities[name] = _probability(entry.get("probability"))
        probabilities = {name: probabilities[name] for name in names}
        if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=PROBABILITY_SUM_TOLERANCE):
            raise ValueError("OpenAI Decisions probabilities do not sum to one")
        if probabilities[label] != max(probabilities.values()):
            raise ValueError("OpenAI Decisions choice is not a maximum-probability class")
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
        raise ValueError("OpenAI Decisions probability/confidence must be finite and in [0,1]")
    return float(value)
