"""TypeSafe Choice adapter. No implicit training, redirects or hidden retries."""
from __future__ import annotations

import hashlib
import json
import math
import os
from time import perf_counter, sleep
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv

from llm_classifier_bench.budgets import UnsupportedConfiguration
from llm_classifier_bench.core import PROBABILITY_SUM_TOLERANCE
from llm_classifier_bench.costs import capture_api_usage, capture_response
from .base import Prediction

DEFAULT_JEV_MODEL = "jev-1.13.0"
DEFAULT_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
INSTRUCTIONS = "Which of the configured classes best describes this text?"
LIMITS_SOURCE = "https://docs.typesafe.ai/models"


class JevAPIError(RuntimeError):
    """Sanitized transport error; provider bodies may contain submitted text."""

    def __init__(self, status_code=None):
        self.status_code = status_code
        super().__init__(f"Jev HTTP {status_code}" if status_code is not None else "Jev transport failure")


class JevClassifier:
    supervision_regime = "zero_shot"
    training_examples_used = 0
    validation_examples_used = 0
    context_examples_used = 0

    def __init__(self, *, model=DEFAULT_JEV_MODEL, endpoint=DEFAULT_JEV_ENDPOINT,
                 api_key=None, timeout_s=30.0, max_retries=0, retry_backoff_s=1.0,
                 max_requests=None, context_window_tokens=32000, session=None,
                 classifier_name="jev-zero-shot"):
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Jev endpoint must be an HTTPS URL without credentials, query or fragment")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Jev model must be nonempty")
        for key, value, minimum in (("max_retries", max_retries, 0),
                                    ("context_window_tokens", context_window_tokens, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if max_requests is not None and (type(max_requests) is not int or max_requests < 1):
            raise ValueError("max_requests must be a positive integer")
        for key, value in (("timeout_s", timeout_s), ("retry_backoff_s", retry_backoff_s)):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        self.name = classifier_name
        self.model, self.endpoint = model, endpoint
        self.timeout_s, self.max_retries = timeout_s, max_retries
        self.retry_backoff_s, self.max_requests = retry_backoff_s, max_requests
        self.context_window_tokens = context_window_tokens
        self._api_key, self._session = api_key, session
        self._injected_session = session is not None
        self._classes = ()
        self._usage_sink = None
        self._request_count = 0

    def _payload(self, classes, text):
        return {"model": self.model, "state": text, "questions": {"classification": {
            "type": "choice", "instructions": INSTRUCTIONS,
            "criteria": {c.name: c.description for c in classes},
        }}}

    def _encoded(self, payload):
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def _check_classes(self, classes):
        if len({c.name for c in classes}) != len(classes) or len(classes) < 2:
            raise ValueError("Jev requires at least two unique class names")
        if len(classes) > 255:
            raise UnsupportedConfiguration("jev_choice_limit: at most 255 labels")

    def preflight(self, classes, inputs, *, matched_budget=None):
        """Local, conservative byte estimate; never truncate labels or input text."""
        reason = None
        try:
            self._check_classes(classes)
        except UnsupportedConfiguration as exc:
            reason = str(exc)
        if matched_budget is not None and (matched_budget.examples or matched_budget.validation_examples):
            reason = "jev_zero_shot_only: nonzero matched fit or validation budgets are unsupported"
        largest = max((len(self._encoded(self._payload(classes, e.text))) + 1024 for e in inputs), default=0)
        if largest > self.context_window_tokens:
            reason = "jev_context_limit: conservative UTF-8 byte estimate plus 1024 framing allowance"
        if self.max_requests is not None and len(inputs) > self.max_requests - self._request_count:
            reason = "jev_request_cap: selected inputs exceed remaining request cap"
        return {"supported": reason is None, "reason": reason,
                "max_estimated_tokens": largest, "estimate_policy": "utf8_bytes_plus_1024_v1",
                "limits": self.inference_metadata()["limits"],
                "requested_model": self.model, "endpoint": self.endpoint}

    def plan_fit(self, classes, examples, *, validation_examples=()):
        self._check_classes(classes)
        return {"live_supported": True, "blocker": None, "planned_remote_operations": [],
                "selection": {"selected_examples": [], "requested_total": 0, "actual_total": 0,
                              "covered_class_count": 0, "class_count": len(classes)},
                **self.fitted_metadata()}

    def prepare(self, classes):
        self._check_classes(classes)
        self._classes = tuple(classes)
        if self._api_key is None:
            load_dotenv()
            self._api_key = os.getenv("JEV_TOKEN")
        if not self._api_key:
            raise ValueError("JEV_TOKEN is required")
        if self._session is None:
            self._session = requests.Session()
            self._session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))

    def fit(self, examples, *, validation_examples=()):
        # The full-training reference passes a pool; zero-shot consumes none of it.
        if not self._classes:
            raise RuntimeError("Call prepare before fit")

    def fitted_metadata(self):
        return {"supervision_regime": "zero_shot", "supported_supervision_modes": ["zero_shot"],
                "training_examples_used": 0, "validation_examples_used": 0,
                "context_examples_used": 0, "total_labeled_examples_used": 0,
                "selected_sample_ids": [], "requested_model": self.model}

    def inference_metadata(self):
        return {"backend": "hosted_api", "batch_execution": "sequential", "server_hardware": None,
                "endpoint": self.endpoint, "requested_model": self.model,
                "timeout_s": self.timeout_s, "transport_max_retries": self.max_retries,
                "retry_backoff_s": self.retry_backoff_s, "max_requests": self.max_requests,
                "transport_attempts_complete": not self._injected_session,
                "tie_policy": "preserve_provider_choice_if_exact_maximum",
                "confidence_semantics": "probability_of_predicted_label; native confidence stored separately",
                "limits": {"choice_options": 255, "configured_single_question_tokens": self.context_window_tokens,
                           "documented_request_tokens": 64000, "documented_state_plus_question_tokens": 32000,
                           "documented_model": DEFAULT_JEV_MODEL, "source": LIMITS_SOURCE,
                           "verified_date": "2026-09-25"}}

    def preparation_metadata(self):
        return {"backend": "hosted_api", "remote_preparation_performed": False}

    def set_usage_sink(self, sink):
        self._usage_sink = sink

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
        started = perf_counter()
        for attempt in range(self.max_retries + 1):
            if self.max_requests is not None and self._request_count >= self.max_requests:
                raise RuntimeError("Jev request cap exhausted")
            try:
                with capture_api_usage(self._usage_sink, provider="jev", sample_id=example.sample_id,
                                       requested_model=self.model,
                                       transport_attempts_complete=not self._injected_session) as usage:
                    usage.update(attempt=attempt + 1, request_sha256=digest, request_bytes=len(encoded),
                                 endpoint=self.endpoint)
                    self._request_count += 1
                    try:
                        response = self._session.post(self.endpoint, data=encoded,
                            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                            timeout=self.timeout_s, allow_redirects=False)
                    except requests.RequestException:
                        raise JevAPIError() from None
                    usage.update(response_received=True, http_status=response.status_code,
                                 request_id=response.headers.get("x-request-id"))
                    try:
                        body = response.json()
                    except ValueError:
                        body = None
                    capture_response(usage, body)
                    if not 200 <= response.status_code < 300:
                        raise JevAPIError(response.status_code)
                    return self._normalize(body, example.sample_id, (perf_counter() - started) * 1000,
                                           usage.get("request_id"), digest)
            except JevAPIError as exc:
                if attempt == self.max_retries or exc.status_code not in (None, 429, 500, 502, 503, 504, 529):
                    raise
                sleep(min(self.retry_backoff_s * (2 ** attempt), 30.0))
        raise AssertionError("Unreachable")

    def _normalize(self, body, sample_id, latency_ms, request_id, digest):
        if not isinstance(body, dict) or not isinstance(body.get("model"), str) or not body["model"]:
            raise ValueError("Jev response requires a resolved model")
        answers = body.get("answers")
        answer = answers.get("classification") if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("Jev response requires the classification Choice answer")
        label = answer.get("choice")
        probabilities = answer.get("probabilities")
        names = [c.name for c in self._classes]
        if not isinstance(label, str) or label not in names:
            raise ValueError("Jev choice must be a configured class")
        if not isinstance(probabilities, dict) or set(probabilities) != set(names):
            raise ValueError("Jev probabilities must contain exactly the configured labels")
        def probability(value):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Jev probability/confidence must be a finite number in [0,1]")
            return float(value)
        probabilities = {name: probability(probabilities[name]) for name in names}
        if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=PROBABILITY_SUM_TOLERANCE):
            raise ValueError("Jev probabilities do not sum to one")
        if probabilities[label] != max(probabilities.values()):
            raise ValueError("Jev choice is not a maximum-probability class")
        native = answer.get("confidence")
        if native is not None:
            native = probability(native)
        return Prediction(sample_id, label, probabilities[label], probabilities, latency_ms,
                          model=body["model"], request_id=request_id,
                          raw_response={"requested_model": self.model, "resolved_model": body["model"],
                                        "provider_choice": label, "provider_confidence": native,
                                        "provider_confidence_unavailable_reason": "not reported" if native is None else None,
                                        "request_sha256": digest,
                                        "tied_maximum_labels": [n for n in names if probabilities[n] == probabilities[label]]})
