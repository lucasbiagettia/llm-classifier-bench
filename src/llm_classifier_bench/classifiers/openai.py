"""Zero-shot and in-context generative classification through the OpenAI API."""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any, Sequence
from hashlib import sha256

from llm_classifier_bench.classifiers.base import Prediction
from llm_classifier_bench.config import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
)
from llm_classifier_bench.costs import capture_api_usage, capture_response
from llm_classifier_bench.core import ClassificationInput, ClassDefinition, LabeledExample
from llm_classifier_bench.budgets import UnsupportedConfiguration
from llm_classifier_bench.datasets.selection import validate_partition_disjointness


class OpenAIClassifier:
    """Closed-set classifier with optional labeled demonstrations in every prompt.

    The adapter intentionally does not ask the model to self-report confidence. The
    API response is normalized to ``Prediction`` with ``confidence=None`` and
    ``probabilities=None``. This keeps calibration metrics unavailable rather than
    fabricating probabilistic evidence that the provider did not expose.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = None,
        reasoning_effort: str = DEFAULT_OPENAI_REASONING_EFFORT,
        client: Any | None = None,
        classifier_name: str | None = None,
        in_context: bool = False,
        context_window_tokens: int | None = None,
        completion_reserve_tokens: int = 4096,
        framing_allowance_tokens: int = 1024,
    ) -> None:
        if classifier_name is None:
            classifier_name = "openai-in-context" if in_context else "openai-zero-shot"
        if not model.strip():
            raise ValueError("model cannot be empty")
        if not classifier_name.strip():
            raise ValueError("classifier_name cannot be empty")
        if not reasoning_effort.strip():
            raise ValueError("reasoning_effort cannot be empty")

        self.model = model
        self.reasoning_effort = reasoning_effort
        self._name = classifier_name
        self._classes: tuple[ClassDefinition, ...] = ()
        self._client = client
        self._api_key = api_key
        self._usage_sink = None
        self.in_context = in_context
        self.context_window_tokens = context_window_tokens
        self.completion_reserve_tokens = completion_reserve_tokens
        self.framing_allowance_tokens = framing_allowance_tokens
        for value in (completion_reserve_tokens, framing_allowance_tokens):
            if type(value) is not int or value < 1:
                raise ValueError("Context allowances must be positive integers")
        if context_window_tokens is not None and (
            type(context_window_tokens) is not int or context_window_tokens < 1
        ):
            raise ValueError("context_window_tokens must be a positive integer")
        self._context_examples: tuple[LabeledExample, ...] = ()
        self.context_examples_used = 0
        self.total_labeled_examples_used = 0

        # Explicit experiment metadata. The runner can persist these fields
        # without special-casing OpenAI by classifier name.
        self.supervision_regime = "zero_shot"
        self.training_examples_used = 0
        self.validation_examples_used = 0

    @property
    def name(self) -> str:
        return self._name

    def set_usage_sink(self, sink) -> None:
        self._usage_sink = sink

    def preparation_metadata(self):
        return {"remote_preparation_performed": False, "reason": "adapter performs no remote prepare/fit operations"}

    def set_preparation_sink(self, sink):
        self._preparation_sink = sink

    def inference_metadata(self) -> dict[str, Any]:
        timeout = getattr(self._client, "timeout", None)
        return {
            "backend": "hosted_api", "batch_execution": "sequential_single_example",
            "transport_max_retries": getattr(self._client, "max_retries", None),
            "transport_attempt_timings": "not exposed; call timing includes any injected-client retries",
            "timeout_seconds": timeout if isinstance(timeout, (int, float)) else {
                key: getattr(timeout, key, None) for key in ("connect", "read", "write", "pool")
            },
            "server_hardware": None,
            "context_policy": self.context_policy(),
        }

    def prepare(self, classes: Sequence[ClassDefinition]) -> None:
        frozen = tuple(classes)
        if len(frozen) < 2:
            raise ValueError("At least two classes are required")
        if len({item.name for item in frozen}) != len(frozen):
            raise ValueError("Class names must be unique")
        self._classes = frozen
        if self._client is None:
            self._client = _build_openai_client(self._api_key)

    def fit(
        self,
        examples: Sequence[LabeledExample],
        *,
        validation_examples: Sequence[LabeledExample] = (),
    ) -> None:
        """Store the entire supplied pool as demonstrations when explicitly enabled."""
        validate_partition_disjointness(examples, validation_examples)
        if self.in_context and not self._classes:
            raise RuntimeError("Call prepare(classes) before fit()")
        if any(e.label not in {c.name for c in self._classes} for e in examples) and self.in_context:
            raise ValueError("Unknown demonstration label")
        self._context_examples = tuple(examples) if self.in_context else ()
        self.context_examples_used = len(self._context_examples)
        self.total_labeled_examples_used = self.context_examples_used
        self.supervision_regime = "in_context" if self.context_examples_used else "zero_shot"

    def context_policy(self):
        return {
            "context_window_tokens": self.context_window_tokens,
            "completion_reserve_tokens": self.completion_reserve_tokens,
            "framing_allowance_tokens": self.framing_allowance_tokens,
            "input_estimator": "UTF-8 bytes of serialized messages/schema (conservative byte-BPE proxy)",
            "limitation": "Not a provider token count; framing allowance is operator supplied. May reject fitting prompts.",
        }

    def fitted_metadata(self):
        return {
            "supervision_regime": self.supervision_regime,
            "training_examples_used": 0, "validation_examples_used": 0,
            "context_examples_used": self.context_examples_used,
            "total_labeled_examples_used": self.total_labeled_examples_used,
            "context_sample_ids": [e.sample_id for e in self._context_examples],
            "context_policy": self.context_policy(),
            "parameter_training": False,
        }

    def plan_context(self, classes, examples, inputs):
        """Check every prospective request before any API call, without gold labels."""
        if self.context_window_tokens is None:
            raise UnsupportedConfiguration("context_window_unknown: set --openai-context-window-tokens")
        demonstrations = tuple(examples) if self.in_context else ()
        requests = []
        for example in inputs:
            payload = self._request_payload(example, classes, demonstrations)
            serialized = json.dumps({k: payload[k] for k in ("messages", "response_format")},
                                    ensure_ascii=False, sort_keys=True).encode("utf-8")
            estimate = len(serialized) + self.framing_allowance_tokens + self.completion_reserve_tokens
            requests.append({"sample_id": example.sample_id, "estimated_total_tokens": estimate,
                             "prompt_sha256": sha256(serialized).hexdigest()})
        return {"policy": self.context_policy(), "requests": requests,
                "supported": all(r["estimated_total_tokens"] <= self.context_window_tokens for r in requests)}

    def predict(self, examples: Sequence[ClassificationInput]) -> list[Prediction]:
        if not self._classes:
            raise RuntimeError("Call prepare(classes) before predict()")
        if self.in_context or self.context_window_tokens is not None:
            plan = self.plan_context(self._classes, self._context_examples, examples)
            if not plan["supported"]:
                raise UnsupportedConfiguration("context_limit_exceeded: conservative preflight estimate")
        if self._client is None:
            self._client = _build_openai_client(self._api_key)
        return [self._predict_one(example) for example in examples]

    def _predict_one(self, example: ClassificationInput) -> Prediction:
        with capture_api_usage(
            self._usage_sink, provider="openai", sample_id=example.sample_id,
            requested_model=self.model,
            transport_attempts_complete=getattr(self._client, "max_retries", None) == 0,
        ) as attempt:
            return self._predict_one_recorded(example, attempt)

    def _predict_one_recorded(self, example: ClassificationInput, attempt: dict) -> Prediction:
        started_at = perf_counter()
        try:
            response = self._client.chat.completions.create(**self._request_payload(
                example, self._classes, self._context_examples,
            ))
        except Exception as exc:
            if getattr(exc, "code", None) == "context_length_exceeded":
                raise UnsupportedConfiguration("context_limit_exceeded: provider rejected request") from exc
            raise
        latency_ms = (perf_counter() - started_at) * 1_000
        capture_response(attempt, _serialize_response(response))
        return self._normalize_response(example, response, latency_ms)

    def _request_payload(self, example, classes, demonstrations):
        class_names = [class_definition.name for class_definition in classes]
        class_block = "\n".join(
            f"- {class_definition.name}: {class_definition.description}"
            for class_definition in classes
        )

        system_prompt = (
            "You are a closed-set text classifier. Classify the user's text into "
            "exactly one of the allowed labels. Use the class descriptions as the "
            "decision rubric. Do not invent labels."
        )
        user_prompt = f"Allowed classes:\n{class_block}\n\nText:\n{example.text}"

        messages = [{"role": "system", "content": system_prompt + (
            f"\n\nAllowed classes:\n{class_block}" if demonstrations else ""
        )}]
        for demonstration in demonstrations:
            messages.extend([
                {"role": "user", "content": f"Text:\n{demonstration.text}"},
                {"role": "assistant", "content": json.dumps({"label": demonstration.label})},
            ])
        messages.append({"role": "user", "content": f"Text:\n{example.text}" if demonstrations else user_prompt})
        payload = dict(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "closed_set_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "enum": class_names,
                            }
                        },
                        "required": ["label"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        if self.context_window_tokens is not None:
            payload["max_completion_tokens"] = self.completion_reserve_tokens
        return payload

    def _normalize_response(self, example, response, latency_ms):
        class_names = [c.name for c in self._classes]

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise ValueError("OpenAI response does not contain message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenAI response contains empty message content")

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenAI structured response is not valid JSON") from exc

        predicted_label = payload.get("label")
        if predicted_label not in class_names:
            raise ValueError(
                f"OpenAI returned label {predicted_label!r}, which is not in the class set"
            )

        response_id = getattr(response, "id", None)
        response_model = getattr(response, "model", None)
        raw_response = _serialize_response(response)

        return Prediction(
            sample_id=example.sample_id,
            predicted_label=str(predicted_label),
            confidence=None,
            probabilities=None,
            latency_ms=latency_ms,
            model=response_model if isinstance(response_model, str) else self.model,
            request_id=response_id if isinstance(response_id, str) else None,
            raw_response=raw_response,
        )


def _build_openai_client(api_key: str | None) -> Any:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "python-dotenv is not installed. Run pip install -r requirements.txt."
        ) from exc

    load_dotenv(override=False)
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OPENAI_API_KEY is required for OpenAIClassifier")

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "openai is not installed. Run pip install -r requirements.txt."
        ) from exc

    # Keep a measured logical call to one SDK attempt; record injected policies.
    return OpenAI(api_key=resolved_api_key, max_retries=0, timeout=120.0)


def _serialize_response(response: Any) -> dict[str, Any]:
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
    }


__all__ = ["OpenAIClassifier"]
