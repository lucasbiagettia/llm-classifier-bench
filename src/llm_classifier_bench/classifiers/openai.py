"""Zero-shot generative classification through the OpenAI API."""

from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any, Sequence

from llm_classifier_bench.classifiers.base import Prediction
from llm_classifier_bench.config import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
)
from llm_classifier_bench.core import ClassificationInput, ClassDefinition, LabeledExample


class OpenAIClassifier:
    """Closed-set zero-shot classifier backed by an OpenAI text model.

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
        classifier_name: str = "openai-zero-shot",
    ) -> None:
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
        self._client = client or _build_openai_client(api_key)

        # Explicit experiment metadata. The runner can persist these fields
        # without special-casing OpenAI by classifier name.
        self.supervision_regime = "zero_shot"
        self.training_examples_used = 0
        self.validation_examples_used = 0

    @property
    def name(self) -> str:
        return self._name

    def prepare(self, classes: Sequence[ClassDefinition]) -> None:
        frozen = tuple(classes)
        if len(frozen) < 2:
            raise ValueError("At least two classes are required")
        if len({item.name for item in frozen}) != len(frozen):
            raise ValueError("Class names must be unique")
        self._classes = frozen

    def fit(
        self,
        examples: Sequence[LabeledExample],
        *,
        validation_examples: Sequence[LabeledExample] = (),
    ) -> None:
        """Zero-shot OpenAI classification uses no labeled examples."""
        return None

    def predict(self, examples: Sequence[ClassificationInput]) -> list[Prediction]:
        if not self._classes:
            raise RuntimeError("Call prepare(classes) before predict()")

        return [self._predict_one(example) for example in examples]

    def _predict_one(self, example: ClassificationInput) -> Prediction:
        class_names = [class_definition.name for class_definition in self._classes]
        class_block = "\n".join(
            f"- {class_definition.name}: {class_definition.description}"
            for class_definition in self._classes
        )

        system_prompt = (
            "You are a closed-set text classifier. Classify the user's text into "
            "exactly one of the allowed labels. Use the class descriptions as the "
            "decision rubric. Do not invent labels."
        )
        user_prompt = f"Allowed classes:\n{class_block}\n\nText:\n{example.text}"

        started_at = perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
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
        latency_ms = (perf_counter() - started_at) * 1_000

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

    return OpenAI(api_key=resolved_api_key)


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
