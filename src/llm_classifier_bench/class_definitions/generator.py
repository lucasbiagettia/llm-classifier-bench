"""Offline builders for versioned class-description profiles."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Sequence

from llm_classifier_bench.class_definitions.base import ClassDefinitionProfile
from llm_classifier_bench.config import (
    DEFAULT_CLASS_DEFINITION_GENERATOR_MODEL,
    DEFAULT_CLASS_DEFINITION_GENERATOR_REASONING_EFFORT,
)
from llm_classifier_bench.core import ClassDefinition


LABEL_DESCRIPTION_PROMPT_VERSION = "label_description_v1"

_SYSTEM_PROMPT = """You prepare ontology descriptions for a closed-set text classification benchmark.

You will receive a dataset name, optional dataset context, and the complete inventory of canonical labels.
For every label, write one concise neutral English sentence explaining the semantic intent of that label.

Rules:
- Copy canonical label names exactly. Never rename, normalize, translate, or merge them.
- Use only the dataset context, the label's wording, and the full label inventory.
- Do not use or request train, validation, or test examples.
- Do not invent policies, eligibility rules, causes, remedies, or domain facts not implied by the label.
- Keep descriptions concise and useful for distinguishing nearby labels.
- Do not include examples of user utterances.
- Return every canonical label exactly once.
"""


def build_minimal_profile(
    *,
    dataset_name: str,
    canonical_names: Sequence[str],
    profile_name: str = "canonical_minimal_v1",
) -> ClassDefinitionProfile:
    """Build the deterministic control profile without an LLM call."""

    names = _validate_canonical_names(canonical_names)
    classes = tuple(
        ClassDefinition(
            name=name,
            description=f"Category: {_readable_label(name)}.",
        )
        for name in names
    )
    return ClassDefinitionProfile(
        dataset=dataset_name,
        profile=profile_name,
        classes=classes,
        generation={
            "method": "deterministic",
            "uses_train_examples": False,
            "uses_validation_examples": False,
            "uses_test_examples": False,
        },
        review_status="not_required",
    )


class OpenAIClassDefinitionGenerator:
    """Generate descriptions once, offline, using an OpenAI model.

    This object is deliberately not a benchmark ``Classifier``. It never sees
    labeled examples and produces a frozen JSON artifact that later benchmark
    runs only load from disk.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_CLASS_DEFINITION_GENERATOR_MODEL,
        reasoning_effort: str = DEFAULT_CLASS_DEFINITION_GENERATOR_REASONING_EFFORT,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        if not reasoning_effort.strip():
            raise ValueError("reasoning_effort cannot be empty")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._client = client or _build_openai_client(api_key)

    def generate(
        self,
        *,
        dataset_name: str,
        canonical_names: Sequence[str],
        dataset_context: str = "",
        profile_name: str = "canonical_llm_enriched_v1",
    ) -> ClassDefinitionProfile:
        names = _validate_canonical_names(canonical_names)
        user_prompt = _build_user_prompt(
            dataset_name=dataset_name,
            dataset_context=dataset_context,
            canonical_names=names,
        )

        response = self._client.chat.completions.create(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_definition_profile",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "classes": {
                                "type": "array",
                                "minItems": len(names),
                                "maxItems": len(names),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "canonical_name": {
                                            "type": "string",
                                            "enum": list(names),
                                        },
                                        "description": {"type": "string"},
                                    },
                                    "required": ["canonical_name", "description"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["classes"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        content = _response_content(response)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Generator returned invalid structured JSON") from exc

        raw_classes = payload.get("classes")
        if not isinstance(raw_classes, list):
            raise ValueError("Generator response does not contain a classes list")

        generated: dict[str, str] = {}
        for index, item in enumerate(raw_classes):
            if not isinstance(item, dict):
                raise ValueError(f"Generator classes[{index}] is not an object")
            name = item.get("canonical_name")
            description = item.get("description")
            if not isinstance(name, str) or name not in names:
                raise ValueError(f"Generator returned unknown canonical name {name!r}")
            if name in generated:
                raise ValueError(f"Generator returned duplicate canonical name {name!r}")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Generator returned empty description for {name!r}")
            generated[name] = description.strip()

        missing = [name for name in names if name not in generated]
        if missing:
            raise ValueError(f"Generator omitted canonical labels: {missing}")

        resolved_model = getattr(response, "model", None)
        request_id = getattr(response, "id", None)
        prompt_material = _SYSTEM_PROMPT + "\n---\n" + user_prompt

        return ClassDefinitionProfile(
            dataset=dataset_name,
            profile=profile_name,
            classes=tuple(
                ClassDefinition(name=name, description=generated[name]) for name in names
            ),
            generation={
                "method": "llm",
                "provider": "openai",
                "requested_model": self.model,
                "resolved_model": (
                    resolved_model if isinstance(resolved_model, str) else None
                ),
                "reasoning_effort": self.reasoning_effort,
                "prompt_version": LABEL_DESCRIPTION_PROMPT_VERSION,
                "prompt_sha256": hashlib.sha256(
                    prompt_material.encode("utf-8")
                ).hexdigest(),
                "dataset_context": dataset_context,
                "uses_train_examples": False,
                "uses_validation_examples": False,
                "uses_test_examples": False,
                "request_id": request_id if isinstance(request_id, str) else None,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            review_status="unreviewed",
        )


def _validate_canonical_names(canonical_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(canonical_names)
    if len(names) < 2:
        raise ValueError("At least two canonical labels are required")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Canonical labels must be non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("Canonical labels must be unique")
    return names


def _readable_label(label: str) -> str:
    return " ".join(label.replace("_", " ").replace("/", " or ").split())


def _build_user_prompt(
    *,
    dataset_name: str,
    dataset_context: str,
    canonical_names: Sequence[str],
) -> str:
    inventory = "\n".join(f"- {name}" for name in canonical_names)
    context = dataset_context.strip() or "No additional dataset context provided."
    return (
        f"Dataset: {dataset_name}\n"
        f"Dataset context: {context}\n\n"
        "Complete canonical label inventory:\n"
        f"{inventory}\n"
    )


def _response_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as exc:
        raise ValueError("Generator response does not contain message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Generator response contains empty message content")
    return content


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
        raise ValueError("OPENAI_API_KEY is required for LLM class-definition generation")

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "openai is not installed. Run pip install -r requirements.txt."
        ) from exc

    return OpenAI(api_key=resolved_api_key)


__all__ = [
    "LABEL_DESCRIPTION_PROMPT_VERSION",
    "OpenAIClassDefinitionGenerator",
    "build_minimal_profile",
]
