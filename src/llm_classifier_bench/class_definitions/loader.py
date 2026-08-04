"""Load and fingerprint frozen class-definition artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm_classifier_bench.class_definitions.base import (
    CLASS_DEFINITION_SCHEMA_VERSION,
    ClassDefinitionProfile,
    reorder_definitions,
)
from llm_classifier_bench.core import ClassDefinition


@dataclass(frozen=True, slots=True)
class LoadedClassDefinitionProfile:
    path: Path
    sha256: str
    profile: ClassDefinitionProfile

    def definitions_for(
        self,
        *,
        dataset_name: str,
        canonical_names: Sequence[str],
    ) -> tuple[ClassDefinition, ...]:
        if self.profile.dataset != dataset_name:
            raise ValueError(
                "Class-definition profile targets dataset "
                f"{self.profile.dataset!r}, not {dataset_name!r}"
            )
        return reorder_definitions(self.profile.classes, canonical_names)

    def benchmark_metadata(self) -> dict[str, Any]:
        return {
            "source": "versioned_profile",
            "path": str(self.path),
            "sha256": self.sha256,
            "schema_version": self.profile.schema_version,
            "dataset": self.profile.dataset,
            "profile": self.profile.profile,
            "review_status": self.profile.review_status,
            "generation": dict(self.profile.generation),
        }


def load_class_definition_profile(path: Path | str) -> LoadedClassDefinitionProfile:
    resolved = Path(path)
    raw = resolved.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid class-definition JSON: {resolved}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Class-definition profile root must be an object")

    schema_version = payload.get("schema_version")
    if schema_version != CLASS_DEFINITION_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported class-definition schema version {schema_version!r}"
        )

    dataset = _required_string(payload, "dataset")
    profile_name = _required_string(payload, "profile")
    review_status = payload.get("review_status", "unreviewed")
    if not isinstance(review_status, str) or not review_status.strip():
        raise ValueError("review_status must be a non-empty string")

    generation = payload.get("generation", {})
    if not isinstance(generation, Mapping):
        raise ValueError("generation must be an object")

    raw_classes = payload.get("classes")
    if not isinstance(raw_classes, list):
        raise ValueError("classes must be a list")

    classes: list[ClassDefinition] = []
    for index, item in enumerate(raw_classes):
        if not isinstance(item, Mapping):
            raise ValueError(f"classes[{index}] must be an object")
        canonical_name = item.get("canonical_name")
        description = item.get("description")
        if not isinstance(canonical_name, str) or not canonical_name.strip():
            raise ValueError(f"classes[{index}].canonical_name must be non-empty")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"classes[{index}].description must be non-empty")
        classes.append(
            ClassDefinition(name=canonical_name, description=description)
        )

    profile = ClassDefinitionProfile(
        dataset=dataset,
        profile=profile_name,
        classes=tuple(classes),
        generation=dict(generation),
        review_status=review_status,
        schema_version=schema_version,
    )
    return LoadedClassDefinitionProfile(
        path=resolved,
        sha256=sha256,
        profile=profile,
    )


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


__all__ = ["LoadedClassDefinitionProfile", "load_class_definition_profile"]
