"""Versioned class-definition artifacts used by zero-shot classifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm_classifier_bench.core import ClassDefinition


CLASS_DEFINITION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ClassDefinitionProfile:
    """A frozen semantic description of one dataset's canonical label space.

    ``ClassDefinition.name`` is always the canonical dataset label. Only the
    description is enriched. The profile is intended to be generated once,
    reviewed, versioned, and then reused identically by every zero-shot
    classifier in a benchmark condition.
    """

    dataset: str
    profile: str
    classes: tuple[ClassDefinition, ...]
    generation: Mapping[str, Any] = field(default_factory=dict)
    review_status: str = "unreviewed"
    schema_version: int = CLASS_DEFINITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CLASS_DEFINITION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported class-definition schema version {self.schema_version!r}"
            )
        if not self.dataset.strip():
            raise ValueError("dataset cannot be empty")
        if not self.profile.strip():
            raise ValueError("profile cannot be empty")
        if len(self.classes) < 2:
            raise ValueError("A class-definition profile requires at least two classes")

        names = [item.name for item in self.classes]
        if len(names) != len(set(names)):
            raise ValueError("Class-definition canonical names must be unique")
        if not self.review_status.strip():
            raise ValueError("review_status cannot be empty")

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.classes)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "profile": self.profile,
            "review_status": self.review_status,
            "generation": dict(self.generation),
            "classes": [
                {
                    "canonical_name": item.name,
                    "description": item.description,
                }
                for item in self.classes
            ],
        }

    def write_json(self, path: Path, *, overwrite: bool = False) -> Path:
        resolved = Path(path)
        if resolved.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing profile: {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(
            json.dumps(self.to_payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return resolved


def reorder_definitions(
    definitions: Sequence[ClassDefinition],
    canonical_names: Sequence[str],
) -> tuple[ClassDefinition, ...]:
    """Validate exact label coverage and return dataset-canonical ordering."""

    expected = tuple(canonical_names)
    by_name = {item.name: item for item in definitions}
    if len(by_name) != len(tuple(definitions)):
        raise ValueError("Class definitions contain duplicate canonical names")

    missing = sorted(set(expected) - set(by_name))
    extra = sorted(set(by_name) - set(expected))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            "Class-definition profile does not match dataset canonical labels: "
            + ", ".join(details)
        )

    return tuple(by_name[name] for name in expected)


__all__ = [
    "CLASS_DEFINITION_SCHEMA_VERSION",
    "ClassDefinitionProfile",
    "reorder_definitions",
]
