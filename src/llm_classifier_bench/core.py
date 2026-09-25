"""Shared domain objects used by datasets and classifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


# Shared absolute tolerance for rounded distributions; values are not renormalized.
PROBABILITY_SUM_TOLERANCE = 1e-4


@dataclass(frozen=True, slots=True)
class ClassDefinition:
    """One class in a closed-set classification task."""

    name: str
    description: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ClassDefinition.name cannot be empty")
        if not self.description.strip():
            raise ValueError("ClassDefinition.description cannot be empty")


@dataclass(frozen=True, slots=True)
class ClassificationInput:
    """One unlabeled input passed to a classifier."""

    sample_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("ClassificationInput.sample_id cannot be empty")
        if not self.text.strip():
            raise ValueError("ClassificationInput.text cannot be empty")


@dataclass(frozen=True, slots=True)
class LabeledExample:
    """One labeled example loaded from a benchmark dataset."""

    sample_id: str
    text: str
    label: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("LabeledExample.sample_id cannot be empty")
        if not self.text.strip():
            raise ValueError("LabeledExample.text cannot be empty")
        if not self.label.strip():
            raise ValueError("LabeledExample.label cannot be empty")

    def as_input(self) -> ClassificationInput:
        """Expose only input provenance, never target-bearing dataset metadata."""

        return ClassificationInput(
            sample_id=self.sample_id,
            text=self.text,
            metadata={key: self.metadata[key] for key in
                      ("dataset", "source", "split", "row_index") if key in self.metadata},
        )
