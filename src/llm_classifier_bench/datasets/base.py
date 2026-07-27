"""Dataset contracts and normalized loaded-dataset representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, runtime_checkable

from llm_classifier_bench.core import (
    ClassificationInput,
    ClassDefinition,
    LabeledExample,
)


SplitName = Literal["train", "test"]


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """A classification dataset normalized for the benchmark.

    Tuples make the loaded artifact stable and prevent accidental mutation while
    different classifiers consume the same examples.
    """

    name: str
    classes: tuple[ClassDefinition, ...]
    train: tuple[LabeledExample, ...]
    test: tuple[LabeledExample, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("DatasetBundle.name cannot be empty")
        if len(self.classes) < 2:
            raise ValueError("A classification dataset requires at least two classes")

        class_names = [item.name for item in self.classes]
        if len(class_names) != len(set(class_names)):
            raise ValueError("Dataset class names must be unique")

        allowed_labels = set(class_names)
        for split_name, examples in (("train", self.train), ("test", self.test)):
            sample_ids = [example.sample_id for example in examples]
            if len(sample_ids) != len(set(sample_ids)):
                raise ValueError(f"Duplicate sample ids in {split_name} split")

            unknown = sorted({example.label for example in examples} - allowed_labels)
            if unknown:
                raise ValueError(
                    f"Unknown labels in {split_name} split: {', '.join(unknown)}"
                )

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.classes)

    def examples(self, split: SplitName) -> tuple[LabeledExample, ...]:
        return self.train if split == "train" else self.test

    def inputs(self, split: SplitName = "test") -> tuple[ClassificationInput, ...]:
        """Return a split without leaking its gold labels to a classifier."""

        return tuple(example.as_input() for example in self.examples(split))

    def gold_labels(self, split: SplitName = "test") -> tuple[str, ...]:
        return tuple(example.label for example in self.examples(split))


@runtime_checkable
class ClassificationDataset(Protocol):
    """Minimal interface implemented by every dataset adapter."""

    @property
    def name(self) -> str:
        """Stable dataset identifier used in configs and artifacts."""
        ...

    def load(self) -> DatasetBundle:
        """Load and normalize train/test splits and class definitions."""
        ...
