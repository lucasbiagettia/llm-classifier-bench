"""Shared helpers for small classifier smoke runs.

These helpers intentionally take the first N examples per class. They are for smoke
validation only, not for the formal benchmark protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from llm_classifier_bench.core import LabeledExample
from llm_classifier_bench.datasets.base import DatasetBundle


@dataclass(frozen=True, slots=True)
class StaticDataset:
    bundle: DatasetBundle

    @property
    def name(self) -> str:
        return self.bundle.name

    def load(self) -> DatasetBundle:
        return self.bundle


def smoke_subset(
    bundle: DatasetBundle,
    *,
    train_per_class: int,
    test_per_class: int,
) -> DatasetBundle:
    return DatasetBundle(
        name=f"{bundle.name}_smoke",
        classes=bundle.classes,
        train=_first_per_class(bundle.train, bundle.class_names, train_per_class),
        test=_first_per_class(bundle.test, bundle.class_names, test_per_class),
        metadata={
            **dict(bundle.metadata),
            "smoke_only": True,
            "source_dataset": bundle.name,
            "train_per_class": train_per_class,
            "test_per_class": test_per_class,
        },
    )


def _first_per_class(
    examples: Sequence[LabeledExample],
    class_names: Sequence[str],
    per_class: int,
) -> tuple[LabeledExample, ...]:
    if per_class < 1:
        raise ValueError("per_class must be at least 1")

    selected: list[LabeledExample] = []
    for class_name in class_names:
        class_examples = [example for example in examples if example.label == class_name]
        if len(class_examples) < per_class:
            raise ValueError(
                f"Class {class_name!r} has only {len(class_examples)} examples; "
                f"requested {per_class}"
            )
        selected.extend(class_examples[:per_class])
    return tuple(selected)
