from __future__ import annotations

from typing import Sequence

import pytest

from llm_classifier_bench.classifiers.base import Prediction
from llm_classifier_bench.core import ClassDefinition, ClassificationInput, LabeledExample
from llm_classifier_bench.datasets.base import DatasetBundle
from llm_classifier_bench.workflows import run_classification_cycle


class StubClassifier:
    name = "stub"

    def __init__(self, labels_by_sample_id: dict[str, str]) -> None:
        self.labels_by_sample_id = labels_by_sample_id
        self.fitted_examples: tuple[LabeledExample, ...] | None = None

    def fit(self, examples: Sequence[LabeledExample]) -> None:
        self.fitted_examples = tuple(examples)

    def predict(
        self, examples: Sequence[ClassificationInput]
    ) -> list[Prediction]:
        return [
            Prediction(
                sample_id=example.sample_id,
                predicted_label=self.labels_by_sample_id[example.sample_id],
                confidence=0.8,
                probabilities={"World": 0.2, "Sports": 0.8},
                latency_ms=1.0,
            )
            for example in examples
        ]


def make_bundle() -> DatasetBundle:
    return DatasetBundle(
        name="tiny-news",
        classes=(
            ClassDefinition("World", "International news."),
            ClassDefinition("Sports", "Sports news."),
        ),
        train=(
            LabeledExample("train-1", "A diplomatic summit.", "World"),
            LabeledExample("train-2", "The team won.", "Sports"),
        ),
        test=(
            LabeledExample("test-1", "A new treaty was signed.", "World"),
            LabeledExample("test-2", "The striker scored twice.", "Sports"),
        ),
    )


def test_cycle_connects_dataset_and_classifier_contracts() -> None:
    bundle = make_bundle()
    classifier = StubClassifier(
        {"test-1": "World", "test-2": "Sports"}
    )

    records = run_classification_cycle(
        bundle=bundle,
        classifier=classifier,
        examples=bundle.test,
    )

    assert classifier.fitted_examples == bundle.train
    assert [record.example.sample_id for record in records] == [
        "test-1",
        "test-2",
    ]
    assert [record.prediction.sample_id for record in records] == [
        "test-1",
        "test-2",
    ]
    assert all(record.is_correct for record in records)


def test_cycle_rejects_unknown_predicted_label() -> None:
    bundle = make_bundle()
    classifier = StubClassifier({"test-1": "NotAClass"})

    with pytest.raises(ValueError, match="unknown label"):
        run_classification_cycle(
            bundle=bundle,
            classifier=classifier,
            examples=bundle.test[:1],
        )
