"""Minimal dataset-to-classifier cycle used before the benchmark runner exists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from llm_classifier_bench.classifiers.base import Classifier, Prediction
from llm_classifier_bench.core import LabeledExample
from llm_classifier_bench.datasets.base import DatasetBundle


@dataclass(frozen=True, slots=True)
class ClassificationRecord:
    """One gold-labeled example paired with its classifier prediction."""

    example: LabeledExample
    prediction: Prediction

    @property
    def is_correct(self) -> bool:
        return self.example.label == self.prediction.predicted_label


def run_classification_cycle(
    *,
    bundle: DatasetBundle,
    classifier: Classifier,
    examples: Sequence[LabeledExample],
) -> tuple[ClassificationRecord, ...]:
    """Fit once, classify explicit examples, and validate interface compatibility.

    ``examples`` is deliberately required instead of defaulting to the full test
    split. This prevents an accidental large or expensive API run.
    """

    selected = tuple(examples)
    if not selected:
        raise ValueError("At least one example is required")

    allowed_labels = set(bundle.class_names)
    unknown_gold_labels = sorted(
        {example.label for example in selected} - allowed_labels
    )
    if unknown_gold_labels:
        raise ValueError(
            "Selected examples contain labels outside the dataset classes: "
            + ", ".join(unknown_gold_labels)
        )

    sample_ids = [example.sample_id for example in selected]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Selected examples must have unique sample ids")

    classifier.fit(bundle.train)
    predictions = tuple(
        classifier.predict(tuple(example.as_input() for example in selected))
    )

    if len(predictions) != len(selected):
        raise ValueError(
            f"Classifier returned {len(predictions)} predictions for "
            f"{len(selected)} inputs"
        )

    records: list[ClassificationRecord] = []
    for example, prediction in zip(selected, predictions, strict=True):
        if prediction.sample_id != example.sample_id:
            raise ValueError(
                "Classifier did not preserve input order/sample ids: "
                f"expected {example.sample_id!r}, got {prediction.sample_id!r}"
            )
        if prediction.predicted_label not in allowed_labels:
            raise ValueError(
                f"Classifier returned unknown label {prediction.predicted_label!r}"
            )
        if (
            prediction.probabilities is not None
            and prediction.predicted_label not in prediction.probabilities
        ):
            raise ValueError(
                "predicted_label is missing from the returned probability mapping"
            )

        records.append(
            ClassificationRecord(example=example, prediction=prediction)
        )

    return tuple(records)
