from __future__ import annotations

import pytest

from llm_classifier_bench.metrics import (
    AccuracyMetric,
    EvaluationRecord,
    MacroF1Metric,
    accuracy_score,
    macro_f1_score,
)


def record(sample_id: str, gold: str, predicted: str) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=sample_id,
        gold_label=gold,
        predicted_label=predicted,
        confidence=None,
        probabilities=None,
        latency_ms=1.0,
    )


def test_accuracy_and_macro_f1() -> None:
    records = (
        record("1", "A", "A"),
        record("2", "A", "B"),
        record("3", "B", "B"),
        record("4", "B", "B"),
    )

    assert accuracy_score(records) == pytest.approx(0.75)

    macro, per_class = macro_f1_score(records)
    assert per_class == pytest.approx({"A": 2 / 3, "B": 0.8})
    assert macro == pytest.approx((2 / 3 + 0.8) / 2)

    assert AccuracyMetric().compute(records).value == pytest.approx(0.75)
    assert MacroF1Metric().compute(records).value == pytest.approx(macro)


def test_macro_f1_can_use_full_declared_label_space() -> None:
    records = (record("1", "A", "A"),)
    result = MacroF1Metric(labels=("A", "B")).compute(records)

    assert result.value == pytest.approx(0.5)
    assert result.metadata["per_class_f1"] == {"A": 1.0, "B": 0.0}
