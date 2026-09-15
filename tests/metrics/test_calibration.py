from __future__ import annotations

import math
from dataclasses import replace

import pytest

from llm_classifier_bench.metrics import (
    AdaptiveECEMetric,
    EvaluationRecord,
    MulticlassBrierScoreMetric,
    MulticlassLogLossMetric,
    TopLabelECEMetric,
    adaptive_ece_score,
    multiclass_brier_score,
    multiclass_log_loss_score,
    top_label_ece_score,
)


def confidence_record(
    sample_id: str,
    *,
    confidence: float,
    correct: bool,
) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=sample_id,
        gold_label="A",
        predicted_label="A" if correct else "B",
        confidence=confidence,
        probabilities=None,
        latency_ms=1.0,
    )


def probability_record(
    sample_id: str,
    *,
    gold: str,
    probabilities: dict[str, float],
) -> EvaluationRecord:
    predicted = max(probabilities, key=probabilities.get)
    return EvaluationRecord(
        sample_id=sample_id,
        gold_label=gold,
        predicted_label=predicted,
        confidence=probabilities[predicted],
        probabilities=probabilities,
        latency_ms=1.0,
    )


def test_fixed_and_adaptive_ece() -> None:
    records = (
        confidence_record("1", confidence=0.9, correct=True),
        confidence_record("2", confidence=0.8, correct=True),
        confidence_record("3", confidence=0.6, correct=False),
        confidence_record("4", confidence=0.2, correct=True),
    )

    fixed = top_label_ece_score(records, n_bins=2)
    adaptive = adaptive_ece_score(records, n_bins=2)

    assert fixed is not None
    assert adaptive is not None
    assert fixed[0] == pytest.approx(0.275)
    assert adaptive[0] == pytest.approx(0.125)
    assert TopLabelECEMetric(n_bins=2).compute(records).value == pytest.approx(0.275)
    assert AdaptiveECEMetric(n_bins=2).compute(records).value == pytest.approx(0.125)


def test_log_loss_and_multiclass_brier() -> None:
    records = (
        probability_record("1", gold="A", probabilities={"A": 0.8, "B": 0.2}),
        probability_record("2", gold="B", probabilities={"A": 0.25, "B": 0.75}),
    )

    expected_log_loss = -(math.log(0.8) + math.log(0.75)) / 2
    assert multiclass_log_loss_score(records) == pytest.approx(expected_log_loss)
    assert multiclass_brier_score(records) == pytest.approx(0.1025)
    assert MulticlassLogLossMetric().compute(records).value == pytest.approx(
        expected_log_loss
    )
    assert MulticlassBrierScoreMetric().compute(records).value == pytest.approx(
        0.1025
    )


def test_probability_metrics_are_unavailable_without_distributions() -> None:
    records = (confidence_record("1", confidence=0.8, correct=True),)

    assert MulticlassLogLossMetric().compute(records).value is None
    assert MulticlassBrierScoreMetric().compute(records).value is None


def test_invalid_probability_sum_is_rejected() -> None:
    records = (
        probability_record("1", gold="A", probabilities={"A": 0.8, "B": 0.3}),
    )

    with pytest.raises(ValueError, match="sum to"):
        multiclass_log_loss_score(records)


@pytest.mark.parametrize("metric", [top_label_ece_score, adaptive_ece_score,
                                  multiclass_log_loss_score, multiclass_brier_score])
@pytest.mark.parametrize("changes, message", [
    ({"confidence": 0.2}, "does not match"),
    ({"predicted_label": "B", "confidence": None}, "argmax"),
    ({"predicted_label": "C", "confidence": None}, "Predicted label"),
    ({"probabilities": {"A": 0.8, "B": 0.3}}, "sum to"),
    ({"probabilities": {"A": float("nan"), "B": 0.2}}, "Invalid probability"),
])
def test_probability_metrics_reject_inconsistent_inputs(metric, changes, message) -> None:
    record = probability_record("1", gold="A", probabilities={"A": 0.8, "B": 0.2})
    with pytest.raises(ValueError, match=message):
        metric((replace(record, **changes),))


def test_probability_map_order_does_not_change_metrics() -> None:
    records = (
        probability_record("1", gold="B", probabilities={"B": 0.8, "A": 0.2}),
        probability_record("2", gold="A", probabilities={"A": 0.7, "B": 0.3}),
    )
    without_confidence = tuple(replace(r, confidence=None) for r in records)
    assert top_label_ece_score(without_confidence)[0] == pytest.approx(0.25)
    assert adaptive_ece_score(without_confidence)[0] == pytest.approx(0.25)
    assert multiclass_log_loss_score(records) == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)
    assert multiclass_brier_score(records) == pytest.approx(0.13)


def test_ece_boundary_confidences_and_tied_argmax() -> None:
    records = (confidence_record("0", confidence=0.0, correct=False),
               confidence_record("1", confidence=1.0, correct=True))
    assert top_label_ece_score(records)[0] == 0.0
    assert adaptive_ece_score(records)[0] == 0.0
    tied = probability_record("tie", gold="B", probabilities={"A": 0.5, "B": 0.5})
    assert top_label_ece_score((replace(tied, predicted_label="B"),))[0] == 0.5
