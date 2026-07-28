"""Probability-quality and calibration metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .base import EvaluationRecord, MetricResult, require_records


def _top_confidence(record: EvaluationRecord) -> float | None:
    if record.confidence is not None:
        return record.confidence
    if record.probabilities:
        return max(record.probabilities.values())
    return None


def _confidence_rows(
    records: Sequence[EvaluationRecord],
) -> tuple[tuple[float, bool], ...] | None:
    rows: list[tuple[float, bool]] = []
    for record in records:
        confidence = _top_confidence(record)
        if confidence is None:
            return None
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Invalid confidence for sample {record.sample_id!r}: {confidence!r}"
            )
        rows.append((confidence, record.gold_label == record.predicted_label))
    return tuple(rows)


def _validate_probability_distributions(
    records: Sequence[EvaluationRecord],
    *,
    sum_tolerance: float,
) -> tuple[str, ...] | None:
    if any(record.probabilities is None for record in records):
        return None

    first = records[0].probabilities
    assert first is not None
    labels = tuple(first.keys())
    if not labels:
        raise ValueError("Probability distributions cannot be empty")
    label_set = set(labels)

    for record in records:
        probabilities = record.probabilities
        assert probabilities is not None
        if set(probabilities) != label_set:
            raise ValueError(
                "Every probability distribution must contain the same class labels"
            )
        total = 0.0
        for label, probability in probabilities.items():
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"Invalid probability for sample {record.sample_id!r}, "
                    f"class {label!r}: {probability!r}"
                )
            total += probability
        if not math.isclose(total, 1.0, abs_tol=sum_tolerance, rel_tol=0.0):
            raise ValueError(
                f"Probabilities for sample {record.sample_id!r} sum to {total}, not 1"
            )
        if record.gold_label not in probabilities:
            raise ValueError(
                f"Gold label {record.gold_label!r} is missing from probabilities "
                f"for sample {record.sample_id!r}"
            )

    return labels


def top_label_ece_score(
    records: Sequence[EvaluationRecord],
    *,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]] | None:
    """Fixed-width top-label Expected Calibration Error."""

    frozen = require_records(records)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    rows = _confidence_rows(frozen)
    if rows is None:
        return None

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for confidence, correct in rows:
        index = min(int(confidence * n_bins), n_bins - 1)
        bins[index].append((confidence, correct))

    details: list[dict[str, Any]] = []
    ece = 0.0
    total_count = len(rows)
    for index, bucket in enumerate(bins):
        if not bucket:
            continue
        count = len(bucket)
        mean_confidence = sum(confidence for confidence, _ in bucket) / count
        accuracy = sum(correct for _, correct in bucket) / count
        gap = abs(accuracy - mean_confidence)
        ece += (count / total_count) * gap
        details.append(
            {
                "bin_index": index,
                "lower": index / n_bins,
                "upper": (index + 1) / n_bins,
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "absolute_gap": gap,
            }
        )

    return ece, details


def adaptive_ece_score(
    records: Sequence[EvaluationRecord],
    *,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]] | None:
    """Equal-frequency top-label ECE.

    Records are sorted by confidence and partitioned into bins whose sizes differ
    by at most one. With fewer samples than requested bins, each sample forms its
    own bin.
    """

    frozen = require_records(records)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    rows = _confidence_rows(frozen)
    if rows is None:
        return None

    sorted_rows = sorted(rows, key=lambda item: item[0])
    effective_bins = min(n_bins, len(sorted_rows))
    base_size, remainder = divmod(len(sorted_rows), effective_bins)

    details: list[dict[str, Any]] = []
    ece = 0.0
    cursor = 0
    for index in range(effective_bins):
        size = base_size + (1 if index < remainder else 0)
        bucket = sorted_rows[cursor : cursor + size]
        cursor += size

        mean_confidence = sum(confidence for confidence, _ in bucket) / size
        accuracy = sum(correct for _, correct in bucket) / size
        gap = abs(accuracy - mean_confidence)
        ece += (size / len(sorted_rows)) * gap
        details.append(
            {
                "bin_index": index,
                "min_confidence": bucket[0][0],
                "max_confidence": bucket[-1][0],
                "count": size,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "absolute_gap": gap,
            }
        )

    return ece, details


def multiclass_log_loss_score(
    records: Sequence[EvaluationRecord],
    *,
    epsilon: float = 1e-15,
    sum_tolerance: float = 1e-5,
) -> float | None:
    frozen = require_records(records)
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5")

    labels = _validate_probability_distributions(
        frozen,
        sum_tolerance=sum_tolerance,
    )
    if labels is None:
        return None

    losses = []
    for record in frozen:
        assert record.probabilities is not None
        probability = max(epsilon, min(1.0 - epsilon, record.probabilities[record.gold_label]))
        losses.append(-math.log(probability))
    return sum(losses) / len(losses)


def multiclass_brier_score(
    records: Sequence[EvaluationRecord],
    *,
    sum_tolerance: float = 1e-5,
) -> float | None:
    """Mean multiclass Brier score using the unnormalized classwise sum.

    For each sample this computes ``sum_k (p_k - y_k)^2`` and then averages over
    samples. Its range is [0, 2], where lower is better.
    """

    frozen = require_records(records)
    labels = _validate_probability_distributions(
        frozen,
        sum_tolerance=sum_tolerance,
    )
    if labels is None:
        return None

    total = 0.0
    for record in frozen:
        assert record.probabilities is not None
        total += sum(
            (
                record.probabilities[label]
                - (1.0 if label == record.gold_label else 0.0)
            )
            ** 2
            for label in labels
        )
    return total / len(frozen)


@dataclass(frozen=True, slots=True)
class TopLabelECEMetric:
    n_bins: int = 10
    name: str = "top_label_ece"

    def compute(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        frozen = require_records(records)
        result = top_label_ece_score(frozen, n_bins=self.n_bins)
        if result is None:
            return MetricResult.unavailable(
                name=self.name,
                reason="confidence or probabilities are missing",
                metadata={"n_samples": len(frozen), "n_bins": self.n_bins},
            )
        value, bins = result
        return MetricResult(
            name=self.name,
            value=value,
            metadata={
                "n_samples": len(frozen),
                "n_bins": self.n_bins,
                "binning": "fixed_width",
                "bins": bins,
            },
        )


@dataclass(frozen=True, slots=True)
class AdaptiveECEMetric:
    n_bins: int = 10
    name: str = "adaptive_ece"

    def compute(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        frozen = require_records(records)
        result = adaptive_ece_score(frozen, n_bins=self.n_bins)
        if result is None:
            return MetricResult.unavailable(
                name=self.name,
                reason="confidence or probabilities are missing",
                metadata={"n_samples": len(frozen), "n_bins": self.n_bins},
            )
        value, bins = result
        return MetricResult(
            name=self.name,
            value=value,
            metadata={
                "n_samples": len(frozen),
                "requested_bins": self.n_bins,
                "effective_bins": len(bins),
                "binning": "equal_frequency",
                "bins": bins,
            },
        )


@dataclass(frozen=True, slots=True)
class MulticlassLogLossMetric:
    epsilon: float = 1e-15
    sum_tolerance: float = 1e-5
    name: str = "multiclass_log_loss"

    def compute(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        frozen = require_records(records)
        value = multiclass_log_loss_score(
            frozen,
            epsilon=self.epsilon,
            sum_tolerance=self.sum_tolerance,
        )
        if value is None:
            return MetricResult.unavailable(
                name=self.name,
                reason="full probability distributions are missing",
                metadata={"n_samples": len(frozen)},
            )
        return MetricResult(
            name=self.name,
            value=value,
            metadata={
                "n_samples": len(frozen),
                "epsilon": self.epsilon,
                "natural_log": True,
            },
        )


@dataclass(frozen=True, slots=True)
class MulticlassBrierScoreMetric:
    sum_tolerance: float = 1e-5
    name: str = "multiclass_brier_score"

    def compute(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        frozen = require_records(records)
        value = multiclass_brier_score(
            frozen,
            sum_tolerance=self.sum_tolerance,
        )
        if value is None:
            return MetricResult.unavailable(
                name=self.name,
                reason="full probability distributions are missing",
                metadata={"n_samples": len(frozen)},
            )
        return MetricResult(
            name=self.name,
            value=value,
            metadata={
                "n_samples": len(frozen),
                "classwise_reduction": "sum",
                "sample_reduction": "mean",
                "range": [0.0, 2.0],
            },
        )


__all__ = [
    "AdaptiveECEMetric",
    "MulticlassBrierScoreMetric",
    "MulticlassLogLossMetric",
    "TopLabelECEMetric",
    "adaptive_ece_score",
    "multiclass_brier_score",
    "multiclass_log_loss_score",
    "top_label_ece_score",
]
