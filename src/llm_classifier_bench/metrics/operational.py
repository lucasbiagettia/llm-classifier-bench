"""Latency and cost metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .base import EvaluationRecord, MetricResult, require_records


def _available_latencies(
    records: Sequence[EvaluationRecord],
) -> tuple[float, ...] | None:
    if any(record.latency_ms is None for record in records):
        return None
    return tuple(float(record.latency_ms) for record in records if record.latency_ms is not None)


def _available_costs(
    records: Sequence[EvaluationRecord],
) -> tuple[float, ...] | None:
    if any(record.cost_usd is None for record in records):
        return None
    return tuple(float(record.cost_usd) for record in records if record.cost_usd is not None)


def percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolated percentile, matching NumPy's default method."""

    frozen = tuple(values)
    if not frozen:
        raise ValueError("At least one value is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if any(not math.isfinite(value) for value in frozen):
        raise ValueError("Percentile values must be finite")

    ordered = sorted(frozen)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * weight


def mean_latency_ms(records: Sequence[EvaluationRecord]) -> float | None:
    frozen = require_records(records)
    values = _available_latencies(frozen)
    return None if values is None else sum(values) / len(values)


def latency_percentile_ms(
    records: Sequence[EvaluationRecord],
    *,
    quantile: float,
) -> float | None:
    frozen = require_records(records)
    values = _available_latencies(frozen)
    return None if values is None else percentile(values, quantile)


def total_cost_usd(records: Sequence[EvaluationRecord]) -> float | None:
    frozen = require_records(records)
    costs = _available_costs(frozen)
    return None if costs is None else sum(costs)


def cost_per_1000_usd(records: Sequence[EvaluationRecord]) -> float | None:
    frozen = require_records(records)
    total = total_cost_usd(frozen)
    return None if total is None else (total / len(frozen)) * 1000.0


@dataclass(frozen=True, slots=True)
class MeanLatencyMetric:
    name: str = "mean_latency_ms"

    def compute(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        frozen = require_records(records)
        value = mean_latency_ms(frozen)
        if value is None:
            return MetricResult.unavailable(
                name=self.name,
                reason="latency is missing for one or more records",
                metadata={"n_samples": len(frozen)},
            )
        return MetricResult(
            name=self.name,
            value=value,
            metadata={"n_samples": len(frozen), "unit": "milliseconds"},
        )


@dataclass(frozen=True, slots=True)
class LatencyP50Metric:
    name: str = "latency_p50_ms"

    def compute(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        frozen = require_records(records)
        value = latency_percentile_ms(frozen, quantile=0.50)
        if value is None:
            return MetricResult.unavailable(
                name=self.name,
                reason="latency is missing for one or more records",
                metadata={"n_samples": len(frozen)},
            )
        return MetricResult(
            name=self.name,
            value=value,
            metadata={
                "n_samples": len(frozen),
                "unit": "milliseconds",
                "quantile": 0.50,
                "interpolation": "linear",
            },
        )


@dataclass(frozen=True, slots=True)
class LatencyP99Metric:
    name: str = "latency_p99_ms"

    def compute(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        frozen = require_records(records)
        value = latency_percentile_ms(frozen, quantile=0.99)
        if value is None:
            return MetricResult.unavailable(
                name=self.name,
                reason="latency is missing for one or more records",
                metadata={"n_samples": len(frozen)},
            )
        return MetricResult(
            name=self.name,
            value=value,
            metadata={
                "n_samples": len(frozen),
                "unit": "milliseconds",
                "quantile": 0.99,
                "interpolation": "linear",
            },
        )


@dataclass(frozen=True, slots=True)
class TotalCostMetric:
    name: str = "total_cost_usd"

    def compute(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        frozen = require_records(records)
        value = total_cost_usd(frozen)
        if value is None:
            return MetricResult.unavailable(
                name=self.name,
                reason="cost_usd is missing for one or more records",
                metadata={"n_samples": len(frozen)},
            )
        return MetricResult(
            name=self.name,
            value=value,
            metadata={"n_samples": len(frozen), "currency": "USD"},
        )


@dataclass(frozen=True, slots=True)
class CostPer1000Metric:
    name: str = "cost_per_1000_usd"

    def compute(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        frozen = require_records(records)
        value = cost_per_1000_usd(frozen)
        if value is None:
            return MetricResult.unavailable(
                name=self.name,
                reason="cost_usd is missing for one or more records",
                metadata={"n_samples": len(frozen)},
            )
        return MetricResult(
            name=self.name,
            value=value,
            metadata={
                "n_samples": len(frozen),
                "currency": "USD",
                "unit": "per_1000_predictions",
            },
        )


__all__ = [
    "CostPer1000Metric",
    "LatencyP50Metric",
    "LatencyP99Metric",
    "MeanLatencyMetric",
    "TotalCostMetric",
    "cost_per_1000_usd",
    "latency_percentile_ms",
    "mean_latency_ms",
    "percentile",
    "total_cost_usd",
]
