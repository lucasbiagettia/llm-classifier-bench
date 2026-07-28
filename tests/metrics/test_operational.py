from __future__ import annotations

import pytest

from llm_classifier_bench.metrics import (
    CostPer1000Metric,
    EvaluationRecord,
    LatencyP50Metric,
    LatencyP99Metric,
    MeanLatencyMetric,
    TotalCostMetric,
    cost_per_1000_usd,
    percentile,
    total_cost_usd,
)


def record(
    sample_id: str,
    *,
    latency_ms: float | None,
    cost_usd: float | None,
) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=sample_id,
        gold_label="A",
        predicted_label="A",
        confidence=1.0,
        probabilities={"A": 1.0},
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


def test_latency_metrics() -> None:
    records = tuple(
        record(str(index), latency_ms=latency, cost_usd=0.001 * index)
        for index, latency in enumerate((10.0, 20.0, 30.0, 40.0), start=1)
    )

    assert percentile((10.0, 20.0, 30.0, 40.0), 0.5) == pytest.approx(25.0)
    assert MeanLatencyMetric().compute(records).value == pytest.approx(25.0)
    assert LatencyP50Metric().compute(records).value == pytest.approx(25.0)
    assert LatencyP99Metric().compute(records).value == pytest.approx(39.7)


def test_cost_metrics() -> None:
    records = (
        record("1", latency_ms=1.0, cost_usd=0.001),
        record("2", latency_ms=1.0, cost_usd=0.002),
        record("3", latency_ms=1.0, cost_usd=0.003),
        record("4", latency_ms=1.0, cost_usd=0.004),
    )

    assert total_cost_usd(records) == pytest.approx(0.01)
    assert cost_per_1000_usd(records) == pytest.approx(2.5)
    assert TotalCostMetric().compute(records).value == pytest.approx(0.01)
    assert CostPer1000Metric().compute(records).value == pytest.approx(2.5)


def test_cost_metrics_are_unavailable_when_cost_is_missing() -> None:
    records = (record("1", latency_ms=1.0, cost_usd=None),)

    assert TotalCostMetric().compute(records).value is None
    assert CostPer1000Metric().compute(records).value is None
