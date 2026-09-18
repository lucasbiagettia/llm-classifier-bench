"""Adapters and orchestration for computing metrics from memory or JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .base import EvaluationRecord, Metric, MetricResult, require_records
from .calibration import (
    AdaptiveECEMetric,
    MulticlassBrierScoreMetric,
    MulticlassLogLossMetric,
    TopLabelECEMetric,
)
from .classification import AccuracyMetric, MacroF1Metric
from .operational import (
    CostPer1000Metric,
    LatencyP50Metric,
    LatencyP95Metric,
    LatencyP99Metric,
    MeanLatencyMetric,
    TotalCostMetric,
)


DEFAULT_METRICS: tuple[Metric, ...] = (
    AccuracyMetric(),
    MacroF1Metric(),
    TopLabelECEMetric(),
    AdaptiveECEMetric(),
    MulticlassLogLossMetric(),
    MulticlassBrierScoreMetric(),
    MeanLatencyMetric(),
    LatencyP50Metric(),
    LatencyP95Metric(),
    LatencyP99Metric(),
    TotalCostMetric(),
    CostPer1000Metric(),
)


def from_classification_records(records: Iterable[Any]) -> tuple[EvaluationRecord, ...]:
    """Adapt existing workflow ``ClassificationRecord`` objects without imports.

    Duck typing keeps this metrics package independent from the temporary workflow
    layer and avoids changing any existing interface.
    """

    normalized: list[EvaluationRecord] = []
    for record in records:
        example = record.example
        prediction = record.prediction
        normalized.append(
            EvaluationRecord(
                sample_id=example.sample_id,
                gold_label=example.label,
                predicted_label=prediction.predicted_label,
                confidence=prediction.confidence,
                probabilities=prediction.probabilities,
                latency_ms=prediction.latency_ms,
                cost_usd=None,
                metadata={
                    "input": example.text,
                    "model": prediction.model,
                    "request_id": prediction.request_id,
                    "raw_response": prediction.raw_response,
                },
            )
        )
    return tuple(normalized)


def evaluation_record_from_mapping(payload: Mapping[str, Any]) -> EvaluationRecord:
    probabilities = payload.get("probabilities")
    if probabilities is not None and not isinstance(probabilities, Mapping):
        raise ValueError("probabilities must be a label-to-probability mapping")
    return EvaluationRecord(
        sample_id=str(payload["sample_id"]),
        gold_label=str(payload["gold_label"]),
        predicted_label=str(payload["predicted_label"]),
        confidence=(
            float(payload["confidence"])
            if payload.get("confidence") is not None
            else None
        ),
        probabilities=(
            {str(label): float(value) for label, value in probabilities.items()}
            if probabilities
            else None
        ),
        latency_ms=(
            float(payload["latency_ms"])
            if payload.get("latency_ms") is not None
            else None
        ),
        cost_usd=(
            float(payload["cost_usd"])
            if payload.get("cost_usd") is not None
            else None
        ),
        metadata={
            key: value
            for key, value in payload.items()
            if key
            not in {
                "sample_id",
                "gold_label",
                "predicted_label",
                "confidence",
                "probabilities",
                "latency_ms",
                "cost_usd",
            }
        },
    )


def load_evaluation_records(path: str | Path) -> tuple[EvaluationRecord, ...]:
    artifact_path = Path(path)
    records: list[EvaluationRecord] = []
    with artifact_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(evaluation_record_from_mapping(payload))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid evaluation artifact at {artifact_path}:{line_number}: {exc}"
                ) from exc
    return require_records(records)


def evaluate_records(
    records: Sequence[EvaluationRecord],
    *,
    metrics: Sequence[Metric] | None = None,
) -> tuple[MetricResult, ...]:
    frozen = require_records(records)
    selected_metrics = tuple(metrics) if metrics is not None else DEFAULT_METRICS
    if not selected_metrics:
        raise ValueError("At least one metric is required")

    names = [metric.name for metric in selected_metrics]
    if len(names) != len(set(names)):
        raise ValueError("Metric names must be unique within one evaluation")

    return tuple(metric.compute(frozen) for metric in selected_metrics)


def evaluate_jsonl(
    path: str | Path,
    *,
    metrics: Sequence[Metric] | None = None,
) -> tuple[MetricResult, ...]:
    records = load_evaluation_records(path)
    results = evaluate_records(records, metrics=metrics)
    artifact = Path(path)
    if artifact.name != "predictions.jsonl" or not (artifact.parent / "usage.jsonl").exists():
        return results
    # Replay the ledger, not the per-prediction sum: warmup and failed attempts
    # have costs too. A copied/subset artifact must not borrow another run's total.
    from llm_classifier_bench.costs import recalculate_costs
    report = recalculate_costs(artifact.parent)
    if {r.sample_id for r in records} != set(report["per_successful_sample"]):
        raise ValueError("Prediction sample IDs do not match the inference usage ledger")
    values = {"total_cost_usd": report["total_cost_usd"],
              "cost_per_1000_usd": report["cost_per_1000_successful_examples_usd"]}
    metadata = {k: report[k] for k in ("scope", "cost_kind", "currency", "coverage_complete",
                "known_cost_subtotal_usd", "failure_rate", "successful_examples", "usage_sha256", "pricing_sha256")}
    metadata["source"] = "usage.jsonl + pricing.json + measurement.json"
    return tuple(MetricResult(r.name, values[r.name], metadata) if r.name in values else r for r in results)


def results_as_dict(results: Sequence[MetricResult]) -> dict[str, Any]:
    return {result.name: result.as_dict() for result in results}


def write_results_json(
    path: str | Path,
    results: Sequence[MetricResult],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results_as_dict(results), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "DEFAULT_METRICS",
    "evaluate_jsonl",
    "evaluate_records",
    "evaluation_record_from_mapping",
    "from_classification_records",
    "load_evaluation_records",
    "results_as_dict",
    "write_results_json",
]
