"""General benchmark-run orchestration.

The runner intentionally depends only on the stable dataset, classifier, and metrics
contracts. It does not know how any concrete dataset is loaded internally or how a
classifier is trained/invoked.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm_classifier_bench.classifiers.base import Classifier, Prediction
from llm_classifier_bench.datasets.base import ClassificationDataset, DatasetBundle
from llm_classifier_bench.metrics.evaluator import evaluate_jsonl, write_results_json


@dataclass(frozen=True, slots=True)
class BenchmarkRunConfig:
    """Configuration for one benchmark run.

    Sampling and class-subset policy deliberately do not live here yet. The formal
    experimental protocol for 5/10/20-label subsets is still an open methodological
    decision. This runner executes the dataset bundle it receives from ``dataset.load``.

    ``metadata`` is the escape hatch for reproducibility information that is specific
    to a classifier or experiment but is not part of the frozen Classifier contract,
    for example model version, prompt version, or training hyperparameters.
    """

    output_root: Path = Path("artifacts/runs")
    run_id: str | None = None
    evaluate: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    """Paths and identifying information produced by a completed run."""

    run_id: str
    run_dir: Path
    config_path: Path
    predictions_path: Path
    status_path: Path
    metrics_path: Path | None
    example_count: int


def run_benchmark(
    dataset: ClassificationDataset,
    classifier: Classifier,
    config: BenchmarkRunConfig | None = None,
) -> BenchmarkRunResult:
    """Execute one dataset/classifier benchmark run.

    Order of operations is intentionally explicit:

    1. load the dataset;
    2. persist the exact run configuration and sample IDs;
    3. call ``classifier.fit`` on the training split (possibly a no-op);
    4. classify the test split without exposing gold labels;
    5. validate the normalized prediction contract;
    6. persist raw normalized predictions as JSONL;
    7. compute metrics from that persisted artifact.

    Failures are written to ``status.json`` and then re-raised. A future matrix
    orchestrator can therefore catch one failed run and continue with the remaining
    jobs without losing the failure record.
    """

    resolved_config = config or BenchmarkRunConfig()
    run_id = resolved_config.run_id or _default_run_id(dataset.name, classifier.name)
    run_dir = Path(resolved_config.output_root) / run_id

    # Refuse to overwrite a previous run. This is especially important when classifier
    # calls are paid: an accidental rerun must be an explicit choice.
    run_dir.mkdir(parents=True, exist_ok=False)

    config_path = run_dir / "config.json"
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    status_path = run_dir / "status.json"

    stage = "loading_dataset"
    _write_status(
        status_path,
        status="running",
        stage=stage,
        run_id=run_id,
        dataset=dataset.name,
        classifier=classifier.name,
    )

    try:
        bundle = dataset.load()

        # Persist the exact data identity before fitting or making inference calls.
        # If a later stage fails, we still know which examples the run intended to use.
        _write_run_config(
            config_path,
            run_id=run_id,
            dataset=bundle,
            classifier_name=classifier.name,
            config=resolved_config,
        )

        stage = "fitting_classifier"
        _write_status(
            status_path,
            status="running",
            stage=stage,
            run_id=run_id,
            dataset=bundle.name,
            classifier=classifier.name,
        )
        classifier.fit(bundle.train)

        stage = "predicting"
        _write_status(
            status_path,
            status="running",
            stage=stage,
            run_id=run_id,
            dataset=bundle.name,
            classifier=classifier.name,
        )
        test_inputs = [example.as_input() for example in bundle.test]
        predictions = classifier.predict(test_inputs)
        _validate_predictions(bundle, predictions)

        stage = "writing_predictions"
        _write_status(
            status_path,
            status="running",
            stage=stage,
            run_id=run_id,
            dataset=bundle.name,
            classifier=classifier.name,
        )
        _write_predictions_jsonl(
            predictions_path,
            bundle=bundle,
            classifier_name=classifier.name,
            predictions=predictions,
        )

        resolved_metrics_path: Path | None = None
        if resolved_config.evaluate:
            stage = "evaluating_metrics"
            _write_status(
                status_path,
                status="running",
                stage=stage,
                run_id=run_id,
                dataset=bundle.name,
                classifier=classifier.name,
            )
            metric_results = evaluate_jsonl(predictions_path)
            write_results_json(metrics_path, metric_results)
            resolved_metrics_path = metrics_path

        _write_status(
            status_path,
            status="completed",
            stage="completed",
            run_id=run_id,
            dataset=bundle.name,
            classifier=classifier.name,
            example_count=len(bundle.test),
        )

        return BenchmarkRunResult(
            run_id=run_id,
            run_dir=run_dir,
            config_path=config_path,
            predictions_path=predictions_path,
            status_path=status_path,
            metrics_path=resolved_metrics_path,
            example_count=len(bundle.test),
        )

    except Exception as exc:
        _write_status(
            status_path,
            status="failed",
            stage=stage,
            run_id=run_id,
            dataset=dataset.name,
            classifier=classifier.name,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


def _validate_predictions(
    bundle: DatasetBundle,
    predictions: Sequence[Prediction],
) -> None:
    if len(predictions) != len(bundle.test):
        raise ValueError(
            "Classifier returned a different number of predictions than test inputs: "
            f"expected {len(bundle.test)}, got {len(predictions)}"
        )

    valid_labels = set(bundle.class_names)

    for index, (example, prediction) in enumerate(zip(bundle.test, predictions)):
        if prediction.sample_id != example.sample_id:
            raise ValueError(
                "Classifier did not preserve input order/sample_id at position "
                f"{index}: expected {example.sample_id!r}, got {prediction.sample_id!r}"
            )

        if prediction.predicted_label not in valid_labels:
            raise ValueError(
                f"Prediction for {prediction.sample_id!r} returned unknown label "
                f"{prediction.predicted_label!r}"
            )

        if prediction.latency_ms < 0 or not math.isfinite(prediction.latency_ms):
            raise ValueError(
                f"Prediction for {prediction.sample_id!r} has invalid latency_ms "
                f"{prediction.latency_ms!r}"
            )

        probabilities = prediction.probabilities
        if probabilities is None:
            continue

        if prediction.predicted_label not in probabilities:
            raise ValueError(
                f"Prediction probabilities for {prediction.sample_id!r} do not contain "
                "the predicted label"
            )

        normalized: dict[str, float] = {}
        for label, value in probabilities.items():
            if label not in valid_labels:
                raise ValueError(
                    f"Prediction probabilities for {prediction.sample_id!r} contain "
                    f"unknown label {label!r}"
                )
            probability = float(value)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"Prediction probabilities for {prediction.sample_id!r} contain "
                    f"invalid value {value!r} for {label!r}"
                )
            normalized[label] = probability

        if not math.isclose(sum(normalized.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"Prediction probabilities for {prediction.sample_id!r} do not sum to 1"
            )

        argmax_label = max(normalized, key=normalized.__getitem__)
        if argmax_label != prediction.predicted_label:
            raise ValueError(
                f"Prediction for {prediction.sample_id!r} is not the argmax of its "
                "probability distribution"
            )

        if prediction.confidence is not None and not math.isclose(
            float(prediction.confidence),
            normalized[prediction.predicted_label],
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"Prediction confidence for {prediction.sample_id!r} does not match "
                "the probability of the predicted label"
            )


def _write_predictions_jsonl(
    path: Path,
    *,
    bundle: DatasetBundle,
    classifier_name: str,
    predictions: Sequence[Prediction],
) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for example, prediction in zip(bundle.test, predictions):
            payload = {
                "dataset": bundle.name,
                "classifier": classifier_name,
                "sample_id": example.sample_id,
                "input": example.text,
                "gold_label": example.label,
                "predicted_label": prediction.predicted_label,
                "correct": example.label == prediction.predicted_label,
                "confidence": prediction.confidence,
                # Preserve the current artifact convention used by the smoke probe.
                "probabilities": dict(prediction.probabilities or {}),
                "latency_ms": prediction.latency_ms,
                "model": prediction.model,
                "request_id": prediction.request_id,
                "raw_response": dict(prediction.raw_response or {}),
            }
            output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_run_config(
    path: Path,
    *,
    run_id: str,
    dataset: DatasetBundle,
    classifier_name: str,
    config: BenchmarkRunConfig,
) -> None:
    payload = {
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "dataset": {
            "name": dataset.name,
            "metadata": dict(dataset.metadata),
            "classes": [
                {"name": class_definition.name, "description": class_definition.description}
                for class_definition in dataset.classes
            ],
            "train_sample_ids": [example.sample_id for example in dataset.train],
            "test_sample_ids": [example.sample_id for example in dataset.test],
            "train_size": len(dataset.train),
            "test_size": len(dataset.test),
        },
        "classifier": {
            "name": classifier_name,
        },
        "run_metadata": dict(config.metadata),
    }
    _write_json(path, payload)


def _write_status(
    path: Path,
    *,
    status: str,
    stage: str,
    run_id: str,
    dataset: str,
    classifier: str,
    **extra: Any,
) -> None:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "stage": stage,
        "updated_at_utc": _utc_now(),
        "dataset": dataset,
        "classifier": classifier,
    }
    payload.update(extra)
    _write_json(path, payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2, default=str)
        output_file.write("\n")


def _default_run_id(dataset_name: str, classifier_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}__{_slug(dataset_name)}__{_slug(classifier_name)}"


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")
    return normalized or "unnamed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
