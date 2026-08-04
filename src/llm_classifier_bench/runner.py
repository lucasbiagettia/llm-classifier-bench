"""General benchmark-run orchestration."""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm_classifier_bench.classifiers.base import Classifier, Prediction
from llm_classifier_bench.class_definitions.loader import load_class_definition_profile
from llm_classifier_bench.config import DEFAULT_SPLIT_SEED, DEFAULT_VALIDATION_FRACTION
from llm_classifier_bench.core import LabeledExample
from llm_classifier_bench.datasets.base import ClassificationDataset, DatasetBundle
from llm_classifier_bench.metrics.evaluator import evaluate_jsonl, write_results_json


@dataclass(frozen=True, slots=True)
class BenchmarkRunConfig:
    """Configuration for one benchmark run.

    The dataset's original ``test`` split is always the final held-out benchmark
    evaluation set. Only ``train`` is partitioned into fit/validation examples.
    """

    output_root: Path = Path("artifacts/runs")
    run_id: str | None = None
    evaluate: bool = True
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    split_seed: int = DEFAULT_SPLIT_SEED
    metadata: Mapping[str, Any] = field(default_factory=dict)
    class_definitions_path: Path | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")


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

    Lifecycle:

    1. load and normalize the dataset;
    2. optionally load a frozen class-definition profile and validate that it
       matches the dataset's canonical label inventory exactly;
    3. split only the dataset train split into fit/validation partitions;
    4. persist the exact data identity and run configuration;
    5. ``classifier.prepare(classes)``;
    6. ``classifier.fit(train, validation_examples=validation)``;
    7. predict the untouched dataset test split;
    8. validate and persist normalized predictions;
    9. compute metrics from the persisted artifact.
    """

    resolved_config = config or BenchmarkRunConfig()
    run_id = resolved_config.run_id or _default_run_id(dataset.name, classifier.name)
    run_dir = Path(resolved_config.output_root) / run_id
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

        class_definitions_metadata: Mapping[str, Any] = {
            "source": "dataset_bundle",
            "profile": None,
        }
        if resolved_config.class_definitions_path is not None:
            stage = "loading_class_definitions"
            loaded_definitions = load_class_definition_profile(
                resolved_config.class_definitions_path
            )
            resolved_classes = loaded_definitions.definitions_for(
                dataset_name=bundle.name,
                canonical_names=bundle.class_names,
            )
            bundle = replace(bundle, classes=resolved_classes)
            class_definitions_metadata = loaded_definitions.benchmark_metadata()

        stage = "splitting_training_data"
        fit_train, validation = split_train_validation(
            bundle.train,
            validation_fraction=resolved_config.validation_fraction,
            seed=resolved_config.split_seed,
        )

        _write_run_config(
            config_path,
            run_id=run_id,
            dataset=bundle,
            classifier=classifier,
            config=resolved_config,
            fit_train=fit_train,
            validation=validation,
            class_definitions_metadata=class_definitions_metadata,
        )

        stage = "preparing_classifier"
        _write_status(
            status_path,
            status="running",
            stage=stage,
            run_id=run_id,
            dataset=bundle.name,
            classifier=classifier.name,
        )
        classifier.prepare(bundle.classes)

        stage = "fitting_classifier"
        _write_status(
            status_path,
            status="running",
            stage=stage,
            run_id=run_id,
            dataset=bundle.name,
            classifier=classifier.name,
        )
        classifier.fit(fit_train, validation_examples=validation)

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


def split_train_validation(
    examples: Sequence[LabeledExample],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[LabeledExample, ...], tuple[LabeledExample, ...]]:
    """Create a deterministic stratified split from the dataset train split.

    At least one example per class is kept in training whenever that class exists.
    Classes with only one training example therefore contribute no validation item.
    The original example order is preserved inside each returned partition.
    """

    frozen = tuple(examples)
    if not frozen:
        raise ValueError("Dataset training split cannot be empty")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0.0:
        return frozen, ()

    by_label: dict[str, list[LabeledExample]] = {}
    for example in frozen:
        by_label.setdefault(example.label, []).append(example)

    validation_ids: set[str] = set()
    rng = random.Random(seed)
    for label in sorted(by_label):
        group = list(by_label[label])
        if len(group) < 2:
            continue
        rng.shuffle(group)
        desired = max(1, int(round(len(group) * validation_fraction)))
        validation_count = min(desired, len(group) - 1)
        validation_ids.update(item.sample_id for item in group[:validation_count])

    train = tuple(item for item in frozen if item.sample_id not in validation_ids)
    validation = tuple(item for item in frozen if item.sample_id in validation_ids)
    if not train:
        raise ValueError("Train/validation split produced an empty training partition")
    return train, validation


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
    for index, (example, prediction) in enumerate(zip(bundle.test, predictions, strict=True)):
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
        if set(probabilities) != valid_labels:
            raise ValueError(
                f"Prediction probabilities for {prediction.sample_id!r} must contain "
                "exactly the configured class set"
            )

        normalized: dict[str, float] = {}
        for label, value in probabilities.items():
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
        for example, prediction in zip(bundle.test, predictions, strict=True):
            payload = {
                "dataset": bundle.name,
                "classifier": classifier_name,
                "sample_id": example.sample_id,
                "input": example.text,
                "gold_label": example.label,
                "predicted_label": prediction.predicted_label,
                "correct": example.label == prediction.predicted_label,
                "confidence": prediction.confidence,
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
    classifier: Classifier,
    config: BenchmarkRunConfig,
    fit_train: Sequence[LabeledExample],
    validation: Sequence[LabeledExample],
    class_definitions_metadata: Mapping[str, Any],
) -> None:
    payload = {
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "dataset": {
            "name": dataset.name,
            "metadata": dict(dataset.metadata),
            "classes": [
                {"name": item.name, "description": item.description}
                for item in dataset.classes
            ],
            "source_train_size": len(dataset.train),
            "fit_train_sample_ids": [item.sample_id for item in fit_train],
            "validation_sample_ids": [item.sample_id for item in validation],
            "test_sample_ids": [item.sample_id for item in dataset.test],
            "fit_train_size": len(fit_train),
            "validation_size": len(validation),
            "test_size": len(dataset.test),
        },
        "class_definitions": dict(class_definitions_metadata),
        "split": {
            "validation_fraction": config.validation_fraction,
            "seed": config.split_seed,
            "strategy": "deterministic_stratified_by_label",
        },
        "classifier": _classifier_metadata(classifier),
        "run_metadata": dict(config.metadata),
    }
    _write_json(path, payload)


def _classifier_metadata(classifier: Classifier) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": classifier.name}
    model = getattr(classifier, "model", None)
    if isinstance(model, str):
        payload["model"] = model
    model_id = getattr(classifier, "model_id", None)
    if isinstance(model_id, str):
        payload["model_id"] = model_id
    training = getattr(classifier, "training", None)
    if training is not None and is_dataclass(training) and not isinstance(training, type):
        payload["training"] = asdict(training)

    # Optional experiment metadata exposed by classifier implementations. Keeping
    # this generic avoids branching on classifier names while making supervision
    # and inference settings explicit in persisted run configuration.
    for attribute in (
        "supervision_regime",
        "training_examples_used",
        "validation_examples_used",
        "reasoning_effort",
    ):
        value = getattr(classifier, attribute, None)
        if value is not None:
            payload[attribute] = value
    return payload


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


__all__ = [
    "BenchmarkRunConfig",
    "BenchmarkRunResult",
    "run_benchmark",
    "split_train_validation",
]
