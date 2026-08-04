"""Run the main Banking77 multiclass scaling pilot across all four classifiers.

Default experiment:
    class counts:        5, 10, 20, 25
    train examples:      100 per class (supervised classifiers)
    validation:          20% of the sampled training data
    test examples:       20 per class
    seed:                42

Before sampling labels, the script filters Banking77 classes by available
train/test support. Label selection is random only within that predeclared pool.
If requested support leaves too few labels, deterministic backoff is allowed
down to explicit minimums and is fully recorded in campaign.json.

Class subsets are nested for a fixed seed:
    labels(5) ⊂ labels(10) ⊂ labels(20) ⊂ labels(25)

For any class shared across conditions, the sampled train and test examples are
also identical. Every classifier within a condition receives the exact same
DatasetBundle and final test examples.

The LLM-generated class descriptions are NEVER regenerated here. A frozen,
versioned class-definition profile is loaded once and deterministically subsetted.

Outputs:
    artifacts/benchmark_runs/<campaign_id>/
        campaign.json
        summary.csv
        summary.json
        definitions/
            ... derived frozen subset profiles ...
        runs/
            <one normal runner directory per classifier/condition>

This is a serious pilot / first benchmark sweep. For publishable claims, run
multiple seeds and add confidence intervals/bootstrap estimates.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from llm_classifier_bench.class_definitions import (
    ClassDefinitionProfile,
    load_class_definition_profile,
)
from llm_classifier_bench.classifiers import (
    BertClassifier,
    EmissaryClassifier,
    EmissaryClient,
    OpenAIClassifier,
    SentenceTransformerLogisticClassifier,
)
from llm_classifier_bench.config import (
    BertTrainingConfig,
    DEFAULT_BERT_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    SentenceTransformerTrainingConfig,
)
from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.datasets import DatasetBundle, get_dataset
from llm_classifier_bench.runner import (
    BenchmarkRunConfig,
    run_benchmark,
    split_train_validation,
)


DEFAULT_DEFINITIONS = Path(
    "class_definitions_data/banking77/canonical_llm_enriched_v1.json"
)
DEFAULT_OUTPUT_ROOT = Path("artifacts/benchmark_runs")
DEFAULT_CLASS_COUNTS = (5, 10, 20, 25)
DEFAULT_CLASSIFIERS = (
    "emissary",
    "openai",
    "sentence-transformer",
    "bert",
)

SCRIPT_VERSION = "support-filter-v2"


@dataclass(frozen=True, slots=True)
class StaticDataset:
    bundle: DatasetBundle

    @property
    def name(self) -> str:
        return self.bundle.name

    def load(self) -> DatasetBundle:
        return self.bundle


@dataclass(frozen=True, slots=True)
class Condition:
    seed: int
    class_count: int
    selected_names: tuple[str, ...]
    definitions_path: Path
    bundle: DatasetBundle
    fit_train_size: int
    validation_size: int


@dataclass(frozen=True, slots=True)
class EligibilityPlan:
    """Dataset-support policy frozen before any classifier is run."""

    requested_train_per_class: int
    effective_train_per_class: int
    requested_test_per_class: int
    effective_test_per_class: int
    required_class_count: int
    eligible_names: tuple[str, ...]
    train_counts: dict[str, int]
    test_counts: dict[str, int]
    auto_backoff_used: bool


def utc_campaign_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def count_examples_by_label(
    examples: Sequence[LabeledExample],
) -> dict[str, int]:
    return dict(Counter(example.label for example in examples))


def resolve_eligibility_plan(
    *,
    canonical_names: Sequence[str],
    train_examples: Sequence[LabeledExample],
    test_examples: Sequence[LabeledExample],
    requested_train_per_class: int,
    requested_test_per_class: int,
    required_class_count: int,
    min_train_per_class: int,
    min_test_per_class: int,
    strict_support: bool,
) -> EligibilityPlan:
    """Choose a support threshold before any model runs.

    Scientific rule:
    - support filtering depends only on dataset counts;
    - the same eligible pool is used for every condition and classifier;
    - labels are sampled randomly *within* that pool later;
    - if the requested thresholds leave too few labels, deterministic backoff
      preserves test size first and lowers train support only as much as needed;
    - if that is still insufficient, test support is lowered minimally;
    - every effective threshold is persisted in campaign metadata.
    """

    if requested_train_per_class < 1 or requested_test_per_class < 1:
        raise ValueError("Requested train/test examples per class must be >= 1")
    if min_train_per_class < 1 or min_test_per_class < 1:
        raise ValueError("Minimum train/test examples per class must be >= 1")
    if min_train_per_class > requested_train_per_class:
        raise ValueError("--min-train-per-class cannot exceed --train-per-class")
    if min_test_per_class > requested_test_per_class:
        raise ValueError("--min-test-per-class cannot exceed --test-per-class")

    train_counts = count_examples_by_label(train_examples)
    test_counts = count_examples_by_label(test_examples)

    def eligible(train_threshold: int, test_threshold: int) -> tuple[str, ...]:
        return tuple(
            name
            for name in canonical_names
            if train_counts.get(name, 0) >= train_threshold
            and test_counts.get(name, 0) >= test_threshold
        )

    requested_eligible = eligible(
        requested_train_per_class,
        requested_test_per_class,
    )
    if len(requested_eligible) >= required_class_count:
        return EligibilityPlan(
            requested_train_per_class=requested_train_per_class,
            effective_train_per_class=requested_train_per_class,
            requested_test_per_class=requested_test_per_class,
            effective_test_per_class=requested_test_per_class,
            required_class_count=required_class_count,
            eligible_names=requested_eligible,
            train_counts=train_counts,
            test_counts=test_counts,
            auto_backoff_used=False,
        )

    if strict_support:
        raise ValueError(
            "Only "
            f"{len(requested_eligible)} labels satisfy "
            f"train>={requested_train_per_class} and "
            f"test>={requested_test_per_class}, but the largest condition "
            f"requires {required_class_count}. Re-run without --strict-support "
            "to allow deterministic support backoff, or lower the requested "
            "per-class sample counts."
        )

    # First preserve the requested TEST size (our metric precision) and lower
    # only the supervised-training threshold until enough classes qualify.
    for train_threshold in range(
        requested_train_per_class - 1,
        min_train_per_class - 1,
        -1,
    ):
        names = eligible(train_threshold, requested_test_per_class)
        if len(names) >= required_class_count:
            return EligibilityPlan(
                requested_train_per_class=requested_train_per_class,
                effective_train_per_class=train_threshold,
                requested_test_per_class=requested_test_per_class,
                effective_test_per_class=requested_test_per_class,
                required_class_count=required_class_count,
                eligible_names=names,
                train_counts=train_counts,
                test_counts=test_counts,
                auto_backoff_used=True,
            )

    # If test support itself is the bottleneck, reduce it minimally. For each
    # candidate test threshold, keep as much training support as possible.
    for test_threshold in range(
        requested_test_per_class - 1,
        min_test_per_class - 1,
        -1,
    ):
        for train_threshold in range(
            requested_train_per_class,
            min_train_per_class - 1,
            -1,
        ):
            names = eligible(train_threshold, test_threshold)
            if len(names) >= required_class_count:
                return EligibilityPlan(
                    requested_train_per_class=requested_train_per_class,
                    effective_train_per_class=train_threshold,
                    requested_test_per_class=requested_test_per_class,
                    effective_test_per_class=test_threshold,
                    required_class_count=required_class_count,
                    eligible_names=names,
                    train_counts=train_counts,
                    test_counts=test_counts,
                    auto_backoff_used=True,
                )

    raise ValueError(
        "Banking77 cannot provide enough supported labels for the requested "
        f"largest condition ({required_class_count} labels), even after "
        f"backing off to train>={min_train_per_class} and "
        f"test>={min_test_per_class} per class."
    )


def choose_master_label_order(
    canonical_names: Sequence[str],
    *,
    seed: int,
) -> tuple[str, ...]:
    names = list(canonical_names)
    random.Random(seed).shuffle(names)
    return tuple(names)


def sample_per_class(
    examples: Sequence[LabeledExample],
    selected_names: Sequence[str],
    *,
    per_class: int,
    seed: int,
    purpose: str,
) -> tuple[LabeledExample, ...]:
    if per_class < 1:
        raise ValueError(f"{purpose} examples per class must be at least 1")

    sampled: list[LabeledExample] = []
    for label in selected_names:
        candidates = [example for example in examples if example.label == label]
        if len(candidates) < per_class:
            raise ValueError(
                f"Class {label!r} has only {len(candidates)} {purpose} examples; "
                f"requested {per_class}. Lower --{purpose}-per-class."
            )

        # Label-specific seed means the examples for a shared class do not change
        # when total class_count changes.
        rng = random.Random(f"{seed}:{purpose}:{label}")
        chosen = rng.sample(candidates, per_class)
        chosen.sort(key=lambda item: item.sample_id)
        sampled.extend(chosen)

    return tuple(sampled)


def subset_definitions(
    definitions: Sequence[ClassDefinition],
    selected_names: Sequence[str],
) -> tuple[ClassDefinition, ...]:
    by_name = {definition.name: definition for definition in definitions}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        raise ValueError(f"Frozen profile is missing selected labels: {missing}")
    return tuple(by_name[name] for name in selected_names)


def write_derived_definition_profile(
    *,
    source_profile: Any,
    selected_definitions: Sequence[ClassDefinition],
    class_count: int,
    seed: int,
    destination: Path,
) -> Path:
    """Write an exact subset artifact so the normal runner can validate it.

    This does NOT regenerate or edit descriptions. It only selects entries from
    the already-frozen source profile.
    """

    profile = ClassDefinitionProfile(
        dataset=source_profile.profile.dataset,
        profile=(
            f"{source_profile.profile.profile}"
            f"__n{class_count}_seed{seed}"
        ),
        classes=tuple(selected_definitions),
        generation={
            **dict(source_profile.profile.generation),
            "derived_from_profile": source_profile.profile.profile,
            "derived_from_sha256": source_profile.sha256,
            "derivation": "exact_subset_only",
            "class_subset_strategy": "support_filtered_then_nested_shuffled_prefix",
            "class_count": class_count,
            "seed": seed,
        },
        review_status=source_profile.profile.review_status,
    )
    return profile.write_json(destination, overwrite=False)


def build_condition(
    *,
    full_bundle: DatasetBundle,
    loaded_profile: Any,
    master_order: Sequence[str],
    class_count: int,
    seed: int,
    train_per_class: int,
    test_per_class: int,
    validation_fraction: float,
    definitions_dir: Path,
) -> Condition:
    if class_count < 2:
        raise ValueError("class counts must be at least 2")
    if class_count > len(master_order):
        raise ValueError(
            f"class_count={class_count} exceeds {len(master_order)} available labels"
        )

    selected_names = tuple(master_order[:class_count])
    all_definitions = loaded_profile.definitions_for(
        dataset_name=full_bundle.name,
        canonical_names=full_bundle.class_names,
    )
    selected_definitions = subset_definitions(all_definitions, selected_names)

    selected_train = sample_per_class(
        full_bundle.train,
        selected_names,
        per_class=train_per_class,
        seed=seed,
        purpose="train",
    )
    selected_test = sample_per_class(
        full_bundle.test,
        selected_names,
        per_class=test_per_class,
        seed=seed,
        purpose="test",
    )

    subset_profile_path = definitions_dir / (
        f"{loaded_profile.profile.profile}"
        f"__n{class_count}_seed{seed}.json"
    )
    write_derived_definition_profile(
        source_profile=loaded_profile,
        selected_definitions=selected_definitions,
        class_count=class_count,
        seed=seed,
        destination=subset_profile_path,
    )

    metadata = {
        **dict(full_bundle.metadata),
        "benchmark_campaign": True,
        "source_dataset": "banking77",
        "class_count": class_count,
        "train_per_class_sampled": train_per_class,
        "test_per_class": test_per_class,
        "class_subset_seed": seed,
        "class_subset_strategy": "support_filtered_then_nested_shuffled_prefix",
        "selected_class_names": list(selected_names),
        "source_definition_profile": loaded_profile.profile.profile,
        "source_definition_profile_path": str(loaded_profile.path),
        "source_definition_profile_sha256": loaded_profile.sha256,
        "definition_review_status": loaded_profile.profile.review_status,
    }

    bundle = DatasetBundle(
        name=full_bundle.name,
        classes=selected_definitions,
        train=selected_train,
        test=selected_test,
        metadata=metadata,
    )

    fit_train, validation = split_train_validation(
        bundle.train,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    return Condition(
        seed=seed,
        class_count=class_count,
        selected_names=selected_names,
        definitions_path=subset_profile_path,
        bundle=bundle,
        fit_train_size=len(fit_train),
        validation_size=len(validation),
    )


def build_classifier(
    name: str,
    *,
    condition: Condition,
    campaign_id: str,
    openai_model: str,
    openai_reasoning_effort: str,
    bert_model: str,
    bert_epochs: int,
    bert_batch_size: int,
    bert_learning_rate: float,
    bert_weight_decay: float,
    bert_max_length: int,
    sentence_transformer_model: str,
    st_embedding_batch_size: int,
    st_c_values: tuple[float, ...],
    st_max_iter: int,
) -> Any:
    if name == "emissary":
        classifier = EmissaryClassifier.create(
            client=EmissaryClient(),
            experiment_name=(
                f"banking77-n{condition.class_count}-seed{condition.seed}-"
                f"{campaign_id}"
            ),
            classes=condition.bundle.classes,
            mode="routing",
            classifier_name="emissary-zero-shot",
        )
        # Generic runner metadata; fit() itself remains a no-op.
        classifier.supervision_regime = "zero_shot"
        classifier.training_examples_used = 0
        classifier.validation_examples_used = 0
        return classifier

    if name == "openai":
        return OpenAIClassifier(
            model=openai_model,
            reasoning_effort=openai_reasoning_effort,
            classifier_name="openai-zero-shot",
        )

    if name == "bert":
        classifier = BertClassifier(
            model=bert_model,
            training=BertTrainingConfig(
                epochs=bert_epochs,
                batch_size=bert_batch_size,
                learning_rate=bert_learning_rate,
                weight_decay=bert_weight_decay,
                max_length=bert_max_length,
                seed=condition.seed,
            ),
            classifier_name="bert-finetuned",
        )
        classifier.supervision_regime = "supervised"
        classifier.training_examples_used = condition.fit_train_size
        classifier.validation_examples_used = condition.validation_size
        return classifier

    if name == "sentence-transformer":
        classifier = SentenceTransformerLogisticClassifier(
            model=sentence_transformer_model,
            training=SentenceTransformerTrainingConfig(
                embedding_batch_size=st_embedding_batch_size,
                c_values=st_c_values,
                max_iter=st_max_iter,
                seed=condition.seed,
            ),
            classifier_name="sentence-transformer-logreg",
        )
        classifier.supervision_regime = "supervised"
        classifier.training_examples_used = condition.fit_train_size
        classifier.validation_examples_used = condition.validation_size
        return classifier

    raise ValueError(f"Unknown classifier {name!r}")


def load_metric_value(metrics: dict[str, Any], name: str) -> float | None:
    item = metrics.get(name)
    if not isinstance(item, dict) or not item.get("available"):
        return None
    value = item.get("value")
    return float(value) if isinstance(value, (int, float)) else None


def summary_row(
    *,
    campaign_id: str,
    condition: Condition,
    classifier_key: str,
    classifier: Any,
    result: Any | None,
    error: Exception | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "campaign_id": campaign_id,
        "seed": condition.seed,
        "class_count": condition.class_count,
        "classifier_key": classifier_key,
        "classifier": getattr(classifier, "name", classifier_key),
        "supervision_regime": getattr(classifier, "supervision_regime", None),
        "training_examples_used": getattr(classifier, "training_examples_used", None),
        "validation_examples_used": getattr(classifier, "validation_examples_used", None),
        "test_examples": len(condition.bundle.test),
        "status": "failed" if error is not None else "completed",
        "run_id": getattr(result, "run_id", None),
        "run_dir": str(getattr(result, "run_dir", "")) if result is not None else None,
        "error_type": type(error).__name__ if error is not None else None,
        "error_message": str(error) if error is not None else None,
    }

    if result is None or result.metrics_path is None:
        return row

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    for metric_name in (
        "accuracy",
        "macro_f1",
        "top_label_ece",
        "adaptive_ece",
        "multiclass_log_loss",
        "multiclass_brier_score",
        "mean_latency_ms",
        "latency_p50_ms",
        "latency_p99_ms",
        "total_cost_usd",
        "cost_per_1000_usd",
    ):
        row[metric_name] = load_metric_value(metrics, metric_name)

    return row


def write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return

    preferred = [
        "campaign_id",
        "seed",
        "class_count",
        "classifier_key",
        "classifier",
        "supervision_regime",
        "training_examples_used",
        "validation_examples_used",
        "test_examples",
        "status",
        "accuracy",
        "macro_f1",
        "top_label_ece",
        "adaptive_ece",
        "multiclass_log_loss",
        "multiclass_brier_score",
        "mean_latency_ms",
        "latency_p50_ms",
        "latency_p99_ms",
        "total_cost_usd",
        "cost_per_1000_usd",
        "run_id",
        "run_dir",
        "error_type",
        "error_message",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extra

    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Banking77 multiclass scaling campaign with all four classifiers."
    )
    parser.add_argument(
        "--class-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_CLASS_COUNTS),
        help="Nested class counts. Default: 5 10 20 25.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=list(DEFAULT_CLASSIFIERS),
        default=list(DEFAULT_CLASSIFIERS),
        help="Classifier families to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Subset/sampling seeds. Use multiple seeds for repeated experiments.",
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=100,
        help="Sampled source-train examples per selected class.",
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=20,
        help="Final held-out test examples per selected class.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Fraction of sampled training data reserved for validation.",
    )
    parser.add_argument(
        "--min-train-per-class",
        type=int,
        default=50,
        help=(
            "Lowest automatic train-per-class backoff allowed when too few "
            "labels satisfy --train-per-class. Default: 50."
        ),
    )
    parser.add_argument(
        "--min-test-per-class",
        type=int,
        default=10,
        help=(
            "Lowest automatic test-per-class backoff allowed when too few "
            "labels satisfy --test-per-class. Default: 10."
        ),
    )
    parser.add_argument(
        "--strict-support",
        action="store_true",
        help=(
            "Disable automatic support backoff. Fail if the requested train/test "
            "counts do not leave enough eligible labels."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Resolve dataset support, eligible labels and nested subsets, then "
            "exit before creating classifiers, calling APIs or training models."
        ),
    )
    parser.add_argument(
        "--definitions",
        type=Path,
        default=DEFAULT_DEFINITIONS,
        help="Frozen full Banking77 class-definition profile.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for benchmark campaigns.",
    )

    # Model overrides.
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument(
        "--openai-reasoning-effort",
        default=DEFAULT_OPENAI_REASONING_EFFORT,
    )
    parser.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--sentence-transformer-model", default=DEFAULT_SENTENCE_TRANSFORMER_MODEL)

    # BERT training.
    parser.add_argument("--bert-epochs", type=int, default=3)
    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--bert-learning-rate", type=float, default=2e-5)
    parser.add_argument("--bert-weight-decay", type=float, default=0.01)
    parser.add_argument("--bert-max-length", type=int, default=128)

    # Frozen embeddings + LR.
    parser.add_argument("--st-embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--st-c-values",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
    )
    parser.add_argument("--st-max-iter", type=int, default=2000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    class_counts = tuple(sorted(set(args.class_counts)))
    if not class_counts:
        raise ValueError("At least one --class-counts value is required")
    if any(value < 2 for value in class_counts):
        raise ValueError("Every class count must be >= 2")
    if max(class_counts) > 25 and "emissary" in args.classifiers:
        print(
            "WARNING: Emissary was only verified up to 25 classes in this project. "
            "Higher values may fail at experiment creation."
        )

    campaign_id = utc_campaign_id()
    campaign_root = args.output_root / campaign_id
    runs_root = campaign_root / "runs"
    definitions_dir = campaign_root / "definitions"
    campaign_root.mkdir(parents=True, exist_ok=False)
    runs_root.mkdir()
    definitions_dir.mkdir()

    full_bundle = get_dataset("banking77").load()
    loaded_profile = load_class_definition_profile(args.definitions)

    # Validate the full frozen ontology before selecting any subset.
    loaded_profile.definitions_for(
        dataset_name=full_bundle.name,
        canonical_names=full_bundle.class_names,
    )

    # Freeze the support-based eligibility pool BEFORE any classifier runs.
    # This uses only Banking77 sample counts, never model performance.
    support_plan = resolve_eligibility_plan(
        canonical_names=full_bundle.class_names,
        train_examples=full_bundle.train,
        test_examples=full_bundle.test,
        requested_train_per_class=args.train_per_class,
        requested_test_per_class=args.test_per_class,
        required_class_count=max(class_counts),
        min_train_per_class=args.min_train_per_class,
        min_test_per_class=args.min_test_per_class,
        strict_support=args.strict_support,
    )

    print(f"script_version={SCRIPT_VERSION}")

    if loaded_profile.profile.review_status != "approved":
        print(
            "\nWARNING: frozen class definitions have review_status="
            f"{loaded_profile.profile.review_status!r}. "
            "Fine for a serious pilot; approve/freeze before publishable runs.\n"
        )

    manifest = {
        "campaign_id": campaign_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "banking77",
        "class_counts": list(class_counts),
        "classifiers": list(args.classifiers),
        "seeds": list(args.seeds),
        "requested_train_per_class": support_plan.requested_train_per_class,
        "effective_train_per_class": support_plan.effective_train_per_class,
        "requested_test_per_class": support_plan.requested_test_per_class,
        "effective_test_per_class": support_plan.effective_test_per_class,
        "validation_fraction": args.validation_fraction,
        "class_subset_strategy": "support_filtered_then_nested_shuffled_prefix",
        "support_policy": {
            "selection_uses_model_performance": False,
            "required_class_count": support_plan.required_class_count,
            "eligible_class_count": len(support_plan.eligible_names),
            "eligible_class_names": list(support_plan.eligible_names),
            "auto_backoff_used": support_plan.auto_backoff_used,
            "min_train_per_class": args.min_train_per_class,
            "min_test_per_class": args.min_test_per_class,
            "train_counts": support_plan.train_counts,
            "test_counts": support_plan.test_counts,
        },
        "sampling_strategy": "label_specific_deterministic_random_sample",
        "source_definition_profile": {
            "path": str(loaded_profile.path),
            "profile": loaded_profile.profile.profile,
            "sha256": loaded_profile.sha256,
            "review_status": loaded_profile.profile.review_status,
        },
        "models": {
            "openai": args.openai_model,
            "openai_reasoning_effort": args.openai_reasoning_effort,
            "bert": args.bert_model,
            "sentence_transformer": args.sentence_transformer_model,
        },
        "bert_training": {
            "epochs": args.bert_epochs,
            "batch_size": args.bert_batch_size,
            "learning_rate": args.bert_learning_rate,
            "weight_decay": args.bert_weight_decay,
            "max_length": args.bert_max_length,
        },
        "sentence_transformer_training": {
            "embedding_batch_size": args.st_embedding_batch_size,
            "c_values": list(args.st_c_values),
            "max_iter": args.st_max_iter,
        },
    }
    write_json(campaign_root / "campaign.json", manifest)

    rows: list[dict[str, Any]] = []

    print(f"campaign_id={campaign_id}")
    print(f"campaign_root={campaign_root}")
    print(f"class_counts={class_counts}")
    print(f"classifiers={tuple(args.classifiers)}")
    print(f"seeds={tuple(args.seeds)}")
    print(
        "train_per_class="
        f"{support_plan.effective_train_per_class} "
        f"(requested={support_plan.requested_train_per_class})"
    )
    print(
        "test_per_class="
        f"{support_plan.effective_test_per_class} "
        f"(requested={support_plan.requested_test_per_class})"
    )
    print(f"eligible_classes={len(support_plan.eligible_names)}")
    print(f"validation_fraction={args.validation_fraction}")

    if support_plan.auto_backoff_used:
        print(
            "WARNING: support backoff was required; effective per-class counts "
            "were frozen before model execution and recorded in campaign.json."
        )

    if args.dry_run:
        print("\nDRY RUN: no classifier will be created and no model will be trained.")
        for seed in args.seeds:
            master_order = choose_master_label_order(
                support_plan.eligible_names,
                seed=seed,
            )
            for class_count in class_counts:
                selected = tuple(master_order[:class_count])
                print(f"seed={seed} classes={class_count} labels={selected}")
        return

    for seed in args.seeds:
        master_order = choose_master_label_order(
            support_plan.eligible_names,
            seed=seed,
        )

        for class_count in class_counts:
            print("\n" + "=" * 80)
            print(f"CONDITION seed={seed} classes={class_count}")
            print("=" * 80)

            condition = build_condition(
                full_bundle=full_bundle,
                loaded_profile=loaded_profile,
                master_order=master_order,
                class_count=class_count,
                seed=seed,
                train_per_class=support_plan.effective_train_per_class,
                test_per_class=support_plan.effective_test_per_class,
                validation_fraction=args.validation_fraction,
                definitions_dir=definitions_dir,
            )

            print(f"fit_train={condition.fit_train_size}")
            print(f"validation={condition.validation_size}")
            print(f"test={len(condition.bundle.test)}")
            print("labels:")
            for label in condition.selected_names:
                print(f"  - {label}")

            for classifier_key in args.classifiers:
                print("\n" + "-" * 80)
                print(
                    f"RUN seed={seed} classes={class_count} "
                    f"classifier={classifier_key}"
                )
                print("-" * 80)

                classifier: Any | None = None
                result: Any | None = None
                error: Exception | None = None

                try:
                    classifier = build_classifier(
                        classifier_key,
                        condition=condition,
                        campaign_id=campaign_id,
                        openai_model=args.openai_model,
                        openai_reasoning_effort=args.openai_reasoning_effort,
                        bert_model=args.bert_model,
                        bert_epochs=args.bert_epochs,
                        bert_batch_size=args.bert_batch_size,
                        bert_learning_rate=args.bert_learning_rate,
                        bert_weight_decay=args.bert_weight_decay,
                        bert_max_length=args.bert_max_length,
                        sentence_transformer_model=args.sentence_transformer_model,
                        st_embedding_batch_size=args.st_embedding_batch_size,
                        st_c_values=tuple(args.st_c_values),
                        st_max_iter=args.st_max_iter,
                    )

                    run_id = (
                        f"{campaign_id}__seed{seed}__n{class_count:02d}"
                        f"__{classifier.name}"
                    )
                    result = run_benchmark(
                        StaticDataset(condition.bundle),
                        classifier,
                        BenchmarkRunConfig(
                            output_root=runs_root,
                            run_id=run_id,
                            validation_fraction=args.validation_fraction,
                            split_seed=seed,
                            class_definitions_path=condition.definitions_path,
                            metadata={
                                "campaign_id": campaign_id,
                                "benchmark_campaign": True,
                                "seed": seed,
                                "class_count": class_count,
                                "class_subset_strategy": "support_filtered_then_nested_shuffled_prefix",
                                "selected_class_names": list(condition.selected_names),
                                "requested_train_per_class": support_plan.requested_train_per_class,
                                "effective_train_per_class": support_plan.effective_train_per_class,
                                "requested_test_per_class": support_plan.requested_test_per_class,
                                "effective_test_per_class": support_plan.effective_test_per_class,
                                "source_definition_profile": loaded_profile.profile.profile,
                                "source_definition_profile_sha256": loaded_profile.sha256,
                            },
                        ),
                    )

                    print(f"completed: {result.run_dir}")

                except Exception as exc:
                    error = exc
                    print(
                        f"FAILED: {type(exc).__name__}: {exc}",
                        flush=True,
                    )

                if classifier is None:
                    # Preserve a row even if construction itself failed.
                    class FailedClassifier:
                        name = classifier_key
                    classifier = FailedClassifier()

                row = summary_row(
                    campaign_id=campaign_id,
                    condition=condition,
                    classifier_key=classifier_key,
                    classifier=classifier,
                    result=result,
                    error=error,
                )
                rows.append(row)

                # Persist after EVERY run so an interrupted campaign still has a
                # usable partial summary.
                write_json(campaign_root / "summary.json", rows)
                write_summary_csv(campaign_root / "summary.csv", rows)

                if error is None:
                    print(
                        "metrics: "
                        f"accuracy={row.get('accuracy')} "
                        f"macro_f1={row.get('macro_f1')} "
                        f"p50_ms={row.get('latency_p50_ms')}"
                    )

    completed = sum(row["status"] == "completed" for row in rows)
    failed = len(rows) - completed

    print("\n" + "=" * 80)
    print("CAMPAIGN COMPLETE")
    print("=" * 80)
    print(f"completed_runs={completed}")
    print(f"failed_runs={failed}")
    print(f"summary_csv={campaign_root / 'summary.csv'}")
    print(f"summary_json={campaign_root / 'summary.json'}")
    print(f"campaign_config={campaign_root / 'campaign.json'}")


if __name__ == "__main__":
    main()
