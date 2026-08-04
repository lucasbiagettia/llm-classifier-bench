"""Pilot Banking77 + Emissary zero-shot benchmark.

This is intentionally a pilot script, not the final multi-run benchmark driver.

What it does:
1. Load the full Banking77 dataset.
2. Load and validate a frozen 77-class definition profile.
3. Deterministically choose a nested class subset from ``--seed``.
4. Deterministically sample the same test examples for overlapping classes.
5. Create a real Emissary experiment using the selected canonical names and
   frozen descriptions.
6. Run the normal benchmark runner so predictions and metrics are persisted.

Important methodological properties:
- No Banking77 train/validation example is used by Emissary.
- The class-definition LLM is NOT called here.
- The same seed gives nested 5/10/20/25 subsets.
- The original Banking77 test split supplies the evaluation examples.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from llm_classifier_bench.class_definitions import load_class_definition_profile
from llm_classifier_bench.classifiers import EmissaryClassifier, EmissaryClient
from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.datasets import DatasetBundle, get_dataset
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark


DEFAULT_DEFINITIONS = Path(
    "class_definitions_data/banking77/canonical_llm_enriched_v1.json"
)


@dataclass(frozen=True, slots=True)
class StaticDataset:
    """Tiny adapter so the normal runner can consume our selected bundle."""

    bundle: DatasetBundle

    @property
    def name(self) -> str:
        return self.bundle.name

    def load(self) -> DatasetBundle:
        return self.bundle


def choose_nested_class_names(
    canonical_names: Sequence[str],
    *,
    class_count: int,
    seed: int,
) -> tuple[str, ...]:
    """Choose a deterministic nested label subset.

    For a fixed seed, asking for 5/10/20/25 classes always takes a prefix of the
    same shuffled 77-label ordering. Therefore:

        labels(5) ⊂ labels(10) ⊂ labels(20) ⊂ labels(25)
    """

    names = list(canonical_names)
    if class_count < 2:
        raise ValueError("--class-count must be at least 2")
    if class_count > len(names):
        raise ValueError(
            f"--class-count={class_count} exceeds the dataset's {len(names)} classes"
        )

    rng = random.Random(seed)
    rng.shuffle(names)
    return tuple(names[:class_count])


def sample_test_examples(
    examples: Sequence[LabeledExample],
    selected_names: Sequence[str],
    *,
    examples_per_class: int,
    seed: int,
) -> tuple[LabeledExample, ...]:
    """Sample a balanced deterministic test subset.

    Each label gets its own RNG seeded from the global seed + canonical label.
    This makes the selected examples for a class independent of how many total
    classes are present in the run.
    """

    if examples_per_class < 1:
        raise ValueError("--examples-per-class must be at least 1")

    result: list[LabeledExample] = []

    for label in selected_names:
        candidates = [example for example in examples if example.label == label]
        if len(candidates) < examples_per_class:
            raise ValueError(
                f"Class {label!r} has only {len(candidates)} test examples; "
                f"requested {examples_per_class}"
            )

        label_rng = random.Random(f"{seed}:test:{label}")
        chosen = label_rng.sample(candidates, examples_per_class)

        # Stable order inside the artifact after random selection.
        chosen.sort(key=lambda item: item.sample_id)
        result.extend(chosen)

    return tuple(result)


def unused_train_placeholders(
    examples: Sequence[LabeledExample],
    selected_names: Sequence[str],
) -> tuple[LabeledExample, ...]:
    """Keep one train item/class only because DatasetBundle/runner expect train data.

    Emissary.fit(...) is a no-op and the run metadata explicitly records
    ``training_examples_used = 0``. These examples are never used to construct
    descriptions or to train Emissary.
    """

    result: list[LabeledExample] = []
    for label in selected_names:
        match = next((example for example in examples if example.label == label), None)
        if match is None:
            raise ValueError(f"No training example found for selected class {label!r}")
        result.append(match)
    return tuple(result)


def subset_definitions(
    all_definitions: Sequence[ClassDefinition],
    selected_names: Sequence[str],
) -> tuple[ClassDefinition, ...]:
    by_name = {definition.name: definition for definition in all_definitions}
    return tuple(by_name[name] for name in selected_names)


def print_metric_summary(metrics_path: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    interesting = (
        "accuracy",
        "macro_f1",
        "top_label_ece",
        "adaptive_ece",
        "multiclass_log_loss",
        "multiclass_brier_score",
        "mean_latency_ms",
        "latency_p50_ms",
        "latency_p99_ms",
    )

    print("\nmetrics:")
    for name in interesting:
        item = metrics.get(name)
        if not isinstance(item, dict):
            continue

        if item.get("available"):
            value = item.get("value")
            if isinstance(value, float):
                print(f"  {name:28s} {value:.6f}")
            else:
                print(f"  {name:28s} {value}")
        else:
            print(f"  {name:28s} unavailable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Banking77 + real Emissary zero-shot pilot"
    )
    parser.add_argument(
        "--class-count",
        type=int,
        default=5,
        help="Number of Banking77 labels to include. Suggested: 5, 10, 20, 25.",
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=10,
        help="Held-out Banking77 test examples per selected class.",
    )
    parser.add_argument(
        "--definitions",
        type=Path,
        default=DEFAULT_DEFINITIONS,
        help="Frozen Banking77 class-definition profile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Controls nested label selection and test sampling.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional Emissary experiment name.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional benchmark run ID.",
    )
    args = parser.parse_args()

    # 1) Load canonical Banking77.
    full_bundle = get_dataset("banking77").load()

    # 2) Load the frozen 77-label ontology and validate it against the dataset.
    loaded_profile = load_class_definition_profile(args.definitions)
    all_definitions = loaded_profile.definitions_for(
        dataset_name=full_bundle.name,
        canonical_names=full_bundle.class_names,
    )

    if loaded_profile.profile.review_status != "approved":
        print(
            "WARNING: class-definition profile review_status="
            f"{loaded_profile.profile.review_status!r}; okay for a pilot, "
            "but approve/freeze it before formal measurements."
        )

    # 3) Select nested labels.
    selected_names = choose_nested_class_names(
        full_bundle.class_names,
        class_count=args.class_count,
        seed=args.seed,
    )
    selected_definitions = subset_definitions(all_definitions, selected_names)

    # 4) Select balanced held-out test examples.
    selected_test = sample_test_examples(
        full_bundle.test,
        selected_names,
        examples_per_class=args.examples_per_class,
        seed=args.seed,
    )

    # The runner expects train data, but Emissary is genuinely zero-shot.
    selected_train = unused_train_placeholders(
        full_bundle.train,
        selected_names,
    )

    pilot_bundle = DatasetBundle(
        # Keep the canonical dataset name: the subset is an experimental condition,
        # not a different source dataset.
        name=full_bundle.name,
        classes=selected_definitions,
        train=selected_train,
        test=selected_test,
        metadata={
            **dict(full_bundle.metadata),
            "pilot": True,
            "source_dataset": "banking77",
            "class_count": args.class_count,
            "examples_per_class": args.examples_per_class,
            "class_subset_seed": args.seed,
            "class_subset_strategy": "nested_shuffled_prefix",
            "selected_class_names": list(selected_names),
            "definition_profile": loaded_profile.profile.profile,
            "definition_profile_path": str(loaded_profile.path),
            "definition_profile_sha256": loaded_profile.sha256,
            "definition_review_status": loaded_profile.profile.review_status,
        },
    )

    # 5) Create the REAL Emissary experiment using the selected frozen ontology.
    experiment_name = args.experiment_name or (
        f"banking77-{args.class_count}labels-seed{args.seed}-pilot"
    )
    client = EmissaryClient()
    classifier = EmissaryClassifier.create(
        client=client,
        experiment_name=experiment_name,
        classes=selected_definitions,
        mode="routing",
        classifier_name="emissary-zero-shot",
    )

    print(f"classes={args.class_count}")
    print(f"examples_per_class={args.examples_per_class}")
    print(f"total_test_examples={len(selected_test)}")
    print(f"seed={args.seed}")
    print(f"profile={loaded_profile.profile.profile}")
    print(f"profile_sha256={loaded_profile.sha256}")
    print(f"experiment_name={experiment_name}")
    print(f"model_id={classifier.model_id}")
    print("selected_labels:")
    for label in selected_names:
        print(f"  - {label}")

    # 6) Use the normal runner for validation, persistence and metrics.
    result = run_benchmark(
        StaticDataset(pilot_bundle),
        classifier,
        BenchmarkRunConfig(
            run_id=args.run_id,
            validation_fraction=0.0,
            split_seed=args.seed,
            metadata={
                "pilot": True,
                "supervision_regime": "zero_shot",
                "training_examples_used": 0,
                "validation_examples_used": 0,
                "class_count": args.class_count,
                "examples_per_class": args.examples_per_class,
                "class_subset_seed": args.seed,
                "class_subset_strategy": "nested_shuffled_prefix",
                "selected_class_names": list(selected_names),
                "definition_profile": loaded_profile.profile.profile,
                "definition_profile_path": str(loaded_profile.path),
                "definition_profile_sha256": loaded_profile.sha256,
            },
            # Deliberately omitted here: the full profile contains 77 labels while
            # this pilot DatasetBundle contains only the selected N labels. We
            # already loaded, validated and subsetted the frozen profile above.
            class_definitions_path=None,
        ),
    )

    print("\ncompleted:")
    print(f"  run_dir={result.run_dir}")
    print(f"  config={result.config_path}")
    print(f"  predictions={result.predictions_path}")
    print(f"  metrics={result.metrics_path}")
    print(f"  status={result.status_path}")

    if result.metrics_path is not None:
        print_metric_summary(result.metrics_path)


if __name__ == "__main__":
    main()
