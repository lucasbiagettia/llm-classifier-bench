from __future__ import annotations

import argparse

from _probe_classifier_utils import StaticDataset, smoke_subset
from llm_classifier_bench.classifiers import SentenceTransformerLogisticClassifier
from llm_classifier_bench.config import (
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    SentenceTransformerTrainingConfig,
)
from llm_classifier_bench.datasets import get_dataset
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small frozen SentenceTransformer + logistic regression smoke run"
    )
    parser.add_argument("--model", default=DEFAULT_SENTENCE_TRANSFORMER_MODEL)
    parser.add_argument("--train-per-class", type=int, default=12)
    parser.add_argument("--test-per-class", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    bundle = get_dataset("ag_news").load()
    smoke = smoke_subset(
        bundle,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
    )
    classifier = SentenceTransformerLogisticClassifier(
        model=args.model,
        training=SentenceTransformerTrainingConfig(
            embedding_batch_size=32,
            c_values=(0.1, 1.0, 10.0),
            max_iter=1_000,
            seed=42,
        ),
    )
    result = run_benchmark(
        StaticDataset(smoke),
        classifier,
        BenchmarkRunConfig(
            run_id=args.run_id,
            validation_fraction=0.25,
            split_seed=42,
            metadata={"smoke": True, "supervision_regime": "supervised"},
        ),
    )
    print(f"run_dir={result.run_dir}")
    print(f"predictions={result.predictions_path}")
    print(f"metrics={result.metrics_path}")


if __name__ == "__main__":
    main()
