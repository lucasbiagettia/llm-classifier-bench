from __future__ import annotations

import argparse

from _probe_classifier_utils import StaticDataset, smoke_subset
from llm_classifier_bench.classifiers import OpenAIClassifier
from llm_classifier_bench.config import DEFAULT_OPENAI_MODEL
from llm_classifier_bench.datasets import get_dataset
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Small paid OpenAI classifier smoke run")
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--test-per-class", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    bundle = get_dataset("ag_news").load()
    smoke = smoke_subset(bundle, train_per_class=2, test_per_class=args.test_per_class)
    result = run_benchmark(
        StaticDataset(smoke),
        OpenAIClassifier(model=args.model),
        BenchmarkRunConfig(
            run_id=args.run_id,
            validation_fraction=0.0,
            metadata={"smoke": True, "supervision_regime": "zero_shot"},
        ),
    )
    print(f"run_dir={result.run_dir}")
    print(f"predictions={result.predictions_path}")
    print(f"metrics={result.metrics_path}")


if __name__ == "__main__":
    main()
