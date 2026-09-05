"""CPU-only local fixture smoke run; no credentials, network, or model downloads."""
from __future__ import annotations

import argparse
from pathlib import Path

from _probe_classifier_utils import StaticDataset
from llm_classifier_bench.classifiers import TfidfLogisticClassifier
from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.datasets import DatasetBundle
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark


def fixture_bundle() -> DatasetBundle:
    return DatasetBundle(
        name="tfidf_local_smoke",
        classes=(ClassDefinition("World", "World news"), ClassDefinition("Sports", "Sports news")),
        train=tuple(
            LabeledExample(f"train-{label}-{i}", text, label)
            for label, texts in (
                ("World", ("world leaders meet", "world treaty signed", "leaders sign treaty", "world leaders negotiate")),
                ("Sports", ("team scored goal", "sports team wins", "team wins match", "sports goal scored")),
            ) for i, text in enumerate(texts)
        ),
        test=(LabeledExample("test-0", "world leaders treaty", "World"),
              LabeledExample("test-1", "sports team goal", "Sports"),
              LabeledExample("test-2", "unseentoken", "World")),
        metadata={"source": "scripts/probe_tfidf_classifier.py:fixture_bundle", "fixture_version": 1},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/tfidf_smoke"))
    args = parser.parse_args()
    result = run_benchmark(
        StaticDataset(fixture_bundle()), TfidfLogisticClassifier(),
        BenchmarkRunConfig(output_root=args.output_root, run_id=args.run_id,
                           validation_fraction=0.25, split_seed=42,
                           metadata={"smoke": True}),
    )
    print(f"run_dir={result.run_dir}")
    print(f"predictions={result.predictions_path}")
    print(f"metrics={result.metrics_path}")
    print(f"fit_metadata={result.run_dir / 'fit_metadata.json'}")


if __name__ == "__main__":
    main()
