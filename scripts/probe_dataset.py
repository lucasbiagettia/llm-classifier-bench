"""Load a small real dataset slice and print its normalized contract."""

from __future__ import annotations

import argparse
from dataclasses import replace

from llm_classifier_bench.datasets import DATASET_REGISTRY, HuggingFaceClassificationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        choices=sorted(DATASET_REGISTRY),
        help="Dataset registry name.",
    )
    parser.add_argument("--train-split", default="train[:8]")
    parser.add_argument("--test-split", default="test[:8]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_spec = DATASET_REGISTRY[args.dataset]
    spec = replace(
        base_spec,
        train_split=args.train_split,
        test_split=args.test_split,
    )
    bundle = HuggingFaceClassificationDataset(spec).load()

    print(f"dataset={bundle.name}")
    print(f"classes={len(bundle.classes)}")
    print(f"train_examples={len(bundle.train)}")
    print(f"test_examples={len(bundle.test)}")
    print("\nClass definitions:")
    for class_definition in bundle.classes:
        print(f"- {class_definition.name}: {class_definition.description}")

    print("\nTest examples:")
    for example in bundle.test:
        print(f"- [{example.label}] {example.text[:180]}")


if __name__ == "__main__":
    main()
