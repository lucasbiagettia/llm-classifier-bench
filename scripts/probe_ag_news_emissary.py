"""Run the first real dataset -> Emissary classification cycle.

This is intentionally a smoke test rather than the final benchmark runner. It:
- loads AG News through the existing dataset interface;
- selects a small balanced sample from the official test split;
- creates or reuses an Emissary experiment from the dataset class definitions;
- classifies the selected examples through the existing classifier interface;
- prints results and writes auditable JSONL artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from llm_classifier_bench.classifiers.emissary import (
    EmissaryClassifier,
    EmissaryClient,
)
from llm_classifier_bench.core import LabeledExample
from llm_classifier_bench.datasets.huggingface import (
    HuggingFaceClassificationDataset,
)
from llm_classifier_bench.datasets.registry import AG_NEWS_SPEC
from llm_classifier_bench.workflows import run_classification_cycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a small balanced AG News sample with Emissary."
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=2,
        help="Number of test examples per AG News class. Default: 2.",
    )
    parser.add_argument(
        "--model-id",
        help=(
            "Reuse an existing Emissary <experiment_id>/<version>. "
            "When omitted, the script creates a new four-class experiment."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL output path. Defaults to artifacts/probes/<timestamp>.jsonl.",
    )
    return parser.parse_args()


def select_balanced_examples(
    examples: Sequence[LabeledExample],
    *,
    class_names: Sequence[str],
    examples_per_class: int,
) -> tuple[LabeledExample, ...]:
    if examples_per_class < 1:
        raise ValueError("examples_per_class must be at least 1")

    selected_by_label: dict[str, list[LabeledExample]] = defaultdict(list)
    wanted = set(class_names)

    for example in examples:
        if (
            example.label in wanted
            and len(selected_by_label[example.label]) < examples_per_class
        ):
            selected_by_label[example.label].append(example)

        if all(
            len(selected_by_label[label]) >= examples_per_class
            for label in class_names
        ):
            break

    missing = {
        label: examples_per_class - len(selected_by_label[label])
        for label in class_names
        if len(selected_by_label[label]) < examples_per_class
    }
    if missing:
        details = ", ".join(f"{label}: {count}" for label, count in missing.items())
        raise RuntimeError(f"Could not build balanced sample; missing {details}")

    # Preserve the dataset's declared class order, making runs deterministic.
    return tuple(
        example
        for label in class_names
        for example in selected_by_label[label]
    )


def default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts/probes") / f"ag_news_emissary_{timestamp}.jsonl"


def main() -> None:
    args = parse_args()

    # Emissary zero-shot does not use train data, so load only a tiny train slice.
    # We load the complete official test split to select a guaranteed balanced
    # sample without changing the dataset interface or its implementation.
    dataset_spec = replace(
        AG_NEWS_SPEC,
        train_split="train[:8]",
        test_split="test",
    )
    bundle = HuggingFaceClassificationDataset(dataset_spec).load()

    selected = select_balanced_examples(
        bundle.test,
        class_names=bundle.class_names,
        examples_per_class=args.examples_per_class,
    )

    client = EmissaryClient()
    if args.model_id:
        classifier = EmissaryClassifier(
            client=client,
            model_id=args.model_id,
            classifier_name="emissary-zero-shot-ag-news",
        )
    else:
        classifier = EmissaryClassifier.create(
            client=client,
            experiment_name=f"ag-news-smoke-{uuid4().hex[:10]}",
            classes=bundle.classes,
            mode="routing",
            classifier_name="emissary-zero-shot-ag-news",
        )

    records = run_classification_cycle(
        bundle=bundle,
        classifier=classifier,
        examples=selected,
    )

    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            prediction = record.prediction
            correct += int(record.is_correct)

            payload = {
                "dataset": bundle.name,
                "classifier": classifier.name,
                "sample_id": record.example.sample_id,
                "input": record.example.text,
                "gold_label": record.example.label,
                "predicted_label": prediction.predicted_label,
                "correct": record.is_correct,
                "confidence": prediction.confidence,
                "probabilities": dict(prediction.probabilities or {}),
                "latency_ms": prediction.latency_ms,
                "model": prediction.model,
                "request_id": prediction.request_id,
                "raw_response": dict(prediction.raw_response or {}),
            }
            output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

            marker = "OK " if record.is_correct else "ERR"
            confidence = (
                f"{prediction.confidence:.3f}"
                if prediction.confidence is not None
                else "n/a"
            )
            text_preview = " ".join(record.example.text.split())[:110]
            print(
                f"{marker} gold={record.example.label:<8} "
                f"pred={prediction.predicted_label:<8} "
                f"confidence={confidence:<5} "
                f"latency_ms={prediction.latency_ms:>7.1f}  "
                f"{text_preview}"
            )

    print()
    print(f"Completed {len(records)} classifications.")
    print(f"Correct in smoke sample: {correct}/{len(records)}")
    print(f"Model: {classifier.model_id}")
    print(f"Saved raw records to {output_path}")


if __name__ == "__main__":
    main()
