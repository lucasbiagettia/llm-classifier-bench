"""Probe Emissary's accepted class counts and raw probability contract.

Example:
    python scripts/probe_emissary.py --class-counts 3 5 20 50
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from llm_classifier_bench.classifiers.base import ClassificationInput, ClassDefinition
from llm_classifier_bench.classifiers.emissary import EmissaryAPIError, EmissaryClassifier, EmissaryClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class-counts",
        nargs="+",
        type=int,
        default=[3, 5, 20, 50],
        help="Class counts to try, in order.",
    )
    parser.add_argument(
        "--input",
        default="This is a synthetic API-contract probe.",
        help="Text classified once for each experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/probes"),
    )
    return parser.parse_args()


def build_classes(class_count: int) -> list[ClassDefinition]:
    return [
        ClassDefinition(
            name=f"class_{index:02d}",
            description=f"Synthetic category number {index:02d} for an API limit probe.",
        )
        for index in range(class_count)
    ]


def main() -> None:
    args = parse_args()
    client = EmissaryClient()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output_dir / f"emissary_probe_{timestamp}.jsonl"

    with output_path.open("w", encoding="utf-8") as output_file:
        for class_count in args.class_counts:
            classes = build_classes(class_count)
            record: dict[str, object] = {
                "timestamp": datetime.now(UTC).isoformat(),
                "class_count": class_count,
                "input": args.input,
                "classes": [
                    {"name": item.name, "description": item.description}
                    for item in classes
                ],
            }

            try:
                classifier = EmissaryClassifier.create(
                    client=client,
                    experiment_name=(
                        f"classification-benchmark-probe-{class_count}-"
                        f"{uuid4().hex[:8]}"
                    ),
                    classes=classes,
                )
                prediction = classifier.predict(
                    [ClassificationInput(sample_id="probe", text=args.input)]
                )[0]

                record.update(
                    {
                        "status": "success",
                        "model_id": classifier.model_id,
                        "predicted_label": prediction.predicted_label,
                        "confidence": prediction.confidence,
                        "probability_sum": sum(
                            prediction.probabilities.values()
                            if prediction.probabilities
                            else []
                        ),
                        "latency_ms": prediction.latency_ms,
                        "raw_response": prediction.raw_response,
                    }
                )
                print(
                    f"OK  classes={class_count:<3} "
                    f"model={classifier.model_id} "
                    f"latency_ms={prediction.latency_ms:.1f}"
                )
            except (EmissaryAPIError, ValueError) as exc:
                record.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"ERR classes={class_count:<3} {type(exc).__name__}: {exc}")

            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()

    print(f"\nSaved probe results to {output_path}")


if __name__ == "__main__":
    main()
