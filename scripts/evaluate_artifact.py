"""Compute all default metrics from a saved prediction JSONL artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_classifier_bench.metrics import (
    evaluate_jsonl,
    write_results_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute benchmark metrics without rerunning any classifier."
    )
    parser.add_argument("artifact", type=Path, help="Prediction JSONL artifact.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional metrics JSON path. Defaults beside the input artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_jsonl(args.artifact)

    for result in results:
        if result.available:
            print(f"{result.name:<28} {result.value:.8f}")
        else:
            print(
                f"{result.name:<28} unavailable "
                f"({result.metadata.get('reason', 'unknown reason')})"
            )

    output = args.output or args.artifact.with_name(
        f"{args.artifact.stem}_metrics.json"
    )
    write_results_json(output, results)
    print(f"\nSaved metrics to {output}")


if __name__ == "__main__":
    main()
