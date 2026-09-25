"""A capped Banking77 pilot using the maintained v2 campaign, not a new protocol."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import run_banking77_scaling_benchmark_v2 as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize at most four paid Jev attempts; default plans only")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/jev_pilot"))
    args = parser.parse_args()
    # One condition, 2 classes x 2 held-out inputs; no warmup or retries.
    forwarded = ["jev-pilot", "--classifiers", "jev", *(["tfidf"] if args.live else []),
                 "--class-counts", "2", "--seeds", "42", "--train-per-class", "8",
                 "--test-per-class", "2", "--min-train-per-class", "8",
                 "--min-test-per-class", "2", "--strict-support", "--validation-fraction", ".25",
                 "--jev-max-requests", "4", "--jev-max-retries", "0", "--warmup-examples", "0",
                 "--pricing", "pricing/inference_2026-09-25.json", "--output-root", str(args.output_root)]
    if not args.live:
        forwarded.append("--dry-run")
    original = sys.argv
    try:
        sys.argv = forwarded
        campaign.main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
