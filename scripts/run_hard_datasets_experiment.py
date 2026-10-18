"""Prepare and immediately execute the full hard-dataset experiment (paid APIs)."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from dotenv import load_dotenv

import run_hard_datasets as campaign

REQUIRED_CREDENTIALS = ("OPENAI_API_KEY", "EMISSARY_API_KEY", "JEV_TOKEN", "PERPLEXITY_API_KEY")


def main(argv=None):
    load_dotenv(campaign.ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emissary-project-id", default=os.getenv("EMISSARY_PROJECT_ID"),
                        help="Existing Projects workspace; alternatively set EMISSARY_PROJECT_ID in .env")
    parser.add_argument("--test-per-class", type=int, default=40,
                        help="Held-out examples/class; 40 matches the previous Banking77 experiment")
    parser.add_argument("--output-dir", type=Path,
                        default=campaign.ROOT / "artifacts/hard_experiments" / campaign.utc_campaign_id())
    parser.add_argument("--max-api-calls", type=int, default=4320)
    parser.add_argument("--max-training-jobs", type=int, default=12)
    parser.add_argument("--max-estimated-usd", type=float, default=2.)
    parser.add_argument("--allow-unpriced", action="store_true",
                        help="Acknowledge unknown Emissary charges; required for actual execution")
    parser.add_argument("--dry-run", action="store_true", help="Prepare only; never train or call inference APIs")
    args = parser.parse_args(argv)
    if args.test_per_class < 1:
        parser.error("--test-per-class must be positive")
    for key in ("max_api_calls", "max_training_jobs", "max_estimated_usd"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"{key} must be finite and nonnegative")
    if not args.dry_run:
        missing = [key for key in REQUIRED_CREDENTIALS if not os.getenv(key, "").strip()]
        if not args.emissary_project_id or not args.emissary_project_id.strip():
            missing.append("EMISSARY_PROJECT_ID (or --emissary-project-id)")
        if missing:
            parser.error("Missing configuration: " + ", ".join(missing))
        if not args.allow_unpriced:
            parser.error("Full execution requires --allow-unpriced because Emissary charges are unknown")
        if 108 * args.test_per_class > args.max_api_calls:
            parser.error("Planned inference exceeds --max-api-calls")
        if 12 > args.max_training_jobs:
            parser.error("Planned SFT exceeds --max-training-jobs")

    # Always create a fresh configured plan. Never edit the user's frozen dry-run.
    prepare_args = ["--datasets", "implicit_hate", "anli_r3", "--methods", *campaign.METHODS,
                    "--train-budgets", "20", "50", "100", "--test-per-class", str(args.test_per_class),
                    "--output-dir", str(args.output_dir)]
    if args.emissary_project_id:
        prepare_args += ["--emissary-project-id", args.emissary_project_id]
    print(f"Preparing both datasets: fit 20/50/100 per class; test {args.test_per_class}/class.", flush=True)
    campaign.main(prepare_args)
    if args.dry_run:
        return
    print(f"Executing 42 conditions. Artifacts: {args.output_dir / 'executions'}", flush=True)
    campaign.main(["--execute", "--plan", str(args.output_dir / "plan.json"),
                   "--max-api-calls", str(args.max_api_calls),
                   "--max-training-jobs", str(args.max_training_jobs),
                   "--max-estimated-usd", str(args.max_estimated_usd), "--allow-unpriced"])


if __name__ == "__main__":
    main()
