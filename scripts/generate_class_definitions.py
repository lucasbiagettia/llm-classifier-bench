#!/usr/bin/env python3
"""Generate a frozen class-definition profile outside the benchmark cycle."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_classifier_bench.class_definitions import (
    OpenAIClassDefinitionGenerator,
    build_minimal_profile,
)
from llm_classifier_bench.config import DEFAULT_CLASS_DEFINITION_GENERATOR_MODEL
from llm_classifier_bench.datasets.registry import get_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--mode",
        choices=("minimal", "llm-enriched"),
        default="llm-enriched",
    )
    parser.add_argument("--profile")
    parser.add_argument("--dataset-context", default="")
    parser.add_argument(
        "--model",
        default=DEFAULT_CLASS_DEFINITION_GENERATOR_MODEL,
        help="Generator model only; independent from the benchmark classifier model.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("class_definitions_data"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Loading resolves canonical names from the dataset adapter. No train/test text
    # is ever passed to the LLM generator.
    bundle = get_dataset(args.dataset).load()
    canonical_names = bundle.class_names

    if args.mode == "minimal":
        profile_name = args.profile or "canonical_minimal_v1"
        profile = build_minimal_profile(
            dataset_name=bundle.name,
            canonical_names=canonical_names,
            profile_name=profile_name,
        )
    else:
        profile_name = args.profile or "canonical_llm_enriched_v1"
        generator = OpenAIClassDefinitionGenerator(model=args.model)
        profile = generator.generate(
            dataset_name=bundle.name,
            canonical_names=canonical_names,
            dataset_context=args.dataset_context,
            profile_name=profile_name,
        )

    output_path = args.output_root / bundle.name / f"{profile.profile}.json"
    profile.write_json(output_path, overwrite=args.overwrite)

    print(f"dataset={bundle.name}")
    print(f"profile={profile.profile}")
    print(f"class_count={len(profile.classes)}")
    print(f"review_status={profile.review_status}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
