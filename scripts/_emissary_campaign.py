"""Shared labeled-example options for the two maintained Banking77 campaigns."""
from argparse import ArgumentParser, Namespace
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

from llm_classifier_bench.config import EmissaryTrainingConfig


def add_emissary_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--emissary-shots", type=int, nargs="+", default=[0],
                        help="Labeled-example budgets; nonzero is currently dry-run only.")
    parser.add_argument("--emissary-shot-unit", choices=["total", "per_class"],
                        help="Required for nonzero budgets; no inferred meaning of 5/100 shots.")
    parser.add_argument("--emissary-selection-seed", type=int, default=42)
    parser.add_argument("--emissary-selection-policy", default="balanced_round_robin_v1")


def emissary_configs(args: Namespace) -> tuple[EmissaryTrainingConfig, ...]:
    return tuple(EmissaryTrainingConfig(
        shots=shots, shot_unit=args.emissary_shot_unit,
        selection_seed=args.emissary_selection_seed,
        selection_policy=args.emissary_selection_policy,
    ) for shots in dict.fromkeys(args.emissary_shots))


def validate_emissary_arguments(args: Namespace) -> None:
    if "emissary" in args.classifiers:
        for config in emissary_configs(args):
            if not args.dry_run:
                config.require_live_support()


def classifier_conditions(args: Namespace) -> Iterator[tuple[str, EmissaryTrainingConfig | None]]:
    for key in args.classifiers:
        if key == "emissary":
            for config in emissary_configs(args):
                yield key, config
        elif not args.dry_run:
            yield key, None


def emissary_manifest(args: Namespace) -> list[dict[str, Any]]:
    return [asdict(config) for config in emissary_configs(args)] if "emissary" in args.classifiers else []


def emissary_name(config: EmissaryTrainingConfig) -> str:
    if not config.shots:
        return "emissary-zero-shot"
    return f"emissary-shots{config.shots}-{config.shot_unit}-seed{config.selection_seed}"
