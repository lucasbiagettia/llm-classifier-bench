"""Shared labeled-example options for the two maintained Banking77 campaigns."""
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from collections.abc import Iterator
from dataclasses import asdict
import os
from typing import Any

from llm_classifier_bench.config import EmissaryTrainingConfig


def add_emissary_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--emissary-shots",
        type=int,
        nargs="+",
        default=[0],
        help="Labeled-example budgets; nonzero requires an explicit mechanism.",
    )
    parser.add_argument("--emissary-shot-unit", choices=["total", "per_class"],
                        help="Required for nonzero budgets; no inferred meaning of 5/100 shots.")
    parser.add_argument("--emissary-selection-seed", type=int, default=42)
    parser.add_argument("--emissary-selection-policy", default="balanced_round_robin_v1")
    parser.add_argument(
        "--emissary-mechanism",
        choices=["project_fine_tuning"],
        help=(
            "Required for live nonzero shots. This trains a separate Projects base "
            "model and is not the routing-experiment mechanism."
        ),
    )
    parser.add_argument(
        "--emissary-project-id",
        default=os.getenv("EMISSARY_PROJECT_ID"),
        help="Existing Projects workspace ID; may also be set by EMISSARY_PROJECT_ID.",
    )
    parser.add_argument("--emissary-base-model")
    parser.add_argument("--emissary-dataset-id")
    parser.add_argument(
        "--emissary-dataset-sha256",
        help="Recorded dataset payload hash required with continuation IDs.",
    )
    parser.add_argument("--emissary-training-job-id")
    parser.add_argument("--emissary-checkpoint", type=int)
    parser.add_argument("--emissary-deployment-id")
    parser.add_argument("--emissary-num-train-epochs", type=int, default=3)
    parser.add_argument("--emissary-learning-rate", type=float, default=2e-4)
    parser.add_argument("--emissary-train-batch-size", type=int, default=2)
    parser.add_argument("--emissary-eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--emissary-dynamic-loss",
        action=BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--emissary-dataset-timeout-s", type=float, default=300.0)
    parser.add_argument("--emissary-training-timeout-s", type=float, default=3600.0)
    parser.add_argument("--emissary-deployment-timeout-s", type=float, default=1800.0)
    parser.add_argument("--emissary-poll-interval-s", type=float, default=10.0)
    parser.add_argument("--emissary-inactive-timeout-s", type=int, default=300)
    parser.add_argument(
        "--emissary-allow-unpriced-training",
        action="store_true",
        help="Explicitly acknowledge that the documented API exposes no cost estimate.",
    )
    parser.add_argument(
        "--emissary-max-training-jobs",
        type=int,
        help="Required live safety bound for newly submitted nonzero conditions.",
    )


def emissary_configs(args: Namespace) -> tuple[EmissaryTrainingConfig, ...]:
    configs = []
    for shots in dict.fromkeys(args.emissary_shots):
        provider = (
            {
                "mechanism": args.emissary_mechanism,
                "project_id": args.emissary_project_id,
                "base_model": args.emissary_base_model,
                "dataset_id": args.emissary_dataset_id,
                "dataset_sha256": args.emissary_dataset_sha256,
                "training_job_id": args.emissary_training_job_id,
                "checkpoint": args.emissary_checkpoint,
                "deployment_id": args.emissary_deployment_id,
                "num_train_epochs": args.emissary_num_train_epochs,
                "learning_rate": args.emissary_learning_rate,
                "per_device_train_batch_size": args.emissary_train_batch_size,
                "per_device_eval_batch_size": args.emissary_eval_batch_size,
                "dynamic_loss": args.emissary_dynamic_loss,
                "dataset_timeout_s": args.emissary_dataset_timeout_s,
                "training_timeout_s": args.emissary_training_timeout_s,
                "deployment_timeout_s": args.emissary_deployment_timeout_s,
                "poll_interval_s": args.emissary_poll_interval_s,
                "inactive_timeout_s": args.emissary_inactive_timeout_s,
            }
            if shots
            else {}
        )
        configs.append(
            EmissaryTrainingConfig(
                shots=shots,
                shot_unit=args.emissary_shot_unit,
                selection_seed=args.emissary_selection_seed,
                selection_policy=args.emissary_selection_policy,
                **provider,
            )
        )
    return tuple(configs)


def validate_emissary_arguments(args: Namespace) -> None:
    if "emissary" in args.classifiers:
        configs = emissary_configs(args)
        nonzero = [config for config in configs if config.shots]
        if not args.dry_run:
            for config in configs:
                config.require_live_support()
            if nonzero and not args.emissary_allow_unpriced_training:
                raise ValueError(
                    "Live project fine-tuning has no published cost estimate. Pass "
                    "--emissary-allow-unpriced-training only with explicit authorization."
                )
            new_jobs = sum(config.training_job_id is None for config in nonzero)
            maximum = args.emissary_max_training_jobs
            if new_jobs and (
                type(maximum) is not int or maximum < 0 or new_jobs > maximum
            ):
                raise ValueError(
                    f"Live run would create {new_jobs} training job(s); set "
                    "--emissary-max-training-jobs to an equal or larger explicit bound."
                )
        continuation = any(
            config.dataset_id or config.training_job_id or config.deployment_id
            for config in nonzero
        )
        if continuation and len(nonzero) != 1:
            raise ValueError("Continuation IDs require exactly one nonzero shot condition")


def classifier_conditions(args: Namespace) -> Iterator[tuple[str, EmissaryTrainingConfig | None]]:
    for key in args.classifiers:
        if key == "emissary":
            for config in emissary_configs(args):
                yield key, config
        elif not args.dry_run:
            yield key, None


def emissary_manifest(args: Namespace) -> list[dict[str, Any]]:
    if "emissary" not in args.classifiers:
        return []
    return [asdict(config) for config in emissary_configs(args)]


def emissary_name(config: EmissaryTrainingConfig) -> str:
    if not config.shots:
        return "emissary-zero-shot"
    mechanism = "project-sft" if config.uses_project_fine_tuning else "unresolved"
    return (
        f"emissary-{mechanism}-shots{config.shots}-{config.shot_unit}"
        f"-seed{config.selection_seed}"
    )
