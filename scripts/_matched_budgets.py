"""Matched-budget options shared by the maintained campaign entry points."""
from argparse import Namespace
from dataclasses import asdict
import json

from _emissary_campaign import classifier_conditions, emissary_configs, validate_emissary_arguments
from llm_classifier_bench.budgets import MatchedBudgetConfig


def add_matched_arguments(parser):
    parser.add_argument("--matched-budgets", nargs="+", type=int,
                        help="Separate matched-pool campaign, e.g. 0 5 100; excludes full-training references")
    parser.add_argument("--budget-unit", choices=["total", "per_class"])
    parser.add_argument("--validation-budget", type=int, default=0,
                        help="Additional TOTAL validation labels, disjoint from the shared fit/context budget")
    parser.add_argument("--openai-context-window-tokens", type=int)
    parser.add_argument("--openai-completion-reserve-tokens", type=int, default=4096)
    parser.add_argument("--openai-framing-allowance-tokens", type=int, default=1024)


def _emissary_args(args, budgets, seed):
    values = vars(args).copy()
    values.update(emissary_shots=budgets, emissary_shot_unit=args.budget_unit,
                  emissary_selection_seed=seed)
    if args.emissary_mechanism is None:
        # Do not silently infer a Projects mechanism from an environment workspace.
        values["emissary_project_id"] = None
    return Namespace(**values)


def validate_budget_arguments(args):
    if args.matched_budgets is None:
        if args.budget_unit is not None or args.validation_budget:
            raise ValueError("Budget options require --matched-budgets")
        validate_emissary_arguments(args)
        return
    for budget in args.matched_budgets:
        MatchedBudgetConfig(budget, args.budget_unit, validation_examples=args.validation_budget)
    if args.validation_budget and not args.validation_fraction:
        raise ValueError("Positive validation budget requires a reserved validation partition")
    if args.emissary_shots != [0] or args.emissary_shot_unit is not None:
        raise ValueError("Use shared --matched-budgets/--budget-unit instead of Emissary-only shot flags")
    if "emissary" in args.classifiers:
        matched = _emissary_args(args, list(dict.fromkeys(args.matched_budgets)), args.seeds[0])
        # Unconfigured SFT cells are recorded as unsupported by the runner.
        if args.emissary_mechanism is None:
            matched.dry_run = True
        validate_emissary_arguments(matched)


def budget_conditions(args, seed):
    if args.matched_budgets is None:
        for key, training in classifier_conditions(args):
            yield key, training, None
        return
    for count in dict.fromkeys(args.matched_budgets):
        budget = MatchedBudgetConfig(count, args.budget_unit, seed, args.validation_budget)
        for key in dict.fromkeys(args.classifiers):
            training = (emissary_configs(_emissary_args(args, [count], seed))[0]
                        if key == "emissary" else None)
            yield key, training, budget


def budget_manifest(args):
    return {
        "comparison_regime": "matched_labeled_budget" if args.matched_budgets is not None else "full_training_reference",
        "matched_budgets": args.matched_budgets,
        "budget_unit": args.budget_unit,
        "validation_budget_total": args.validation_budget if args.matched_budgets is not None else None,
        "selection_seed_policy": "campaign seed" if args.matched_budgets is not None else "adapter configuration",
        "openai_context_window_tokens": args.openai_context_window_tokens,
        "openai_completion_reserve_tokens": args.openai_completion_reserve_tokens,
        "openai_framing_allowance_tokens": args.openai_framing_allowance_tokens,
        "conditions_by_seed": {
            str(seed): [{"classifier": key, "budget": asdict(budget),
                         "emissary_training": asdict(training) if training else None}
                        for key, training, budget in budget_conditions(args, seed)]
            for seed in dict.fromkeys(args.seeds)
        } if args.matched_budgets is not None else None,
    }


def annotate_budget_row(row, result, budget):
    row["comparison_regime"] = "matched_labeled_budget" if budget else "full_training_reference"
    if budget:
        row.update(labeled_budget=budget.examples, budget_unit=budget.unit,
                   validation_budget_total=budget.validation_examples)
    if result is None:
        return
    status = json.loads(result.status_path.read_text())
    row["status"] = status["status"]
    row["error_type"] = status.get("error_type")
    row["error_message"] = status.get("error_message")
    path = result.run_dir / "labeled_budget.json"
    if path.exists():
        data = json.loads(path.read_text())
        row["pool_sha256"] = data.get("pool_sha256")
        row["class_coverage"] = data.get("selection", {}).get("class_coverage")
        row.update(data.get("consumption", {}))
        if "consumption" not in data:
            # Unsupported/planned cells have no measured consumption.
            for field in ("training_examples_used", "validation_examples_used", "context_examples_used", "total_labeled_examples_used"):
                row[field] = None
