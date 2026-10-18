"""Execute all hard-task methods, reusing prepared data and provisioning Emissary."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from dotenv import load_dotenv

import run_hard_datasets as campaign
from llm_classifier_bench.classifiers.emissary import EmissaryClient
from llm_classifier_bench.datasets.hard_tasks import explanation_coverage
from llm_classifier_bench.runner import _classifier_metadata
from _hard_recovery import validated_pending

REQUIRED_CREDENTIALS = ("OPENAI_API_KEY", "EMISSARY_API_KEY", "JEV_TOKEN", "PERPLEXITY_API_KEY")
DEFAULT_PREPARED_PLAN = campaign.ROOT / "artifacts/hard_fit_sweep/plan.json"
MISSING_PROJECT = "Configure emissary.project_id or --emissary-project-id before preparation"


def refresh_cells(plan, bundles, card):
    """Recheck current request support and estimates without remote calls."""
    cfg = plan["configuration"]
    for cell in plan["cells"]:
        method, budget = cell["method"], cell["fit_per_class"]
        bundle = campaign.fit_budget_bundle(bundles[cell["dataset"]], budget or min(cfg["train_budgets"]))
        cell_cfg = {**cfg, "train_per_class": budget or min(cfg["train_budgets"])}
        missing_project = method in campaign.SFT and not cfg["emissary"].get("project_id")
        classifier = None if missing_project else campaign.build_classifier(
            method, cell_cfg, run_id=f"{cell['dataset']}-{method}-fit{budget}-plan", test_count=len(bundle.test))
        preflight = getattr(classifier, "preflight", None)
        support = preflight(bundle.classes, bundle.inputs()) if preflight else {"supported": True}
        if method in ("openai", "openai-context"):
            support = classifier.plan_context(bundle.classes,
                campaign.context_pool(bundle, cfg) if method == "openai-context" else (), bundle.inputs())
        cell.update(test_examples=len(bundle.test), preflight=support,
                    blocker=MISSING_PROJECT if missing_project else
                    None if support["supported"] else support.get("reason", "context_limit_exceeded"),
                    cost=campaign.planned_cost(method, classifier, bundle, cell_cfg, card))
    plan["planned_api_calls"] = sum(c["cost"]["api_calls"] for c in plan["cells"])
    plan["planned_training_jobs"] = sum(c["method"] in campaign.SFT for c in plan["cells"])
    plan["estimated_priced_subtotal_usd"] = sum(c["cost"]["estimated_usd"] or 0 for c in plan["cells"])


def save_plan(root, plan):
    campaign.write_json(root / "plan.json", plan)
    (root / "plan.sha256").write_text(campaign.digest(root / "plan.json") + "\n", encoding="utf-8")


def reuse_prepared(args):
    """Make a NEW plan from integrity-checked data; never execute/edit the old plan.

    Older implementation hashes are retained as provenance. Replanning checks the
    current support/configuration and freezes current code against the same data.
    """
    source_path = args.prepared_plan.resolve()
    source_root = source_path.parent
    plan = deepcopy(campaign.read_json(source_path))
    if campaign.digest(source_path) != (source_root / "plan.sha256").read_text().strip():
        raise ValueError("Prepared source plan checksum mismatch")
    for name, expected in plan["files_sha256"].items():
        if campaign.digest(source_root / name) != expected:
            raise ValueError(f"Prepared source file changed: {name}")
    if set(plan["datasets"]) != {"implicit_hate", "anli_r3"} or plan["configuration"]["train_budgets"] != [20, 50, 100]:
        raise ValueError("Prepared input must contain both hard datasets and fit budgets 20/50/100")
    expected = {(name, method, b) for name in plan["datasets"] for method in campaign.METHODS
                for b in ([20, 50, 100] if method in campaign.LOCAL | campaign.SFT else [0])}
    plan["cells"] = deepcopy(plan.get("all_cells", plan["cells"]))
    # Completion evidence is reconstructed only by --resume-from. Plain data
    # replanning must not inherit both old summary rows and a full fresh matrix.
    for field in ("reused_results", "reuse_files_sha256", "resume_from"):
        plan.pop(field, None)
    actual = [(c["dataset"], c["method"], c["fit_per_class"]) for c in plan["cells"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Prepared input must contain the complete 42-condition matrix")
    # Retain the full blueprint for future replanning; exclusions persist on resume.
    plan["all_cells"] = [{k: v for k, v in c.items() if k not in ("resume_run", "reused_predictions")}
                         for c in deepcopy(plan["cells"])]
    excluded = set(plan.get("excluded_methods", []))
    if args.skip_perplexity:
        excluded.add("perplexity-decisions")
    if args.skip_emissary_llama:
        excluded.add("emissary-llama")
    plan["excluded_methods"] = sorted(excluded)
    plan["cells"] = [c for c in plan["cells"] if c["method"] not in excluded]
    bundles = {}
    for name, info in plan["datasets"].items():
        bundle = campaign.load_snapshot(source_root / info["snapshot"])
        campaign.validate_partition_disjointness(bundle.train, bundle.validation, bundle.test)
        if [e.sample_id for e in bundle.test] != info["test_sample_ids"]:
            raise ValueError("Prepared test IDs do not match source audit")
        for budget in (20, 50, 100):
            ids = [e.sample_id for e in campaign.fit_budget_bundle(bundle, budget).train]
            if ids != info["fit_budgets"][str(budget)]["fit_sample_ids"]:
                raise ValueError("Prepared fit IDs do not match source audit")
        test = []
        for label in bundle.class_names:
            group = [e for e in bundle.test if e.label == label]
            if len(group) < args.test_per_class:
                raise ValueError("Prepared test pool is too small; prepare a larger source test pool first")
            test.extend(group[:args.test_per_class])
        bundles[name] = replace(bundle, test=tuple(test))
    # Everything above is read-only; reject corrupt/inadequate input before writes.
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source_root / "pricing.json", root / "pricing.json")
    plan["source_plan_sha256"] = campaign.digest(source_path)
    plan["source_git_revision"] = plan["git_revision"]
    plan["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=campaign.ROOT, text=True).strip()
    plan["git_worktree_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=campaign.ROOT, text=True))
    plan["source_implementation_sha256"] = plan["implementation_sha256"]
    plan["preparation_source"] = "verified frozen snapshots; nested prefixes only, no dataset download or reshuffle"
    plan["configuration"]["test_per_class"] = args.test_per_class
    plan["configuration"]["local_device"] = None if args.device == "auto" else args.device
    plan["configuration"]["jev"] = {"probability_sum_tolerance": .05, "invalid_sum_policy": "label_only"}
    plan["configuration"]["openai"]["refusal_policy"] = "null"
    plan["configuration"]["openai-decisions"] = {"refusal_policy": "null"}
    if args.emissary_project_id:
        plan["configuration"]["emissary"]["project_id"] = args.emissary_project_id
    plan["implementation_sha256"] = {
        str(p.relative_to(campaign.ROOT)): campaign.digest(p) for p in (
            Path(campaign.__file__).resolve(), Path(__file__).resolve(),
            Path(__file__).resolve().parent / "_hard_recovery.py",
            campaign.ROOT / "src/llm_classifier_bench/datasets/hard_tasks.py",
            campaign.ROOT / "src/llm_classifier_bench/datasets/base.py",
            campaign.ROOT / "src/llm_classifier_bench/runner.py",
            campaign.ROOT / "src/llm_classifier_bench/classifiers/emissary.py",
            campaign.ROOT / "src/llm_classifier_bench/classifiers/jev.py",
            campaign.ROOT / "src/llm_classifier_bench/classifiers/openai.py",
            campaign.ROOT / "src/llm_classifier_bench/classifiers/openai_decisions.py",
            campaign.ROOT / "src/llm_classifier_bench/classifiers/base.py",
            campaign.ROOT / "src/llm_classifier_bench/metrics/base.py",
            campaign.ROOT / "src/llm_classifier_bench/metrics/classification.py",
            campaign.ROOT / "src/llm_classifier_bench/metrics/evaluator.py",
            campaign.ROOT / "src/llm_classifier_bench/classifiers/bert.py",
            campaign.ROOT / "src/llm_classifier_bench/classifiers/sentence_transformer.py")}
    plan["files_sha256"] = {"pricing.json": campaign.digest(root / "pricing.json")}
    for name, bundle in bundles.items():
        info = plan["datasets"][name]
        filename = f"{name}.json"
        snapshot = {"name": bundle.name, "classes": [asdict(c) for c in bundle.classes],
                    "metadata": dict(bundle.metadata), **{s: [asdict(e) for e in getattr(bundle, s) or ()]
                    for s in ("train", "validation", "test")}}
        campaign.write_json(root / filename, snapshot)
        info.update(snapshot=filename, test_sample_ids=[e.sample_id for e in bundle.test],
                    input_characters={"min": min(len(e.text) for e in bundle.test),
                                      "max": max(len(e.text) for e in bundle.test)})
        info["selected_counts"]["test"] = dict(Counter(e.label for e in bundle.test))
        info["selected_explanation_coverage"]["test"] = explanation_coverage(bundle.test)
        plan["files_sha256"][filename] = campaign.digest(root / filename)
    refresh_cells(plan, bundles, campaign.read_json(root / "pricing.json"))
    save_plan(root, plan)
    print(f"Reused frozen samples from {source_path}; {len(plan['cells'])} conditions.", flush=True)


def reuse_completed(args, plan):
    """Reuse only completed full-cohort results with matching inputs/settings."""
    summaries = list(args.resume_from.glob("executions/*/summary.json"))
    if len(summaries) != 1:
        raise ValueError("--resume-from must contain exactly one execution summary")
    cfg = plan["configuration"]
    original_project = campaign.read_json(args.resume_from / "plan.json")["configuration"]["emissary"].get("project_id")
    if cfg["emissary"].get("project_id") != original_project:
        raise ValueError("Resume must preserve the original Emissary project")
    bundles = {name: campaign.load_snapshot(args.output_dir / info["snapshot"])
               for name, info in plan["datasets"].items()}
    lookup = {(c["dataset"], c["method"], c["fit_per_class"]): c for c in plan["cells"]}
    reused, fingerprints, completed = [], {str(summaries[0].resolve()): campaign.digest(summaries[0])}, set()
    for row in campaign.read_json(summaries[0]):
        key = (row["dataset"], row["classifier_key"], row["fit_per_class"])
        if row["classifier_key"] in plan.get("excluded_methods", []):
            continue
        if key not in lookup:
            raise ValueError(f"Unexpected historical condition: {key}")
        run = Path(row["run_dir"]).resolve()
        partial = row["status"] != "completed"
        if partial:
            if row["classifier_key"] in ("jev", "openai-decisions"):
                pass  # Validated below; resume only missing inputs, with separate ledgers.
            elif row["classifier_key"] in campaign.LOCAL:
                continue
            else:
                raise ValueError("An incomplete hosted condition requires explicit recovery; refusing to repeat paid attempts")
        if key in completed:
            raise ValueError("Duplicate completed historical condition")
        status = campaign.read_json(run / "status.json")
        if status["status"] != row["status"]:
            raise ValueError("Summary disagrees with saved condition status")
        cell = lookup[key]
        budget, method = cell["fit_per_class"], cell["method"]
        bundle = campaign.fit_budget_bundle(bundles[key[0]], budget or min(cfg["train_budgets"]))
        cell_cfg = {**cfg, "train_per_class": budget or min(cfg["train_budgets"])}
        if method not in campaign.LOCAL | campaign.SFT:
            bundle = replace(bundle, train=campaign.context_pool(bundle, cfg) if method == "openai-context" else (), validation=())
        config = campaign.read_json(run / "config.json")
        for field, expected in (
            ("classes", [asdict(c) for c in bundle.classes]),
            ("fit_train_sample_ids", [e.sample_id for e in bundle.train]),
            ("validation_sample_ids", [e.sample_id for e in bundle.validation]),
            ("test_sample_ids", [e.sample_id for e in bundle.test]),
        ):
            if config["dataset"][field] != expected:
                raise ValueError(f"Completed condition inputs differ: {key}/{field}")
        classifier = campaign.build_classifier(method, cell_cfg, run_id="resume-audit", test_count=len(bundle.test))
        expected = json.loads(json.dumps(_classifier_metadata(classifier)))
        for field in ("name", "model", "training", "reasoning_effort", "in_context"):
            if config["classifier"].get(field) != expected.get(field):
                raise ValueError(f"Completed classifier settings differ: {key}/{field}")
        if partial:
            pending, old = validated_pending(bundle, classifier, run)
            prior_plan = campaign.read_json(run / "inference_plan.json") if (run / "inference_plan.json").exists() else None
            if prior_plan and prior_plan["endpoint"] != classifier.endpoint:
                raise ValueError("Persisted Jev endpoint differs")
            cell["resume_run"] = str(run)
            cell["reused_predictions"] = len(old)
            # This is a new-call cap; failed historical attempts remain in the source ledger.
            cell["cost"]["api_calls"] = len(pending)
            for evidence in run.rglob("*"):
                if evidence.is_file():
                    fingerprints[str(evidence)] = campaign.digest(evidence)
            print(f"{method} recovery: {len(old)} results reused "
                  f"({sum(r['predicted_label'] is None for r in old.values())} refusals); "
                  f"{len(pending)} new requests.", flush=True)
            continue
        predictions = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
        if [(p["sample_id"], p["input"], p["gold_label"]) for p in predictions] != [
            (e.sample_id, e.text, e.label) for e in bundle.test]:
            raise ValueError("Completed predictions do not cover the exact shared test cohort")
        campaign.read_json(run / "metrics.json")
        for filename in ("config.json", "status.json", "predictions.jsonl", "metrics.json", "fit_metadata.json",
                         "cost_report.json", "operational_report.json", "pricing.json", "recovery.json", "coverage.json"):
            path = run / filename
            if path.exists():
                fingerprints[str(path)] = campaign.digest(path)
        reused.append({**row, "run_dir": str(run), "reused": True})
        completed.add(key)
    plan["cells"] = [c for c in plan["cells"] if (c["dataset"], c["method"], c["fit_per_class"]) not in completed]
    plan["reused_results"] = reused
    plan["reuse_files_sha256"] = fingerprints
    plan["resume_from"] = str(args.resume_from.resolve())
    plan["emissary_project_source"] = "reused_from_prior_campaign"
    plan["planned_api_calls"] = sum(c["cost"]["api_calls"] for c in plan["cells"])
    plan["planned_training_jobs"] = sum(c["method"] in campaign.SFT for c in plan["cells"])
    plan["estimated_priced_subtotal_usd"] = sum(c["cost"]["estimated_usd"] or 0 for c in plan["cells"])
    save_plan(args.output_dir, plan)
    print(f"Resume: {len(reused)} completed conditions reused; {len(plan['cells'])} pending.", flush=True)


def provision_project(root, plan):
    """At most one project POST; retain intent even after an ambiguous timeout."""
    name = f"hard-datasets-{root.name}"
    intent = root / "project_creation.json"
    with intent.open("x", encoding="utf-8") as handle:
        json.dump({"status": "submitting", "name": name, "operation": "POST /v1/projects"}, handle)
    try:
        project = EmissaryClient().create_project(name)
    except BaseException as exc:
        campaign.write_json(intent, {"status": "failed_or_ambiguous", "name": name,
                                     "error_type": type(exc).__name__, "automatic_retry": False})
        raise
    # Save the returned ID before any dataset upload, training or deployment.
    campaign.write_json(intent, {"status": "created", "name": name, "project_id": project["id"]})
    plan["files_sha256"][intent.name] = campaign.digest(intent)
    plan["configuration"]["emissary"]["project_id"] = project["id"]
    for cell in plan["cells"]:
        if cell["blocker"] == MISSING_PROJECT:
            cell["blocker"] = None
    plan["emissary_project_source"] = "created_programmatically_for_this_campaign"
    save_plan(root, plan)
    print(f"Emissary project created: {project['id']}", flush=True)


def main(argv=None):
    load_dotenv(campaign.ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emissary-project-id", default=os.getenv("EMISSARY_PROJECT_ID"),
                        help="Optional existing workspace; omitted means create a fresh project automatically")
    parser.add_argument("--prepared-plan", type=Path, default=DEFAULT_PREPARED_PLAN,
                        help="Reuse the datasets from this frozen plan; default artifacts/hard_fit_sweep/plan.json")
    parser.add_argument("--resume-from", type=Path, help="Campaign root: reuse completed results and its Emissary project")
    parser.add_argument("--skip-perplexity", action="store_true",
                        help="Exclude Perplexity on both datasets; persists in subsequent resumes")
    parser.add_argument("--skip-emissary-llama", action="store_true",
                        help="Exclude all Emissary Llama budgets on both datasets; persists on resume")
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu",
                        help="Local transformer device; cpu matches historical campaigns and supports older GPUs")
    parser.add_argument("--test-per-class", type=int, default=40)
    parser.add_argument("--output-dir", type=Path,
                        default=campaign.ROOT / "artifacts/hard_experiments" / campaign.utc_campaign_id())
    parser.add_argument("--max-api-calls", type=int, default=4320)
    parser.add_argument("--max-training-jobs", type=int, default=12)
    parser.add_argument("--max-estimated-usd", type=float, default=2.)
    parser.add_argument("--allow-unpriced", action="store_true", help="Acknowledge unknown Emissary charges")
    parser.add_argument("--dry-run", action="store_true", help="Prepare only; no remote resources, training or inference")
    args = parser.parse_args(argv)
    if args.resume_from:
        args.prepared_plan = args.resume_from / "plan.json"
        project_file = args.resume_from / "project_creation.json"
        if project_file.exists() and campaign.read_json(project_file)["status"] != "created":
            parser.error("Previous project creation was ambiguous; reconcile it before resuming")
    if args.test_per_class < 1:
        parser.error("--test-per-class must be positive")
    for key in ("max_api_calls", "max_training_jobs", "max_estimated_usd"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"{key} must be finite and nonnegative")
    if not args.dry_run:
        prior_exclusions = (campaign.read_json(args.prepared_plan).get("excluded_methods", [])
                            if args.prepared_plan.is_file() else [])
        skip_perplexity = args.skip_perplexity or "perplexity-decisions" in prior_exclusions
        skip_llama = args.skip_emissary_llama or "emissary-llama" in prior_exclusions
        missing = [key for key in REQUIRED_CREDENTIALS
                   if not (skip_perplexity and key == "PERPLEXITY_API_KEY") and not os.getenv(key, "").strip()]
        if missing:
            parser.error("Missing configuration: " + ", ".join(missing))
        if not args.allow_unpriced:
            parser.error("Full execution requires --allow-unpriced because Emissary charges are unknown")
        if not args.resume_from and ((108 - 9 * skip_perplexity - 27 * skip_llama) * args.test_per_class > args.max_api_calls
                                    or args.max_training_jobs < 12 - 6 * skip_llama):
            parser.error("Planned inference/training exceeds configured caps")
    if args.prepared_plan.is_file():
        reuse_prepared(args)
    else:
        parser.error(f"Prepared data not found: {args.prepared_plan}; run run_hard_datasets.py --train-budgets 20 50 100 first")
    root = args.output_dir
    plan = campaign.read_json(root / "plan.json")
    if args.resume_from:
        reuse_completed(args, plan)
    auto_project = not plan["configuration"]["emissary"].get("project_id")
    plan["planned_new_projects"] = int(auto_project)
    save_plan(root, plan)
    if args.dry_run:
        print(f"Dry run only: {plan['planned_api_calls']} inference calls, {plan['planned_training_jobs']} SFT jobs, "
              f"{int(auto_project)} new project planned; priced subtotal ${plan['estimated_priced_subtotal_usd']:.6f}.", flush=True)
        return
    blockers = [c["blocker"] for c in plan["cells"] if c["blocker"] and c["blocker"] != MISSING_PROJECT]
    if blockers or plan["planned_api_calls"] > args.max_api_calls or plan["planned_training_jobs"] > args.max_training_jobs:
        raise ValueError(f"Prepared campaign is blocked or exceeds caps: {blockers}")
    if plan["estimated_priced_subtotal_usd"] > args.max_estimated_usd:
        raise ValueError("Priced estimate exceeds --max-estimated-usd; no project was created")
    if auto_project:
        provision_project(root, plan)
    print(f"Executing {len(plan['cells'])} pending conditions on local device {args.device}. "
          f"Artifacts: {root / 'executions'}", flush=True)
    campaign.main(["--execute", "--plan", str(root / "plan.json"),
                   "--max-api-calls", str(args.max_api_calls),
                   "--max-training-jobs", str(args.max_training_jobs),
                   "--max-estimated-usd", str(args.max_estimated_usd), "--allow-unpriced"])


if __name__ == "__main__":
    main()
