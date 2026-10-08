"""Replay completed Banking77 zero-shot cohorts with GPT-6 Decisions; offline by default."""
from __future__ import annotations

import argparse
import fcntl
from dataclasses import asdict, replace
import os
from pathlib import Path

import run_budget_extension as shared
from _decisions_campaign import add_decisions_arguments, decisions_options
from _decisions_recovery import inspect_cell, next_segment_id, segment_prefix, write_cell_artifacts
from llm_classifier_bench.classifiers.openai_decisions import OpenAIDecisionsRefusalError
from _matched_budgets import annotate_budget_row
from llm_classifier_bench.classifiers import OpenAIDecisionsClassifier
from llm_classifier_bench.costs import load_pricing
from run_banking77_scaling_benchmark_v2 import (
    Condition, StaticDataset, summary_row, utc_campaign_id,
    write_derived_definition_profile, write_summary_csv,
)

ROOT = Path(__file__).resolve().parents[1]
COUNTS = (5, 10, 15, 20)
DEFAULT_PRICING = Path("pricing/decisions_2026-10-07.json")


def locate_run(evidence):
    return evidence if (evidence / "config.json").is_file() else evidence / "runs/new"


def prepare_plan(args):
    """Verify frozen inputs and every completed comparable cohort without network."""
    fingerprints = {}

    def record(path):
        fingerprints[str(path)] = shared.sha256(path)
        return shared.read_json(path)

    manifest = record(args.reference_manifest)
    results = record(args.reference_results)
    shared.require(manifest["dataset"] == "banking77" and manifest["seed"] == 42,
                   "Expected completed Banking77 seed-42 study")
    shared.require(tuple(manifest["class_counts"]) == COUNTS and manifest["test_examples_per_class"] == 40,
                   "Completed study matrix changed")
    source = shared.load_source(args.source_dir, manifest)
    for split in ("train", "test"):
        fingerprints[str(args.source_dir / f"{split}.csv")] = shared.sha256(args.source_dir / f"{split}.csv")
    definition_path = Path(manifest["definitions"]["path"])
    shared.require(shared.sha256(definition_path) == manifest["definitions"]["sha256"], "Frozen definitions changed")
    fingerprints[str(definition_path)] = shared.sha256(definition_path)
    profile = shared.load_class_definition_profile(definition_path)
    definitions = {c.name: c for c in profile.profile.classes}
    classifier = OpenAIDecisionsClassifier(**decisions_options(args))
    load_pricing(args.pricing)
    fingerprints[str(args.pricing)] = shared.sha256(args.pricing)
    conditions, audit = {}, []
    for k in COUNTS:
        classes = tuple(definitions[name] for name in manifest["label_sets"][str(k)])
        bundle = shared.build_bundle(source, classes)
        fit, reserved = shared.pools(bundle, 100)[:2]
        shared.audit_partitions(fit, reserved, source["test"])
        references = [r for r in results["conditions"] if r["k"] == k]
        shared.require(bool(references) and all(r["status"] == "completed" for r in references),
                       f"Incomplete comparable references at K={k}")
        for reference in references:
            predictions = Path(reference["evidence"]) / "predictions.jsonl"
            shared.verify_predictions(predictions, bundle.test)
            fingerprints[str(predictions)] = shared.sha256(predictions)
        baseline = [r for r in references if r["method"] == "openai" and r["fit_per_class"] == 0]
        shared.require(len(baseline) == 1, f"Missing/duplicate completed OpenAI zero-shot reference at K={k}")
        config = record(locate_run(Path(baseline[0]["evidence"])) / "config.json")
        shared.require(config["dataset"]["classes"] == [asdict(c) for c in classes], "Reference definitions differ")
        shared.require(config["split"] == {"validation_fraction": .2, "seed": 42,
                                         "strategy": "deterministic_stratified_by_label"}, "Reference split changed")
        shared.require(config["matched_budget"] == {"examples": 0, "unit": "per_class", "seed": 42,
                                                  "validation_examples": 0}, "Reference budget changed")
        measurement = config["measurement"]
        shared.require(measurement["batch_size"] == 1 and measurement["warmup_examples"] == 0,
                       "Reference measurement policy changed")
        preflight = classifier.preflight(classes, bundle.inputs(),
                                         matched_budget=shared.MatchedBudgetConfig(0, "per_class", 42, 0))
        shared.require(preflight["supported"], preflight["reason"])
        conditions[str(k)] = {"classes": [asdict(c) for c in classes], "test_sample_ids": shared.ids(bundle.test),
                              "measurement": measurement, "reference_evidence": baseline[0]["evidence"],
                              "preflight": preflight}
        audit.append({"k": k, "comparable_conditions_checked": len(references),
                      "test_examples": len(bundle.test), "test_identity": "IDs, order, text and gold labels matched",
                      **shared.audit_partitions(fit, reserved, bundle.test)})
    cells = [{"id": f"openai-decisions__s42__k{k}__b0", "method": "openai-decisions", "k": k,
              "budget": 0, "test_examples": 40*k} for k in COUNTS]
    return {"schema_version": 1, "dataset": "banking77", "seed": 42, "cells": cells,
            "class_counts": list(COUNTS), "test_examples_per_class": 40,
            "comparison_regime": "matched_labeled_budget", "budget_unit": "per_class", "fit_per_class": 0,
            "source_dir": str(args.source_dir), "source": manifest["source"],
            "definitions_path": str(definition_path), "conditions": conditions,
            "decisions": classifier.inference_metadata(), "pricing_path": str(args.pricing),
            "input_sha256": fingerprints, "audit": {"status": "passed", "conditions": audit},
            "maximum_predictions": sum(c["test_examples"] for c in cells), "retries": 0, "warmup": 0,
            "stop_on_first_failure": True, "refusal_policy": "record_and_continue_without_retry",
            "scope": "Four zero-shot cells match completed OpenAI/Jev conditions. Supervised budgets "
                     "20/50/100 reuse these test cohorts but are not Decisions training conditions."}


def run_campaign(args, plan):
    resume = getattr(args, "resume", None)
    if resume:
        root = Path(resume)
        previous = shared.read_json(root / "campaign.json")
        for key in ("source", "definitions_path", "conditions", "decisions", "pricing_path", "input_sha256", "cells"):
            shared.require(previous[key] == plan[key], f"Resume configuration changed: {key}")
        campaign_id = previous["campaign_id"]
    else:
        campaign_id = utc_campaign_id()
        root = args.output_root / campaign_id
        root.mkdir(parents=True, exist_ok=False)
        (root / "runs").mkdir()
        (root / "definitions").mkdir()
        shared.write_json(root / "campaign.json", {**plan, "campaign_id": campaign_id, "dry_run": not args.execute})
        shared.write_json(root / "audit.json", plan["audit"])
    print(f"campaign_root={root}", flush=True)
    # Protect against two local executions selecting the same pending inputs.
    with (root / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_campaign_locked(args, plan, root, campaign_id, bool(resume))


def _run_campaign_locked(args, plan, root, campaign_id, resume):
    for path, digest in plan["input_sha256"].items():
        shared.require(shared.sha256(path) == digest, f"Audited input changed: {path}")
    source = shared.load_source(Path(plan["source_dir"]), plan)
    profile = shared.load_class_definition_profile(plan["definitions_path"])
    prepared = []
    # Audit ALL existing cells before spending anything on pending ones.
    for cell in plan["cells"]:
        k = cell["k"]
        reference = plan["conditions"][str(k)]
        bundle = shared.build_bundle(source, [shared.ClassDefinition(**c) for c in reference["classes"]])
        bundle = replace(bundle, metadata={"source": plan["source"], "class_count": k,
                                          "selected_class_names": [c.name for c in bundle.classes]})
        shared.require(shared.ids(bundle.test) == reference["test_sample_ids"], "Test IDs changed")
        coverage_path = root / "cells" / cell["id"] / "coverage.json"
        if coverage_path.exists():
            for name, digest in shared.read_json(coverage_path)["evidence_sha256"].items():
                shared.require(shared.sha256(name) == digest, f"Saved resume evidence changed: {name}")
        state = inspect_cell(root, campaign_id, cell, bundle, plan)
        prepared.append((cell, reference, bundle, state))
    if resume and not args.execute:
        for cell, reference, bundle, state in prepared:
            print(f"PLAN {cell['id']}: saved={len(state['predictions'])}; refusals={len(state['refusals'])}; "
                  f"pending={len(state['pending'])}", flush=True)
        return root
    rows = []
    if args.execute:
        shared.write_json(root / "continuation_policy.json", {
            "refusals": "record_and_continue_without_retry", "resume": resume,
            "original_segments": "immutable", "full_cohort_quality": "unavailable when any sample is refused",
            "transport_errors": "stop; no automatic retries; explicit resume accepts recorded HTTP 503 failures"})
    for cell, reference, bundle, state in prepared:
        k = cell["k"]
        path = root / "definitions" / f"{profile.profile.profile}__n{k}_seed42.json"
        if not path.exists():
            write_derived_definition_profile(source_profile=profile, selected_definitions=bundle.classes,
                                             class_count=k, seed=42, destination=path)
        # Validate saved subset definitions too, before handing them to the runner.
        subset = shared.load_class_definition_profile(path)
        shared.require(subset.profile.classes == bundle.classes, "Saved subset definitions changed")
        condition = Condition(42, k, bundle.class_names, path, bundle, 0, 0)
        budget = shared.MatchedBudgetConfig(0, "per_class", 42, 0)
        classifier = OpenAIDecisionsClassifier(model=plan["decisions"]["requested_model"],
            base_url=plan["decisions"]["endpoint"].removesuffix("/decisions"),
            timeout_s=plan["decisions"]["timeout_s"])
        result, error = None, None
        run_id = None
        print(f"{'RUN' if args.execute else 'PLAN'} {cell['id']}: pending={len(state['pending'])}; "
              f"saved={len(state['predictions'])}; refusals={len(state['refusals'])}", flush=True)
        try:
            while state["pending"]:
                run_id = next_segment_id(root, segment_prefix(campaign_id, k))
                try:
                    result = shared.run_benchmark(StaticDataset(replace(bundle, test=state["pending"])), classifier,
                        shared.BenchmarkRunConfig(output_root=root / "runs", run_id=run_id,
                            dry_run=not args.execute, matched_budget=budget, validation_fraction=.2, split_seed=42,
                            # A continuation segment must not present subset scores as full-cohort quality.
                            evaluate=len(state["pending"]) == len(bundle.test),
                            class_definitions_path=path, pricing_path=args.pricing,
                            measurement=shared.MeasurementConfig(**reference["measurement"]),
                            metadata={"campaign_id": campaign_id, "benchmark_campaign": True, "seed": 42, "class_count": k,
                                      "cell_id": cell["id"], "reference_evidence": reference["reference_evidence"],
                                      "source": plan["source"], "full_test_examples": len(bundle.test)}))
                except OpenAIDecisionsRefusalError as exc:
                    print(f"REFUSAL {exc.sample_id}: retained; continuing with the next sample", flush=True)
                    result = None
                if not args.execute:
                    break
                state = inspect_cell(root, campaign_id, cell, bundle, plan)
                write_cell_artifacts(root, cell, bundle, state)
            if args.execute:
                directory, coverage, metrics = write_cell_artifacts(root, cell, bundle, state)
                print(f"{coverage['status']}: {len(state['predictions'])} predictions, "
                      f"{len(state['refusals'])} refusals; {directory}", flush=True)
        except BaseException as exc:
            error = exc
            raise
        finally:
            row = summary_row(campaign_id=campaign_id, condition=condition, classifier_key="openai-decisions",
                              classifier=classifier, result=result, error=error)
            if result is None and run_id and (root / "runs" / run_id / "status.json").is_file():
                row.update(run_id=run_id, run_dir=str(root / "runs" / run_id))
            annotate_budget_row(row, result, budget)
            if args.execute and error is None:
                row.update(status=coverage["status"], successful_examples=coverage["successful_examples"],
                           refused_examples=coverage["refused_examples"], pending_examples=coverage["pending_examples"],
                           evidence=str(directory), segment_run_dirs=state["runs"],
                           training_examples_used=0, validation_examples_used=0, context_examples_used=0,
                           total_labeled_examples_used=0)
                for name, metric in metrics.items():
                    row[name] = metric["value"] if metric["available"] else None
                if len(state["runs"]) == 1:
                    row.update(run_dir=state["runs"][0], run_id=Path(state["runs"][0]).name)
                else:
                    # No single segment represents this complete test cohort.
                    row.update(run_dir=None, run_id=None, inference_throughput_examples_per_second=None)
                row["inference_call_count"] = len(state["predictions"])
            rows.append(row)
            shared.write_json(root / "summary.json", rows)
            write_summary_csv(root / "summary.csv", rows)
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute paid pending conditions; omitted means offline audit/plan")
    parser.add_argument("--resume", type=Path, help="Continue this existing campaign without repeating saved predictions or refusals")
    parser.add_argument("--reference-manifest", type=Path, default=Path("reports/v2/manifest.json"))
    parser.add_argument("--reference-results", type=Path, default=Path("reports/v2/results.json"))
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/v2_source"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/benchmark_runs"))
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    add_decisions_arguments(parser)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    plan = prepare_plan(args)
    print(f"AUDIT PASSED: K=5/10/15/20, seed 42, 40 test/class; {plan['maximum_predictions']} predictions.", flush=True)
    run_campaign(args, plan)


if __name__ == "__main__":
    main()
