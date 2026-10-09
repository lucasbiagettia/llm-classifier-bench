"""Replay frozen Banking77 zero-shot cohorts with Perplexity Decisions; offline by default."""
from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

import run_gpt6_decisions as reference
from _decisions_campaign import add_perplexity_arguments, perplexity_options
from _matched_budgets import annotate_budget_row
from llm_classifier_bench.classifiers import PerplexityDecisionsClassifier
from run_banking77_scaling_benchmark_v2 import (
    Condition, StaticDataset, summary_row, utc_campaign_id,
    write_derived_definition_profile, write_summary_csv,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRICING = Path("pricing/perplexity_decisions_2026-10-09.json")
shared = reference.shared


def prepare_plan(args):
    # Share the existing reference audit: identical definitions, IDs, order,
    # input text, gold labels, split policy and measurement configuration.
    plan = reference.prepare_plan(args, classifier=PerplexityDecisionsClassifier(**perplexity_options(args)),
                                  method="perplexity-decisions")
    plan["failure_policy"] = "stop_on_first_failure_without_retry; preserve earlier predictions and usage"
    plan.pop("refusal_policy")
    return plan


def run_campaign(args, plan):
    # Revalidate everything after planning and before the first request.
    for path, digest in plan["input_sha256"].items():
        shared.require(shared.sha256(path) == digest, f"Audited input changed: {path}")
    source = shared.load_source(Path(plan["source_dir"]), plan)
    profile = shared.load_class_definition_profile(plan["definitions_path"])
    campaign_id = utc_campaign_id()
    root = args.output_root / campaign_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "runs").mkdir()
    (root / "definitions").mkdir()
    shared.write_json(root / "campaign.json", {**plan, "campaign_id": campaign_id, "dry_run": not args.execute})
    shared.write_json(root / "audit.json", plan["audit"])
    print(f"campaign_root={root}", flush=True)
    rows = []
    for cell in plan["cells"]:
        k = cell["k"]
        cohort = plan["conditions"][str(k)]
        bundle = shared.build_bundle(source, [shared.ClassDefinition(**c) for c in cohort["classes"]])
        bundle = replace(bundle, metadata={"source": plan["source"], "class_count": k,
                                          "selected_class_names": list(bundle.class_names)})
        shared.require(shared.ids(bundle.test) == cohort["test_sample_ids"], "Test IDs changed")
        path = root / "definitions" / f"{profile.profile.profile}__n{k}_seed42.json"
        write_derived_definition_profile(source_profile=profile, selected_definitions=bundle.classes,
                                         class_count=k, seed=42, destination=path)
        condition = Condition(42, k, bundle.class_names, path, bundle, 0, 0)
        budget = shared.MatchedBudgetConfig(0, "per_class", 42, 0)
        classifier = PerplexityDecisionsClassifier(model=plan["decisions"]["requested_model"],
            base_url=plan["decisions"]["endpoint"].removesuffix("/decisions"),
            timeout_s=plan["decisions"]["timeout_s"])
        run_id = f"{campaign_id}__{cell['id']}"
        result, error = None, None
        print(f"{'RUN' if args.execute else 'PLAN'} {cell['id']}: {len(bundle.test)} predictions", flush=True)
        try:
            result = shared.run_benchmark(StaticDataset(bundle), classifier,
                shared.BenchmarkRunConfig(output_root=root / "runs", run_id=run_id,
                    dry_run=not args.execute, matched_budget=budget, validation_fraction=.2, split_seed=42,
                    class_definitions_path=path, pricing_path=Path(plan["pricing_path"]),
                    measurement=shared.MeasurementConfig(**cohort["measurement"]),
                    metadata={"campaign_id": campaign_id, "benchmark_campaign": True, "seed": 42,
                              "class_count": k, "cell_id": cell["id"],
                              "reference_evidence": cohort["reference_evidence"], "source": plan["source"]}))
        except BaseException as exc:
            error = exc
            raise
        finally:
            row = summary_row(campaign_id=campaign_id, condition=condition,
                              classifier_key="perplexity-decisions", classifier=classifier,
                              result=result, error=error)
            if result is None:
                row.update(run_id=run_id, run_dir=str(root / "runs" / run_id))
            annotate_budget_row(row, result, budget)
            rows.append(row)
            shared.write_json(root / "summary.json", rows)
            write_summary_csv(root / "summary.csv", rows)
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute paid requests; omitted means offline audit/plan")
    parser.add_argument("--reference-manifest", type=Path, default=Path("reports/v2/manifest.json"))
    parser.add_argument("--reference-results", type=Path, default=Path("reports/v2/results.json"))
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/v2_source"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/perplexity_decisions"))
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    add_perplexity_arguments(parser)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    plan = prepare_plan(args)
    print(f"AUDIT PASSED: K=5/10/15/20, seed 42, 40 test/class; {plan['maximum_predictions']} predictions.", flush=True)
    run_campaign(args, plan)


if __name__ == "__main__":
    main()
