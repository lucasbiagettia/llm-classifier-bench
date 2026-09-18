"""Reprice saved inference usage without changing artifacts or calling providers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_classifier_bench.costs import recalculate_costs


def render_report(report: dict, run_name: str) -> str:
    def number(value):
        return "unavailable" if value is None else f"{value:.10g}"

    lines = [f"# Inference cost: {run_name}", "", report["scope"], "",
             f"Run status: {report['run_status']}. Cost kind: **{report['cost_kind']}**. Currency: USD.", "",
             "| Quantity | Value |", "| --- | ---: |"]
    for label, key in (
        ("Total inference cost", "total_cost_usd"),
        ("Known subtotal (not a complete total when coverage is unknown)", "known_cost_subtotal_usd"),
        ("Observed charges subtotal", "observed_charges_subtotal_usd"),
        ("Estimated cost subtotal", "estimated_cost_subtotal_usd"),
        ("Cost / 1,000 successfully classified evaluation examples", "cost_per_1000_successful_examples_usd"),
        ("Attempted evaluation examples", "attempted_examples"),
        ("Successfully classified evaluation examples", "successful_examples"),
        ("Evaluation failure fraction", "failure_rate"),
    ):
        lines.append(f"| {label} | {number(report[key])} |")
    lines += ["", "Success means a valid classification, whether or not the label is correct.",
              "A failed batch has no accepted outputs; its submitted examples count as unsuccessful.",
              "Warmup costs enter the total, but warmup examples never enter the denominator.", "",
              "## Coverage and assumptions", "",
              f"- Complete cost coverage: {report['coverage_complete']}.",
              f"- Unpriced entries: {report['unpriced_entries']}.",
              f"- Hidden retry coverage unknown: {report['hidden_retry_coverage_unknown']}.",
              f"- Pricing: {report['pricing_mode']}.",
              f"- Usage SHA-256: `{report['usage_sha256']}`.",
              f"- Selected pricing SHA-256: `{report['pricing_sha256']}`.", "",
              f"- Measurement metadata SHA-256: `{report['measurement_sha256']}`.", "",
              "Full pricing assumptions and per-attempt evidence are in the JSON report.", ""]
    reasons = sorted({e["reason"] for e in report["entries"] if e["reason"]})
    if reasons:
        lines += ["Unavailable cost reasons:", "", *[f"- {r}" for r in reasons], ""]
    scenario = report.get("self_hosted")
    if scenario is not None:
        lines += ["## Self-hosted inference projections", ""]
        if not scenario["available"]:
            lines += [f"Unavailable: {scenario['reason']}.", ""]
        else:
            measured = scenario["observed_evaluation"]
            conditions = scenario["conditions"]
            lines += [
                f"Interpretation: **{scenario['interpretation']}**; all projected costs are estimates.", "",
                f"- Measured valid classifications/hour: {number(scenario['measured_successful_examples_per_hour'])}.",
                f"- Continuous USD / 1,000 valid classifications: {number(scenario['continuous']['cost_per_1000_successful_examples_usd'])}.",
                f"- Evaluation calls: {measured['successful_call_latency']['observation_count']}; failure fraction: {number(measured['unsuccessful_example_fraction'])}.",
                f"- P50/P95 call latency ms: {number(measured['successful_call_latency']['p50_ms'])} / {number(measured['successful_call_latency']['p95_ms'])}.",
                f"- Batch size / concurrency: {conditions['measurement']['batch_size']} / {conditions['concurrency']}.",
                f"- Phase: {measured['phase_label']}.",
                f"- Cache: {conditions['measurement']['cache_condition']}.",
                f"- Rate allocation: {scenario['rate']['resource']}.",
                f"- Rate source/date: {scenario['rate']['source']} / {scenario['rate']['effective_date']}.",
                f"- Rate assumptions: {scenario['rate'].get('assumptions', 'See selected rate card')}.",
                f"- Measured CPU / GPUs: {conditions['runtime'].get('cpu_model')} / {conditions['runtime'].get('gpu_devices')}.",
                f"- Input characters: {json.dumps(conditions['input_characters'])}; token-length distribution unavailable.",
                "", "Each volume scenario includes one measured warmup; preparation remains excluded.", "",
                "| Valid classifications | Continuous hours | Continuous USD | Continuous USD/1,000 | Deployed USD/1,000 | Busy fraction |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            deployed = {row['successful_examples']: row for row in scenario['deployed']['scenarios']}
            for row in scenario['continuous']['scenarios']:
                fixed = deployed.get(row['successful_examples'], {})
                fixed_cost = number(fixed.get('cost_per_1000_successful_examples_usd'))
                if fixed and not fixed['feasible']:
                    fixed_cost = 'infeasible at measured throughput'
                lines.append(f"| {row['successful_examples']} | {number(row['total_hours_including_one_warmup'])} | {number(row['total_cost_usd'])} | {number(row['cost_per_1000_successful_examples_usd'])} | {fixed_cost} | {number(fixed.get('projected_busy_fraction'))} |")
            lines += ["", f"Deployed allocation hours: {number(scenario['deployment_hours'])}.", ""]
            if scenario['deployed']['reason']:
                lines += [scenario['deployed']['reason'] + '.', ""]
            lines += ["Assumptions:", "", *[f"- {item}." for item in scenario['assumptions']], ""]
        lines += [f"FLOPs per prediction: unavailable. {scenario['flops_reason']}.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pricing", type=Path, help="Alternative rate card; saved usage is unchanged")
    parser.add_argument("--deployment-hours", type=float,
                        help="One self-hosted allocation's hours, including idle time; no schedule assumed by default")
    parser.add_argument("--volumes", nargs="+", type=int, default=[1000, 10000, 100000],
                        help="Projected counts of valid classifications")
    parser.add_argument("--output", type=Path, help="Optional Markdown report; defaults to stdout")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    # Repricing must never overwrite the original evidence or its original reports.
    protected = {p.resolve() for p in args.run_dir.iterdir() if p.is_file()}
    if args.pricing:
        protected.add(args.pricing.resolve())
    if args.output and args.json_output and args.output.resolve() == args.json_output.resolve():
        parser.error("Markdown and JSON outputs must be different files")
    for output in (args.output, args.json_output):
        if output is not None and output.resolve() in protected:
            parser.error("Output must be a new file outside the saved run's original files")
    try:
        report = recalculate_costs(args.run_dir, args.pricing,
                                   deployment_hours=args.deployment_hours, volumes=tuple(args.volumes))
    except ValueError as exc:
        parser.error(str(exc))
    markdown = render_report(report, args.run_dir.name)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown)
    else:
        print(markdown)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
