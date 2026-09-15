"""Regenerate an operational report from saved timing observations, without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_classifier_bench.measurement import regenerate_report


def render_report(report: dict, run_name: str) -> str:
    metadata = report["measurement"]
    config, runtime = metadata["config"], metadata["runtime"]
    evaluation = report["evaluation"]

    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"

    lines = [
        f"# Operational report: {run_name}", "",
        "Generated from `measurement.json` and `timings.jsonl`; no inference was repeated.", "",
        f"Run status: **{metadata['run_status']}**. Phase: {evaluation['phase_label']}.", "",
        "## Measurement conditions", "",
        "| Condition | Value |", "| --- | --- |",
        f"| Boundary | {metadata['boundary']} |",
        f"| Backend | {runtime['classifier'].get('backend')} |",
        f"| Classifier / model | {metadata['classifier_name']} / {metadata['model']} |",
        f"| Device | {runtime['classifier'].get('device', 'provider hardware unavailable')} |",
        f"| CPU | {runtime['cpu_model']} |",
        f"| GPUs initialized in process | {runtime['gpu_devices']} |",
        f"| Platform / Python | {runtime['platform']} / {runtime['python']} |",
        f"| Client location | {config['client_location'] or 'unavailable'} |",
        f"| Batch size / concurrency | {config['batch_size']} / {metadata['concurrency']} |",
        f"| Batch execution | {runtime['classifier'].get('batch_execution')} |",
        f"| Cache conditions | {config['cache_condition']} |",
        f"| Warmup calls / failures | {report['warmup']['observation_count']} / {report['warmup_failed_calls']} |",
        f"| Warmup examples | {report['warmup_attempted_examples']} |",
        f"| Application retries | {metadata['application_retries']} |",
        "", "Transport retry/timeout settings, library versions, affinity and thread settings:", "",
        "```json", json.dumps({key: runtime[key] for key in
                               ('classifier', 'library_versions', 'cpu_affinity', 'thread_environment', 'torch_threads')}, indent=2), "```", "",
        "## Preparation (excluded from inference)", "",
        "| Stage | Status | Wall ms |", "| --- | --- | ---: |",
    ]
    lines.extend(f"| {r['phase']} | {r['status']} | {number(r['elapsed_ms'])} |" for r in report["preparation"])
    lines += ["", "Classifier construction occurred before the runner and is not timed here.",
              "Model loading performed inside prepare/fit is included in those stage durations.", "",
              "## Evaluation call latency", "",
              "| Population | Calls | Mean ms | P50 ms | P95 ms | P99 ms |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, summary in (("Successful", evaluation["successful_call_latency"]),
                           ("Failed or invalid", evaluation["failed_call_latency"])):
        lines.append(f"| {label} | {summary['observation_count']} | {number(summary['mean_ms'])} | "
                     f"{number(summary['p50_ms'])} | {number(summary['p95_ms'])} | {number(summary['p99_ms'])} |")
    for size, summary in evaluation["successful_call_latency_by_batch_size"].items():
        lines.append(f"| Successful, batch={size} | {summary['observation_count']} | {number(summary['mean_ms'])} | "
                     f"{number(summary['p50_ms'])} | {number(summary['p95_ms'])} | {number(summary['p99_ms'])} |")
    lines += ["", "P99 requires at least 1,000 calls in its population; this heuristic is not a confidence interval.", "",
              "## Throughput and coverage", "",
              f"- Successful / attempted / unattempted examples: {evaluation['successful_examples']} / {evaluation['attempted_examples']} / {evaluation['unattempted_examples']}.",
              f"- Failed-call fraction: {number(evaluation['failed_call_fraction'])}.",
              f"- Successful examples per second, observed window: {number(evaluation['throughput_successful_examples_per_second'])}.",
              f"- Successful examples per second, active calls only: {number(evaluation['active_call_throughput_successful_examples_per_second'])}.",
              f"- Observed window / active call ms: {number(evaluation['window_wall_ms'])} / {number(evaluation['active_call_ms'])}.",
              f"- Amortized successful-call ms per example: {number(evaluation['amortized_successful_call_ms_per_example'])}.", "",
              f"Throughput denominator: {evaluation['throughput_denominator']}.", "",
              "Amortized time is a processing rate, not an individual request latency. Existing",
              "adapters process examples sequentially even when predict receives a larger list.",
              "Hosted calls include client/network/service time; local calls include preprocessing,",
              "local compute and output materialization. These are client-observed boundaries, not",
              "identical measures of server compute. Unknown provider cache, hardware and injected",
              "transport-attempt timings remain unavailable.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = regenerate_report(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(report, args.run_dir.name))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
