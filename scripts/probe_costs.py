"""Offline cost validation: simulated known API usage plus real local TF-IDF inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from probe_tfidf_classifier import fixture_bundle, StaticDataset
from report_costs import render_report
from llm_classifier_bench.classifiers import OpenAIClassifier, TfidfLogisticClassifier
from llm_classifier_bench.costs import recalculate_costs
from llm_classifier_bench.measurement import MeasurementConfig
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark


class SimulatedUsageClient:
    """No SDK, credentials or network: deterministic response/usage fixtures."""
    max_retries = 0

    def __init__(self, *, invalid_second=False):
        self.chat = SimpleNamespace(completions=self)
        self.calls = 0
        self.invalid_second = invalid_second

    def create(self, **kwargs):
        self.calls += 1
        invalid = self.invalid_second and self.calls == 2
        usage = {
            'prompt_tokens': 200 if invalid else 1000,
            'completion_tokens': 10 if invalid else 100,
            'prompt_tokens_details': {'cached_tokens': 0 if invalid else 400},
            'completion_tokens_details': {'reasoning_tokens': 0 if invalid else 20},
        }
        payload = {'id': f'simulated-{self.calls}', 'model': 'gpt-5-nano-2025-08-07',
                   'service_tier': 'default', 'usage': usage}
        label = 'invalid' if invalid else 'World'
        return SimpleNamespace(**payload, model_dump=lambda: payload,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'label': label})))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('artifacts/cost_validation/issue10'))
    args = parser.parse_args()
    pricing = Path(__file__).resolve().parents[1] / 'pricing/inference_2026-09-16.json'
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout)
    bundle = fixture_bundle()
    conditions = [
        ('simulated-success', OpenAIClassifier(client=SimulatedUsageClient()), 1),
        ('simulated-invalid', OpenAIClassifier(client=SimulatedUsageClient(invalid_second=True)), 0),
        ('local-tfidf', TfidfLogisticClassifier(), 1),
    ]
    reports = {}
    for name, classifier, warmup in conditions:
        try:
            run_benchmark(StaticDataset(bundle), classifier, BenchmarkRunConfig(
                output_root=args.output_root, run_id=name, validation_fraction=.25,
                pricing_path=pricing, measurement=MeasurementConfig(warmup_examples=warmup),
                metadata={'smoke': True, 'simulated_api': name.startswith('simulated'),
                          'code_revision': revision, 'working_tree_dirty': dirty,
                          'limitation': 'Known fabricated API usage and tiny local fixture; no paid requests or quality claims'},
            ))
        except ValueError as exc:
            if name != 'simulated-invalid' or 'not in the class set' not in str(exc):
                raise
        run_dir = args.output_root / name
        report = recalculate_costs(run_dir)
        (run_dir / 'cost_report.md').write_text(render_report(report, name))
        reports[name] = report
    # Independent arithmetic, including cached-input discount and reasoning subset.
    assert abs(reports['simulated-success']['total_cost_usd'] - .000288) < 1e-12
    assert abs(reports['simulated-success']['cost_per_1000_successful_examples_usd'] - .096) < 1e-12
    assert abs(reports['simulated-invalid']['total_cost_usd'] - .000086) < 1e-12
    assert abs(reports['simulated-invalid']['cost_per_1000_successful_examples_usd'] - .086) < 1e-12
    assert reports['simulated-invalid']['failure_rate'] == .5
    timings = [json.loads(line) for line in (args.output_root / 'local-tfidf/timings.jsonl').read_text().splitlines()]
    active_ms = sum(row['elapsed_ms'] for row in timings if row['kind'] == 'inference')
    assert abs(reports['local-tfidf']['total_cost_usd'] - active_ms / 3_600_000 * (164 / 730)) < 1e-12
    summary = {name: {key: report[key] for key in (
        'run_status', 'total_cost_usd', 'cost_kind', 'successful_examples',
        'cost_per_1000_successful_examples_usd', 'failure_rate', 'usage_sha256', 'pricing_sha256',
    )} for name, report in reports.items()}
    (args.output_root / 'validation.json').write_text(json.dumps({
        'offline_only': True, 'checks_passed': True, 'code_revision': revision,
        'working_tree_dirty': dirty, 'reports': summary,
    }, indent=2) + '\n')
    print(args.output_root / 'validation.json')


if __name__ == '__main__':
    main()
