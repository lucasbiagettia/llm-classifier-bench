"""Recalculate preparation cost and an explicitly selected amortization scenario."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_classifier_bench.costs import recalculate_costs
from llm_classifier_bench.preparation import recalculate_preparation, amortize_preparation


def render_report(report):
    prep, amortization = report['preparation'], report['amortization']
    def number(value):
        return 'unavailable' if value is None else f'{value:.10g}'
    lines = ['# Preparation investment and amortization', '',
             f"Complete preparation: {prep['complete']}. Cost kind: **{prep['cost_kind']}**.", '',
             f"- Preparation wall milliseconds: {number(prep.get('total_elapsed_ms'))}.",
             f"- Total incremental preparation USD: {number(prep['total_cost_usd'])}.",
             f"- Cost basis: {prep.get('basis', prep.get('reason'))}.", '',
             'Parent prepare/fit intervals are counted once. Stage components use exclusive time;',
             'nested child time is subtracted from its parent before grouping. Warmup belongs to inference.', '',
             '## Measured components', '', '| Category | Exclusive ms |', '| --- | ---: |']
    for category, ms in prep.get('exclusive_ms_by_category',{}).items():
        lines.append(f'| {category} | {number(ms)} |')
    selected = prep.get('selected_configuration')
    lines += ['', '## Selected configuration', '']
    if selected:
        lines += [f"- Configuration: `{json.dumps(selected['selected_configuration'])}`.",
                  f"- Final refit performed: {selected['final_refit_performed']}.",
                  f"- Selected fit exclusive ms: {number(selected['selected_fit_exclusive_ms'])}.",
                  f"- Selected fit cost estimate USD: {number(selected['selected_fit_cost_estimate_usd'])}.",
                  selected['role'] + '.', '']
    else:
        lines += ['Selected-fit component unavailable or not applicable.', '']
    lines += ['## Amortization', '', f"Inference scenario: **{amortization['scenario']}**.", '',
              '| Valid predictions | Preparation USD | Inference scenario USD | Total USD/prediction | Availability |',
              '| ---: | ---: | ---: | ---: | --- |']
    for row in amortization['rows']:
        lines.append(f"| {row['successful_examples']} | {number(row['preparation_cost_usd'])} | {number(row['inference_scenario_cost_usd'])} | {number(row['amortized_cost_per_valid_prediction_usd'])} | {row['reason'] or 'estimated projection'} |")
    lines += ['', amortization['assumptions']+'.', '',
              'Preparation charges remain separate from inference charges. Unknown costs and infeasible',
              'inference scenarios do not become zero. Stage estimates do not allocate an observed invoice.', '',
              'Timing includes instrumentation overhead. Hardware, configuration, cache provenance,',
              'rate assumptions, billing evidence and hashes are in the JSON report and source artifacts.', '']
    if prep.get('reason'):
        lines += [f"Preparation cost limitation: {prep['reason']}.", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--pricing', type=Path)
    parser.add_argument('--billing', type=Path, help='Documented entire-preparation charge or billable allocation evidence')
    parser.add_argument('--inference-scenario', required=True, choices=['continuous','deployed','recorded_api'])
    parser.add_argument('--deployment-hours', type=float)
    parser.add_argument('--volumes', type=int, nargs='+', default=[1000,10000,100000])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()
    if args.inference_scenario == 'deployed' and args.deployment_hours is None:
        parser.error('deployed inference requires --deployment-hours')
    if args.inference_scenario != 'deployed' and args.deployment_hours is not None:
        parser.error('--deployment-hours applies only to the deployed scenario')
    protected = {p.resolve() for p in args.run_dir.rglob('*') if p.is_file()}
    protected |= {p.resolve() for p in (args.pricing,args.billing) if p is not None}
    outputs = [p.resolve() for p in (args.output,args.json_output) if p is not None]
    if len(set(outputs)) != len(outputs) or any(p in protected for p in outputs):
        parser.error('Use distinct report output paths; source evidence cannot be overwritten')
    try:
        prep = recalculate_preparation(args.run_dir, args.pricing, args.billing)
        inference = recalculate_costs(args.run_dir, args.pricing,
                                      deployment_hours=args.deployment_hours, volumes=tuple(args.volumes))
        report = {'preparation':prep,
                  'amortization':amortize_preparation(prep,inference,scenario=args.inference_scenario,volumes=tuple(args.volumes)),
                  'inference':inference}
    except ValueError as exc:
        parser.error(str(exc))
    rendered = render_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True,exist_ok=True)
        args.json_output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
