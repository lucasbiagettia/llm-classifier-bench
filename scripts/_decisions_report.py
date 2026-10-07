"""Offline publication of Decisions coverage, without scoring accepted-only subsets."""
from pathlib import Path
import json

import numpy as np

from _decisions_recovery import inspect_cell
from llm_classifier_bench.metrics.evaluator import evaluate_records, evaluation_record_from_mapping
from llm_classifier_bench.metrics.uncertainty import stratified_quality_bootstrap
from run_budget_extension import read_json, sha256, require, load_source, build_bundle, ClassDefinition, ids


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def collect_decisions(root, historical, quality):
    """Audit immutable campaign segments and recompute only full-cohort metrics."""
    root = Path(root)
    plan = read_json(root / 'campaign.json')
    require(plan['seed'] == 42 and plan['class_counts'] == [5, 10, 15, 20], 'Unexpected Decisions matrix')
    require([(c['k'], c['budget']) for c in plan['cells']] == [(k, 0) for k in (5, 10, 15, 20)],
            'Unexpected Decisions cells')
    fingerprints = {str(root / 'campaign.json'): sha256(root / 'campaign.json')}
    for name, digest in plan['input_sha256'].items():
        require(sha256(name) == digest, f'Decisions reference evidence changed: {name}')
    source = load_source(Path(plan['source_dir']), plan)
    conditions, paired = [], []
    for cell in plan['cells']:
        k = cell['k']
        reference = plan['conditions'][str(k)]
        bundle = build_bundle(source, [ClassDefinition(**c) for c in reference['classes']])
        require(ids(bundle.test) == reference['test_sample_ids'], 'Decisions cohort changed')
        state = inspect_cell(root, plan['campaign_id'], cell, bundle, plan)
        require(not state['pending'], f'Decisions still has pending samples at K={k}')
        directory = root / 'cells' / cell['id']
        coverage = read_json(directory / 'coverage.json')
        require(coverage['planned_sample_ids'] == ids(bundle.test)
                and coverage['pending_examples'] == 0
                and coverage['successful_examples'] == len(state['predictions'])
                and coverage['refused_examples'] == len(state['refusals'])
                and coverage['evidence_sha256'] == state['evidence_sha256'], 'Decisions coverage evidence differs')
        fingerprints.update(state['evidence_sha256'])
        for name in ('coverage.json', 'outcomes.jsonl', 'predictions.jsonl', 'refusals.jsonl', 'metrics.json', 'cost_report.json'):
            path = directory / name
            fingerprints[str(path)] = sha256(path)
        predictions = [state['predictions'][e.sample_id] for e in bundle.test if e.sample_id in state['predictions']]
        require(records(directory / 'predictions.jsonl') == predictions, 'Decisions aggregate predictions differ')
        refused = len(state['refusals'])
        status = 'completed_with_refusals' if refused else 'completed'
        require(coverage['status'] == status, 'Decisions coverage status differs')
        row = {'method': 'openai-decisions', 'title': 'GPT-6 Decisions zero-shot', 'k': k,
               'fit_per_class': 0, 'n': len(predictions), 'expected': len(bundle.test),
               'probability_vectors': len(predictions), 'status': status, 'evidence': str(directory),
               'refused_examples': refused, 'pending_examples': 0,
               'coverage': len(predictions) / len(bundle.test),
               'refused_sample_ids': [e.sample_id for e in bundle.test if e.sample_id in state['refusals']],
               'metrics': {m.name: {'name': m.name, 'value': None, 'available': False,
                           'metadata': {'reason': 'Full-cohort predictions unavailable due to refusals'}} for m in quality}}
        if not refused:
            rec = [evaluation_record_from_mapping(r) for r in predictions]
            row['metrics'] = {m.name: m.as_dict() for m in evaluate_records(rec, metrics=quality)}
            row['uncertainty'] = stratified_quality_bootstrap(rec)
            for method in ('openai', 'jev'):
                baseline, = [r for r in historical['conditions'] if r['method'] == method and r['k'] == k]
                other = [evaluation_record_from_mapping(r) for r in records(Path(baseline['evidence']) / 'predictions.jsonl')]
                paired.append({'reference': method, 'k': k,
                               'uncertainty': stratified_quality_bootstrap(rec, paired_records=other)})
        saved_metrics = read_json(directory / 'metrics.json')
        require(all(saved_metrics[m.name]['value'] == row['metrics'][m.name]['value'] for m in quality),
                'Decisions saved quality metrics differ')
        attempts, phases = [], []
        for run in map(Path, state['runs']):
            attempts.extend(e for e in records(run / 'usage.jsonl') if e['kind'] == 'api_attempt')
            times = [r['latency_ms'] for r in records(run / 'predictions.jsonl')]
            phases.append({'evidence': str(run), 'successful_examples': len(times),
                           'latency_p50_ms': float(np.median(times)) if times else None,
                           'latency_p95_ms': float(np.quantile(times, .95)) if times else None})
        costs = state['costs']
        total = sum(r['total_cost_usd'] for r in costs) if all(r['total_cost_usd'] is not None for r in costs) else None
        known = sum(r['known_cost_subtotal_usd'] for r in costs)
        saved_cost = read_json(directory / 'cost_report.json')
        require(saved_cost['total_cost_usd'] == total and saved_cost['known_cost_subtotal_usd'] == known,
                'Decisions saved costs differ')
        row['operations'] = {'api_attempts': len(attempts),
            'http_503_attempts': sum(e.get('http_status') == 503 for e in attempts),
            'total_cost_usd': total, 'known_cost_subtotal_usd': known,
            'cost_per_1000_successful_usd': total * 1000 / len(predictions) if total is not None and predictions else None,
            'latency_p50_ms': saved_metrics['latency_p50_ms']['value'],
            'latency_p95_ms': saved_metrics['latency_p95_ms']['value'],
            'measurement_phases': phases}
        conditions.append(row)
    return {'schema_version': 1, 'campaign_root': str(root), 'campaign_id': plan['campaign_id'],
            'model': plan['decisions']['requested_model'], 'seed': plan['seed'],
            'scope': 'One resumed campaign; earlier abandoned campaign excluded. Refusals were not retried.',
            'audit': plan['audit'], 'conditions': conditions, 'paired_effects': paired,
            'input_sha256': plan['input_sha256'], 'evidence_sha256': fingerprints}


def validate_summary(summary):
    rows = summary['conditions']
    require([r['k'] for r in rows] == [5, 10, 15, 20], 'Decisions summary must contain four ordered cells')
    for row in rows:
        require(row['n'] + row['refused_examples'] == row['expected'] == 40 * row['k']
                and row['pending_examples'] == 0 and row['probability_vectors'] == row['n'],
                'Invalid Decisions coverage')
        require(row['status'] == ('completed_with_refusals' if row['refused_examples'] else 'completed'),
                'Invalid Decisions status')
        require(row['coverage'] == row['n'] / row['expected'], 'Invalid Decisions coverage fraction')
        if row['refused_examples']:
            require(all(m['value'] is None for m in row['metrics'].values()) and 'uncertainty' not in row,
                    'Refused cohorts cannot publish accepted-only quality metrics')


def narrative(summary, *, brief=False):
    """Render the public supplement exclusively from versioned audited numbers."""
    validate_summary(summary)
    rows = summary['conditions']
    good = sum(r['n'] for r in rows)
    refused = sum(r['refused_examples'] for r in rows)
    lines = ['## GPT-6 Decisions supplement', '',
        f"The user-run `{summary['model']}` campaign produced {good:,} classifications and {refused} refusals "
        'across the same 2,000 planned evaluations (K=5/10/15/20, seed 42, 40 test/class). '
        'All inputs were attempted; none remain pending. The 64 historical conditions are unchanged. '
        'Only one of the four additional conditions has full classification coverage.', '',
        '| K | Classifications / planned | Refusals | Coverage | Accuracy | Macro-F1 | ECE | Adaptive ECE | Log loss | Brier |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        vals = [r['metrics'][m]['value'] for m in ('accuracy', 'macro_f1', 'top_label_ece', 'adaptive_ece',
                                                 'multiclass_log_loss', 'multiclass_brier_score')]
        rendered = [('—' if v is None else f'{v*100:.2f}%' if i == 0 else f'{v:.4f}') for i, v in enumerate(vals)]
        lines.append(f"| {r['k']} | {r['n']}/{r['expected']} | {r['refused_examples']} | {100*r['coverage']:.3f}% | "
                     + ' | '.join(rendered) + ' |')
    lines += ['', 'A dash means full-cohort quality is unavailable. Refusals have no label or probabilities; '
        'they are neither removed to score an accepted-only subset nor assigned invented predictions. '
        'Their cause is unknown; ambiguity has not been established as the cause. '
        'The generative OpenAI baseline (GPT-5 nano) remains a separate method without probabilities. '
        'Decisions supplies native probabilities for every accepted classification.', '']
    if not brief:
        lines += ['At K=5, the following pointwise paired intervals compare Decisions with the existing zero-shot references. '
                  'They use the same 2,000 gold-stratified bootstrap resamples and seed as the historical report. '
                  'No paired quality comparison is made for cohorts with refusals.', '',
                  '| Reference | K | Δ accuracy [95% CI], pp | Δ macro-F1 [95% CI], pp |',
                  '| --- | ---: | --- | --- |']
        for pair in summary['paired_effects']:
            values = [pair['uncertainty']['metrics'][m] for m in ('accuracy', 'macro_f1')]
            formatted = [f"{100*v['estimate']:+.2f} [{100*v['lower']:+.2f}, {100*v['upper']:+.2f}]" for v in values]
            lines.append(f"| {pair['reference']} zero-shot | {pair['k']} | " + ' | '.join(formatted) + ' |')
        lines += ['', '| K | API attempts | HTTP 503 | Estimated total USD | Known subtotal USD | p50 / p95 latency, ms |',
                  '| ---: | ---: | ---: | ---: | ---: | --- |']
        for r in rows:
            o = r['operations']
            total = '—' if o['total_cost_usd'] is None else f"{o['total_cost_usd']:.7f}"
            latency = '—' if o['latency_p50_ms'] is None else f"{o['latency_p50_ms']:.2f} / {o['latency_p95_ms']:.2f}"
            lines.append(f"| {r['k']} | {o['api_attempts']} | {o['http_503_attempts']} | {total} | "
                         f"{o['known_cost_subtotal_usd']:.7f} | {latency} |")
        lines += ['', 'Costs use the recorded input-token rate card, include refused attempts, and describe this campaign only. '
                  'K=20 includes a resumed HTTP 503 attempt whose usage is unknown: its total cost is unavailable, '
                  'not equal to the known subtotal. The failed attempt and its subsequent retry are both retained. '
                  'Latency is not pooled across resumed segments; segment-specific measurements remain in the summary. '
                  'No cross-provider cost or latency winner is established.', '']
    lines += ['Evidence and reproduction: [Decisions summary](decisions_results.json) records coverage, metrics, '
              'paired intervals, segment provenance and SHA-256 fingerprints. '
              '[Execution and recovery instructions](../../docs/gpt6_decisions.md). '
              'This supplement uses one resumed campaign; the earlier abandoned run is excluded, and refused '
              'inputs were not retried. The tests overlap across K, so the 2,000 evaluations are not independent samples.', '']
    return lines
