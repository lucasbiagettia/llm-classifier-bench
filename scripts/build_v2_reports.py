"""Render the v1.2 Banking77 reports offline; optionally recompute saved evidence."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from llm_classifier_bench.metrics.evaluator import (
    DEFAULT_METRICS, evaluate_records, evaluation_record_from_mapping,
)
from llm_classifier_bench.metrics.uncertainty import stratified_quality_bootstrap

REPORT = Path('reports/v2')
CACHE = Path('.local/report-cache/v1.2')
COUNTS = (5, 10, 15, 20)
METHODS = {
    'tfidf': 'TF-IDF + LR', 'sentence-transformer': 'MiniLM + LR', 'bert': 'BERT',
    'openai': 'OpenAI zero-shot', 'emissary': 'Emissary routing zero-shot',
    'jev': 'Jev zero-shot', 'emissary-qwen': 'Emissary Qwen SFT',
}
BUDGETS = {m: (20, 50, 100) if m in ('tfidf', 'sentence-transformer', 'bert')
           else (20, 100) if m == 'emissary-qwen' else (0,) for m in METHODS}
QUALITY = [m for m in DEFAULT_METRICS if m.name in (
    'accuracy', 'macro_f1', 'top_label_ece', 'adaptive_ece', 'multiclass_log_loss', 'multiclass_brier_score')]
CALIBRATION = ('top_label_ece', 'adaptive_ece', 'multiclass_log_loss', 'multiclass_brier_score')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def num(value):
    return '—' if value is None else f'{value:.3f}'


def pct(value):
    return '—' if value is None else f'{100*value:.2f}%'


def title(method, budget):
    return METHODS[method] + (f' ({budget}/class)' if budget else '')


def conditions():
    return [(m, b, k) for m in METHODS for b in BUDGETS[m] for k in COUNTS]


def key(method, budget, k):
    return f'{method}__s42__k{k}__b{budget}'


def validate_prediction_identity(rows, expected):
    ids = [r['sample_id'] for r in rows]
    if len(ids) != len(set(ids)) or any(
        expected.get(r['sample_id']) != (r['input'], r['gold_label']) for r in rows
    ):
        raise ValueError('Duplicate or changed prediction identities')
    return set(ids) == set(expected)


def bootstrap(cache_key, rec, fingerprints, paired=None):
    path = CACHE / f'{cache_key}.json'
    if path.exists() and read_json(path).get('prediction_sha256') == fingerprints:
        return read_json(path)
    result = {**stratified_quality_bootstrap(rec, paired_records=paired), 'prediction_sha256': fingerprints}
    write_json(path, result)
    return result


def audit_extension_run(root, cell, plan, source):
    """Verify consumed pools, no test leakage, fixed settings and actual SFT upload."""
    from run_budget_extension import (
        build_bundle, pools, audit_partitions, verify_selection, provider_payload,
        ClassDefinition, ids, require,
    )
    k, budget = cell['k'], cell['budget']
    condition = plan['conditions'][str(k)]
    bundle = build_bundle(source, [ClassDefinition(**c) for c in condition['classes']])
    selected, reserved, _ = pools(bundle, budget)
    overlap = audit_partitions(selected, reserved, source['test'])
    config = read_json(root/'config.json')
    fit = read_json(root/'fit_metadata.json')
    pool = read_json(root/'labeled_budget.json')
    require(config['dataset']['classes'] == condition['classes'], 'Extension class definitions differ')
    require(config['dataset']['test_sample_ids'] == ids(bundle.test), 'Extension test IDs differ')
    require(config['dataset']['fit_train_sample_ids'] == ids(selected), 'Extension fit IDs differ')
    require(ids(selected) == condition['training_sample_ids'][str(budget)], 'Frozen fit pool differs')
    require(not config['dataset']['validation_sample_ids'], 'Unexpected validation consumption')
    require(fit['training_examples_used'] == budget*k and fit['validation_examples_used'] == 0,
            'Extension consumed a different training budget')
    verify_selection(pool, selected)
    reference = next(r for r in plan['references'] if (r['method'], r['k']) == (cell['method'], k))
    if cell['method'] == 'emissary-qwen':
        payload = provider_payload(selected, bundle.classes, budget)
        digest = hashlib.sha256(payload).hexdigest()
        require(digest == fit['dataset_payload_sha256'], 'Extension SFT payload differs')
        require(sha256(root.parent/'remote/train.jsonl') == digest, 'Actual SFT upload differs')
        require({e['sample_id'] for e in fit['selection']['selected_examples']} == set(ids(selected)),
                'SFT consumed a different pool')
        remote = fit['training_response']
        require(remote['base_model'] == 'Qwen3-4B-Base' and not remote.get('base_fine_tuned_model')
                and not remote.get('test_dataset'), 'Unexpected SFT continuation/test dataset')
        require(remote['hyper_parameters'] == reference['parameters'], 'SFT hyperparameters changed')
        for field in ('dataset_id', 'training_job_id', 'deployment_id'):
            require(config['classifier']['training'][field] is None, 'SFT reused historical resources')
    else:
        require(config['classifier']['training'] == reference['config']['training'], 'Training settings changed')
        require(config['classifier']['model'] == reference['config']['model'], 'Model snapshot changed')
    return {'cell_id': key(cell['method'], budget, k), 'k': k, 'fit_per_class': budget,
            'status': 'passed', **overlap, 'test_scope': 'entire official test split',
            'consumed_validation': 0, 'pool_sha256': pool['pool_sha256']}


def collect(completion_root, extension_root=Path('artifacts/v2_budget_extension')):
    from run_budget_extension import load_source
    plan = read_json(extension_root/'manifest.json')
    source = load_source(Path(plan['source_dir']), plan)
    extension = read_json(extension_root/'summary.json')
    expected = {}
    for k in COUNTS:
        rows = records(Path(f'artifacts/v2_small/cells/openai__s42__k{k}__b0/predictions.jsonl'))
        expected[k] = {r['sample_id']: (r['input'], r['gold_label']) for r in rows}
        if len(rows) != 40*k or len(expected[k]) != len(rows):
            raise ValueError('Invalid original test cohort')
    results, completed_records, measurements, fingerprints, audits = [], {}, [], {}, []
    for method, budget, k in conditions():
        name = key(method, budget, k)
        new = budget in (20, 100) and method in ('tfidf', 'sentence-transformer', 'bert') or (
            method == 'emissary-qwen' and budget == 20)
        if new:
            root = extension_root/'cells'/name/'run'
            saved = read_json(root/'status.json') if (root/'status.json').exists() else {}
            state = extension.get(name, {})
            saved_complete = saved.get('status') == 'completed' and state.get('status') == 'completed'
            phases = [{'phase': 'budget_extension', 'run_dir': str(root)}]
            if saved_complete:
                audits.append(audit_extension_run(root, state['cell'], plan, source))
        else:
            root = Path('artifacts/v2_small/cells')/name
            candidate = completion_root/'cells'/name
            if method == 'emissary-qwen' or (candidate/'result.json').exists():
                root = candidate
            saved = read_json(root/'result.json') if (root/'result.json').exists() else {}
            saved_complete = saved.get('status') == 'completed'
            phases = saved.get('evidence', [])
        path = root/'predictions.jsonl'
        rows = records(path) if path.exists() else []
        full = validate_prediction_identity(rows, expected[k])
        valid = saved_complete and full
        item = {'method': method, 'title': title(method, budget), 'k': k, 'fit_per_class': budget,
                'n': len(rows), 'expected': 40*k, 'status': 'completed' if valid else 'incomplete' if rows else 'pending',
                'evidence': str(root), 'probability_vectors': sum(bool(r.get('probabilities')) for r in rows),
                'phase': 'budget_extension' if new else 'original'}
        if path.exists():
            fingerprints[str(path)] = sha256(path)
        if valid:
            rec = tuple(evaluation_record_from_mapping(r) for r in rows)
            metrics = {m.name: m.as_dict() for m in evaluate_records(rec, metrics=QUALITY)}
            if any(not r.get('probabilities') for r in rows):
                for metric in CALIBRATION:
                    metrics[metric] = {'name': metric, 'value': None, 'available': False,
                                       'metadata': {'reason': 'Full-cohort probabilities unavailable'}}
            item['metrics'] = metrics
            item['uncertainty'] = bootstrap(name, rec, sha256(path))
            completed_records[method, budget, k] = rec
            for phase in phases:
                if not phase.get('run_dir'):
                    continue
                run = Path(phase['run_dir'])
                pred = run if run.is_file() else run/'predictions.jsonl'
                if not pred.exists():
                    continue
                phase_rows = records(pred)
                latencies = [r['latency_ms'] for r in phase_rows if r.get('latency_ms') is not None]
                measurements.append({'method': method, 'fit_per_class': budget, 'k': k,
                    'phase': phase['phase'], 'evidence': str(run), 'n': len(phase_rows),
                    'latency_p50_ms': float(np.median(latencies)) if latencies else None,
                    'latency_p95_ms': float(np.quantile(latencies, .95)) if latencies else None,
                    'cost_note': 'Retain original ledgers, including failures. Unknown charges are not zero.'})
        results.append(item)
        print(f'Collected {name}: {item["status"]}', flush=True)
    by_key = {(r['method'], r['fit_per_class'], r['k']): r for r in results}
    paired = []

    def pair(first, second, comparison):
        if first not in completed_records or second not in completed_records:
            return
        hashes = {key(*c): by_key[c]['uncertainty']['prediction_sha256'] for c in (first, second)}
        ci = bootstrap(f'paired-{key(*first)}-minus-{key(*second)}', completed_records[first], hashes,
                       completed_records[second])
        paired.append({'method': first[0], 'fit_per_class': first[1], 'k': first[2],
                       'reference': second[0], 'reference_fit_per_class': second[1],
                       'comparison': comparison, 'uncertainty': ci})

    for method, budget, k in conditions():
        if method != 'openai':
            pair((method, budget, k), ('openai', 0, k), 'versus_zero_shot_openai')
        if budget in (20, 100) and method in ('tfidf', 'bert', 'emissary-qwen'):
            pair((method, budget, k), ('sentence-transformer', budget, k), 'matched_budget_vs_minilm')
        if budget == 100:
            pair((method, 100, k), (method, 20, k), 'within_method_100_minus_20')
    complete = sum(r['status'] == 'completed' for r in results)
    return {'schema_version': 2, 'report_version': 'v1.2', 'status': 'evaluated' if complete == 56 else 'interim',
            'conditions': results, 'paired_effects': paired, 'measurement_phases': measurements,
            'evidence_sha256': fingerprints,
            'data_integrity': {'historical': plan['audit'], 'extension': audits,
                               'limits': 'No audit of base pretraining, semantic duplicates or provider internals'},
            'extension_manifest_sha256': sha256(extension_root/'manifest.json')}


def value(row, metric='accuracy'):
    return row.get('metrics', {}).get(metric, {}).get('value')


def accuracy_table(results):
    index = {(r['method'], r['fit_per_class'], r['k']): r for r in results}
    lines = ['| Method | Fit/class | K=5 | K=10 | K=15 | K=20 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for method in METHODS:
        for budget in BUDGETS[method]:
            vals = [pct(value(index[method, budget, k])) for k in COUNTS]
            lines.append(f'| {METHODS[method]} | {budget} | ' + ' | '.join(vals) + ' |')
    return lines


def paired_table(payload, comparison):
    lines = ['| Method | Fit/class | K | Δ accuracy [95% CI], pp | Δ macro-F1 [95% CI], pp |',
             '| --- | ---: | ---: | --- | --- |']
    for row in payload['paired_effects']:
        if row['comparison'] != comparison:
            continue
        def ci(name):
            metric = row['uncertainty']['metrics'][name]
            return f"{100*metric['estimate']:+.2f} [{100*metric['lower']:+.2f}, {100*metric['upper']:+.2f}]"
        lines.append(f"| {METHODS[row['method']]} | {row['fit_per_class']} | {row['k']} | {ci('accuracy')} | {ci('macro_f1')} |")
    return lines


def render(payload):
    results = payload['conditions']
    expected = set(conditions())
    actual = {(r['method'], r['fit_per_class'], r['k']) for r in results}
    if len(results) != len(expected) or actual != expected:
        raise ValueError('Summary must account for all 56 method/budget/cardinality conditions')
    REPORT.mkdir(exist_ok=True, parents=True)
    completed = sum(r['status'] == 'completed' for r in results)
    write_json(REPORT/'results.json', payload)
    write_json(REPORT/'data_integrity.json', payload['data_integrity'])
    with (REPORT/'results.csv').open('w') as stream:
        fields = ['method', 'k', 'fit_per_class', 'status', 'n', 'expected', 'probability_vectors', *[m.name for m in QUALITY]]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in results:
            writer.writerow({**{k: row[k] for k in fields if k in row},
                             **{k: v['value'] for k, v in row.get('metrics', {}).items()}})
    design = (
        'Seed 42; nested label sets of 5/10/15/20 classes; 40 official held-out test examples/class '
        '(200/400/600/800 predictions per condition). The 28 new budget conditions add 14,000 predictions '
        'to the 28 earlier conditions. All methods within each K share identical test IDs, text and labels. '
        'The eligible pool contains 42 labels. See the [protocol](protocol.md) and [manifest](manifest.json).')
    budgets = (
        'TF-IDF+LR, frozen MiniLM+LR and BERT use 20/50/100 fit examples per class. Emissary Qwen3-4B-Base SFT '
        'uses 20/100 per class. OpenAI, Jev and Emissary routing remain zero-shot. Within each K, '
        '20 ⊂ 50 ⊂ 100 training pools, with identical members across methods at equal budgets. '
        'Twenty-five additional examples/class remain reserved and unused for validation. Both regressions '
        'retain C=1; BERT retains two CPU epochs; Qwen retains one epoch and selects the last checkpoint. '
        'Hyperparameters are fixed rather than tuned independently for each budget. Qwen 20 starts from the '
        'pretrained base, not from the 100-shot model. These are training labels, not prompt demonstrations.')
    integrity = (
        'Offline checks passed for 16 historical supervised conditions and all 28 new conditions. '
        'There is no overlap of IDs, exact text, or text normalized with NFKC/casefold/collapsed whitespace '
        'between fit, reserved validation and test. Checks include the entire official test split and '
        'the recorded Qwen training uploads. [Audit evidence](data_integrity.json). This does not audit '
        'base-model pretraining, semantic duplicates or the provider’s internal processing.')
    limitations = (
        'One seed and previously observed test results make this an exploratory extension. Bootstrap intervals '
        'condition on fitted models, labels, class balance and campaign seed; they do not estimate training-seed '
        'variability. The 2,000 gold-stratified resamples use seed 20260925 and pointwise 95% percentile '
        'intervals without multiplicity adjustment. Test cohorts overlap across K and must not be pooled as '
        'independent observations. Definitions were assistant-reviewed, not independently human-validated. '
        'Changing K adds classes and examples, so trends do not isolate a causal cardinality effect.')
    availability = (
        'OpenAI has no probabilities. Jev retains complete accuracy/F1 coverage but has 1/2/3 unavailable '
        'probability vectors at K=10/15/20; full-cohort calibration is unavailable there. The earlier recovery '
        'kept the 0.0001 sum tolerance and retained a class choice only when the sum was the sole invalidity '
        'and the choice was a maximum of finite nonnegative scores with positive total mass. No probabilities '
        'were renormalized. Recovery requested 970 missing IDs; historical and recovered timings remain separate.')
    # Statements are scoped to the completed published extension, never inferred for partial runs.
    findings = []
    if completed == 56:
        findings = [
            'At equal 20/class budgets, MiniLM+LR has the highest accuracy and macro-F1 among the four supervised '
            'methods at every K. At equal 100/class budgets, MiniLM leads accuracy at K=10/15/20 and ties Qwen '
            'at K=5 (98.00%); Qwen has the slightly higher macro-F1 at K=5. These are point estimates; paired '
            'intervals below quantify uncertainty.',
            'Qwen 100/class no longer has the highest accuracy at K=20 after adding MiniLM 100/class: '
            '92.25% versus 92.38%. The difference is one prediction out of 800. At K=10/15, MiniLM 100/class '
            'scores 96.50%/95.83%, versus Qwen 93.25%/93.83%.',
            'Qwen 20/class scores 84.00%, 91.50%, 89.50% and 84.50% accuracy. More labels improve Qwen in '
            'every K under this fixed one-epoch recipe. The low-budget K=5 result shows that quality is not '
            'monotonic in class cardinality.',
            'BERT at 20/class scores 56.00%, 21.25%, 17.67% and 16.25% accuracy, while its 100/class variants '
            'score 96.00%, 92.50%, 87.67% and 85.12%. This characterizes the fixed two-epoch recipe; it does '
            'not establish the best achievable performance of a separately tuned BERT model.',
        ]
    lines = ['# Banking77 v2 experiment — v1.2 extended report', '',
             f'**{completed}/56 quality conditions complete.** Fourteen method/budget variants across four label sets.',
             '', '## Design and data integrity', '', design, '', budgets, '', integrity,
             '', '## Accuracy by label budget', '', *accuracy_table(results), '',
             '![Accuracy by cardinality and training budget](accuracy.png)', '',
             '## Interpretation', '']
    for finding in findings:
        lines += [finding, '']
    lines += ['Qwen SFT and Emissary routing are different models/mechanisms. Equal training budgets improve '
              'comparability among supervised methods; comparisons against zero-shot services still differ '
              'in supervision. Quick Train is not part of this benchmark.', '',
              '## Full quality and calibration', '', availability, '',
              '| Method | K | Coverage | Accuracy [95% CI] | Macro-F1 | ECE | Adaptive ECE | Log loss | Brier |',
              '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in results:
        ci = row.get('uncertainty', {}).get('metrics', {}).get('accuracy')
        acc = f"{pct(value(row))} [{pct(ci['lower'])}, {pct(ci['upper'])}]" if ci else '—'
        metrics = [num(value(row, name)) for name in ('macro_f1', *CALIBRATION)]
        lines.append(f"| {row['title']} | {row['k']} | {row['n']}/{row['expected']} ({row['status']}) | {acc} | " + ' | '.join(metrics) + ' |')
    lines += ['', '## Paired effects at equal budgets', '',
              'Differences are method minus MiniLM+LR using the same training budget and test IDs. Positive '
              'values favor the named method. These exploratory comparisons do not adjust for multiplicity.', '',
              *paired_table(payload, 'matched_budget_vs_minilm'), '',
              '## Effect of increasing training labels', '',
              'Differences are 100/class minus 20/class within a method, on the same test IDs. Epoch counts '
              'are fixed, so increasing the budget also increases optimization steps; this is not a '
              'compute-matched experiment.', '', *paired_table(payload, 'within_method_100_minus_20'), '',
              '## Operational measurements and costs', '',
              'Inference uses batch size 1, concurrency 1, no warmup and no automatic retries. Model loading, '
              'training and deployment remain separate from prediction latency. `results.json` retains '
              'phase-specific p50/p95 latency and evidence locations with the training budget. Historical '
              'and new measurements are not pooled. Local methods use CPU with four numeric-library threads; '
              'provider hardware and hosted cache state are uncontrolled. No cross-provider speed or cost '
              'winner is established here. Unknown Emissary charges are unavailable, not zero. Original '
              'ledgers retain failed attempts and preparation costs.', '',
              '## Limitations and reproducibility', '', limitations, '',
              'All new measurements were executed by the user. Report generation only reads saved evidence '
              'and makes no provider calls. Original predictions remain unchanged. The publication version '
              '`v1.2` is distinct from the experiment directory name `v2`.', '',
              'Render from the committed summary: `PYTHONPATH=src venv/bin/python scripts/build_v2_reports.py`. '
              'Add `--recompute` to recalculate from the local evidence bundle, including '
              '`artifacts/v2_budget_extension`. [Extension instructions](../../docs/budget_extension.md). '
              'The committed summary supports clean-clone rendering without raw datasets or credentials. '
              'Paired OpenAI comparisons are retained in `results.json`; source/model revisions and '
              'cohort identity are in the manifest. Report generation does not create tags or releases.']
    (REPORT/'report.md').write_text('\n'.join(lines) + '\n')
    brief = ['# Banking77 — v1.2 brief', '', f'**{completed}/56 quality conditions complete.** '
             'The extension adds equal training budgets for four supervised methods, retaining three '
             'zero-shot references and all original results.', '', *accuracy_table(results), '']
    brief += [text for finding in findings for text in (finding, '')]
    brief += [integrity, '', availability, '', limitations, '',
              'Emissary pricing remains unknown. This brief makes no cross-provider cost or latency claim. '
              'See the [extended report](report.md) for full calibration results, paired uncertainty, '
              'measurement scope and reproduction.']
    (REPORT/'brief.md').write_text('\n'.join(brief) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = {method: f'C{i}' for i, method in enumerate(METHODS)}
    for ax, budget in zip(axes, (20, 50, 100)):
        for method in METHODS:
            actual_budget = 0 if BUDGETS[method] == (0,) else budget
            points = [r for r in results if (r['method'], r['fit_per_class']) == (method, actual_budget)
                      and r['status'] == 'completed']
            if points:
                ax.plot([r['k'] for r in points], [value(r) for r in points], marker='o',
                        color=colors[method], linestyle='--' if actual_budget == 0 else '-',
                        alpha=.65 if actual_budget == 0 else 1, label=METHODS[method])
        ax.set(xlabel='Number of classes', xticks=COUNTS, ylim=(0, 1.02), title=f'{budget} training examples/class')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Accuracy')
    fig.suptitle(f'Banking77 v1.2 — {completed}/56 conditions; dashed lines are zero-shot references')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, .13, 1, .96))
    fig.savefig(REPORT/'accuracy.png', dpi=180)
    plt.close(fig)
    print(f'Reports rebuilt offline: {completed}/56 complete; report.md, brief.md, results.csv/json and accuracy.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion-root', type=Path, default=Path('artifacts/v2_completion'))
    parser.add_argument('--extension-root', type=Path, default=Path('artifacts/v2_budget_extension'))
    parser.add_argument('--summary', type=Path, default=Path('reports/v2/results.json'))
    parser.add_argument('--output-dir', type=Path, default=REPORT)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    REPORT = args.output_dir
    render(collect(args.completion_root, args.extension_root) if args.recompute else read_json(args.summary))
