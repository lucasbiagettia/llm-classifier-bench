"""Offline collection and publication of the fixed-label hard-task supplement."""
from collections import Counter
from dataclasses import asdict, replace
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from llm_classifier_bench.classifiers.base import Prediction
from llm_classifier_bench.classifiers.emissary import serialize_classification_dataset
from llm_classifier_bench.costs import recalculate_costs
from llm_classifier_bench.metrics.evaluator import DEFAULT_METRICS, evaluate_records, evaluation_record_from_mapping, results_as_dict
from llm_classifier_bench.metrics.uncertainty import stratified_quality_bootstrap
from llm_classifier_bench.runner import _validate_predictions

METHODS = {
    'tfidf': 'TF-IDF + LR', 'sentence-transformer': 'MiniLM + LR', 'bert': 'BERT',
    'openai': 'OpenAI zero-shot', 'openai-context': 'OpenAI in-context (5 total)',
    'jev': 'Jev zero-shot', 'openai-decisions': 'GPT-6 Decisions zero-shot',
    'emissary': 'Emissary routing zero-shot', 'emissary-qwen': 'Emissary Qwen SFT',
}
SUPERVISED = {'tfidf', 'sentence-transformer', 'bert', 'emissary-qwen'}
DATASETS = {'implicit_hate': 'Implicit Hate (6 classes)', 'anli_r3': 'ANLI R3 (3 classes)'}
QUALITY = DEFAULT_METRICS[:6]
START, END = '<!-- HARD DATASETS START -->', '<!-- HARD DATASETS END -->'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def relative(path):
    """Keep repository-relative evidence locations, without machine-specific prefixes."""
    path = Path(path).resolve()
    root = Path(__file__).resolve().parents[1]
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def segments(run, seen=None):
    seen = set() if seen is None else seen
    run = Path(run).resolve()
    if run in seen:
        raise ValueError('Repeated recovery source')
    seen.add(run)
    if (run / 'recovery.json').exists():
        source = read(run / 'recovery.json')['source_run']
        result = segments(source, seen)
        pending = run / 'continuations/pending'
        return result + (segments(pending, seen) if pending.exists() else [])
    return [run]


def collect(root):
    """Recompute quality from exact full cohorts; audit frozen fit pools and SFT bytes."""
    import run_hard_datasets as campaign
    root = Path(root).resolve()
    plan = read(root / 'plan.json')
    if digest(root / 'plan.json') != (root / 'plan.sha256').read_text().strip():
        raise ValueError('Source plan checksum differs')
    for name, expected in plan['files_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    summaries = list(root.glob('executions/*/summary.json'))
    if len(summaries) != 1:
        raise ValueError('Expected exactly one execution summary')
    summary = read(summaries[0])
    expected = {(d, m, b) for d in DATASETS for m in METHODS
                for b in ((20, 50, 100) if m in SUPERVISED else (0,))}
    actual = [(r['dataset'], r['classifier_key'], r['fit_per_class']) for r in summary]
    if len(actual) != 34 or set(actual) != expected or any(r['status'] != 'completed' for r in summary):
        raise ValueError('Expected exactly 34 completed hard-task conditions')
    cfg = plan['configuration']
    bundles = {d: campaign.load_snapshot(root / info['snapshot']) for d, info in plan['datasets'].items()}
    datasets = {}
    for name, bundle in bundles.items():
        campaign.validate_partition_disjointness(bundle.train, bundle.validation, bundle.test)
        info = plan['datasets'][name]
        if [e.sample_id for e in bundle.test] != info['test_sample_ids']:
            raise ValueError('Frozen test identity differs')
        datasets[name] = {**info, 'classes': [asdict(c) for c in bundle.classes],
                          'metadata': dict(bundle.metadata), 'snapshot_sha256': digest(root / info['snapshot']),
                          'partition_audit': 'passed: selected fit/dev/test IDs and exact text disjoint'}
    conditions = []
    for row in summary:
        dataset, method, budget = row['dataset'], row['classifier_key'], row['fit_per_class']
        bundle = campaign.fit_budget_bundle(bundles[dataset], budget or 20)
        if method not in SUPERVISED:
            bundle = replace(bundle, train=campaign.context_pool(bundle, cfg) if method == 'openai-context' else (), validation=())
        run = Path(row['run_dir'])
        config = read(run / 'config.json')
        if read(run / 'status.json')['status'] != 'completed':
            raise ValueError('Incomplete source run')
        for key, expected_value in (
            ('classes', [asdict(c) for c in bundle.classes]),
            ('fit_train_sample_ids', [e.sample_id for e in bundle.train]),
            ('validation_sample_ids', [e.sample_id for e in bundle.validation]),
            ('test_sample_ids', [e.sample_id for e in bundle.test]),
        ):
            if config['dataset'][key] != expected_value:
                raise ValueError(f'Frozen condition inputs differ: {dataset}/{method}/{key}')
        predictions = rows(run / 'predictions.jsonl')
        if [(p['sample_id'], p['input'], p['gold_label']) for p in predictions] != [
                (e.sample_id, e.text, e.label) for e in bundle.test]:
            raise ValueError('Predictions differ from the complete shared test cohort')
        _validate_predictions(bundle, [Prediction(p['sample_id'], p['predicted_label'], p['confidence'],
                              p['probabilities'] or None, p['adapter_latency_ms'], raw_response=p['raw_response'])
                                      for p in predictions])
        records = tuple(evaluation_record_from_mapping(p) for p in predictions)
        metrics = results_as_dict(evaluate_records(records, metrics=QUALITY))
        saved = read(run / 'metrics.json')
        for name, metric in metrics.items():
            a, b = metric['value'], saved[name]['value']
            if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, abs_tol=1e-12)):
                raise ValueError(f'Recomputed metric differs: {dataset}/{method}/{name}')
        nulls = sum(p['predicted_label'] is None for p in predictions)
        fit_audit = None
        if method == 'emissary-qwen':
            fit = read(run / 'fit_metadata.json')
            selected = fit['selection']['selected_examples']
            by_id = {e.sample_id: e for e in bundle.train}
            if len(selected) != len(by_id) or {e['sample_id'] for e in selected} != set(by_id):
                raise ValueError('SFT fit selection differs')
            payload = serialize_classification_dataset([by_id[e['sample_id']] for e in selected], bundle.classes)
            if hashlib.sha256(payload).hexdigest() != fit['dataset_payload_sha256']:
                raise ValueError('SFT submitted payload checksum differs')
            if fit['training_response']['status'] != 'Success' or fit['training_response']['base_model'] != 'Qwen3-4B-Base':
                raise ValueError('Unexpected SFT training result')
            fit_audit = {'status': 'passed', 'dataset_payload_sha256': fit['dataset_payload_sha256'],
                         'training_job_id': fit['training_job_id'], 'deployment_id': fit['deployment_id'],
                         'reported_parameters': fit['training_response']['hyper_parameters']}
        evidence = {}
        source_runs = segments(run)
        operations = []
        for source in dict.fromkeys([run, *source_runs]):
            for path in sorted(source.glob('*')):
                if path.is_file():
                    evidence[relative(path)] = digest(path)
        for source in source_runs:
            cost = recalculate_costs(source)
            ledger = rows(source / 'usage.jsonl')
            attempts = [e for e in ledger if e['kind'] == 'api_attempt']
            operational = read(source / 'operational_report.json')['evaluation']
            operations.append({'run': relative(source), 'status': read(source / 'status.json')['status'],
                               'api_attempts': len(attempts), 'failed_api_attempts': sum(e['status'] != 'success' for e in attempts),
                               'known_cost_subtotal_usd': cost['known_cost_subtotal_usd'],
                               'total_cost_usd': cost['total_cost_usd'], 'cost_kind': cost['cost_kind'],
                               'latency': operational['successful_call_latency']})
        conditions.append({'dataset': dataset, 'method': method, 'fit_per_class': budget,
            'context_examples': len(bundle.train) if method == 'openai-context' else 0,
            'fit_examples': len(bundle.train) if method in SUPERVISED else 0,
            'validation_examples': len(bundle.validation), 'n': len(predictions),
            'classifications': len(predictions) - nulls, 'refusals': nulls,
            'classification_coverage': 1 - nulls / len(predictions),
            'status': 'completed_with_refusals' if nulls else 'completed',
            'missing_probability_vectors': sum(not p['probabilities'] for p in predictions),
            'jev_sum_policies': dict(Counter(p['raw_response'].get('probability_sum_policy', 'unchanged') for p in predictions)) if method == 'jev' else None,
            'metrics': metrics, 'uncertainty': stratified_quality_bootstrap(records),
            'operations': operations, 'fit_audit': fit_audit,
            'classifier_configuration': config['classifier'], 'measurement': config['measurement'],
            'evidence': evidence})
    return {'schema_version': 1, 'status': 'completed', 'campaign_root': relative(root),
            'summary_sha256': digest(summaries[0]), 'plan_sha256': digest(root / 'plan.json'),
            'seed': cfg['seed'], 'test_per_class': cfg['test_per_class'],
            'configuration': cfg, 'datasets': datasets, 'conditions': conditions,
            'excluded_methods': plan['excluded_methods'],
            'llama_failure': {'dataset': 'implicit_hate', 'fit_per_class': 20,
                              'training_job_id': 'tr-DfFCVXuyPg6bawDpqSKTFR',
                              'status': 'Failed', 'progress': 0, 'cause': 'not returned by provider API'},
            'policy': {'null_predictions': 'count as incorrect; full test denominator; no null class in macro-F1',
                       'jev': cfg['jev'], 'uncertainty': '2000 gold-stratified draws, seed 20260925; fixed model and campaign seed; pointwise intervals',
                       'costs': 'inference only; retain failed attempts and separate resumed segments; unknown costs unavailable'}}


def accuracy_table(payload):
    table = ['| Method | Fit/class | Implicit Hate (6 classes) | ANLI R3 (3 classes) |',
             '| --- | ---: | ---: | ---: |']
    lookup = {(r['dataset'], r['method'], r['fit_per_class']): r for r in payload['conditions']}
    for method, title in METHODS.items():
        for budget in ((20, 50, 100) if method in SUPERVISED else (0,)):
            values = [f"{100 * lookup[d, method, budget]['metrics']['accuracy']['value']:.2f}%" for d in DATASETS]
            table.append(f'| {title} | {budget} | ' + ' | '.join(values) + ' |')
    return '\n'.join(table)


def summary_text(payload):
    return ('**34/34 included conditions completed**, 17 per dataset; seed 42 and 40 held-out test examples/class '
            '(240 Implicit Hate, 120 ANLI R3). Supervised fit budgets are 20/50/100 per class; OpenAI in-context '
            'uses five demonstrations total and zero-shot methods use no fit labels. Validation labels consumed: zero. '
            'These fixed-label tasks test domain difficulty, not label-cardinality scaling.\n\n'
            + accuracy_table(payload) + '\n\n'
            'Implicit Hate: MiniLM+LR at 100/class leads the observed accuracy (57.92%); Qwen at 100/class reaches '
            '55.83%. ANLI R3: Jev zero-shot leads the observed accuracy (66.67%). These are exploratory point '
            'estimates on small cohorts; increasing fit labels does not consistently improve every method. '
            'Comparisons between supervised and zero-shot methods use different label budgets.\n\n'
            'GPT-6 Decisions refused 18/240 Implicit Hate inputs (92.50% classification coverage); its 49.17% '
            'accuracy and 0.5070 macro-F1 include those refusals as errors on all 240 inputs. ANLI Decisions has '
            '120/120 classification coverage. This full-denominator policy differs from the historical Banking77 '
            'accepted-only supplement above; those older results remain unchanged.\n\n'
            'Jev Implicit Hate has three probability vectors renormalized for sums within 0.05 of one; original '
            'values and the correction policy remain in raw evidence. All other Jev vectors are unchanged and '
            'none are missing. This differs from the historical strict sum policy. Emissary Llama is excluded '
            'after the Implicit Hate 20/class job failed; its remaining jobs were not attempted. Perplexity is '
            'excluded by user instruction. Neither exclusion is a zero score.\n\n'
            'Implicit Hate uses a synthetic stratified 60/20/20 split after grouping identical posts; ANLI '
            'uses official R3 train/dev/test boundaries, removing cross-split exact duplicates from lower-priority '
            'pools. Class descriptions are fixed assistant-authored rubrics, not independently human-reviewed. '
            'Per-example explanations stay outside classifier inputs. Local transformers use CPU, BERT two '
            'epochs, and Qwen SFT one epoch; BERT length 512 differs from the earlier Banking77 recipe. '
            'One seed, model-specific truncation and uncontrolled hosted hardware limit interpretation. '
            'No cross-provider cost or latency winner is claimed; Emissary prices remain unavailable.')


def replace_block(path, content):
    path = Path(path)
    text = path.read_text()
    block = START + '\n' + content + '\n' + END
    if START in text:
        before, after = text.split(START, 1)
        _, tail = after.split(END, 1)
        text = before + block + tail
    else:
        text = text.rstrip() + '\n\n' + block + '\n'
    path.write_text(text)


def render(payload, output, readme=None):
    expected = {(d, m, b) for d in DATASETS for m in METHODS
                for b in ((20, 50, 100) if m in SUPERVISED else (0,))}
    keys = [(r['dataset'], r['method'], r['fit_per_class']) for r in payload['conditions']]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Publication requires the complete 34-condition matrix')
    for row in payload['conditions']:
        expected_n = 40 * len(payload['datasets'][row['dataset']]['classes'])
        if (row['n'] != expected_n or row['classifications'] + row['refusals'] != expected_n
                or row['metrics']['accuracy']['metadata']['n_samples'] != expected_n
                or row['metrics']['accuracy']['metadata']['null_predictions'] != row['refusals']):
            raise ValueError('Publication must retain the full test denominator and refusals')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'hard_datasets_results.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    stream = io.StringIO()
    fields = ['dataset', 'method', 'fit_per_class', 'context_examples', 'n', 'classifications', 'refusals', 'classification_coverage',
              *[m.name for m in QUALITY]]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in payload['conditions']:
        writer.writerow({**{k: row[k] for k in fields if k in row}, **{k: v['value'] for k, v in row['metrics'].items()}})
    (output / 'hard_datasets_results.csv').write_text(stream.getvalue())
    detail = ['# Hard classification tasks — Implicit Hate and ANLI R3', '', summary_text(payload), '',
              '## Dataset protocol', '', '| Dataset | Source revision | Labels | Fit totals | Test | Split |',
              '| --- | --- | ---: | --- | ---: | --- |']
    for name, info in payload['datasets'].items():
        k = len(info['classes'])
        split = 'synthetic 60/20/20' if name == 'implicit_hate' else 'official R3'
        detail.append(f"| {DATASETS[name]} | `{info['metadata']['revision']}` | {k} | {20*k}/{50*k}/{100*k} | {40*k} | {split} |")
    detail += ['', 'Implicit Hate excludes primary `other` and retains six fine labels. ANLI labels are entailment, '
               'neutral and contradiction; all methods see `Premise: …` followed by `Hypothesis: …`. The frozen '
               'summary includes exact labels/rubrics, source counts, explanation coverage, test IDs, fit-pool IDs, '
               'partition audits and source hashes. Selection followed the user-specified tasks, replacing the '
               'issue’s proposed SELU-first candidate search; no SELU review is claimed. Difficulty is motivated '
               'by implicit meaning and overlapping hate categories, and adversarial premise–hypothesis reasoning, '
               'rather than selecting an observed accuracy range. The fixed sampling preceded the user’s API '
               'runs; no test-driven resampling or tuning was performed. Implicit Hate license/access information '
               'remains unresolved in the public mirror, ANLI documents CC BY-NC 4.0; no raw corpus is committed. '
               '[Source and partition documentation](../../docs/hard_datasets.md).', '',
               '## Full-cohort quality', '',
               'Accuracy intervals are pointwise 95% gold-stratified bootstrap intervals (2,000 draws, seed '
               '20260925), conditional on the fixed models, label set and campaign seed. They do not estimate '
               'training-seed variation. No multiplicity-adjusted significance claim is made. Missing probabilities '
               'leave calibration unavailable for the full cohort; they are never replaced by a fabricated vector.', '',
               '| Dataset | Method | Fit/class | Accuracy [95% CI] | Macro-F1 | ECE | Adaptive ECE | Log loss | Brier |',
               '| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in payload['conditions']:
        ci = r['uncertainty']['metrics']['accuracy']
        values = ['—' if r['metrics'][m]['value'] is None else f"{r['metrics'][m]['value']:.4f}" for m in
                  ('macro_f1', 'top_label_ece', 'adaptive_ece', 'multiclass_log_loss', 'multiclass_brier_score')]
        detail.append(f"| {DATASETS[r['dataset']]} | {METHODS[r['method']]} | {r['fit_per_class']} | "
                      f"{100*ci['estimate']:.2f}% [{100*ci['lower']:.2f}, {100*ci['upper']:.2f}] | " + ' | '.join(values) + ' |')
    detail += ['', '## Refusals and probability coverage', '',
               '| Dataset | Method | Classified / planned | Refusals | Missing probability vectors |',
               '| --- | --- | ---: | ---: | ---: |']
    for r in payload['conditions']:
        if r['method'] in ('openai', 'openai-context', 'openai-decisions', 'jev'):
            detail.append(f"| {DATASETS[r['dataset']]} | {METHODS[r['method']]} | {r['classifications']}/{r['n']} | {r['refusals']} | {r['missing_probability_vectors']} |")
    detail += ['', '## Operations and evidence', '',
               'There are 6,120 planned evaluation outcomes, including 18 refusals. Those outcomes are not '
               '6,120 independent input texts: every method uses the same test cohort within a dataset. '
               'Six Qwen trainings succeeded, one per dataset/budget. Local methods have fixed recipes and '
               'consume the same nested fit pools at equal budgets. SFT selection and serialized upload '
               'hashes were checked against saved metadata; successful training/deployment IDs and provider '
               'hyperparameters are retained in the JSON summary. This verifies client-side evidence, not '
               'provider internals or base-model pretraining contamination.', '',
               'Inference uses batch size one, no warmup and no automatic transport retries. Interrupted '
               'conditions were resumed at sample level, preserving earlier predictions. The failed Jev '
               'normalization input was requested again; the historical Decisions refusal was recovered from '
               'its usage ledger without another request. Cost/latency populations remain separated by '
               'segment. `operations` in the JSON retains original attempt counts, failed attempts, known '
               'inference subtotals and latency observations for each segment; preparation and training '
               'are excluded from inference prices. Unknown Emissary and local hardware prices remain '
               'unavailable. No total campaign bill or pooled resumed throughput is asserted.', '',
               'The Llama failure was `tr-DfFCVXuyPg6bawDpqSKTFR`, Implicit Hate 20/class, status Failed at '
               '0% progress, with no checkpoints. The provider detail API returned no cause and `/logs` '
               'returned 404; remaining Llama budgets were excluded, not retried or scored.', '',
               '## Reproduction', '',
               'All model/API runs were performed by the user. This report only reads saved artifacts and '
               'recomputes metrics, integrity checks and uncertainty offline. It does not train or call providers.', '',
               'Render from the committed summary (no credentials or raw datasets):', '', '```bash',
               'PYTHONPATH=src venv/bin/python scripts/build_hard_datasets_report.py --update-readme', '```', '',
               'Recompute from the locally retained raw campaign:', '', '```bash',
               'PYTHONPATH=src venv/bin/python scripts/build_hard_datasets_report.py \\',
               f"  --campaign-root {payload['campaign_root']} --update-readme", '```', '',
               'See [machine-readable results and SHA-256 evidence](hard_datasets_results.json), '
               '[flat quality table](hard_datasets_results.csv), and [execution instructions](../../docs/hard_datasets.md). '
               'The 64 Banking77 historical condition objects and four Decisions supplement conditions remain separate and unchanged.']
    (output / 'hard_datasets.md').write_text('\n'.join(detail) + '\n')
    block = '## Hard-dataset supplement\n\n' + summary_text(payload) + '\n\n' + (
        '[Full hard-task report](hard_datasets.md), [results JSON](hard_datasets_results.json) and '
        '[CSV](hard_datasets_results.csv).')
    for name in ('brief.md', 'report.md'):
        if (output / name).exists():
            replace_block(output / name, block)
    if readme:
        replace_block(readme, '## Hard-dataset results\n\n' + summary_text(payload) + '\n\n'
                      '[Full hard-task report](reports/v2/hard_datasets.md), '
                      '[brief](reports/v2/brief.md) and [results JSON](reports/v2/hard_datasets_results.json).')
