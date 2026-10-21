"""Hard-task publication remains portable and keeps refusal/repair policies visible."""
from copy import deepcopy
import csv
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def report(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    return importlib.import_module('_hard_report')


def test_portable_render_from_versioned_summary_without_artifacts(tmp_path):
    summary = ROOT / 'reports/v2/hard_datasets_results.json'
    output = tmp_path / 'reports'
    subprocess.run([sys.executable, str(ROOT / 'scripts/build_hard_datasets_report.py'),
                    '--summary', str(summary), '--output-dir', str(output)], cwd=tmp_path,
                   env={**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'HF_HUB_OFFLINE': '1'},
                   check=True, capture_output=True, text=True)
    for name in ('hard_datasets.md', 'hard_datasets_results.json', 'hard_datasets_results.csv'):
        assert (output / name).read_bytes() == (ROOT / 'reports/v2' / name).read_bytes()
    assert not (tmp_path / 'artifacts').exists()


def test_published_full_cohorts_refusals_repairs_and_sft_audits():
    payload = json.loads((ROOT / 'reports/v2/hard_datasets_results.json').read_text())
    conditions = payload['conditions']
    assert len(conditions) == 40 and sum(r['n'] for r in conditions) == 7200
    decisions = next(r for r in conditions if (r['dataset'], r['method']) == ('implicit_hate', 'openai-decisions'))
    assert (decisions['classifications'], decisions['refusals'], decisions['n']) == (222, 18, 240)
    assert decisions['metrics']['accuracy']['value'] == pytest.approx(118 / 240)
    assert decisions['classification_coverage'] == .925
    assert all(decisions['metrics'][name]['value'] is None for name in
               ('top_label_ece', 'adaptive_ece', 'multiclass_log_loss', 'multiclass_brier_score'))
    jev = next(r for r in conditions if (r['dataset'], r['method']) == ('implicit_hate', 'jev'))
    assert jev['jev_sum_policies'] == {'unchanged': 237, 'renormalized_rounding': 3}
    assert jev['missing_probability_vectors'] == 0
    assert len(jev['operations']) == 2  # Failed attempt retained, measurements kept separate.
    assert sum(s['api_attempts'] for s in jev['operations']) == 241
    assert sum(s['api_attempts'] for s in decisions['operations']) == 240  # Refusal not retried.
    sft = [r for r in conditions if r['method'] in ('emissary-qwen', 'emissary-llama')]
    assert len(sft) == 12 and all(r['fit_audit']['status'] == 'passed' for r in sft)
    assert all(s['total_cost_usd'] is None for r in sft for s in r['operations'])
    assert set(payload['excluded_methods']) == {'perplexity-decisions'}
    with (ROOT / 'reports/v2/hard_datasets_results.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 40
    assert b'\r' not in (ROOT / 'reports/v2/hard_datasets_results.csv').read_bytes()


def test_supplement_is_identical_in_readme_brief_and_report(report):
    for path in ('README.md', 'reports/v2/brief.md', 'reports/v2/report.md'):
        content = (ROOT / path).read_text().split(report.START)[1].split(report.END)[0]
        payload = report.read(ROOT / 'reports/v2/hard_datasets_results.json')
        assert report.accuracy_table(payload) in content
        assert '18/240' in content and '92.50%' in content
        assert 'different label budgets' in content


@pytest.mark.parametrize('mutation', ['drop_cell', 'drop_refusal'])
def test_render_rejects_incomplete_or_filtered_cohorts(report, tmp_path, mutation):
    payload = deepcopy(report.read(ROOT / 'reports/v2/hard_datasets_results.json'))
    if mutation == 'drop_cell':
        payload['conditions'].pop()
    else:
        row = next(r for r in payload['conditions'] if r['refusals'])
        row['n'] -= row['refusals']
    with pytest.raises(ValueError, match='matrix|denominator'):
        report.render(payload, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_collection_rejects_mutated_plan_before_reading_any_predictions(report, tmp_path):
    (tmp_path / 'plan.json').write_text('{}')
    (tmp_path / 'plan.sha256').write_text('wrong')
    with pytest.raises(ValueError, match='checksum'):
        report.collect(tmp_path)


def test_llama_retry_matches_paired_training_uploads():
    payload = json.loads((ROOT / 'reports/v2/hard_datasets_results.json').read_text())
    llama = [r for r in payload['conditions'] if r['method'] == 'emissary-llama']
    assert len(llama) == 6
    for row in llama:
        qwen = next(r for r in payload['conditions'] if
                    (r['dataset'], r['method'], r['fit_per_class']) ==
                    (row['dataset'], 'emissary-qwen', row['fit_per_class']))
        assert row['fit_audit']['dataset_payload_sha256'] == qwen['fit_audit']['dataset_payload_sha256']
        assert row['classifications'] == row['n'] and row['refusals'] == 0
        assert row['missing_probability_vectors'] == 0
        assert row['classifier_configuration']['training']['base_model'] == 'Llama-3.2-1B-Instruct'
    assert payload['llama_failure']['status'] == 'Failed'


@pytest.mark.parametrize('mutation', ['source_plan', 'dataset', 'upload', 'duplicate'])
def test_retry_rejects_unpaired_evidence(report, monkeypatch, tmp_path, mutation):
    payload = report.read(ROOT / 'reports/v2/hard_datasets_results.json')
    retry = {**payload, 'conditions': [r for r in payload['conditions'] if r['method'] == 'emissary-llama']}
    original = deepcopy(payload)
    original['conditions'] = [r for r in original['conditions'] if r['method'] != 'emissary-llama']
    plan = {'source_plan_sha256': payload['plan_sha256'], 'retry_policy': 'fresh jobs'}
    if mutation == 'source_plan':
        plan['source_plan_sha256'] = 'wrong'
    elif mutation == 'dataset':
        retry['datasets'] = deepcopy(retry['datasets'])
        retry['datasets']['implicit_hate']['test_sample_ids'].reverse()
    elif mutation == 'upload':
        retry['conditions'][0]['fit_audit']['dataset_payload_sha256'] = 'wrong'
    else:
        original = payload
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(report, 'collect', lambda *args, **kwargs: retry)
    with pytest.raises(ValueError, match='differs|differ|already'):
        report.add_llama_retry(original, tmp_path)
