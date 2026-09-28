"""Publication rendering must work from a clean clone without private artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).parents[1]


def test_report_renders_in_empty_workdir_from_versioned_summary(tmp_path):
    source=ROOT/'reports/v2/results.json'
    output=tmp_path/'rendered'
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'}
    result=subprocess.run([sys.executable,str(ROOT/'scripts/build_v2_reports.py'),
        '--summary',str(source),'--output-dir',str(output)],cwd=tmp_path,env=env,
        capture_output=True,text=True,check=True)
    assert '56/56 complete' in result.stdout
    for name in ('brief.md','report.md','results.csv'):
        assert (output/name).read_text()==(ROOT/'reports/v2'/name).read_text()
    assert json.loads((output/'results.json').read_text())==json.loads(source.read_text())
    assert (output/'accuracy.png').stat().st_size>1000
    assert not (tmp_path/'artifacts').exists()


def test_published_summary_accounts_for_coverage_and_calibration():
    payload=json.loads((ROOT/'reports/v2/results.json').read_text())
    cells=payload['conditions']
    assert len(cells)==len({(r['method'],r['fit_per_class'],r['k']) for r in cells})==56
    for row in cells:
        assert row['status']=='completed' and row['n']==row['expected']==40*row['k']
        assert row['uncertainty']['metrics']['accuracy']['estimate']==row['metrics']['accuracy']['value']
        if row['probability_vectors']<row['n']:
            assert all(row['metrics'][key]['value'] is None for key in
                       ('top_label_ece','adaptive_ece','multiclass_log_loss','multiclass_brier_score'))
    jev={r['k']:r for r in cells if r['method']=='jev'}
    assert {k:r['n']-r['probability_vectors'] for k,r in jev.items()}=={5:0,10:1,15:2,20:3}


def test_publication_has_expected_budgets_and_completed_extension_audit():
    payload=json.loads((ROOT/'reports/v2/results.json').read_text())
    for method in ('bert','tfidf','sentence-transformer'):
        assert {r['fit_per_class'] for r in payload['conditions'] if r['method']==method}=={20,50,100}
    assert {r['fit_per_class'] for r in payload['conditions'] if r['method']=='emissary-qwen'}=={20,100}
    audit=payload['data_integrity']
    assert audit['historical']['historical_supervised_conditions_checked']==16
    assert len(audit['extension'])==28
    assert all(r['status']=='passed' and r['overlapping_ids']==r['overlapping_exact_texts']==r['overlapping_normalized_texts']==0 for r in audit['extension'])
    assert len([p for p in payload['paired_effects'] if p['comparison']=='matched_budget_vs_minilm'])==24
    assert len([p for p in payload['paired_effects'] if p['comparison']=='within_method_100_minus_20'])==16


def test_prediction_identity_rejects_duplicate_or_changed_text(monkeypatch):
    import importlib
    import pytest
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    report=importlib.import_module('build_v2_reports')
    row={'sample_id':'a','input':'hello','gold_label':'X'}
    expected={'a':('hello','X')}
    assert report.validate_prediction_identity([row],expected)
    assert not report.validate_prediction_identity([],expected)
    with pytest.raises(ValueError):report.validate_prediction_identity([row,row],expected)
    with pytest.raises(ValueError):report.validate_prediction_identity([{**row,'input':'changed'}],expected)


def test_render_rejects_missing_budget_variant(tmp_path,monkeypatch):
    import importlib
    import pytest
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    report=importlib.import_module('build_v2_reports')
    monkeypatch.setattr(report,'REPORT',tmp_path)
    payload=json.loads((ROOT/'reports/v2/results.json').read_text())
    payload['conditions'].pop()
    with pytest.raises(ValueError,match='56'):report.render(payload)
