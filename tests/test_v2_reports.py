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
    assert '28/28 complete' in result.stdout
    for name in ('brief.md','report.md','results.csv'):
        assert (output/name).read_text()==(ROOT/'reports/v2'/name).read_text()
    assert json.loads((output/'results.json').read_text())==json.loads(source.read_text())
    assert (output/'accuracy.png').stat().st_size>1000
    assert not (tmp_path/'artifacts').exists()


def test_published_summary_accounts_for_coverage_and_calibration():
    payload=json.loads((ROOT/'reports/v2/results.json').read_text())
    cells=payload['conditions']
    assert len(cells)==len({(r['method'],r['k']) for r in cells})==28
    for row in cells:
        assert row['status']=='completed' and row['n']==row['expected']==40*row['k']
        assert row['uncertainty']['metrics']['accuracy']['estimate']==row['metrics']['accuracy']['value']
        if row['probability_vectors']<row['n']:
            assert all(row['metrics'][key]['value'] is None for key in
                       ('top_label_ece','adaptive_ece','multiclass_log_loss','multiclass_brier_score'))
    jev={r['k']:r for r in cells if r['method']=='jev'}
    assert {k:r['n']-r['probability_vectors'] for k,r in jev.items()}=={5:0,10:1,15:2,20:3}
