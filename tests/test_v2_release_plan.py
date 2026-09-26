import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('release_plan', ROOT/'scripts/run_v2_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_small_matrix_bounds_and_openai_zero_shot():
    cells = list(release.cells())
    manifest = json.loads((ROOT/'reports/v2/matrix_small.json').read_text())
    assert manifest['cells'] == cells
    assert len({c['id'] for c in cells}) == 32
    ready = [c for c in cells if c['availability']=='ready']
    assert len(ready) == 24
    assert {c['seed'] for c in ready} == {42}
    assert {c['class_count'] for c in ready} == {5,10,15,20}
    for method in release.METHODS:
        selected = [c for c in ready if c['method']==method]
        assert len(selected) == 4
        assert sum(c['test_examples'] for c in selected) == 1500
    for cell in cells:
        if cell['availability'] != 'ready':
            with pytest.raises(ValueError, match='Blocked'):
                release.command_for(cell, manifest, Path('/tmp/never-executed'))
            continue
        command = release.command_for(cell, manifest, Path('/tmp/never-executed'))
        if cell['method']=='openai':
            assert cell['budget']==0
            assert command[command.index('--matched-budgets')+1]=='0'
        if cell['method']=='bert':
            assert command[command.index('--bert-epochs')+1]=='2'
            assert command[command.index('--matched-budgets')+1]=='50'
    assert manifest['limits']['bert_total_wall_seconds']==1800
    assert manifest['limits']['new_emissary_training_jobs']==0


def test_timeout_stops_child_without_running_a_model(tmp_path):
    with (tmp_path/'sleeper.log').open('w') as log:
        code, timed_out = release.run_condition(
            [sys.executable, '-c', 'import time; time.sleep(60)'],
            env=os.environ.copy(), log=log, timeout_s=.2)
    assert timed_out
    assert code != 0


def test_interruption_stops_active_process(monkeypatch, tmp_path):
    class Process:
        pid = 99999
        def __init__(self): self.waited = 0
        def wait(self, timeout=None):
            self.waited += 1
            if self.waited==1: raise KeyboardInterrupt()
            return -2
        def poll(self): return None
    process=Process(); signals=[]
    monkeypatch.setattr(release.subprocess,'Popen',lambda *a,**kw: process)
    monkeypatch.setattr(release.os,'killpg',lambda pid,sig: signals.append((pid,sig)))
    with pytest.raises(KeyboardInterrupt):
        release.run_condition(['never-executed'],env={},log=None,timeout_s=1)
    assert signals==[(process.pid,release.signal.SIGINT)]
    assert process.waited==2


def test_default_is_plan_only_and_creates_no_run_directory(tmp_path):
    output=tmp_path/'no-execution'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_v2_release.py'),
                           '--only','openai','--root',str(output)],
                          cwd=ROOT,capture_output=True,text=True,check=True)
    assert '4 executable conditions, 1500 maximum evaluation calls' in result.stdout
    assert not output.exists()


def test_exhausted_bert_budget_cannot_launch_more_work(monkeypatch, tmp_path):
    import hashlib
    manifest = ROOT/'reports/v2/matrix_small.json'
    state = {'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
             'bert_accounted_wall_seconds':1800., 'cells':{}}
    (tmp_path/'execution.json').write_text(json.dumps(state))
    monkeypatch.setattr(sys,'argv',['run_v2_release.py','--root',str(tmp_path),'--execute','--only','bert'])
    monkeypatch.setattr(release.signal,'signal',lambda *args: None)
    def forbidden(*args,**kwargs):
        pytest.fail('Exhausted budget must never launch a condition')
    monkeypatch.setattr(release,'run_condition',forbidden)
    release.main()
    saved=json.loads((tmp_path/'execution.json').read_text())
    assert len(saved['cells'])==4
    assert all(c['status']=='not_run' for c in saved['cells'].values())
