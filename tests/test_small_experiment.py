"""Offline regressions: synthetic outputs only, never provider/model execution."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import _small_experiment as small
import run_small_condition as worker
from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.datasets.base import DatasetBundle


def row(key, label='a'):
    return {'sample_id':key,'input':f'text {key}','gold_label':label,
            'predicted_label':label,'model':'fixture','probabilities':None}


def prepare_merge(root, old, new, expected=('1','2'), status='completed'):
    small.write_json(root/'condition.json',{'test_sample_ids':list(expected),
        'reused_run':'retained-fixture','pending_examples':len(expected)-len(old)})
    (root/'reused_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in old))
    target=root/'runs/new';target.mkdir(parents=True)
    (target/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in new))
    small.write_json(target/'status.json',{'status':status})


def test_merge_preserves_order_and_separates_measurement_evidence(tmp_path):
    prepare_merge(tmp_path,[row('2','b')],[row('1')])
    result=small.consolidate(tmp_path)
    assert result['status']=='completed'
    assert result['valid_examples']==2 and result['reused_examples']==1
    assert [r['sample_id'] for r in map(json.loads,(tmp_path/'predictions.jsonl').read_text().splitlines())]==['1','2']
    assert [e['phase'] for e in result['evidence']]==['reused','new']
    assert 'accuracy' in result['metrics']
    assert not any('latency' in k or 'cost' in k for k in result['metrics'])


@pytest.mark.parametrize('new,status', [([], 'failed'), ([row('2','b')], 'failed')])
def test_incomplete_or_failed_finalization_never_gets_quality_score(tmp_path,new,status):
    prepare_merge(tmp_path,[row('1')],new,status=status)
    result=small.consolidate(tmp_path)
    assert result['status']=='failed'
    assert 'metrics' not in result
    assert not (tmp_path/'metrics.json').exists()


@pytest.mark.parametrize('new',[row('1'),row('unexpected')])
def test_merge_rejects_duplicates_and_unknown_ids(tmp_path,new):
    prepare_merge(tmp_path,[row('1')],[new])
    with pytest.raises(ValueError,match='Duplicate or unexpected'):
        small.consolidate(tmp_path)


def test_changed_reuse_file_fails_before_classifier_construction(tmp_path,monkeypatch):
    artifact=tmp_path/'predictions.jsonl';artifact.write_text('original')
    manifest={'reuse':{'fixture':{'run_dir':str(tmp_path),'files_sha256':{'predictions.jsonl':small.sha256(artifact)}}}}
    artifact.write_text('changed')
    monkeypatch.setattr(small,'validate_reuse',lambda *a:pytest.fail('Must verify hashes first'))
    with pytest.raises(ValueError,match='Reuse evidence changed'):
        small.reused_rows(manifest,{'id':'fixture'},None,None)


def test_worker_requests_only_missing_examples_and_pins_reused_emissary_model(tmp_path,monkeypatch):
    cell={'id':'fixture','method':'emissary','availability':'ready','seed':42,'class_count':2,'budget':0,'test_examples':2}
    manifest={'cells':[cell],'reuse':{'fixture':{'run_dir':'retained-fixture'}}}
    manifest_path=tmp_path/'matrix.json';small.write_json(manifest_path,manifest)
    bundle=DatasetBundle('fixture',(ClassDefinition('a','A'),ClassDefinition('b','B')),(),
        (LabeledExample('1','text 1','a'),LabeledExample('2','text 2','b')))
    condition=SimpleNamespace(bundle=bundle,definitions_path=None)
    args=SimpleNamespace(validation_fraction=0,pricing=None)
    monkeypatch.setattr(worker,'condition_for',lambda *a:(args,condition))
    monkeypatch.setattr(worker,'reused_rows',lambda *a:([row('1')],'pinned/version'))
    captured={}
    def classifier(*a,**kw):
        captured.update(kw)
        return object()
    monkeypatch.setattr(worker,'classifier_for',classifier)
    monkeypatch.setattr(worker,'measurement_from_args',lambda *a:None)
    def benchmark(dataset,classifier,config):
        assert [e.sample_id for e in dataset.load().test]==['2']
        assert config.matched_budget.examples==0
        root=config.output_root/config.run_id;root.mkdir(parents=True)
        (root/'predictions.jsonl').write_text(json.dumps(row('2','b'))+'\n')
        small.write_json(root/'status.json',{'status':'completed'})
    monkeypatch.setattr(worker,'run_benchmark',benchmark)
    output=tmp_path/'result'
    monkeypatch.setattr(sys,'argv',['worker','--manifest',str(manifest_path),'--cell','fixture','--output',str(output)])
    worker.main()
    assert captured=={'emissary_model':'pinned/version','request_cap':1}
    assert small.read_json(output/'result.json')['valid_examples']==2
