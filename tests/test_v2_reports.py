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
    assert '64/64 complete' in result.stdout
    for name in ('brief.md','report.md','results.csv'):
        assert (output/name).read_text()==(ROOT/'reports/v2'/name).read_text()
    assert json.loads((output/'results.json').read_text())==json.loads(source.read_text())
    assert (output/'accuracy.png').stat().st_size>1000
    assert not (tmp_path/'artifacts').exists()


def test_published_summary_accounts_for_coverage_and_calibration():
    payload=json.loads((ROOT/'reports/v2/results.json').read_text())
    cells=payload['conditions']
    assert len(cells)==len({(r['method'],r['fit_per_class'],r['k']) for r in cells})==64
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
    assert len([p for p in payload['paired_effects'] if p['comparison']=='matched_budget_vs_minilm'])==32
    assert len([p for p in payload['paired_effects'] if p['comparison']=='within_method_100_minus_20'])==20
    assert len([p for p in payload['paired_effects'] if p['comparison']=='matched_budget_llama_vs_qwen'])==8
    assert len([p for p in payload['paired_effects'] if p['comparison']=='emissary_vs_jev'])==20
    llama = audit['llama_extension']
    assert len(llama)==8
    assert all(r['status']=='passed' and r['same_upload_as_qwen'] and r['reported_parameters_match_qwen'] for r in llama)
    assert all(r['overlapping_ids']==r['overlapping_exact_texts']==r['overlapping_normalized_texts']==0 for r in llama)
    assert all(r['omitted_qwen_parameters']=={'max_grad_norm': .3, 'warmup_ratio': .03} for r in llama)


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
    with pytest.raises(ValueError,match='64'):report.render(payload)


def test_previous_56_measurements_remain_unchanged():
    # Digest of the complete condition objects in the published v1.2 snapshot.
    import hashlib
    payload=json.loads((ROOT/'reports/v2/results.json').read_text())
    original=[r for r in payload['conditions'] if r['method']!='emissary-llama']
    assert len(original)==56
    digest=hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()
    assert digest=='b612896365f1ab5e7dae5f91b96485638b715ea75eab45c37a4d27790d7d60c4'


def test_llama_audit_rejects_changed_upload_or_parameters(tmp_path, monkeypatch):
    import importlib
    import pytest
    from dataclasses import asdict
    from llm_classifier_bench.core import ClassDefinition, LabeledExample
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    report=importlib.import_module('build_v2_reports')
    shared=importlib.import_module('run_budget_extension')
    source={split: tuple(LabeledExample(f'{split}-{c}-{i}', f'{split} class {c} row {i}', c)
                        for c in ('a','b') for i in range(n))
            for split,n in (('train',125),('test',40))}
    classes=[ClassDefinition(c,c) for c in ('a','b')]
    bundle=shared.build_bundle(source,classes)
    fit,reserved,pool=shared.pools(bundle,20)
    run=tmp_path/'run';remote=tmp_path/'remote'
    run.mkdir();remote.mkdir()
    upload=shared.provider_payload(fit,classes,20)
    (remote/'train.jsonl').write_bytes(upload)
    (run/'predictions.jsonl').write_text('fixture')
    digest=report.sha256(remote/'train.jsonl')
    params={'num_train_epochs':1}
    ref={'classes':[asdict(c) for c in classes], 'fit_sample_ids':shared.ids(fit),
         'reserved_validation_sample_ids':shared.ids(reserved), 'test_sample_ids':shared.ids(bundle.test),
         'payload_sha256':digest, 'training_parameters':params, 'parameters':{**params,'warmup_ratio':.03},
         'omitted_qwen_parameters':{'warmup_ratio':.03}, 'measurement':{}}
    cell={'id':'fixture','k':2,'budget':20}
    plan={'references':{'fixture':ref}}
    state={'prediction_sha256':report.sha256(run/'predictions.jsonl')}
    config={'dataset':{'classes':ref['classes'],'fit_train_sample_ids':ref['fit_sample_ids'],
            'test_sample_ids':ref['test_sample_ids'],'validation_sample_ids':[]},
            'measurement':{},'classifier':{'training':{k:None for k in ('dataset_id','training_job_id','deployment_id')}}}
    metadata={'training_examples_used':40,'validation_examples_used':0,'selection':{'selected_examples':
              [{'sample_id':e.sample_id} for e in fit]},'dataset_payload_sha256':digest,
              'dataset_id':'ds-fixture','training_job_id':'tr-fixture','deployment_id':'dp-fixture',
              'training_response':{'status':'Success','base_model':'Llama-3.2-1B-Instruct',
                                   'hyper_parameters':ref['parameters']}}
    for name,value in (('config',config),('fit_metadata',metadata),('labeled_budget',pool)):
        report.write_json(run/f'{name}.json',value)
    report.write_json(remote/'training_intent.json',{'parameters':params})
    assert report.audit_llama_run(run,cell,plan,source,state)['reported_parameters_match_qwen']
    (remote/'train.jsonl').write_text('tampered')
    with pytest.raises(ValueError,match='uploaded bytes'):
        report.audit_llama_run(run,cell,plan,source,state)
    (remote/'train.jsonl').write_bytes(upload)
    metadata['training_response']['hyper_parameters']={'num_train_epochs':3}
    report.write_json(run/'fit_metadata.json',metadata)
    with pytest.raises(ValueError,match='reported parameters'):
        report.audit_llama_run(run,cell,plan,source,state)
