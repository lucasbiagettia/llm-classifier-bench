from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from llm_classifier_bench.preparation import (
    PreparationRecorder, preparation_stage, recalculate_preparation, amortize_preparation,
)
from llm_classifier_bench.costs import load_pricing, recalculate_costs
from llm_classifier_bench.classifiers import TfidfLogisticClassifier, SentenceTransformerLogisticClassifier, BertClassifier
from llm_classifier_bench.config import SentenceTransformerTrainingConfig, BertTrainingConfig
from llm_classifier_bench.measurement import MeasurementConfig
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark
from test_runner import FakeDataset, build_bundle

RATE = Path(__file__).resolve().parents[1]/'pricing/self_hosted_reference.json'


def fixture(tmp_path, backend='local'):
    def event(ident,name,category,start,end,parent=None,details=None):
        return dict(event_id=ident,name=name,category=category,start_ns=start*10**9,end_ns=end*10**9,
                    elapsed_ms=(end-start)*1000,parent_id=parent,status='success',details=details or {})
    # Four hours of preparation. Nested subintervals must not be added to parents.
    events = [event('p','prepare','parent',0,3600), event('f','fit','parent',3600,14400),
              event('features','features','features',3600,7200,'f'),
              event('a','fit.a','fit',7200,10800,'f'), event('b','fit.b','fit',10800,12600,'f'),
              event('s','select','selection',12600,14400,'f',
                    {'selected_configuration':{'c':1}, 'selected_fit_names':['fit.b'], 'final_refit_performed':False})]
    (tmp_path/'preparation.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    (tmp_path/'preparation_metadata.json').write_text(json.dumps({'runtime':{'classifier':{'backend':backend,'device':'cpu'}},'adapter':{}}))
    (tmp_path/'config.json').write_text(json.dumps({'measurement':{'hardware_profile':'fixture'}}))
    card=load_pricing(RATE)
    card['local_hardware'].update(usd_per_hour='2',interpretation='measured_hardware',hardware_profile='fixture')
    (tmp_path/'pricing.json').write_text(json.dumps(card))
    return events


def test_parent_intervals_and_selected_fit_are_not_double_counted(tmp_path):
    fixture(tmp_path)
    report = recalculate_preparation(tmp_path)
    assert report['complete'] and report['available']
    assert report['total_cost_usd'] == 8
    assert report['total_elapsed_ms'] == 4*3600000
    assert sum(report['exclusive_ms_by_category'].values()) == report['total_elapsed_ms']
    assert report['selected_configuration']['selected_fit_cost_estimate_usd'] == 1
    assert not report['selected_configuration']['final_refit_performed']


@pytest.mark.parametrize('corruption',['duplicate','overlap','missing_parent','wrong_duration','selection'])
def test_corrupted_stage_evidence_cannot_be_silently_priced(tmp_path,corruption):
    events=fixture(tmp_path)
    if corruption=='duplicate': events.append(events[-1])
    if corruption=='overlap': events[3]['start_ns']=events[2]['start_ns'];events[3]['elapsed_ms']=7200000
    if corruption=='missing_parent': events[2]['parent_id']='missing'
    if corruption=='wrong_duration': events[2]['elapsed_ms']=1
    if corruption=='selection': events[-1]['details']['selected_fit_names']=['nonexistent']
    (tmp_path/'preparation.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(ValueError): recalculate_preparation(tmp_path)


def test_observed_and_billable_evidence_replace_runtime_estimate(tmp_path):
    fixture(tmp_path)
    billing=tmp_path/'billing.json'
    billing.write_text(json.dumps({'kind':'observed_charge','scope':'entire_preparation','source':'fixture invoice line 1',
                                   'currency':'USD','amount':'12'}))
    result=recalculate_preparation(tmp_path,billing_path=billing)
    assert result['total_cost_usd']==12 and result['cost_kind']=='observed'
    assert result['selected_configuration']['selected_fit_cost_estimate_usd']==1  # diagnostic estimate, not invoice allocation
    billing.write_text(json.dumps({'kind':'billable_allocation','scope':'entire_preparation','source':'fixture resource meter',
                                   'device':'cpu','hardware_profile':'fixture','billable_hours':'5'}))
    assert recalculate_preparation(tmp_path,billing_path=billing)['total_cost_usd']==10
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    assert recalculate_preparation(tmp_path)['total_cost_usd']==8
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}


def test_remote_wait_is_not_billed_without_provider_evidence(tmp_path):
    fixture(tmp_path,backend='hosted_api')
    report=recalculate_preparation(tmp_path)
    assert report['total_elapsed_ms']>0 and report['total_cost_usd'] is None
    assert report['cost_kind']=='unavailable'
    meta=tmp_path/'preparation_metadata.json'; value=json.loads(meta.read_text())
    value['adapter']['billing']={'kind':'observed_charge','scope':'entire_preparation','source':'provider receipt','currency':'USD','amount':3}
    meta.write_text(json.dumps(value))
    assert recalculate_preparation(tmp_path)['total_cost_usd']==3


def test_amortization_uses_one_preparation_and_one_inference_warmup():
    prep={'complete':True,'total_cost_usd':10}
    inference={'self_hosted':{'available':True,'continuous':{'scenarios':[
        {'successful_examples':1000,'total_cost_usd':3}]},'deployed':{'scenarios':[
        {'successful_examples':1000,'allocation_cost_usd':8,'feasible':True},
        {'successful_examples':10000,'allocation_cost_usd':8,'feasible':False}]}},
        'backend':'hosted_api','run_status':'completed','successful_examples':100,
        'evaluation':{'coverage_complete':True,'total_cost_usd':2},
        'warmup':{'coverage_complete':True,'total_cost_usd':1}}
    continuous=amortize_preparation(prep,inference,scenario='continuous',volumes=(1000,))['rows'][0]
    assert continuous['amortized_cost_per_valid_prediction_usd']==.013
    deployed=amortize_preparation(prep,inference,scenario='deployed',volumes=(1000,10000))['rows']
    assert deployed[0]['amortized_cost_per_valid_prediction_usd']==.018
    assert deployed[1]['amortized_cost_per_valid_prediction_usd'] is None
    api=amortize_preparation(prep,inference,scenario='recorded_api',volumes=(1000,))['rows'][0]
    assert api['inference_scenario_cost_usd']==21  # 2*10 + warmup 1, not 3*10
    assert api['amortized_cost_per_valid_prediction_usd']==.031
    assert amortize_preparation({'complete':True,'total_cost_usd':None},inference,scenario='continuous',volumes=(1000,))['rows'][0]['amortized_cost_per_valid_prediction_usd'] is None
    assert amortize_preparation({'complete':False,'total_cost_usd':10},inference,scenario='continuous',volumes=(1000,))['rows'][0]['amortized_cost_per_valid_prediction_usd'] is None


def richer_bundle():
    bundle=build_bundle()
    train=bundle.train+tuple(replace(e,sample_id=e.sample_id+'-extra',text=e.text+' again') for e in bundle.train)
    return replace(bundle,train=train)


def run_local(tmp_path,classifier,name='run'):
    return run_benchmark(FakeDataset(richer_bundle()),classifier,BenchmarkRunConfig(
        output_root=tmp_path,run_id=name,pricing_path=RATE,validation_fraction=.5,
        measurement=MeasurementConfig(warmup_examples=1)))


def test_real_tfidf_selection_replay_and_amortization_cli(tmp_path):
    classifier=TfidfLogisticClassifier()
    result=run_local(tmp_path,classifier)
    prep=recalculate_preparation(result.run_dir)
    assert prep==json.loads((result.run_dir/'preparation_report.json').read_text())
    assert prep['complete'] and prep['total_cost_usd']>0
    assert prep['selected_configuration']['selected_configuration']['c']==classifier.selected_c
    assert len([e for e in prep['events'] if e['category']=='fit'])==3
    assert prep['selected_configuration']['selected_fit_cost_estimate_usd'] < prep['total_cost_usd']
    assert classifier._preparation_sink is None
    from llm_classifier_bench.runner import split_train_validation
    train, validation = split_train_validation(richer_bundle().train, validation_fraction=.5, seed=42)
    reference = TfidfLogisticClassifier()
    reference.prepare(richer_bundle().classes)
    reference.fit(train, validation_examples=validation)
    inputs = [e.as_input() for e in richer_bundle().test]
    assert reference.selected_c == classifier.selected_c
    assert [p.probabilities for p in reference.predict(inputs)] == [p.probabilities for p in classifier.predict(inputs)]
    before={p.name:p.read_bytes() for p in result.run_dir.iterdir()}
    script=Path(__file__).resolve().parents[1]/'scripts/report_preparation.py'
    out=tmp_path/'combined.json'
    subprocess.run([sys.executable,str(script),str(result.run_dir),'--inference-scenario','deployed','--deployment-hours','24',
                    '--output',str(tmp_path/'combined.md'),'--json-output',str(out)],check=True,capture_output=True)
    combined=json.loads(out.read_text())
    assert len(combined['amortization']['rows'])==3
    for row in combined['amortization']['rows']:
        assert row['amortized_cost_per_valid_prediction_usd']==pytest.approx((prep['total_cost_usd']+row['inference_scenario_cost_usd'])/row['successful_examples'])
    assert before=={p.name:p.read_bytes() for p in result.run_dir.iterdir()}


def test_minilm_reused_encoder_does_not_refit_selected_candidate(tmp_path):
    class Encoder:
        device='cpu'
        def encode(self,texts,**kwargs):
            return [[float('goal' in text),float(len(text))] for text in texts]
    classifier=SentenceTransformerLogisticClassifier(encoder=Encoder(),model='offline-fixture',
        training=SentenceTransformerTrainingConfig(c_values=(.1,1.,10.)))
    result=run_local(tmp_path,classifier)
    prep=recalculate_preparation(result.run_dir)
    assert prep['complete']
    load=next(e for e in prep['events'] if e['name']=='encoder.load')
    assert load['details']['cache']=='reused_in_memory'
    assert len([e for e in prep['events'] if e['category']=='fit'])==3
    assert prep['selected_configuration']['selected_configuration']['c']==classifier.selected_c
    assert prep['selected_configuration']['final_refit_performed'] is False


def test_failed_fit_preserves_spending_but_cannot_be_amortized(tmp_path):
    class Failure(TfidfLogisticClassifier):
        def fit(self,*args,**kwargs):
            with preparation_stage(self,'failed-fit','fit'):
                raise RuntimeError('fixture failure')
    with pytest.raises(RuntimeError): run_local(tmp_path,Failure())
    prep=recalculate_preparation(tmp_path/'run')
    assert not prep['complete'] and prep['total_cost_usd'] is not None
    assert next(e for e in prep['events'] if e['name']=='failed-fit')['status']=='failed'


def test_historical_preparation_is_unknown_not_zero(tmp_path):
    assert recalculate_preparation(tmp_path)['total_cost_usd'] is None


def test_bert_instrumentation_preserves_fit_and_selection_without_downloads(tmp_path,monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    class Tokenizer:
        @classmethod
        def from_pretrained(cls,*args,**kwargs): return cls()
        def __call__(self,texts,**kwargs):
            texts=[texts] if isinstance(texts,str) else texts
            return {'input_ids':torch.tensor([[float('goal' in text),1.] for text in texts])}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=torch.nn.Linear(2,2)
            self.config=SimpleNamespace(_commit_hash='fixture-revision')
        @classmethod
        def from_pretrained(cls,*args,**kwargs): return cls()
        def forward(self,input_ids,labels=None):
            logits=self.layer(input_ids)
            loss=torch.nn.functional.cross_entropy(logits,labels) if labels is not None else None
            return SimpleNamespace(logits=logits,loss=loss)
    monkeypatch.setattr('llm_classifier_bench.classifiers.bert._load_transformer_stack',lambda:(torch,Tokenizer,Model))
    classifier=BertClassifier(model='offline-fixture',training=BertTrainingConfig(epochs=2,batch_size=2))
    result=run_local(tmp_path,classifier)
    prep=recalculate_preparation(result.run_dir)
    assert prep['complete']
    assert len([e for e in prep['events'] if e['category']=='fit'])==2
    assert any(e['category']=='tokenization' for e in prep['events'])
    assert any(e['name']=='bert.restore' for e in prep['events'])
    assert prep['selected_configuration']['selected_configuration']['epoch']==classifier.fitted_metadata()['selected_epoch']
    assert prep['selected_configuration']['final_refit_performed'] is False
    assert sum(prep['exclusive_ms_by_category'].values())==pytest.approx(prep['total_elapsed_ms'])


def test_stage_synchronization_brackets_work_and_survives_failure(tmp_path):
    classifier=TfidfLogisticClassifier()
    recorder=PreparationRecorder(tmp_path,classifier)
    calls=[]
    with pytest.raises(RuntimeError):
        with preparation_stage(classifier,'gpu-stage','fit',synchronize=lambda:calls.append('sync')):
            calls.append('work')
            raise RuntimeError('fixture')
    assert calls==['sync','work','sync']
    assert recorder.events[0]['status']=='failed'
    recorder.close()


@pytest.mark.parametrize('billing',[
    {'kind':'observed_charge','source':'receipt','scope':'entire_preparation','currency':'EUR','amount':1},
    {'kind':'observed_charge','source':'receipt','scope':'entire_preparation','currency':'USD','amount':-1},
    {'kind':'observed_charge','source':'receipt','scope':'fit_only','currency':'USD','amount':1},
    {'kind':'billable_allocation','source':'meter','scope':'entire_preparation','device':'cpu','hardware_profile':'wrong','billable_hours':1},
])
def test_invalid_billing_evidence_is_rejected(tmp_path,billing):
    fixture(tmp_path)
    path=tmp_path/'billing.json';path.write_text(json.dumps(billing))
    with pytest.raises(ValueError): recalculate_preparation(tmp_path,billing_path=path)
