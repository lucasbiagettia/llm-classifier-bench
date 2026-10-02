"""Llama campaign checks: no network, real training, or changes to old campaigns."""
from dataclasses import asdict
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_classifier_bench.class_definitions import ClassDefinitionProfile
from llm_classifier_bench.config import EmissaryTrainingConfig
from llm_classifier_bench.core import ClassDefinition
from llm_classifier_bench.measurement import MeasurementConfig


@pytest.fixture
def llama(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module = importlib.import_module('run_llama_comparison')
    def forbidden(*args, **kwargs):
        pytest.fail('Tests must not contact any provider')
    monkeypatch.setattr('requests.sessions.Session.request', forbidden)
    monkeypatch.setattr('socket.create_connection', forbidden)
    return module


def test_matrix_and_fresh_base_configuration(llama, tmp_path):
    assert len(llama.cells()) == 8
    assert sum(c['test_examples'] for c in llama.cells()) == 4000
    assert (llama.cells()[0]['k'], llama.cells()[0]['budget']) == (5, 100)
    assert {(c['budget'], c['k']) for c in llama.cells()} == {(b, k) for b in (20, 100) for k in (5, 10, 15, 20)}
    reference = {'project_id': 'workspace', 'parameters': EmissaryTrainingConfig(num_train_epochs=1).training_parameters()}
    for cell in llama.cells():
        classifier = llama.classifier_for(cell, reference, tmp_path, live=False)
        assert classifier.client is None
        assert classifier.training.base_model == 'Llama-3.2-1B-Instruct'
        assert classifier.training.shots == cell['budget']
        assert classifier.training.num_train_epochs == 1
        assert classifier.training.dataset_id is None
        assert classifier.training.training_job_id is None
        assert classifier.training.deployment_id is None


@pytest.fixture
def history(llama, tmp_path, monkeypatch):
    shared = llama.shared
    monkeypatch.chdir(tmp_path)
    classes = tuple(ClassDefinition(f'class{i:02d}', f'Frozen {i}') for i in range(20))
    source_dir = tmp_path/'artifacts/v2_source'
    source_dir.mkdir(parents=True)
    import csv
    source = {'files': {}}
    for split, size in (('train', 130), ('test', 40)):
        path = source_dir/f'{split}.csv'
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['text', 'category'])
            writer.writerows((f'{split} unique {c.name} {i}', c.name) for c in classes for i in range(size))
        source['files'][path.name] = {'sha256': shared.sha256(path)}
    definitions = tmp_path/'definitions.json'
    ClassDefinitionProfile(dataset='banking77', profile='fixture', classes=classes,
                           review_status='approved').write_json(definitions)
    manifest = {'source': source, 'definitions': {'path': str(definitions), 'sha256': shared.sha256(definitions)},
                'label_sets': {str(k): [c.name for c in classes[:k]] for k in (5, 10, 15, 20)},
                'raw_evidence_roots': []}
    source = shared.load_source(source_dir, manifest)
    results = {'conditions': []}
    for cell in llama.cells():
        k, budget = cell['k'], cell['budget']
        bundle = shared.build_bundle(source, classes[:k])
        selected, _, selection = shared.pools(bundle, budget)
        evidence = tmp_path/'qwen'/cell['id']/'run'
        config_training = EmissaryTrainingConfig(shots=budget, shot_unit='per_class', num_train_epochs=1,
                                                mechanism='project_fine_tuning', project_id='workspace', base_model='Qwen3-4B-Base')
        parameters = {**config_training.training_parameters(), 'max_grad_norm': .3, 'warmup_ratio': .03, 'lora_rank': 16}
        config = {'dataset': {'classes': [asdict(c) for c in bundle.classes],
                             'test_sample_ids': shared.ids(bundle.test), 'fit_train_sample_ids': shared.ids(selected),
                             'validation_sample_ids': []},
                  'classifier': {'training': asdict(config_training)}, 'measurement': asdict(MeasurementConfig())}
        payload = shared.provider_payload(selected, bundle.classes, budget)
        upload = evidence.parent/'remote/train.jsonl'
        upload.parent.mkdir(parents=True)
        upload.write_bytes(payload)
        fit = {'training_examples_used': budget*k, 'validation_examples_used': 0, 'project_id': 'workspace',
               'dataset_payload_sha256': shared.sha256(upload),
               'training_response': {'base_model': 'Qwen3-4B-Base', 'hyper_parameters': parameters}}
        for name, value in [('config', config), ('fit_metadata', fit), ('labeled_budget', selection),
                            ('status', {'status': 'completed'})]:
            shared.write_json(evidence/f'{name}.json', value)
        (evidence/'predictions.jsonl').write_text(''.join(json.dumps(
            {'sample_id': e.sample_id, 'input': e.text, 'gold_label': e.label})+'\n' for e in bundle.test))
        results['conditions'].append({'method': 'emissary-qwen', 'fit_per_class': budget, 'k': k,
                                      'status': 'completed', 'evidence': str(evidence)})
    shared.write_json(tmp_path/'reports/v2/manifest.json', manifest)
    shared.write_json(tmp_path/'reports/v2/results.json', results)
    return tmp_path


def test_audit_pairs_every_qwen_upload_and_is_offline(llama, history, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Auditing must not instantiate a provider client')
    monkeypatch.setattr(llama, 'LlamaClient', forbidden)
    plan = llama.prepare_plan(14400)
    assert len(plan['references']) == len(plan['audit']['conditions']) == 8
    assert plan['audit']['status'] == 'passed'
    for reference in plan['references'].values():
        assert reference['omitted_qwen_parameters'] == {'max_grad_norm': .3, 'warmup_ratio': .03}
        assert reference['training_parameters']['lora_rank'] == 16
        assert set(reference['parameters']) - set(reference['training_parameters']) == set(llama.OMITTED_QWEN_PARAMETERS)
    for row in plan['audit']['conditions']:
        assert row['same_upload_as_qwen']
        assert row['overlapping_ids'] == row['overlapping_exact_texts'] == row['overlapping_normalized_texts'] == 0
    llama.file_hashes_valid(plan['input_sha256'])


def test_audit_rejects_changed_upload(llama, history):
    upload = next((history/'qwen').glob('*/remote/train.jsonl'))
    upload.write_text('changed')
    with pytest.raises(ValueError, match='upload not found'):
        llama.prepare_plan(14400)


def test_model_parameter_support_checked_before_upload(llama, tmp_path, monkeypatch):
    client = llama.LlamaClient(tmp_path, {'num_train_epochs': 1, 'custom': 10}, 'digest', api_key='fixture')
    monkeypatch.setattr(llama.EmissaryClient, 'retrieve_base_model',
                        lambda *a: {'parameter_template': {'num_train_epochs': 1}})
    with pytest.raises(ValueError, match='advertise'):
        client.retrieve_base_model(llama.MODEL)
    assert not list(tmp_path.glob('*intent*'))


def test_job_recipe_is_frozen_and_ambiguous_submission_cannot_repeat(llama, tmp_path, monkeypatch):
    params = {'num_train_epochs': 1}
    client = llama.LlamaClient(tmp_path, params, 'digest', api_key='fixture')
    calls = []
    def failed(self, **kwargs):
        calls.append(kwargs)
        raise TimeoutError('response lost')
    monkeypatch.setattr(llama.EmissaryClient, 'create_training_job', failed)
    kwargs = {'base_model': llama.MODEL, 'parameters': params}
    with pytest.raises(TimeoutError):client.create_training_job(**kwargs)
    with pytest.raises(FileExistsError):client.create_training_job(**kwargs)
    assert len(calls) == 1
    with pytest.raises(ValueError, match='base model'):
        client.create_training_job(base_model='Qwen3-4B-Base', parameters=params)


def test_provider_cannot_silently_change_parameters(llama, tmp_path, monkeypatch):
    client = llama.LlamaClient(tmp_path, {'num_train_epochs': 1}, 'digest', api_key='fixture')
    monkeypatch.setattr(llama.EmissaryClient, 'retrieve_training_job',
                        lambda *a, **kw: {'status': 'Success', 'hyper_parameters': {'num_train_epochs': 3}})
    with pytest.raises(ValueError, match='hyperparameters'):
        client.retrieve_training_job(project_id='fixture', training_job_id='fixture')


def test_execution_stops_on_first_failure_and_never_retries(llama, tmp_path, monkeypatch):
    monkeypatch.setattr(llama, 'OUTPUT', tmp_path)
    plan = {'cells': llama.cells(), 'audit': {}, 'environment': {}, 'timeout_s': 20}
    calls = []
    def failed(command, **kwargs):
        calls.append(command)
        return 2, False
    monkeypatch.setattr(llama.shared, 'run_process', failed)
    assert llama.execute(plan) == 2
    assert len(calls) == 1
    state = llama.read_json(tmp_path/'summary.json')
    assert sum(c['status'] == 'failed' for c in state.values()) == 1
    assert sum(c['status'] == 'pending' for c in state.values()) == 7
    assert llama.execute(plan) == 2
    assert len(calls) == 1


def test_pilot_success_is_skipped_when_continuing_rest(llama, tmp_path, monkeypatch):
    monkeypatch.setattr(llama, 'OUTPUT', tmp_path)
    plan = {'cells': llama.cells(), 'audit': {}, 'environment': {}, 'timeout_s': 20}
    calls = []
    def completed(command, **kwargs):
        name = command[-1]
        calls.append(name)
        output = tmp_path/'cells'/name/'run'
        llama.write_json(output/'status.json', {'status': 'completed'})
        (output/'predictions.jsonl').write_text('fixture\n')
        return 0, False
    monkeypatch.setattr(llama.shared, 'run_process', completed)
    assert llama.execute(plan, pilot_only=True) == 0
    assert len(calls) == 1
    assert llama.execute(plan) == 0
    assert len(calls) == 8
    assert llama.execute(plan) == 0
    assert len(calls) == 8


def test_worker_only_prepares_training_pool_and_matching_test(llama, history, monkeypatch):
    plan = llama.prepare_plan(14400)
    monkeypatch.setattr(llama, 'OUTPUT', history/'new-output')
    llama.write_json(llama.OUTPUT/'manifest.json', plan)
    monkeypatch.setattr(llama, 'classifier_for', lambda *a, **kw: object())
    calls = []
    def fake_run(dataset, classifier, config):
        bundle = dataset.load()
        selected, reserved, _ = llama.shared.pools(bundle, config.matched_budget.examples)
        assert set(llama.shared.ids(selected)).isdisjoint(llama.shared.ids(bundle.test))
        assert not config.matched_budget.validation_examples
        calls.append(config)
        run = config.output_root/config.run_id
        llama.write_json(run/'status.json', {'status': 'completed'})
        digest = llama.hashlib.sha256(llama.shared.provider_payload(selected, bundle.classes, config.matched_budget.examples)).hexdigest()
        llama.write_json(run/'fit_metadata.json', {'training_examples_used': len(selected),
            'validation_examples_used': 0, 'dataset_payload_sha256': digest})
        path = run/'predictions.jsonl'
        path.write_text(''.join(json.dumps({'sample_id': e.sample_id, 'input': e.text, 'gold_label': e.label})+'\n' for e in bundle.test))
        return SimpleNamespace(run_dir=run, status_path=run/'status.json', predictions_path=path)
    monkeypatch.setattr(llama.shared, 'run_benchmark', fake_run)
    for cell in (plan['cells'][0], plan['cells'][4]):
        llama.worker(plan, cell)
    assert {c.matched_budget.examples for c in calls} == {20, 100}


def test_llama_recipe_omits_only_documented_parameters_before_submission(llama, tmp_path, monkeypatch):
    qwen = {**EmissaryTrainingConfig(num_train_epochs=1).training_parameters(),
            'max_grad_norm': .3, 'warmup_ratio': .03, 'lora_rank': 16}
    expected = {k: v for k, v in qwen.items() if k not in ('max_grad_norm', 'warmup_ratio')}
    monkeypatch.setenv('EMISSARY_API_KEY', 'fixture')
    reference = {'project_id': 'fixture', 'parameters': qwen, 'payload_sha256': 'digest',
                 'training_parameters': llama.llama_parameters(qwen)}
    classifier = llama.classifier_for(llama.cells()[0], reference, tmp_path, live=True)
    client = classifier.client
    monkeypatch.setattr(llama.EmissaryClient, 'retrieve_base_model',
                        lambda *a: {'parameter_template': expected})
    client.retrieve_base_model(llama.MODEL)
    assert llama.read_json(tmp_path/'remote/base_model_response.json')['parameter_template'] == expected
    sent = []
    def submit(self, **kwargs):
        sent.append(kwargs)
        return {'id': 'fixture-job'}
    monkeypatch.setattr(llama.EmissaryClient, 'create_training_job', submit)
    client.create_training_job(base_model=llama.MODEL, parameters=classifier.training.training_parameters())
    assert sent[0]['parameters'] == expected
    monkeypatch.setattr(llama.EmissaryClient, 'retrieve_training_job',
                        lambda *a, **kw: {'status': 'Success', 'hyper_parameters': expected})
    assert client.retrieve_training_job(project_id='fixture', training_job_id='fixture')['status'] == 'Success'
