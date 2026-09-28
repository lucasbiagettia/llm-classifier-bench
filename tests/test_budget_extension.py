"""Offline guards for the v2 supervised extension; no model loads or provider calls."""
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from llm_classifier_bench.class_definitions import ClassDefinitionProfile
from llm_classifier_bench.config import TfidfTrainingConfig
from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.measurement import MeasurementConfig


@pytest.fixture
def extension(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('run_budget_extension')
    def forbidden(*args, **kwargs):
        pytest.fail('Offline checks must not contact a provider')
    monkeypatch.setattr('requests.sessions.Session.request', forbidden)
    monkeypatch.setattr('socket.create_connection', forbidden)
    return module


@pytest.fixture
def history(extension, tmp_path, monkeypatch):
    m = extension
    monkeypatch.setattr(m, 'COUNTS', (2, 3))
    classes = tuple(ClassDefinition(f'class{i}', f'Frozen description {i}') for i in range(3))
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    import csv
    source = {'files': {}}
    for split, size in (('train', 130), ('test', 40)):
        path = source_dir / f'{split}.csv'
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['text', 'category'])
            writer.writerows((f'{split} unique {c.name} row {i}', c.name) for c in classes for i in range(size))
        source['files'][path.name] = {'sha256': m.sha256(path)}
    definitions = tmp_path / 'definitions.json'
    ClassDefinitionProfile(dataset='banking77', profile='fixture', classes=classes,
                           review_status='approved').write_json(definitions)
    upload_root = tmp_path / 'uploads'
    manifest = {'source': source, 'definitions': {'path': str(definitions), 'sha256': m.sha256(definitions)},
                'label_sets': {str(k): [c.name for c in classes[:k]] for k in m.COUNTS},
                'raw_evidence_roots': [str(upload_root)]}
    loaded = m.load_source(source_dir, manifest)
    results = {'conditions': []}
    for k in m.COUNTS:
        bundle = m.build_bundle(loaded, classes[:k])
        for method in m.METHODS:
            budget = 100 if method == 'emissary-qwen' else 50
            selected, _, selection = m.pools(bundle, budget)
            evidence = tmp_path / f'{method}-{k}'
            run = evidence / 'runs/new'
            config = {'dataset': {'classes': [asdict(c) for c in bundle.classes],
                                 'test_sample_ids': m.ids(bundle.test),
                                 'fit_train_sample_ids': m.ids(selected), 'validation_sample_ids': []},
                      'classifier': {'training': asdict(TfidfTrainingConfig()), 'model': 'fixture'},
                      'measurement': asdict(MeasurementConfig())}
            fit = {'training_examples_used': budget*k, 'validation_examples_used': 0}
            if method == 'emissary-qwen':
                payload = m.serialize_classification_dataset(selected, bundle.classes)
                path = upload_root / f'k{k}/train.jsonl'
                path.parent.mkdir(parents=True)
                path.write_bytes(payload)
                fit.update(selection=selection['selection'], project_id=f'project-{k}',
                           dataset_payload_sha256=m.sha256(path),
                           training_response={'base_model': 'Qwen3-4B-Base', 'test_dataset': '',
                                              'hyper_parameters': {'num_train_epochs': 1}})
            for name, value in [('config', config), ('fit_metadata', fit), ('labeled_budget', selection),
                                ('status', {'status': 'completed'})]:
                m.write_json(run / f'{name}.json', value)
            (evidence / 'predictions.jsonl').write_text(''.join(json.dumps(
                {'sample_id': e.sample_id, 'input': e.text, 'gold_label': e.label}) + '\n' for e in bundle.test))
            results['conditions'].append({'method': method, 'k': k, 'fit_per_class': budget,
                                          'status': 'completed', 'evidence': str(evidence)})
    args = SimpleNamespace(reference_manifest=tmp_path/'manifest.json', reference_results=tmp_path/'results.json',
                           source_dir=source_dir, local_timeout=1800, remote_timeout=14400,
                           output_root=tmp_path/'output', only=['tfidf'])
    m.write_json(args.reference_manifest, manifest)
    m.write_json(args.reference_results, results)
    return args


def test_offline_plan_checks_history_and_nested_pools(extension, history, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Audit must not instantiate a client or train')
    monkeypatch.setattr(extension, 'RecordedTrainingClient', forbidden)
    monkeypatch.setattr(extension, 'run_benchmark', forbidden)
    plan = extension.prepare_plan(history)
    assert plan['audit']['status'] == 'passed'
    assert plan['audit']['historical_supervised_conditions_checked'] == 8
    assert len(plan['cells']) == 14
    for k, condition in plan['conditions'].items():
        n = int(k)
        pools = condition['training_sample_ids']
        assert pools['20'] == pools['50'][:20*n] == pools['100'][:20*n]
        assert pools['50'] == pools['100'][:50*n]
        assert set(pools['100']).isdisjoint(condition['test_sample_ids'])
        assert len(condition['reserved_validation_sample_ids']) == 25*n
    assert sum(p.endswith('train.jsonl') for p in plan['input_sha256']) == 2


@pytest.mark.parametrize('mutation', ['source', 'fit', 'prediction', 'upload'])
def test_audit_rejects_changed_evidence(extension, history, mutation):
    m = extension
    result = m.read_json(history.reference_results)['conditions'][0]
    evidence = Path(result['evidence'])
    if mutation == 'source':
        with (history.source_dir / 'train.csv').open('a') as stream:
            stream.write('changed,class0\n')
    elif mutation == 'fit':
        path = evidence / 'runs/new/labeled_budget.json'
        data = m.read_json(path)
        data['selection']['selected_examples'][0]['text'] = 'changed'
        m.write_json(path, data)
    elif mutation == 'prediction':
        path = evidence / 'predictions.jsonl'
        path.write_text(path.read_text().replace('test unique', 'changed unique', 1))
    else:
        (history.source_dir.parent / 'uploads/k2/train.jsonl').write_text('changed')
    with pytest.raises(ValueError):
        m.prepare_plan(history)


@pytest.mark.parametrize('same_id,text', [(True, 'different'), (False, 'train text'), (False, 'ＴＲＡＩＮ   TEXT')])
def test_rejects_leakage_by_id_exact_or_normalized_text(extension, same_id, text):
    fit = (LabeledExample('fit', 'train text', 'A'),)
    test = (LabeledExample('fit' if same_id else 'test', text, 'A'),)
    with pytest.raises(ValueError, match='overlap|duplicate'):
        extension.audit_partitions(fit, (), test)


def test_upload_hash_matches_adapter_order_for_20(extension, history):
    m = extension
    plan = m.prepare_plan(history)
    source = m.load_source(history.source_dir, plan)
    bundle = m.build_bundle(source, [ClassDefinition(**c) for c in plan['conditions']['2']['classes']])
    selected, _, _ = m.pools(bundle, 20)
    from llm_classifier_bench.config import EmissaryTrainingConfig
    classifier = m.EmissaryClassifier(experiment_name='offline-fixture', training=EmissaryTrainingConfig(shots=20, shot_unit='per_class',
        mechanism='project_fine_tuning', project_id='fixture', base_model='Qwen3-4B-Base'))
    recorded = classifier.plan_fit(bundle.classes, selected)
    import hashlib
    assert hashlib.sha256(m.provider_payload(selected, bundle.classes, 20)).hexdigest() == recorded['selection']['provider_dataset']['sha256']
    assert {e['sample_id'] for e in recorded['selection']['selected_examples']} == set(m.ids(selected))


def test_fresh_qwen_does_not_continue_historical_training(extension, tmp_path, monkeypatch):
    monkeypatch.setattr(extension, 'RecordedTrainingClient', lambda *a, **kw: object())
    parameters = {'num_train_epochs': 1, 'learning_rate': .0002, 'per_device_train_batch_size': 2,
                  'per_device_eval_batch_size': 1, 'dynamic_loss': False}
    ref = {'config': {'training': {}}, 'project_id': 'existing-workspace', 'parameters': parameters}
    classifier = extension.classifier_for({'method': 'emissary-qwen', 'k': 5}, ref, tmp_path, 'digest')
    assert classifier.training.shots == 20
    assert classifier.training.num_train_epochs == 1
    assert classifier.training.base_model == 'Qwen3-4B-Base'
    assert classifier.training.dataset_id is None
    assert classifier.training.training_job_id is None
    assert classifier.training.deployment_id is None


def test_remote_intent_blocks_repeat_even_after_ambiguous_error(extension, tmp_path):
    client = extension.RecordedTrainingClient(tmp_path, {}, 'digest', api_key='fixture')
    calls = []
    def ambiguous():
        calls.append(1)
        raise TimeoutError('response lost')
    with pytest.raises(TimeoutError):
        client.once('training', {'name': 'fixture'}, ambiguous)
    with pytest.raises(FileExistsError):
        client.once('training', {'name': 'fixture'}, ambiguous)
    assert len(calls) == 1


def test_execution_skips_completed_and_failed_attempts(extension, history, monkeypatch):
    m = extension
    plan = m.prepare_plan(history)
    calls = []
    def fake_run(command, **kwargs):
        key = command[-1]
        calls.append(key)
        m.write_json(history.output_root / 'cells' / key / 'run/status.json', {'status': 'completed'})
        return (2 if len(calls) == 1 else 0), False
    monkeypatch.setattr(m, 'run_process', fake_run)
    assert m.execute(plan, history) == 2
    assert len(calls) == 4
    assert m.execute(plan, history) == 2
    assert len(calls) == 4
