"""New Llama SFT campaign, paired with Qwen; default is offline audit only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import run_budget_extension as shared
from llm_classifier_bench.classifiers.emissary import EmissaryClient, EmissaryClassifier
from llm_classifier_bench.config import EmissaryTrainingConfig

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/emissary_llama_comparison'
MODEL = 'Llama-3.2-1B-Instruct'
# Not exposed by the Llama template observed on the first attempt.
# Keep the omission explicit and fixed across all eight conditions.
OMITTED_QWEN_PARAMETERS = ('max_grad_norm', 'warmup_ratio')
CORE_PARAMETERS = ('num_train_epochs', 'learning_rate', 'per_device_train_batch_size',
                   'per_device_eval_batch_size', 'dynamic_loss')
read_json, write_json, require, sha256 = shared.read_json, shared.write_json, shared.require, shared.sha256


def llama_parameters(qwen_parameters):
    return {k: v for k, v in qwen_parameters.items() if k not in OMITTED_QWEN_PARAMETERS}


def cells():
    # Validate the reported provider fix on a complete 100/class condition first.
    return [{'id': f'emissary-llama__s42__k{k}__b{budget}', 'k': k, 'budget': budget,
             'method': 'emissary-llama', 'test_examples': 40*k}
            for budget in (100, 20) for k in (5, 10, 15, 20)]


def file_hashes_valid(fingerprints):
    for name, digest in fingerprints.items():
        require(sha256(name) == digest, f'Frozen input changed: {name}')


def locate_run(evidence):
    return evidence if (evidence/'config.json').is_file() else evidence/'runs/new'


def prepare_plan(timeout_s):
    """Reconstruct all eight Qwen conditions from local evidence, with no API client."""
    fingerprints = {}

    def record(path):
        fingerprints[str(path)] = sha256(path)
        return read_json(path)

    manifest = record(Path('reports/v2/manifest.json'))
    results = record(Path('reports/v2/results.json'))
    source_dir = Path('artifacts/v2_source')
    source = shared.load_source(source_dir, manifest)
    for split in ('train', 'test'):
        path = source_dir/f'{split}.csv'
        fingerprints[str(path)] = sha256(path)
    definition_path = Path(manifest['definitions']['path'])
    require(sha256(definition_path) == manifest['definitions']['sha256'], 'Definitions changed')
    fingerprints[str(definition_path)] = sha256(definition_path)
    definitions = {c.name: c for c in shared.load_class_definition_profile(definition_path).profile.classes}
    references, audit = {}, []
    common_parameters = None
    for cell in cells():
        k, budget = cell['k'], cell['budget']
        candidates = [r for r in results['conditions'] if
                      (r['method'], r['fit_per_class'], r['k']) == ('emissary-qwen', budget, k)]
        require(len(candidates) == 1 and candidates[0]['status'] == 'completed',
                f'Missing completed Qwen reference: K={k}, budget={budget}')
        evidence = Path(candidates[0]['evidence'])
        run = locate_run(evidence)
        config = record(run/'config.json')
        fit = record(run/'fit_metadata.json')
        pool = record(run/'labeled_budget.json')
        require(record(run/'status.json')['status'] == 'completed', f'Incomplete reference: {run}')
        classes = tuple(definitions[name] for name in manifest['label_sets'][str(k)])
        bundle = shared.build_bundle(source, classes)
        selected, reserved, _ = shared.pools(bundle, budget)
        overlap = shared.audit_partitions(selected, reserved, source['test'])
        require(config['dataset']['classes'] == [asdict(c) for c in classes], 'Qwen class definitions differ')
        require(config['dataset']['fit_train_sample_ids'] == shared.ids(selected), 'Qwen fit IDs differ')
        require(config['dataset']['test_sample_ids'] == shared.ids(bundle.test), 'Qwen test IDs differ')
        require(not config['dataset']['validation_sample_ids'], 'Qwen used validation labels')
        require(fit['training_examples_used'] == budget*k and fit['validation_examples_used'] == 0,
                'Qwen training consumption differs')
        shared.verify_selection(pool, selected)
        predictions = evidence/'predictions.jsonl'
        shared.verify_predictions(predictions, bundle.test)
        fingerprints[str(predictions)] = sha256(predictions)
        payload = shared.provider_payload(selected, classes, budget)
        digest = hashlib.sha256(payload).hexdigest()
        require(digest == fit['dataset_payload_sha256'], 'Qwen upload does not match reconstructed data')
        upload_candidates = [evidence.parent/'remote/train.jsonl'] + [
            Path(root)/f'k{k}'/'train.jsonl' for root in manifest['raw_evidence_roots']]
        uploads = [p for p in upload_candidates if p.is_file() and sha256(p) == digest]
        require(bool(uploads), f'Original Qwen upload not found: {cell["id"]}')
        for path in uploads:
            fingerprints[str(path)] = sha256(path)
        remote = fit['training_response']
        require(remote['base_model'] == 'Qwen3-4B-Base' and not remote.get('base_fine_tuned_model')
                and not remote.get('test_dataset'), 'Unexpected Qwen training inputs')
        parameters = remote['hyper_parameters']
        require(parameters['num_train_epochs'] == 1, 'Expected the fixed one-epoch Qwen recipe')
        require(all(config['classifier']['training'][n] == parameters[n] for n in CORE_PARAMETERS),
                'Qwen requested/recorded hyperparameters differ')
        if common_parameters is None:
            common_parameters = parameters
        require(parameters == common_parameters, 'Qwen recipes differ between conditions')
        pricing_path = config.get('pricing_path')
        if pricing_path:
            record(Path(pricing_path))
        measurement = config['measurement']
        require(measurement['batch_size'] == 1 and measurement['warmup_examples'] == 0,
                'Qwen measurement policy changed')
        references[cell['id']] = {
            'classes': [asdict(c) for c in classes], 'fit_sample_ids': shared.ids(selected),
            'reserved_validation_sample_ids': shared.ids(reserved), 'test_sample_ids': shared.ids(bundle.test),
            'payload_sha256': digest, 'project_id': fit['project_id'], 'parameters': parameters,
            'measurement': measurement, 'pricing_path': pricing_path, 'qwen_evidence': str(evidence),
            'training_parameters': llama_parameters(parameters),
            'omitted_qwen_parameters': {k: parameters[k] for k in OMITTED_QWEN_PARAMETERS if k in parameters},
        }
        audit.append({'cell_id': cell['id'], 'status': 'passed', 'qwen_budget_per_class': budget,
                      **overlap, 'test_scope': 'entire official test', 'same_upload_as_qwen': True})
    for k in (5, 10, 15, 20):
        small = references[f'emissary-llama__s42__k{k}__b20']['fit_sample_ids']
        large = references[f'emissary-llama__s42__k{k}__b100']['fit_sample_ids']
        require(small == large[:20*k], 'Training budgets are not nested')
    code = [Path(__file__), Path(shared.__file__), *sorted((ROOT/'src').rglob('*.py')),
            ROOT/'scripts/run_banking77_scaling_benchmark_v2.py']
    return {'schema_version': 2, 'campaign': 'llama-qwen-matched-v2', 'base_model': MODEL,
            'seed': 42, 'cells': cells(), 'references': references,
            'source_dir': str(source_dir), 'source': manifest['source'],
            'input_sha256': fingerprints, 'code_sha256': {str(p): sha256(p) for p in code},
            'audit': {'status': 'passed', 'conditions': audit,
                      'scope': 'Recorded benchmark inputs; not base pretraining or provider internals'},
            'maximum_new_datasets': 8, 'maximum_new_training_jobs': 8,
            'maximum_new_deployments': 8, 'maximum_predictions': 4000,
            'first_condition': cells()[0]['id'], 'stop_on_first_failure': True,
            'retries': 0, 'warmup': 0, 'timeout_s': timeout_s,
            'environment': shared.ENVIRONMENT,
            'comparison_note': 'Same Qwen pools, upload bytes and shared hyperparameters. '
                'max_grad_norm and warmup_ratio are not requested because the Llama template does not expose them; '
                'their effective values are provider-controlled and not assumed equal to Qwen. '
                'Llama is an instruction-tuned base.'}


class LlamaClient(shared.RecordedTrainingClient):
    """New datasets/jobs only; fail before submission on incompatible model settings."""

    def retrieve_base_model(self, name):
        require(name == MODEL, 'Only the requested Llama base is allowed')
        response = super().retrieve_base_model(name)
        write_json(self.directory/'base_model_response.json', shared._redact_download_urls(response))
        template = response.get('parameter_template')
        require(isinstance(template, dict), 'Llama parameter template is missing')
        unsupported = sorted(set(self.parameters) - set(template))
        require(not unsupported, f'Llama does not advertise the fixed recipe parameters: {unsupported}')
        return response

    def upload_dataset(self, **kwargs):
        digest = hashlib.sha256(kwargs['content']).hexdigest()
        require(digest == self.expected_payload_sha256, 'Llama upload differs from paired Qwen data')
        with (self.directory/'train.jsonl').open('xb') as stream:
            stream.write(kwargs['content'])
        metadata = {k: v for k, v in kwargs.items() if k != 'content'}
        return self.once('dataset', {**metadata, 'sha256': digest},
                         lambda: EmissaryClient.upload_dataset(self, **kwargs))

    def create_training_job(self, **kwargs):
        require(kwargs['base_model'] == MODEL, 'Unexpected base model')
        require(all(self.parameters.get(k) == v for k, v in kwargs['parameters'].items()),
                'Llama training parameters differ from the paired recipe')
        kwargs['parameters'] = self.parameters
        return self.once('training', kwargs, lambda: EmissaryClient.create_training_job(self, **kwargs))

    def retrieve_training_job(self, **kwargs):
        response = super().retrieve_training_job(**kwargs)
        if response.get('status') == 'Success':
            require(not response.get('base_fine_tuned_model') and not response.get('test_dataset'),
                    'Unexpected continuation or test dataset')
            actual = response.get('hyper_parameters', {})
            require(all(actual.get(k) == v for k, v in self.parameters.items()),
                    'Provider changed requested hyperparameters; do not deploy or evaluate')
        return response


def classifier_for(cell, reference, output, *, live):
    parameters = reference.get('training_parameters', llama_parameters(reference['parameters']))
    training = EmissaryTrainingConfig(
        shots=cell['budget'], shot_unit='per_class', selection_seed=42,
        mechanism='project_fine_tuning', project_id=reference['project_id'], base_model=MODEL,
        **{k: parameters[k] for k in CORE_PARAMETERS},
        training_timeout_s=7200, deployment_timeout_s=1800, inactive_timeout_s=300)
    # No dataset/job/deployment continuation IDs: these are eight fresh base fits.
    client = LlamaClient(output/'remote', parameters, reference['payload_sha256'],
                        read_timeout_s=120) if live else None
    return EmissaryClassifier(training=training, client=client, classifier_name='emissary-llama-sft',
        experiment_name=f'llama-k{cell["k"]}-b{cell["budget"]}-{time.time_ns()}')


def worker(plan, cell):
    file_hashes_valid(plan['input_sha256'])
    file_hashes_valid(plan['code_sha256'])
    output = OUTPUT/'cells'/cell['id']
    output.mkdir(parents=True, exist_ok=False)
    reference = plan['references'][cell['id']]
    source = shared.load_source(Path(plan['source_dir']), plan)
    bundle = shared.build_bundle(source, [shared.ClassDefinition(**c) for c in reference['classes']])
    selected, reserved, _ = shared.pools(bundle, cell['budget'])
    shared.audit_partitions(selected, reserved, source['test'])
    require(shared.ids(selected) == reference['fit_sample_ids'], 'Fit IDs changed')
    require(shared.ids(bundle.test) == reference['test_sample_ids'], 'Test IDs changed')
    digest = hashlib.sha256(shared.provider_payload(selected, bundle.classes, cell['budget'])).hexdigest()
    require(digest == reference['payload_sha256'], 'Training payload changed')
    write_json(output/'condition.json', {'cell': cell, **reference})
    result = shared.run_benchmark(shared.StaticDataset(bundle), classifier_for(cell, reference, output, live=True),
        shared.BenchmarkRunConfig(output_root=output, run_id='run', validation_fraction=.2, split_seed=42,
            matched_budget=shared.MatchedBudgetConfig(cell['budget'], 'per_class', 42, 0),
            measurement=shared.MeasurementConfig(**reference['measurement']),
            pricing_path=Path(reference['pricing_path']) if reference['pricing_path'] else None,
            metadata={'llama_comparison': True, 'cell': cell, 'qwen_reference': reference['qwen_evidence'],
                      'source': plan['source'], 'manifest_sha256': sha256(OUTPUT/'manifest.json')}))
    require(read_json(result.status_path)['status'] == 'completed', 'Llama condition did not complete')
    shared.verify_predictions(result.predictions_path, bundle.test)
    fit = read_json(result.run_dir/'fit_metadata.json')
    require(fit['training_examples_used'] == cell['budget']*cell['k'] and fit['validation_examples_used'] == 0,
            'Actual label consumption differs')
    require(fit['dataset_payload_sha256'] == reference['payload_sha256'], 'Actual upload differs')


def select_action(cell, state, output):
    previous = state.get(cell['id'], {'status': 'pending'})
    if previous['status'] == 'completed':
        require(sha256(output/'run/predictions.jsonl') == previous['prediction_sha256'],
                'Completed evidence changed; refusing to repeat')
        return 'skip'
    if previous['status'] != 'pending' or output.exists():
        return 'stop'
    return 'run'


def execute(plan, *, pilot_only=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = OUTPUT/'manifest.json'
        if path.exists():
            require(read_json(path) == plan, 'Existing Llama campaign differs; do not restart it under a new folder')
        else:
            write_json(path, plan)
        write_json(OUTPUT/'audit.json', plan['audit'])
        state_path = OUTPUT/'summary.json'
        state = read_json(state_path) if state_path.exists() else {
            c['id']: {'status': 'pending', 'cell': c} for c in plan['cells']}
        write_json(state_path, state)
        selected = plan['cells'][:1] if pilot_only else plan['cells']
        env = {**os.environ, **plan['environment'], 'PYTHONPATH': str(ROOT/'src'), 'PYTHONUNBUFFERED': '1'}
        logs = OUTPUT/'logs'
        logs.mkdir(exist_ok=True)
        for cell in selected:
            name = cell['id']
            output = OUTPUT/'cells'/name
            action = select_action(cell, state, output)
            if action == 'skip':
                print(f'SKIP {name}: complete, unchanged', flush=True)
                continue
            if action == 'stop':
                print(f'STOP {name}: previous incomplete attempt retained; inspect its log, no automatic retry', flush=True)
                return 2
            state[name] = {'cell': cell, 'status': 'running', 'evidence': str(output/'run')}
            write_json(state_path, state)
            started = time.monotonic()
            print(f'START {name}: fresh Llama job, {cell["budget"]*cell["k"]} training rows', flush=True)

            def progress():
                status_path = output/'run/fit_metadata.json'
                try:
                    stage = read_json(status_path).get('status', 'loading') if status_path.exists() else 'loading'
                except json.JSONDecodeError:
                    stage = 'updating'
                pred = output/'run/predictions.jsonl'
                count = len(pred.read_text().splitlines()) if pred.exists() else 0
                print(f'PROGRESS {name}: {stage}; predictions={count}/{cell["test_examples"]}; '
                      f'elapsed={time.monotonic()-started:.0f}s', flush=True)

            try:
                with (logs/f'{name}.log').open('x') as log:
                    code, timeout = shared.run_process(
                        [sys.executable, '-u', str(Path(__file__).resolve()), '--execute', '--worker', name],
                        env=env, log=log, timeout_s=plan['timeout_s'], progress=progress)
                path = output/'run/status.json'
                status = read_json(path) if path.exists() else {}
                complete = code == 0 and not timeout and status.get('status') == 'completed'
                state[name].update(status='completed' if complete else 'failed', returncode=code, timeout=timeout)
                if complete:
                    state[name]['prediction_sha256'] = sha256(output/'run/predictions.jsonl')
            except BaseException as error:
                state[name].update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                   error=f'{type(error).__name__}: {error}')
                if not isinstance(error, Exception):
                    raise
            finally:
                state[name]['wall_seconds'] = time.monotonic()-started
                write_json(state_path, state)
            print(f'END {name}: {state[name]["status"]}; log={logs/f"{name}.log"}', flush=True)
            if state[name]['status'] != 'completed':
                print('STOP: no further jobs submitted. Keep the evidence for inspection.', flush=True)
                return 2
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Submit new Llama jobs, deploy and measure')
    parser.add_argument('--pilot-only', action='store_true', help='Execute only K=5, 100/class before the rest')
    parser.add_argument('--timeout-s', type=int, default=14400, help='Wall-time limit per condition')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    require(args.timeout_s > 0, 'Timeout must be positive')
    if args.worker:
        require(args.execute, 'Worker requires --execute')
        plan = read_json(OUTPUT/'manifest.json')
        return worker(plan, next(c for c in plan['cells'] if c['id'] == args.worker))
    plan = prepare_plan(args.timeout_s)
    print('AUDIT PASSED: all 8 Qwen references matched, train/test disjoint, upload bytes identical.')
    print('PARAMETERS: match Qwen except max_grad_norm/warmup_ratio (not exposed by Llama; provider-controlled).')
    print('PLAN: Llama-3.2-1B-Instruct; 20/100 per class; K=5/10/15/20; 8 new jobs; 4000 predictions.')
    if not args.execute:
        write_json(OUTPUT/'plan.json', plan)
        write_json(OUTPUT/'audit.json', plan['audit'])
        print(f'Offline only: {OUTPUT}/plan.json; no provider calls, training or inference.')
        return 0
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, shared.interrupted)
    return execute(plan, pilot_only=args.pilot_only)


if __name__ == '__main__':
    raise SystemExit(main())
