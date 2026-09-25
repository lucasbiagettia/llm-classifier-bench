"""Regression cases from the September audit; no downloads or remote calls."""
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_classifier_bench.classifiers.bert import _mean_validation_loss
from llm_classifier_bench.core import LabeledExample
from llm_classifier_bench.costs import CostRecorder, capture_api_usage
from llm_classifier_bench.datasets.huggingface import HFDatasetSpec, HuggingFaceClassificationDataset
from llm_classifier_bench.measurement import MeasurementConfig
from llm_classifier_bench.metrics.base import EvaluationRecord
from llm_classifier_bench.metrics.calibration import AdaptiveECEMetric, adaptive_ece_score
from llm_classifier_bench.metrics.evaluator import evaluate_jsonl, evaluate_records
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark
from test_runner import FakeClassifier, FakeDataset, build_bundle


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("error_type", [TimeoutError, KeyboardInterrupt])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_valid_batches_survive_later_failure(tmp_path, error_type, batch_size):
    bundle = build_bundle()
    bundle = replace(bundle, test=bundle.test + tuple(
        replace(e, sample_id=e.sample_id + '-extra') for e in bundle.test))
    run_dir = tmp_path / 'partial'

    class FailsSecondBatch(FakeClassifier):
        calls = 0

        def predict(self, examples):
            self.calls += 1
            if self.calls == 2:
                # Results must already be on disk before entering this call.
                assert len(read_rows(run_dir / 'predictions.jsonl')) == batch_size
                raise error_type('simulated failure')
            return super().predict(examples)

    with pytest.raises(error_type):
        run_benchmark(FakeDataset(bundle), FailsSecondBatch(), BenchmarkRunConfig(
            output_root=tmp_path, run_id='partial', measurement=MeasurementConfig(batch_size=batch_size)))
    predictions = read_rows(run_dir / 'predictions.jsonl')
    assert [p['sample_id'] for p in predictions] == [e.sample_id for e in bundle.test[:batch_size]]
    assert all(p['raw_response'] == {'fixture': True} for p in predictions)
    assert all(p['cost_usd'] is None and not p['cost_reconciled'] for p in predictions)
    status = json.loads((run_dir / 'status.json').read_text())
    assert status['status'] == 'failed'
    assert status['error_type'] == error_type.__name__
    assert status['persisted_predictions'] == batch_size
    assert not (run_dir / 'metrics.json').exists()
    report = json.loads((run_dir / 'operational_report.json').read_text())
    assert report['evaluation']['successful_examples'] == batch_size
    assert report['evaluation']['failed_calls'] == 1
    assert read_rows(run_dir / 'timings.jsonl')[-1]['status'] == 'exception'
    costs = json.loads((run_dir / 'cost_report.json').read_text())
    assert costs['run_status'] == 'failed'
    assert costs['successful_examples'] == batch_size
    with pytest.raises(ValueError, match='incomplete benchmark'):
        evaluate_jsonl(run_dir / 'predictions.jsonl')


@pytest.mark.parametrize('failure_stage', ['cost_report', 'final_replace'])
def test_post_inference_failure_keeps_incremental_predictions(tmp_path, monkeypatch, failure_stage):
    def fail(*args, **kwargs):
        raise OSError('simulated report or finalization failure')
    if failure_stage == 'cost_report':
        monkeypatch.setattr(CostRecorder, 'finish', fail)
    else:
        monkeypatch.setattr(Path, 'replace', fail)
    with pytest.raises(OSError):
        run_benchmark(FakeDataset(build_bundle()), FakeClassifier(), BenchmarkRunConfig(
            output_root=tmp_path, run_id='preserved'))
    run_dir = tmp_path / 'preserved'
    assert len(read_rows(run_dir / 'predictions.jsonl')) == 2
    assert all(not p['cost_reconciled'] for p in read_rows(run_dir / 'predictions.jsonl'))
    assert json.loads((run_dir / 'status.json').read_text())['status'] == 'failed'


def test_success_reconciles_predictions_and_duplicate_replay_is_rejected(tmp_path):
    result = run_benchmark(FakeDataset(build_bundle()), FakeClassifier(), BenchmarkRunConfig(
        output_root=tmp_path, run_id='complete'))
    rows = read_rows(result.predictions_path)
    assert len(rows) == 2
    assert all(p['cost_reconciled'] for p in rows)
    assert not result.predictions_path.with_suffix('.jsonl.tmp').exists()
    with result.predictions_path.open('a') as stream:
        stream.write(json.dumps(rows[0]) + '\n')
    with pytest.raises(ValueError, match='Duplicate evaluation sample IDs'):
        evaluate_jsonl(result.predictions_path)


def test_interrupt_marks_api_attempt_failed_and_preserves_observed_usage():
    attempts = []
    with pytest.raises(KeyboardInterrupt):
        with capture_api_usage(attempts.append, provider='openai', sample_id='1',
                               requested_model='fixture', transport_attempts_complete=True) as attempt:
            attempt['usage'] = {'prompt_tokens': 12}
            raise KeyboardInterrupt()
    assert len(attempts) == 1
    assert attempts[0]['status'] == 'failed'
    assert attempts[0]['error_type'] == 'KeyboardInterrupt'
    assert attempts[0]['usage'] == {'prompt_tokens': 12}


@pytest.mark.parametrize('batch_size', [1, 8, 16, 17])
def test_bert_validation_is_mean_over_examples(batch_size):
    # A deterministic model exposes known per-example losses without downloads.
    class Tensor:
        def __init__(self, values): self.values = values
        def to(self, device): return self
    class Loss:
        def __init__(self, value): self.value = value
        def detach(self): return self
        def cpu(self): return self
        def item(self): return self.value
    class Model:
        def eval(self): pass
        def train(self): pass
        def __call__(self, values, labels):
            return SimpleNamespace(loss=Loss(sum(values.values) / len(values.values)))
    torch = SimpleNamespace(no_grad=nullcontext, long='long', tensor=lambda values, **kw: Tensor(values))
    tokenizer = lambda texts, **kw: {'values': Tensor([float(t) for t in texts])}
    examples = tuple(LabeledExample(str(i), str(v), 'A') for i, v in enumerate([.1] * 16 + [1.0]))
    loss = _mean_validation_loss(torch=torch, model=Model(), tokenizer=tokenizer, examples=examples,
        label_to_id={'A': 0}, batch_size=batch_size, max_length=128, device='cpu')
    assert loss == pytest.approx(2.6 / 17)


def test_dataset_target_metadata_does_not_reach_classifier_inputs():
    rows = [{'text': 'alpha', 'label': 'A'}, {'text': 'beta', 'label': 'B'}]
    dataset = HuggingFaceClassificationDataset(HFDatasetSpec(name='fixture', path='fixture'),
                                              loader=lambda **kw: rows)
    example = dataset.load().test[0]
    example.metadata['nested_target'] = {'gold': 'A'}
    inp = example.as_input()
    assert inp.metadata == {'dataset': 'fixture', 'source': 'fixture', 'split': 'test', 'row_index': 0}
    assert example.metadata['raw_label'] == example.label
    inp.metadata['source'] = 'changed'
    assert example.metadata['source'] == 'fixture'


def confidence_record(i, confidence, correct):
    return EvaluationRecord(str(i), 'A', 'A' if correct else 'B', confidence, None, 1)


def test_adaptive_ece_is_invariant_to_tied_row_order():
    correct = [confidence_record(i, .5, True) for i in range(10)]
    wrong = [confidence_record(i + 10, .5, False) for i in range(10)]
    grouped = correct + wrong
    interleaved = [r for pair in zip(correct, wrong) for r in pair]
    assert adaptive_ece_score(grouped) == adaptive_ece_score(interleaved)
    value, bins = adaptive_ece_score(grouped)
    assert value == 0
    assert len(bins) == 1 and bins[0]['count'] == 20
    assert AdaptiveECEMetric().compute(grouped).metadata['binning'] == 'equal_frequency_preserve_ties_v2'


def test_adaptive_ece_preserves_ties_across_multiple_quantiles():
    records = [confidence_record(i, c, True) for i, c in enumerate([.1] * 7 + [.5] * 2 + [.9])]
    value, bins = adaptive_ece_score(records, n_bins=5)
    assert sum(b['count'] for b in bins) == len(records)
    assert len(bins) <= 5
    assert all(a['max_confidence'] < b['min_confidence'] for a, b in zip(bins, bins[1:]))
    assert value == pytest.approx(.74)
    assert adaptive_ece_score(records, n_bins=5) == adaptive_ece_score(list(reversed(records)), n_bins=5)


def test_duplicate_records_rejected_without_ledger(tmp_path):
    record = confidence_record('same', .5, True)
    with pytest.raises(ValueError, match='Duplicate evaluation sample IDs'):
        evaluate_records([record, record])
    path = tmp_path / 'historical.jsonl'
    row = {'sample_id': 'same', 'gold_label': 'A', 'predicted_label': 'A'}
    path.write_text((json.dumps(row) + '\n') * 2)
    with pytest.raises(ValueError, match='Duplicate evaluation sample IDs'):
        evaluate_jsonl(path)


def test_independent_calibration_audit_uses_same_tie_policy(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from audit_calibration import reference_metrics
    probabilities = [[.5, .5]] * 20
    reference = reference_metrics(probabilities, ['A'] * 10 + ['B'] * 10, ['A', 'B'])
    assert reference['adaptive_ece'] == 0


@pytest.mark.parametrize('script', ['run_banking77_scaling_benchmark', 'run_banking77_scaling_benchmark_v2'])
@pytest.mark.parametrize('matched', [False, True])
def test_training_cap_covers_whole_campaign_before_loading_data(script, matched, monkeypatch):
    import importlib
    import sys
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from _matched_budgets import validate_budget_arguments
    campaign = importlib.import_module(script)
    args = [script, '--classifiers', 'emissary', '--class-counts', '5', '10', '20', '25',
            '--seeds', '1', '2', '3', '--emissary-mechanism', 'project_fine_tuning',
            '--emissary-project-id', 'fixture', '--emissary-base-model', 'fixture',
            '--emissary-allow-unpriced-training']
    args += (['--matched-budgets', '0', '100', '--budget-unit', 'total'] if matched else
             ['--emissary-shots', '0', '100', '--emissary-shot-unit', 'total'])
    monkeypatch.setattr(campaign, 'get_dataset', lambda *_: pytest.fail('Must stop before loading data or calling APIs'))
    for maximum in (1, 11):
        monkeypatch.setattr(sys, 'argv', args + ['--emissary-max-training-jobs', str(maximum)])
        with pytest.raises(ValueError, match='at least 12'):
            campaign.main()
    # The exact bound is sufficient; validate without launching any work.
    monkeypatch.setattr(sys, 'argv', args + ['--emissary-max-training-jobs', '12'])
    validate_budget_arguments(campaign.parse_args())
    monkeypatch.setattr(sys, 'argv', args + ['--dry-run'])
    validate_budget_arguments(campaign.parse_args())


@pytest.mark.parametrize('matched', [False, True])
def test_training_continuation_requires_one_cell_and_no_new_job_budget(matched, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    import run_banking77_scaling_benchmark_v2 as campaign
    from _matched_budgets import validate_budget_arguments
    args = ['audit', '--classifiers', 'emissary', '--class-counts', '5', '--seeds', '1',
            '--emissary-mechanism', 'project_fine_tuning', '--emissary-project-id', 'fixture',
            '--emissary-base-model', 'fixture', '--emissary-allow-unpriced-training',
            '--emissary-dataset-id', 'dataset', '--emissary-dataset-sha256', 'a' * 64,
            '--emissary-training-job-id', 'training', '--emissary-max-training-jobs', '0']
    args += (['--matched-budgets', '100', '--budget-unit', 'total'] if matched else
             ['--emissary-shots', '100', '--emissary-shot-unit', 'total'])
    monkeypatch.setattr(sys, 'argv', args)
    validate_budget_arguments(campaign.parse_args())
    monkeypatch.setattr(sys, 'argv', args + ['--seeds', '1', '2'])
    with pytest.raises(ValueError, match='exactly one seed/class-count'):
        validate_budget_arguments(campaign.parse_args())


@pytest.mark.parametrize('delta,accepted', [(5e-6, True), (9e-5, True), (-9e-5, True),
                                           (1e-4, True), (1.00005e-4, False),
                                           (1.1e-4, False), (-1.1e-4, False)])
def test_probability_rounding_contract_agrees_across_all_layers(delta, accepted):
    from llm_classifier_bench.classifiers.emissary import parse_classification_response
    from llm_classifier_bench.classifiers.base import Prediction
    from llm_classifier_bench.runner import _validate_predictions
    bundle = build_bundle()
    example = bundle.test[0]
    probs = {'Sports': .6 + delta, 'World': .4}
    raw = {'data': [{'probs': probs}]}
    prediction = Prediction(example.sample_id, 'Sports', probs['Sports'], probs, 1)
    record = EvaluationRecord(example.sample_id, example.label, 'Sports', probs['Sports'], probs, 1)
    operations = [
        lambda: parse_classification_response(raw, sample_id=example.sample_id, latency_ms=1),
        lambda: _validate_predictions(bundle, [prediction], examples=bundle.test[:1]),
        lambda: evaluate_records([record]),
    ]
    for operation in operations:
        if accepted:
            operation()
        else:
            with pytest.raises(ValueError, match='sum'):
                operation()


def test_rounded_emissary_probabilities_survive_runner_and_saved_replay(tmp_path):
    from llm_classifier_bench.classifiers.emissary import parse_classification_response
    probs = {'Sports': .500005, 'World': .5}

    class RoundedClassifier(FakeClassifier):
        def predict(self, examples):
            return [parse_classification_response({'data': [{'probs': probs}]},
                    sample_id=e.sample_id, latency_ms=1) for e in examples]

    result = run_benchmark(FakeDataset(build_bundle()), RoundedClassifier(), BenchmarkRunConfig(
        output_root=tmp_path, run_id='rounded'))
    assert json.loads(result.status_path.read_text())['status'] == 'completed'
    assert all(p['probabilities'] == probs for p in read_rows(result.predictions_path))
    metrics = {m.name: m.as_dict() for m in evaluate_jsonl(result.predictions_path)}
    assert metrics == json.loads(result.metrics_path.read_text())
    assert metrics['multiclass_log_loss']['available']
    assert metrics['top_label_ece']['available']
