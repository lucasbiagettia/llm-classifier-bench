"""Independent billing arithmetic plus adapter/runner failure and replay checks."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_classifier_bench.classifiers import OpenAIClassifier, TfidfLogisticClassifier
from llm_classifier_bench.costs import build_cost_report, load_pricing, price_entry, recalculate_costs
from llm_classifier_bench.measurement import MeasurementConfig
from llm_classifier_bench.metrics.evaluator import evaluate_jsonl, results_as_dict
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark
from test_runner import FakeDataset, build_bundle

CARD_PATH = Path(__file__).resolve().parents[1] / 'pricing/inference_2026-09-16.json'
METADATA = {'run_status': 'completed', 'runtime': {'classifier': {'device': 'cpu'}}}


def attempt(**updates):
    entry = {
        'event_id': 'a', 'kind': 'api_attempt', 'phase': 'evaluation', 'sample_id': 's',
        'observation_index': 2, 'provider': 'openai', 'model': 'gpt-5-nano-2025-08-07',
        'service_tier': 'default', 'transport_attempts_complete': True,
        'usage': {'prompt_tokens': 1000, 'completion_tokens': 100, 'total_tokens': 1100,
                  'prompt_tokens_details': {'cached_tokens': 400, 'cache_write_tokens': None},
                  'completion_tokens_details': {'reasoning_tokens': 20}},
    }
    entry.update(updates)
    return entry


def call(**updates):
    entry = {'event_id': 'call', 'kind': 'inference_call', 'phase': 'evaluation',
             'observation_index': 2, 'sample_ids': ['s'], 'status': 'success'}
    entry.update(updates)
    return entry


def report(entries, card=None):
    return build_cost_report(entries, card or load_pricing(CARD_PATH), METADATA)


def test_cached_input_is_subtracted_and_reasoning_is_not_added_twice():
    priced = price_entry(attempt(), load_pricing(CARD_PATH), None)
    # 600*.05 + 400*.005 + 100*.40, divided by one million.
    assert priced['cost_usd'] == pytest.approx(.000072)
    assert priced['kind'] == 'estimated'
    assert priced['billing_units']['output_tokens_including_reasoning'] == 100


@pytest.mark.parametrize('update', [
    {'model': 'unpriced-model'}, {'service_tier': 'flex'}, {'usage': None},
    {'usage': {'prompt_tokens': 2, 'completion_tokens': 1}},
    {'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'prompt_tokens_details': {'cached_tokens': 3}}},
    {'usage': {'prompt_tokens': True, 'completion_tokens': 1, 'prompt_tokens_details': {'cached_tokens': 0}}},
    {'usage': {'prompt_tokens': -1, 'completion_tokens': 1, 'prompt_tokens_details': {'cached_tokens': 0}}},
    {'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'prompt_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 1}}},
    {'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 1}}},
    {'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'prompt_tokens_details': {'cached_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 2}}},
    {'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 9, 'prompt_tokens_details': {'cached_tokens': 0}}},
    {'observed_charge': {'amount': 1, 'currency': 'EUR', 'source': 'invoice'}},
])
def test_unknown_or_invalid_usage_never_becomes_free(update):
    priced = price_entry(attempt(**update), load_pricing(CARD_PATH), None)
    assert priced['cost_usd'] is None
    assert priced['kind'] == 'unavailable'
    assert priced['reason']


def test_billed_retry_failure_and_warmup_enter_total_once():
    entries = [attempt(status='failed'), attempt(event_id='retry', status='success'), call(),
               attempt(event_id='warm', phase='warmup', sample_id='train', observation_index=1),
               call(event_id='warmcall', phase='warmup', sample_ids=['train'], observation_index=1),
               attempt(event_id='fail', sample_id='bad', observation_index=3, status='failed'),
               call(event_id='failedcall', sample_ids=['bad'], observation_index=3, status='exception')]
    result = report(entries)
    assert result['total_cost_usd'] == pytest.approx(.000288)
    assert result['cost_per_1000_successful_examples_usd'] == pytest.approx(.288)
    assert result['failure_rate'] == .5
    assert result['successful_examples'] == 1
    assert result['per_successful_sample']['s']['cost_usd'] == pytest.approx(.000144)


@pytest.mark.parametrize('missing', [attempt(usage=None), attempt(transport_attempts_complete=False)])
def test_unknown_charge_or_hidden_retry_makes_total_unavailable(missing):
    result = report([attempt(event_id='known'), missing, call()])
    assert result['total_cost_usd'] is None
    assert result['known_cost_subtotal_usd'] >= .000072
    assert result['cost_per_1000_successful_examples_usd'] is None


def test_observed_charge_replaces_estimate_with_provenance():
    result = report([attempt(observed_charge={'amount': '.02', 'currency': 'USD', 'source': 'test invoice line 1'}), call()])
    assert result['cost_kind'] == 'observed'
    assert result['total_cost_usd'] == .02
    assert result['estimated_cost_subtotal_usd'] == 0


def test_local_batch_cost_uses_runtime_once_and_requires_matching_hardware():
    card = load_pricing(CARD_PATH)
    card['local_hardware']['usd_per_hour'] = '1'
    local = call(event_id='local', kind='local_call', elapsed_ms=3600, sample_ids=['s', 't'])
    result = report([local, call(sample_ids=['s', 't'])], card)
    assert result['total_cost_usd'] == .001
    assert result['per_successful_sample']['s']['cost_usd'] == .0005
    assert result['cost_per_1000_successful_examples_usd'] == .5
    assert price_entry(local, card, 'cuda:0')['cost_usd'] is None
    assert price_entry(local, load_pricing(None), 'cpu')['cost_usd'] is None


def test_missing_call_usage_or_interrupted_run_cannot_report_zero():
    assert report([call()])['total_cost_usd'] is None
    assert report([attempt(observation_index=99), call()])['total_cost_usd'] is None
    result = build_cost_report([], load_pricing(CARD_PATH), {'run_status': 'running'})
    assert result['total_cost_usd'] is None
    assert result['cost_kind'] == 'unavailable'
    with pytest.raises(ValueError, match='Duplicate'):
        report([attempt(), attempt(), call()])


@pytest.mark.parametrize('bad', ['NaN', '-1', 'Infinity', True])
def test_invalid_prices_rejected_before_inference(tmp_path, bad):
    card = load_pricing(CARD_PATH)
    card['api_rates'][0]['input_usd_per_million'] = bad
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(card))
    with pytest.raises(ValueError):
        load_pricing(path)


class KnownUsageClient:
    max_retries = 0

    def __init__(self, labels):
        self.labels = iter(labels)
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        label = next(self.labels)
        self.calls += 1
        if isinstance(label, Exception):
            raise label
        payload = {'id': f'response-{self.calls}', 'model': 'gpt-5-nano-2025-08-07',
                   'service_tier': 'default', 'usage': attempt()['usage']}
        return SimpleNamespace(**payload, model_dump=lambda: payload,
                               choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'label': label})))])


def run_api(tmp_path, labels, **config):
    client = KnownUsageClient(labels)
    classifier = OpenAIClassifier(client=client)
    result = run_benchmark(FakeDataset(build_bundle()), classifier, BenchmarkRunConfig(
        output_root=tmp_path, run_id='api', pricing_path=CARD_PATH, **config))
    return result, client


def test_runner_persists_usage_and_replays_metrics_including_warmup(tmp_path):
    result, client = run_api(tmp_path, ['World', 'Sports', 'World'],
                             measurement=MeasurementConfig(warmup_examples=1))
    saved = json.loads((result.run_dir / 'cost_report.json').read_text())
    assert saved == recalculate_costs(result.run_dir)
    assert saved['total_cost_usd'] == pytest.approx(.000216)
    assert saved['cost_per_1000_successful_examples_usd'] == pytest.approx(.108)
    assert saved['successful_examples'] == 2
    predictions = [json.loads(line) for line in result.predictions_path.read_text().splitlines()]
    assert sum(p['cost_usd'] for p in predictions) == pytest.approx(.000144)
    assert all(p['usage_event_ids'] and p['cost_kind'] == 'estimated' for p in predictions)
    metrics = results_as_dict(evaluate_jsonl(result.predictions_path))
    assert metrics == json.loads(result.metrics_path.read_text())
    assert metrics['total_cost_usd']['value'] == saved['total_cost_usd']
    assert metrics['cost_per_1000_usd']['value'] == saved['cost_per_1000_successful_examples_usd']
    # Reprice all token rates, without changing any saved file or making another call.
    before = {p.name: p.read_bytes() for p in result.run_dir.iterdir() if p.is_file()}
    card = load_pricing(CARD_PATH)
    for key in ('input_usd_per_million', 'cached_input_usd_per_million', 'output_usd_per_million'):
        card['api_rates'][0][key] = str(float(card['api_rates'][0][key]) * 2)
    alternative = tmp_path / 'alternative.json'
    alternative.write_text(json.dumps(card))
    repriced = recalculate_costs(result.run_dir, alternative)
    assert repriced['total_cost_usd'] == pytest.approx(saved['total_cost_usd'] * 2)
    assert repriced['usage_sha256'] == saved['usage_sha256']
    assert repriced['pricing_sha256'] != saved['pricing_sha256']
    assert client.calls == 3
    assert before == {p.name: p.read_bytes() for p in result.run_dir.iterdir() if p.is_file()}
    # A copied subset must not inherit the complete run's cost denominator.
    result.predictions_path.write_text(json.dumps(predictions[0]) + '\n')
    with pytest.raises(ValueError, match='sample IDs'):
        evaluate_jsonl(result.predictions_path)


def test_invalid_prediction_is_billed_and_saved_when_run_fails(tmp_path):
    with pytest.raises(ValueError, match='not in the class set'):
        run_api(tmp_path, ['Sports', 'invalid'])
    result = recalculate_costs(tmp_path / 'api')
    assert result['run_status'] == 'failed'
    assert result['total_cost_usd'] == pytest.approx(.000144)
    assert result['successful_examples'] == 1
    assert result['failure_rate'] == .5
    assert result['cost_per_1000_successful_examples_usd'] == pytest.approx(.144)
    rows = [json.loads(line) for line in (tmp_path / 'api/usage.jsonl').read_text().splitlines()]
    assert [r['status'] for r in rows if r['kind'] == 'api_attempt'] == ['success', 'failed']


def test_timeout_keeps_known_subtotal_and_partial_batch_counts_no_success(tmp_path):
    with pytest.raises(TimeoutError):
        run_api(tmp_path, ['Sports', TimeoutError()], measurement=MeasurementConfig(batch_size=2))
    result = recalculate_costs(tmp_path / 'api')
    assert result['total_cost_usd'] is None
    assert result['known_cost_subtotal_usd'] == pytest.approx(.000072)
    assert result['successful_examples'] == 0
    assert result['failure_rate'] == 1


def test_real_local_inference_uses_measured_device_and_batch_runtime(tmp_path):
    # More fit examples are unnecessary: validation disabled, one fit per class.
    result = run_benchmark(FakeDataset(build_bundle()), TfidfLogisticClassifier(), BenchmarkRunConfig(
        output_root=tmp_path, run_id='local', validation_fraction=0,
        measurement=MeasurementConfig(batch_size=2, warmup_examples=1), pricing_path=CARD_PATH))
    costs = recalculate_costs(result.run_dir)
    timings = [json.loads(line) for line in (result.run_dir / 'timings.jsonl').read_text().splitlines()]
    inference_ms = sum(r['elapsed_ms'] for r in timings if r['kind'] == 'inference')
    expected = inference_ms / 3_600_000 * (164 / 730)
    assert costs['total_cost_usd'] == pytest.approx(expected)
    assert costs['cost_kind'] == 'estimated'
    assert len(costs['entries']) == 2  # One warmup call, one evaluation batch.


def test_billed_api_error_body_is_retained(tmp_path):
    error = RuntimeError('simulated provider failure')
    error.body = {'model': 'gpt-5-nano-2025-08-07', 'service_tier': 'default', 'usage': attempt()['usage']}
    error.request_id = 'failed-request'
    with pytest.raises(RuntimeError):
        run_api(tmp_path, [error])
    costs = recalculate_costs(tmp_path / 'api')
    assert costs['total_cost_usd'] == pytest.approx(.000072)
    assert costs['successful_examples'] == 0
    assert costs['cost_per_1000_successful_examples_usd'] is None
    assert costs['failure_rate'] == 1
    row = json.loads((tmp_path / 'api/usage.jsonl').read_text().splitlines()[0])
    assert row['request_id'] == 'failed-request'


def test_usage_sink_is_detached_after_runner_finishes(tmp_path):
    client = KnownUsageClient(['Sports', 'World', 'Sports'])
    classifier = OpenAIClassifier(client=client)
    result = run_benchmark(FakeDataset(build_bundle()), classifier, BenchmarkRunConfig(
        output_root=tmp_path, run_id='closed', pricing_path=CARD_PATH))
    before = (result.run_dir / 'usage.jsonl').read_bytes()
    classifier.predict([build_bundle().test[0].as_input()])
    assert (result.run_dir / 'usage.jsonl').read_bytes() == before
