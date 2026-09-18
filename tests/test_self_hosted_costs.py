"""Independent scenario arithmetic and evidence/CLI integration checks."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from llm_classifier_bench.costs import load_pricing, price_entry, recalculate_costs
from llm_classifier_bench.measurement import MeasurementConfig
from llm_classifier_bench.self_hosted_costs import self_hosted_cost_report
from llm_classifier_bench.classifiers import TfidfLogisticClassifier
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark
from test_runner import FakeDataset, build_bundle

CARD_PATH = Path(__file__).resolve().parents[1] / 'pricing/inference_2026-09-16.json'


def fixture_run(tmp_path, *, backend='local', status='completed', failed=False):
    metadata = {
        'config': {'warmup_examples': 1, 'batch_size': 1, 'cache_condition': 'fixture', 'hardware_profile': 'fixture-cpu'},
        'planned_test_examples': 2, 'run_status': status, 'concurrency': 1,
        'runtime': {'classifier': {'backend': backend, 'device': 'cpu'}, 'cpu_model': 'fixture-cpu'},
    }
    # One hour of warmup; two evaluation calls spanning one hour, each .25 h,
    # with .5 h of gaps. Active-time throughput would be twice the window rate.
    durations = [('warmup', 0, 3600, 'warm'), ('evaluation', 3600, 4500, 'a'),
                 ('evaluation', 6300, 7200, 'b')]
    rows = []
    for phase, start, end, sample in durations:
        rows.append({'kind': 'inference', 'phase': phase, 'sample_ids': [sample], 'batch_size': 1,
                     'start_offset_ns': start * 10**9, 'end_offset_ns': end * 10**9,
                     'elapsed_ms': (end-start)*1000,
                     'status': 'exception' if failed and sample == 'b' else 'success'})
    (tmp_path / 'measurement.json').write_text(json.dumps(metadata))
    (tmp_path / 'timings.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path / 'config.json').write_text(json.dumps({'dataset': {'test_input_characters': {'count': 2, 'p50': 15}}}))
    card = load_pricing(CARD_PATH)
    card['local_hardware'].update(usd_per_hour='2', interpretation='measured_hardware', hardware_profile='fixture-cpu')
    return card


def test_continuous_and_deployed_costs_include_warmup_once_and_idle(tmp_path):
    card = fixture_run(tmp_path)
    result = self_hosted_cost_report(tmp_path, card, deployment_hours=4, volumes=(2, 4, 8))
    assert result['available']
    assert result['measured_successful_examples_per_hour'] == 2
    assert result['continuous']['cost_per_1000_successful_examples_usd'] == 1000
    continuous = result['continuous']['scenarios']
    assert [r['total_cost_usd'] for r in continuous] == [4, 6, 10]  # warmup USD 2 once
    assert [r['total_hours_including_one_warmup'] for r in continuous] == [2, 3, 5]
    deployed = result['deployed']['scenarios']
    assert result['deployed']['projected_success_capacity'] == 6
    assert deployed[0]['allocation_cost_usd'] == 8
    assert deployed[0]['cost_per_1000_successful_examples_usd'] == 4000
    assert deployed[0]['projected_idle_hours'] == 2
    assert deployed[0]['projected_busy_fraction'] == .5
    assert deployed[1]['cost_per_1000_successful_examples_usd'] == 2000
    assert deployed[2]['feasible'] is False
    assert deployed[2]['cost_per_1000_successful_examples_usd'] is None
    assert deployed[2]['allocation_cost_usd'] == 8  # allocation remains a known cost
    assert result['flops_per_prediction'] is None
    assert result['conditions']['input_characters']['p50'] == 15


def test_failure_fraction_reduces_success_throughput(tmp_path):
    card = fixture_run(tmp_path, failed=True)
    result = self_hosted_cost_report(tmp_path, card, deployment_hours=4, volumes=(2,))
    assert result['observed_evaluation']['unsuccessful_example_fraction'] == .5
    assert result['measured_successful_examples_per_hour'] == 1
    assert result['continuous']['cost_per_1000_successful_examples_usd'] == 2000
    assert result['continuous']['scenarios'][0]['total_cost_usd'] == 6


def test_schedule_is_explicit_and_scenario_inputs_are_validated(tmp_path):
    card = fixture_run(tmp_path)
    result = self_hosted_cost_report(tmp_path, card)
    assert result['available']
    assert not result['deployed']['available']
    assert result['deployment_hours'] is None
    for hours in (0, -1, float('inf'), float('nan'), True):
        with pytest.raises(ValueError):
            self_hosted_cost_report(tmp_path, card, deployment_hours=hours)
    for volumes in ((), (0,), (1, 1), (False,), (-1,), (1.2,)):
        with pytest.raises(ValueError):
            self_hosted_cost_report(tmp_path, card, volumes=volumes)


@pytest.mark.parametrize('backend,status', [('hosted_api','completed'), ('unknown','completed'), ('local','failed'), ('local','running')])
def test_hosted_or_incomplete_runs_cannot_claim_deployment_economics(tmp_path, backend, status):
    card = fixture_run(tmp_path, backend=backend, status=status)
    result = self_hosted_cost_report(tmp_path, card)
    assert not result['available']
    assert result['continuous'] is None
    assert result['reason']


def test_unknown_prices_and_wrong_hardware_remain_unavailable(tmp_path):
    card = fixture_run(tmp_path)
    assert not self_hosted_cost_report(tmp_path, load_pricing(None))['available']
    mismatch = deepcopy(card)
    mismatch['local_hardware']['hardware_profile'] = 'other-allocation'
    assert not self_hosted_cost_report(tmp_path, mismatch)['available']
    card['local_hardware']['device'] = 'cuda'
    assert not self_hosted_cost_report(tmp_path, card)['available']


def test_identity_is_required_for_measured_hardware_but_not_for_proxy(tmp_path):
    card = fixture_run(tmp_path)
    entry = {'event_id': 'local', 'phase': 'evaluation', 'kind': 'local_call', 'elapsed_ms': 3600000}
    assert price_entry(entry, card, 'cpu', 'fixture-cpu')['cost_usd'] == 2
    assert price_entry(entry, card, 'cpu')['cost_usd'] is None
    assert price_entry(entry, card, 'cpu', 'other-allocation')['cost_usd'] is None
    card['local_hardware'].pop('hardware_profile')
    path = tmp_path / 'card.json'
    path.write_text(json.dumps(card))
    with pytest.raises(ValueError, match='hardware_profile'):
        load_pricing(path)
    card['local_hardware']['interpretation'] = 'cloud_equivalent'
    assert self_hosted_cost_report(tmp_path, card)['available']
    assert self_hosted_cost_report(tmp_path, card)['interpretation'] == 'cloud_equivalent'


def test_less_than_warmup_and_zero_success_have_no_price_per_valid_result(tmp_path):
    card = fixture_run(tmp_path)
    result = self_hosted_cost_report(tmp_path, card, deployment_hours=.5, volumes=(1,))
    assert result['deployed']['projected_success_capacity'] == 0
    assert result['deployed']['scenarios'][0]['cost_per_1000_successful_examples_usd'] is None
    path = tmp_path / 'timings.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row['phase'] == 'evaluation':
            row['status'] = 'exception'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    assert not self_hosted_cost_report(tmp_path, card)['available']


def test_runner_and_cli_reprice_without_mutating_original_evidence(tmp_path):
    card = load_pricing(CARD_PATH)
    card['local_hardware'].update(usd_per_hour='2', interpretation='measured_hardware', hardware_profile='fixture-cpu')
    pricing = tmp_path / 'pricing.json'
    pricing.write_text(json.dumps(card))
    result = run_benchmark(FakeDataset(build_bundle()), TfidfLogisticClassifier(), BenchmarkRunConfig(
        output_root=tmp_path, run_id='run', validation_fraction=0, pricing_path=pricing,
        measurement=MeasurementConfig(warmup_examples=1, hardware_profile='fixture-cpu')))
    saved = json.loads((result.run_dir / 'cost_report.json').read_text())
    assert saved == recalculate_costs(result.run_dir)
    assert saved['self_hosted']['available']
    assert saved['self_hosted']['conditions']['measurement']['hardware_profile'] == 'fixture-cpu'
    lengths = saved['self_hosted']['conditions']['input_characters']
    expected = sorted(len(item.text) for item in build_bundle().test)
    assert lengths['min'] == expected[0] and lengths['max'] == expected[-1]
    assert lengths['count'] == 2
    before = {p.name:p.read_bytes() for p in result.run_dir.iterdir() if p.is_file()}
    script = Path(__file__).resolve().parents[1] / 'scripts/report_costs.py'
    out = tmp_path / 'scenarios.json'
    markdown = tmp_path / 'scenarios.md'
    command = [sys.executable, str(script), str(result.run_dir), '--deployment-hours', '24',
               '--volumes', '1000', '10000', '--json-output', str(out), '--output', str(markdown)]
    subprocess.run(command, check=True, capture_output=True)
    replayed = json.loads(out.read_text())
    assert replayed['total_cost_usd'] == saved['total_cost_usd']
    assert replayed['self_hosted']['deployed']['available']
    assert len(replayed['self_hosted']['continuous']['scenarios']) == 2
    assert 'Self-hosted inference projections' in markdown.read_text()
    # Alternate price changes both estimates proportionally, not measured throughput.
    card['local_hardware']['usd_per_hour'] = '4'
    pricing.write_text(json.dumps(card))
    changed = recalculate_costs(result.run_dir, pricing, deployment_hours=24)
    assert changed['self_hosted']['continuous']['cost_per_1000_successful_examples_usd'] == pytest.approx(saved['self_hosted']['continuous']['cost_per_1000_successful_examples_usd'] * 2)
    assert changed['self_hosted']['deployed']['scenarios'][0]['allocation_cost_usd'] == 96
    assert before == {p.name:p.read_bytes() for p in result.run_dir.iterdir() if p.is_file()}


def test_missing_configured_warmup_is_not_assumed_free(tmp_path):
    card = fixture_run(tmp_path)
    path = tmp_path / 'timings.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['phase'] != 'warmup'))
    result = self_hosted_cost_report(tmp_path, card)
    assert not result['available']
    assert 'warmup' in result['reason']
