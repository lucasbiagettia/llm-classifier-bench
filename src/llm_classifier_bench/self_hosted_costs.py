"""Transparent self-hosted inference projections from saved measurement evidence."""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

from .costs import amount, local_hardware_rate
from .measurement import regenerate_report


def _number(value: Decimal) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Scenario result is too large to represent')
    return result


def self_hosted_cost_report(run_dir: Path, card: dict, *, deployment_hours=None,
                           volumes=(1000, 10000, 100000)) -> dict:
    """Estimate one measured allocation; never infer GPU capacity from peak TFLOPS.

    A scenario volume means successfully classified examples, not raw requests.
    Throughput includes failed-call time and gaps in the measured evaluation window.
    One measured warmup is charged per scenario, separate from the continuous rate.
    No preparation, extra replicas, traffic model or automatic scaling is assumed.
    """
    if not volumes or any(type(n) is not int or n <= 0 for n in volumes) or len(set(volumes)) != len(volumes):
        raise ValueError('Scenario volumes must be unique positive integers')
    hours = amount(deployment_hours) if deployment_hours is not None else None
    if hours is not None and (not hours or not math.isfinite(float(hours))):
        raise ValueError('deployment_hours must be finite and positive')

    operational = regenerate_report(run_dir)
    metadata, evaluation = operational['measurement'], operational['evaluation']
    runtime = metadata['runtime']
    config_path = run_dir / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    result = {
        'schema_version': 1, 'available': False, 'cost_kind': 'unavailable', 'reason': None,
        'currency': 'USD', 'rate': card.get('local_hardware'),
        'conditions': {
            'runtime': runtime, 'measurement': metadata['config'],
            'concurrency': metadata.get('concurrency'),
            'input_characters': config.get('dataset', {}).get('test_input_characters'),
            'input_token_lengths': None,
            'input_token_lengths_reason': 'Not measured; character counts are not token counts',
            'run_metadata': config.get('run_metadata', {}),
        },
        'observed_evaluation': evaluation,
        'assumptions': [
            'Same measured hardware allocation, input distribution, batching, concurrency and cache conditions',
            'Measured throughput is extrapolated, not theoretical maximum throughput or a latency SLA',
            'Rate is USD/hour for the entire measured allocation, not USD/hour per GPU or per core',
            'One measured warmup per scenario; model loading/training and other preparation excluded',
            'Flat prorated hourly pricing; no minimum charges, rounding, traffic bursts, autoscaling or availability guarantees',
            'All scenario volumes are valid classifications; predicted labels need not be correct',
        ],
        'flops_per_prediction': None,
        'flops_reason': 'Optional diagnostic; no comparable operation count measured; peak TFLOPS are not a dollar conversion',
        'deployment_hours': _number(hours) if hours is not None else None,
        'volumes': list(volumes), 'continuous': None, 'deployed': None,
        'timings_sha256': hashlib.sha256((run_dir / 'timings.jsonl').read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else None,
    }
    try:
        if runtime.get('classifier', {}).get('backend') != 'local':
            raise ValueError('Self-hosted projections require a local classifier, not hosted API latency')
        if metadata['run_status'] != 'completed' or evaluation['unattempted_examples']:
            raise ValueError('An incomplete or failed benchmark is not a complete projection sample')
        if operational['warmup_failed_calls']:
            raise ValueError('Warmup did not complete successfully')
        if operational['warmup_attempted_examples'] != metadata['config']['warmup_examples']:
            raise ValueError('Recorded warmup does not match the configured warmup workload')
        if not evaluation['successful_examples'] or evaluation['window_wall_ms'] <= 0:
            raise ValueError('Positive measured throughput from valid classifications is required')
        rate = local_hardware_rate(card, runtime['classifier'].get('device'),
                                   metadata['config'].get('hardware_profile'))
        hourly_price = amount(rate['usd_per_hour'])
        throughput = Decimal(evaluation['successful_examples']) * 3600000 / amount(evaluation['window_wall_ms'])
        # Use the whole warmup window rather than summing batch means. This also
        # retains inter-call recording overhead, just as the evaluation window does.
        rows = [json.loads(line) for line in (run_dir / 'timings.jsonl').read_text().splitlines() if line.strip()]
        warmup = [r for r in rows if r['kind'] == 'inference' and r['phase'] == 'warmup']
        warmup_hours = (Decimal(max(r['end_offset_ns'] for r in warmup) - min(r['start_offset_ns'] for r in warmup))
                        / Decimal(3600000000000)) if warmup else Decimal(0)
        continuous = []
        deployed = []
        for volume in volumes:
            active_hours = Decimal(volume) / throughput
            required_hours = active_hours + warmup_hours
            cost = hourly_price * required_hours
            continuous.append({
                'successful_examples': volume, 'projected_evaluation_hours': _number(active_hours),
                'total_hours_including_one_warmup': _number(required_hours),
                'total_cost_usd': _number(cost),
                'cost_per_1000_successful_examples_usd': _number(cost * 1000 / volume),
            })
            if hours is not None:
                feasible = required_hours <= hours
                deployed.append({
                    'successful_examples': volume, 'feasible': feasible,
                    'reason': None if feasible else 'Volume exceeds projected throughput within the allocated hours',
                    'allocation_cost_usd': _number(hourly_price * hours),
                    'cost_per_1000_successful_examples_usd': _number(hourly_price * hours * 1000 / volume) if feasible else None,
                    'required_hours_including_one_warmup': _number(required_hours),
                    'projected_busy_fraction': _number(required_hours / hours),
                    'projected_idle_hours': _number(hours - required_hours) if feasible else None,
                })
        result.update(
            available=True, cost_kind='estimated', interpretation=rate['interpretation'],
            measured_successful_examples_per_hour=_number(throughput),
            warmup_once_hours=_number(warmup_hours),
            continuous={
                'cost_per_1000_successful_examples_usd': _number(hourly_price * 1000 / throughput),
                'basis': 'Observed evaluation-window throughput; warmup excluded from this rate and included once in each volume scenario',
                'scenarios': continuous,
            },
            deployed={
                'available': hours is not None,
                'reason': None if hours is not None else 'Specify deployment_hours; no operating schedule is assumed',
                'projected_success_capacity': _number(max(Decimal(0), hours - warmup_hours) * throughput) if hours is not None else None,
                'scenarios': deployed,
            },
        )
    except ValueError as exc:
        result['reason'] = str(exc)
    return result
