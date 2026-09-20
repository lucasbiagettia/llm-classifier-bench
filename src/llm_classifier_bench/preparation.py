"""Preparation stage evidence, exclusive-time accounting and offline amortization."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

from .costs import amount, load_pricing, local_hardware_rate
from .measurement import runtime_metadata


@contextmanager
def preparation_stage(classifier, name, category, synchronize=None, **details):
    sink = getattr(classifier, '_preparation_sink', None)
    if sink is None:
        yield details
        return
    if synchronize is not None:
        synchronize()
    with sink(name, category, **details) as recorded:
        try:
            yield recorded
        finally:
            if synchronize is not None:
                synchronize()


class PreparationRecorder:
    def __init__(self, run_dir, classifier):
        self.run_dir, self.classifier = Path(run_dir), classifier
        self.origin = perf_counter_ns()
        self.stack, self.events = [], []
        self.next_id = 0
        (self.run_dir / 'preparation.jsonl').touch(exist_ok=False)
        hook = getattr(classifier, 'set_preparation_sink', None)
        self.supports_stages = callable(hook)
        if self.supports_stages:
            hook(self.stage)

    @contextmanager
    def stage(self, name, category, **details):
        event = {'event_id': f'prep-{self.next_id:06d}', 'parent_id': self.stack[-1] if self.stack else None,
                 'name': name, 'category': category, 'details': details, 'status': 'success'}
        self.next_id += 1
        self.stack.append(event['event_id'])
        started = perf_counter_ns()
        try:
            yield details
        except BaseException as exc:
            event.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            ended = perf_counter_ns()
            self.stack.pop()
            event.update(start_ns=started-self.origin, end_ns=ended-self.origin,
                         elapsed_ms=(ended-started)/1e6)
            self.events.append(event)
            with (self.run_dir / 'preparation.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')

    def measure(self, operation, name):
        with self.stage(name, 'parent'):
            return operation()

    def finish(self):
        hook = getattr(self.classifier, 'preparation_metadata', None)
        metadata = {'schema_version': 1, 'supports_stages': self.supports_stages,
                    'runtime': runtime_metadata(self.classifier),
                    'adapter': hook() if callable(hook) else {},
                    'boundary': 'prepare and fit only; construction before runner, dataset loading and inference warmup excluded',
                    'unavailable_resource_details': ['RAM allocation quota', 'GPU sharing/reservation'],
                    'allocation': 'one whole allocation retained throughout each prepare/fit call; gaps between calls excluded',
                    'cache_scope': 'incremental work in this run; prior artifact creation is never inferred to be free'}
        (self.run_dir / 'preparation_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
        report = recalculate_preparation(self.run_dir)
        (self.run_dir / 'preparation_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        return report

    def close(self):
        hook = getattr(self.classifier, 'set_preparation_sink', None)
        if callable(hook):
            hook(None)


def _exclusive_times(events):
    by_id = {e['event_id']: e for e in events}
    if len(by_id) != len(events):
        raise ValueError('Duplicate preparation event IDs')
    children = defaultdict(list)
    for event in events:
        if event['status'] not in {'success', 'failed'}:
            raise ValueError('Invalid preparation status')
        start, end = event['start_ns'], event['end_ns']
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError('Invalid preparation clock interval')
        if amount(event['elapsed_ms']) != Decimal(end-start)/1000000:
            raise ValueError('Preparation duration differs from interval')
        parent = event['parent_id']
        if parent is not None:
            if parent not in by_id or parent == event['event_id']:
                raise ValueError('Missing/self-referencing preparation parent')
            p = by_id[parent]
            if not (p['start_ns'] <= start <= end <= p['end_ns']):
                raise ValueError('Preparation child lies outside parent')
        children[parent].append(event)
    for group in children.values():
        ordered = sorted(group, key=lambda e:e['start_ns'])
        if any(a['end_ns'] > b['start_ns'] for a,b in zip(ordered, ordered[1:])):
            raise ValueError('Preparation siblings overlap')
    for event in events:
        seen = set()
        current = event
        while current['parent_id'] is not None:
            if current['event_id'] in seen:
                raise ValueError('Preparation parent cycle')
            seen.add(current['event_id'])
            current = by_id[current['parent_id']]
    exclusive = {e['event_id']: Decimal(e['end_ns']-e['start_ns']-
                 sum(c['end_ns']-c['start_ns'] for c in children[e['event_id']]))/1000000 for e in events}
    return children[None], exclusive


def recalculate_preparation(run_dir: Path, pricing_path: Path | None = None,
                            billing_path: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    paths = {name: run_dir/name for name in ('preparation.jsonl','preparation_metadata.json','config.json')}
    if any(not p.exists() for p in paths.values()):
        return {'available':False, 'complete':False, 'total_cost_usd':None, 'cost_kind':'unavailable',
                'reason':'Preparation evidence missing; historical runs need a new preparation measurement', 'events':[]}
    events = [json.loads(line) for line in paths['preparation.jsonl'].read_text().splitlines() if line.strip()]
    metadata = json.loads(paths['preparation_metadata.json'].read_text())
    config = json.loads(paths['config.json'].read_text())
    card = load_pricing(pricing_path or run_dir/'pricing.json')
    roots, exclusive = _exclusive_times(events)
    complete = sorted(e['name'] for e in roots) == ['fit','prepare'] and all(e['status']=='success' for e in roots)
    total_ms = sum((amount(e['elapsed_ms']) for e in roots), Decimal(0))
    backend = metadata['runtime']['classifier'].get('backend')
    device = metadata['runtime']['classifier'].get('device')
    profile = config.get('measurement',{}).get('hardware_profile')
    unit_rate = None
    rate_reason = None
    if backend == 'local':
        try:
            unit_rate = amount(local_hardware_rate(card, device, profile)['usd_per_hour'])/3600000
        except ValueError as exc:
            rate_reason = str(exc)
    cost = total_ms*unit_rate if unit_rate is not None and roots else None
    kind = 'estimated' if cost is not None else 'unavailable'
    reason = None if cost is not None else rate_reason or 'Provider preparation charges/billable resource time unavailable'
    basis = 'active prepare/fit wall time at whole-allocation rate; children are explanatory components'
    if metadata['adapter'].get('remote_preparation_performed') is False and backend == 'hosted_api':
        cost, kind, reason = Decimal(0), 'no_billable_work', None
        basis = 'Adapter performs no remote preparation; client construction/compute excluded'
    billing = json.loads(Path(billing_path).read_text()) if billing_path is not None else metadata['adapter'].get('billing')
    if billing is not None:
        if billing.get('scope') != 'entire_preparation' or not billing.get('source'):
            raise ValueError('Billing evidence requires entire_preparation scope and source')
        if billing.get('kind') == 'observed_charge':
            if billing.get('currency') != 'USD':
                raise ValueError('Observed preparation charges must be USD')
            cost, kind, reason = amount(billing.get('amount')), 'observed', None
            basis = 'Observed charge for entire preparation; not added to runtime estimate'
        elif billing.get('kind') == 'billable_allocation':
            rate = local_hardware_rate(card, billing.get('device'), billing.get('hardware_profile'))
            if rate['interpretation'] != 'measured_hardware':
                raise ValueError('Billable allocation requires a measured_hardware rate')
            if backend == 'local' and (billing.get('hardware_profile') != profile or str(billing.get('device')).split(':')[0] != str(device).split(':')[0]):
                raise ValueError('Billable allocation does not match measured local hardware')
            cost = amount(billing.get('billable_hours'))*amount(rate['usd_per_hour'])
            kind, reason, basis = 'estimated', None, 'Recorded billable allocation hours including declared idle/rounding; not remote wait time'
        else:
            raise ValueError('Unknown preparation billing evidence kind')
    if not roots:
        cost, kind, reason = None, 'unavailable', 'No recorded preparation stages'
    categories = defaultdict(Decimal)
    priced_events = []
    for event in events:
        ms = exclusive[event['event_id']]
        categories[event['category']] += ms
        priced_events.append({**event, 'exclusive_ms':float(ms),
                              'exclusive_cost_estimate_usd':float(ms*unit_rate) if unit_rate is not None else None})
    selections = [e for e in events if 'selected_fit_names' in e['details']]
    selected = None
    if selections:
        if len(selections) != 1:
            raise ValueError('Ambiguous preparation selection')
        selection = selections[0]['details']
        names = selection['selected_fit_names']
        fits = [e for e in events if e['name'] in names and e['category']=='fit']
        if len(set(names)) != len(names) or len(fits) != len(names) or any(e['status'] != 'success' for e in fits):
            raise ValueError('Selected fits must refer to unique measured fit stages')
        ms = sum((exclusive[e['event_id']] for e in fits), Decimal(0))
        selected = {**selection, 'selected_fit_exclusive_ms':float(ms),
                    'selected_fit_cost_estimate_usd':float(ms*unit_rate) if unit_rate is not None else None,
                    'role':'subset of total investment; excludes shared features, selection evaluation and restoration; never add to total'}
    hashes = {name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in paths.items()}
    price_path = Path(pricing_path) if pricing_path is not None else run_dir/'pricing.json'
    hashes['selected_pricing'] = hashlib.sha256(price_path.read_bytes()).hexdigest()
    if billing_path is not None:
        hashes['billing'] = hashlib.sha256(Path(billing_path).read_bytes()).hexdigest()
    return {'schema_version':1, 'available':cost is not None, 'complete':complete,
            'scope':'incremental preparation in this run; inference and its warmup excluded',
            'total_elapsed_ms':float(total_ms), 'total_cost_usd':float(cost) if cost is not None else None,
            'cost_kind':kind, 'currency':'USD', 'reason':reason, 'basis':basis,
            'metadata':metadata, 'pricing':card, 'billing':billing, 'source_sha256':hashes,
            'exclusive_ms_by_category':{k:float(v) for k,v in categories.items()},
            'selected_configuration':selected,
            'cache_provenance':[{'stage':e['name'], **e['details']} for e in events if 'cache' in e['details'] or e['details'].get('reused')],
            'prior_artifact_creation_cost_usd':None, 'events':priced_events}


def amortize_preparation(preparation: dict, inference: dict, *, scenario: str,
                         volumes=(1000,10000,100000)) -> dict:
    if scenario not in {'continuous','deployed','recorded_api'}:
        raise ValueError('Choose continuous, deployed or recorded_api inference')
    if not volumes or any(type(v) is not int or v <= 0 for v in volumes) or len(set(volumes)) != len(volumes):
        raise ValueError('Volumes must be unique positive integers')
    rows = []
    for volume in volumes:
        inference_cost, reason = None, None
        if scenario in {'continuous','deployed'}:
            hosted = inference.get('self_hosted') or {}
            scenarios = (hosted.get(scenario) or {}).get('scenarios', []) if hosted.get('available') else []
            entry = next((e for e in scenarios if e['successful_examples']==volume), None)
            if entry is not None and (scenario=='continuous' or entry.get('feasible')):
                inference_cost = entry['total_cost_usd'] if scenario=='continuous' else entry['allocation_cost_usd']
            else:
                reason = 'Inference scenario unavailable or infeasible'
        else:
            # Warmup is paid once, not once per extrapolated test-sized block.
            if (inference.get('backend') == 'hosted_api' and inference.get('run_status')=='completed'
                    and inference.get('successful_examples',0) and inference['evaluation']['coverage_complete']
                    and inference['warmup']['coverage_complete']):
                inference_cost = float(amount(inference['evaluation']['total_cost_usd'])*volume /
                                       inference['successful_examples'] + amount(inference['warmup']['total_cost_usd']))
            else:
                reason = 'Complete hosted API usage/price evidence unavailable'
        prep_cost = preparation['total_cost_usd']
        total = None
        if preparation.get('complete') and prep_cost is not None and inference_cost is not None:
            total = float((amount(prep_cost)+amount(inference_cost))/volume)
        elif reason is None:
            reason = 'Preparation incomplete or cost unavailable'
        rows.append({'successful_examples':volume, 'preparation_cost_usd':prep_cost,
                     'inference_scenario_cost_usd':inference_cost,
                     'amortized_cost_per_valid_prediction_usd':total, 'reason':reason})
    return {'scenario':scenario, 'rows':rows, 'assumptions':'Same measured conditions/rates and failure mix; one inference warmup; preparation added once'}
