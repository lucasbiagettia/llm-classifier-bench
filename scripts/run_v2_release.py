"""Plan the small v2 experiment by default; execute only with --execute."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

METHODS = ('tfidf', 'sentence-transformer', 'openai', 'emissary', 'jev', 'bert')
CLASS_COUNTS = (5, 10, 15, 20)
SEED = 42
TEST_PER_CLASS = 30


def cells():
    for method in METHODS:
        for k in CLASS_COUNTS:
            budget = 50 if method in ('tfidf', 'sentence-transformer', 'bert') else 0
            yield {'id': f'{method}__s42__k{k}__b{budget}', 'method': method,
                   'seed': SEED, 'class_count': k, 'budget': budget,
                   'test_examples': TEST_PER_CLASS * k, 'availability': 'ready'}
    for technique, budget, reason in (
        ('quick-train', 5, 'Quick Train is UI-only in the recorded provider contract; do not substitute SFT'),
        ('projects-sft', 100, 'Previous provider training job failed; readiness has not been established'),
    ):
        for k in CLASS_COUNTS:
            yield {'id': f'emissary-{technique}__s42__k{k}__b{budget}', 'method': 'emissary',
                   'seed': SEED, 'class_count': k, 'budget': budget,
                   'test_examples': TEST_PER_CLASS * k, 'availability': 'blocked', 'reason': reason}


def command_for(cell, manifest, output):
    if cell['availability'] != 'ready':
        raise ValueError('Blocked techniques cannot be executed by this launcher')
    return [sys.executable, 'scripts/run_banking77_scaling_benchmark_v2.py',
            *manifest['common_arguments'], '--classifiers', cell['method'],
            '--seeds', str(cell['seed']), '--class-counts', str(cell['class_count']),
            '--matched-budgets', str(cell['budget']), '--budget-unit', 'per_class',
            '--validation-budget', '0', '--output-root', str(output),
            '--jev-max-requests', str(cell['test_examples'])]


def stop_process_group(process):
    """Stop the active condition and its children before returning control."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_condition(command, *, env, log, timeout_s):
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        return process.wait(timeout=timeout_s), False
    except subprocess.TimeoutExpired:
        stop_process_group(process)
        return process.returncode, True
    except BaseException:
        stop_process_group(process)
        raise


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Launcher received signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/v2_small'))
    parser.add_argument('--manifest', type=Path, default=Path('reports/v2/matrix_small.json'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--only', nargs='+', choices=METHODS)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['scope'] != 'small-v2-2026-09-26' or manifest['cells'] != list(cells()):
        raise ValueError('This launcher accepts only the small frozen matrix')
    for path, digest in manifest['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen input changed: {path}')
    if args.root.resolve() == Path('artifacts/v2_release').resolve():
        raise ValueError('Preserve the historical campaign; use a separate small-campaign directory')
    selected = [c for c in manifest['cells'] if not args.only or c['method'] in args.only]
    ready = [c for c in selected if c['availability'] == 'ready']
    print(f"PLAN: {len(ready)} executable conditions, {sum(c['test_examples'] for c in ready)} "
          'maximum evaluation calls; retries=0, warmup=0', flush=True)
    if not args.execute:
        for cell in selected:
            print(json.dumps({'cell': cell, 'command': command_for(cell, manifest, args.root/'cells'/cell['id'])
                              if cell['availability']=='ready' else None}))
        return
    env = {**os.environ, **manifest['environment'], 'PYTHONPATH': 'src'}
    args.root.mkdir(parents=True, exist_ok=True)
    # A locked campaign cannot be launched again while its first process is alive.
    with (args.root/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGHUP, interrupted)
        state_path = args.root/'execution.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['manifest_sha256'] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
                raise ValueError('Existing execution uses a different manifest')
        else:
            state = {'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                     'bert_accounted_wall_seconds': 0., 'cells': {}}
        for cell in selected:
            key = cell['id']; output = args.root/'cells'/key
            if key in state['cells'] or output.exists():
                print(f'SKIP {key}: previous attempt retained; no automatic repetition', flush=True)
                continue
            if cell['availability'] != 'ready':
                state['cells'][key] = {'status':'blocked', 'reason':cell['reason']}
                state_path.write_text(json.dumps(state, indent=2)+'\n')
                continue
            timeout_s = manifest['limits']['condition_wall_seconds']
            if cell['method'] == 'bert':
                remaining = manifest['limits']['bert_total_wall_seconds'] - state['bert_accounted_wall_seconds']
                timeout_s = min(timeout_s, remaining)
                if timeout_s <= 0:
                    state['cells'][key] = {'status':'not_run', 'reason':'BERT campaign time budget exhausted'}
                    state_path.write_text(json.dumps(state, indent=2)+'\n')
                    print(f'SKIP {key}: BERT campaign time budget exhausted', flush=True)
                    continue
            logs = args.root/'logs'; logs.mkdir(exist_ok=True)
            state['cells'][key] = {'status':'running'}
            if cell['method']=='bert':
                # Reserve first: an uncatchable kill must not reset the time budget.
                state['bert_accounted_wall_seconds'] += timeout_s
            state_path.write_text(json.dumps(state, indent=2)+'\n')
            started = time.monotonic()
            try:
                with (logs/f'{key}.log').open('x') as log:
                    print(f'START {key}; wall-time limit {timeout_s:.0f}s', flush=True)
                    code, timed_out = run_condition(command_for(cell, manifest, output),
                                                   env=env, log=log, timeout_s=timeout_s)
                statuses = sorted(output.glob('*/runs/*/status.json'))
                status = json.loads(statuses[-1].read_text())['status'] if statuses else 'failed'
                state['cells'][key] = {'status':'failed' if timed_out else status,
                                      'returncode':code, 'time_limit_exceeded':timed_out}
            except BaseException:
                state['cells'][key] = {'status':'interrupted'}
                raise
            finally:
                elapsed = time.monotonic()-started
                state['cells'][key]['wall_seconds'] = elapsed
                if cell['method']=='bert':
                    state['bert_accounted_wall_seconds'] += elapsed - timeout_s
                state_path.write_text(json.dumps(state, indent=2)+'\n')
            print(f"END {key}: {state['cells'][key]['status']}", flush=True)
            if code and not timed_out:
                raise RuntimeError('Condition process failed; remaining conditions were not started')


if __name__ == '__main__':
    main()
