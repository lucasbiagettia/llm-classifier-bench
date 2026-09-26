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
TEST_PER_CLASS = 40


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


def command_for(cell, manifest, output, manifest_path=Path('reports/v2/matrix_small.json')):
    if cell['availability'] != 'ready':
        raise ValueError('Blocked techniques cannot be executed by this launcher')
    return [sys.executable, '-u', 'scripts/run_small_condition.py', '--manifest', str(manifest_path),
            '--cell', cell['id'], '--output', str(output)]


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


def run_condition(command, *, env, log, timeout_s, on_progress=None):
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stop_process_group(process)
                return process.returncode, True
            try:
                return process.wait(timeout=min(15, remaining)), False
            except subprocess.TimeoutExpired:
                if on_progress is not None:
                    on_progress()
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
    total = sum(c['test_examples'] for c in ready)
    reused = sum(manifest.get('reuse', {}).get(c['id'], {}).get('examples', 0) for c in ready)
    print(f'PLAN: {len(ready)} executable conditions, {total} evaluation examples; '
          f'reused={reused}, maximum new prediction calls={total-reused}; retries=0, warmup=0', flush=True)
    if not args.execute:
        for cell in selected:
            print(json.dumps({'cell': cell, 'command': command_for(cell, manifest, args.root/'cells'/cell['id'], args.manifest)
                              if cell['availability']=='ready' else None}))
        return
    env = {**os.environ, **manifest['environment'], 'PYTHONPATH': 'src', 'PYTHONUNBUFFERED':'1'}
    args.root.mkdir(parents=True, exist_ok=True)
    def report(message):
        from datetime import datetime
        line=f'{datetime.now().astimezone().isoformat(timespec="seconds")} | {message}'
        print(line,flush=True)
        with (args.root/'run.log').open('a') as stream: stream.write(line+'\n')
    def save_state(state):
        from _small_experiment import write_json
        write_json(args.root/'execution.json',state)
        write_json(args.root/'summary.json',list({'cell':key,**value} for key,value in state['cells'].items()))
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
        from _small_experiment import write_json
        if not (args.root/'manifest.json').exists():
            write_json(args.root/'manifest.json', manifest)
            from datetime import datetime, timezone
            write_json(args.root/'execution_provenance.json', {
                'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'python': sys.version, 'argv': sys.argv,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
                'code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for root in ('src', 'scripts') for p in Path(root).rglob('*.py')},
            })
        report(f'RUN started | selected={len(selected)} | manifest={args.manifest}')
        for cell in selected:
            key = cell['id']; output = args.root/'cells'/key
            if key in state['cells'] or output.exists():
                report(f'SKIP {key}: previous attempt retained; no automatic repetition')
                continue
            if cell['availability'] != 'ready':
                state['cells'][key] = {'status':'blocked', 'reason':cell['reason']}
                report(f'BLOCKED {key}: {cell["reason"]}')
                save_state(state)
                continue
            timeout_s = manifest['limits']['hosted_condition_wall_seconds'] if cell['method'] in ('openai','emissary','jev') else manifest['limits']['condition_wall_seconds']
            if cell['method'] == 'bert':
                remaining = manifest['limits']['bert_total_wall_seconds'] - state['bert_accounted_wall_seconds']
                timeout_s = min(timeout_s, remaining)
                if timeout_s <= 0:
                    state['cells'][key] = {'status':'not_run', 'reason':'BERT campaign time budget exhausted'}
                    save_state(state)
                    report(f'SKIP {key}: BERT campaign time budget exhausted')
                    continue
            logs = args.root/'logs'; logs.mkdir(exist_ok=True)
            state['cells'][key] = {'status':'running'}
            if cell['method']=='bert':
                # Reserve first: an uncatchable kill must not reset the time budget.
                state['bert_accounted_wall_seconds'] += timeout_s
            save_state(state)
            started = time.monotonic()
            try:
                with (logs/f'{key}.log').open('x') as log:
                    reused=manifest.get('reuse',{}).get(key,{}).get('examples',0)
                    report(f'START {key} | total={cell["test_examples"]} reused={reused} new={cell["test_examples"]-reused} | limit={timeout_s:.0f}s | log={log.name}')
                    def progress():
                        stage='loading'
                        statuses=sorted(output.glob('runs/*/status.json'))
                        if statuses:
                            try:stage=json.loads(statuses[-1].read_text()).get('stage','unknown')
                            except (ValueError,OSError):stage='updating status'
                        count=sum(sum(1 for line in p.open() if line.strip()) for p in output.glob('runs/*/predictions.jsonl'))
                        report(f'PROGRESS {key} | stage={stage} | new={count}/{cell["test_examples"]-reused} reused={reused} | elapsed={time.monotonic()-started:.0f}s')
                    code, timed_out = run_condition(command_for(cell, manifest, output, args.manifest),
                                                   env=env, log=log, timeout_s=timeout_s,on_progress=progress)
                if (output/'condition.json').exists():
                    from _small_experiment import consolidate
                    try:
                        result=consolidate(output)
                    except (ValueError, OSError) as exc:
                        report(f'ERROR {key}: cannot consolidate evidence: {type(exc).__name__}: {exc}')
                        result={'status':'failed','valid_examples':None,'reused_examples':reused}
                else:
                    result={'status':'failed','valid_examples':0,'reused_examples':0}
                state['cells'][key] = {'status':'failed' if timed_out or code else result['status'],
                    'valid_examples':result['valid_examples'],'reused_examples':result['reused_examples'],
                    'planned_examples':cell['test_examples'],'returncode':code,'time_limit_exceeded':timed_out,
                    'result_path':str(output/'result.json'),'log':str(logs/f'{key}.log')}
                if result['status']!='completed' or timed_out:
                    detail=result.get('new_attempt_status',{})
                    report(f'ERROR {key} | type={detail.get("error_type")} stage={detail.get("stage")} | time_limit={timed_out}; see {logs/key}.log')
            except BaseException:
                state['cells'][key] = {'status':'interrupted'}
                report(f'STOPPED {key}; child process group terminated')
                raise
            finally:
                elapsed = time.monotonic()-started
                state['cells'][key]['wall_seconds'] = elapsed
                if cell['method']=='bert':
                    state['bert_accounted_wall_seconds'] += elapsed - timeout_s
                save_state(state)
            report(f"END {key}: {state['cells'][key]['status']} | elapsed={elapsed:.1f}s")
            if code and not timed_out:
                report('Condition process exited with an error; evidence retained; continuing other conditions')
        from collections import Counter
        counts=dict(Counter(v['status'] for v in state['cells'].values()))
        report(f'RUN FINISHED | {counts} | summary={args.root/"summary.json"}')
        if any(state['cells'].get(c['id'],{}).get('status')!='completed' for c in ready):
            raise SystemExit(2)


if __name__ == '__main__':
    main()
