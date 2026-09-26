"""Execute the frozen release matrix one cell at a time; retain terminal cells on restart."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

METHODS = ('tfidf', 'jev', 'emissary', 'openai', 'sentence-transformer', 'bert')


def cells():
    for method in METHODS:
        for seed in (42, 43, 44):
            for k in (5, 10, 20, 25):
                for budget in (0, 5, 100, None):
                    if method == "openai" and budget != 0:
                        continue
                    # Zero-shot controls are shared with the reference tables, not repeated.
                    if budget is None and method in ('jev', 'emissary', 'openai'):
                        continue
                    yield {'id': f'{method}__s{seed}__k{k}__b{budget if budget is not None else "reference"}',
                           'method': method, 'seed': seed, 'class_count': k, 'budget': budget}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/v2_release'))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest', type=Path, default=Path('reports/v2/matrix_next.json'))
    parser.add_argument('--restart-interrupted', action='store_true', help='Restart only KeyboardInterrupt cells with zero saved predictions; retain original attempt')
    parser.add_argument('--only', nargs='+', choices=METHODS)
    args = parser.parse_args()
    root = args.root
    manifest_path = args.manifest
    if not manifest_path.exists():
        raise ValueError('Freeze matrix.json and its input hashes before execution')
    manifest = json.loads(manifest_path.read_text())
    if manifest['cells'] != list(cells()):
        raise ValueError('Matrix differs from the frozen plan')
    for path, expected in manifest['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen input changed: {path}')
    env = {**os.environ, 'PYTHONPATH': 'src', 'CUDA_VISIBLE_DEVICES': '',
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
           'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1'}
    for cell in manifest['cells']:
        if args.only and cell['method'] not in args.only:
            continue
        output = root / 'cells' / cell['id']
        previous = sorted(output.glob('*/runs/*/status.json'))
        if previous:
            status_path = previous[-1]
            saved = json.loads(status_path.read_text())
            predictions = status_path.with_name('predictions.jsonl')
            has_predictions = predictions.exists() and bool(predictions.read_text().strip())
            restartable = (args.restart_interrupted and saved['status'] == 'failed'
                           and saved.get('error_type') == 'KeyboardInterrupt' and not has_predictions)
            if not restartable:
                print(f"SKIP {cell['id']} ({saved['status']}); retained evidence", flush=True)
                continue
        elif output.exists() and any(output.iterdir()):
            raise ValueError(f'Interrupted cell needs explicit recovery; refusing duplicate inference: {output}')
        command = [sys.executable, 'scripts/run_banking77_scaling_benchmark_v2.py',
                   *manifest['common_arguments'], '--classifiers', cell['method'],
                   '--seeds', str(cell['seed']), '--class-counts', str(cell['class_count']),
                   '--output-root', str(output)]
        if cell['budget'] is not None:
            command += ['--matched-budgets', str(cell['budget']), '--budget-unit', 'per_class', '--validation-budget', '0']
        if not args.execute:
            print(json.dumps(command)); continue
        (root / 'logs').mkdir(parents=True, exist_ok=True)
        print(f"START {cell['id']}", flush=True)
        attempt = len(previous) + 1
        with (root / 'logs' / f"{cell['id']}__attempt{attempt:03d}.log").open('x') as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"END {cell['id']} exit={process.returncode}", flush=True)
        if process.returncode:
            raise RuntimeError(f'Cell process failed; inspect {cell["id"]}')


if __name__ == '__main__':
    main()
