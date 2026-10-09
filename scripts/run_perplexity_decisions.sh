#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
exec venv/bin/python -u scripts/run_perplexity_decisions.py "$@"
