"""Validate persisted outputs before selecting unpredicted inputs for continuation."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence
from .core import LabeledExample


def pending_examples(examples: Sequence[LabeledExample], artifacts: Sequence[Path]):
    expected = {e.sample_id: e for e in examples}
    if len(expected) != len(examples):
        raise ValueError('Duplicate planned sample IDs')
    completed = {}
    for artifact in artifacts:
        if not artifact.exists():
            continue
        for line in artifact.read_text().splitlines():
            row = json.loads(line)
            key = row['sample_id']
            if key not in expected or key in completed:
                raise ValueError('Unexpected or repeated persisted sample ID')
            example = expected[key]
            if row['gold_label'] != example.label or row['input'] != example.text:
                raise ValueError('Persisted sample text/gold differs from frozen source')
            completed[key] = row
    return tuple(e for e in examples if e.sample_id not in completed), completed
