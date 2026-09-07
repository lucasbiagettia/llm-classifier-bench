"""Small deterministic fit-partition selector, independent of provider HTTP."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import random
from typing import Any, Sequence

from llm_classifier_bench.config import EmissaryTrainingConfig
from llm_classifier_bench.core import ClassDefinition, LabeledExample


def validate_partition_disjointness(*partitions: Sequence[LabeledExample]) -> None:
    """Reject repeated IDs within/across partitions and exact text across partitions.

    Exact UTF-8 content equality is checked; this is not semantic deduplication.
    Duplicate text within one source partition is retained as distinct source IDs.
    """
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for partition in partitions:
        ids = [item.sample_id for item in partition]
        content = {sha256(item.text.encode("utf-8")).hexdigest() for item in partition}
        if len(ids) != len(set(ids)) or seen_ids.intersection(ids):
            raise ValueError("Partitions contain duplicate or overlapping sample IDs")
        if seen_content.intersection(content):
            raise ValueError("Partitions contain overlapping text content")
        seen_ids.update(ids)
        seen_content.update(content)


def select_labeled_examples(
    examples: Sequence[LabeledExample],
    classes: Sequence[ClassDefinition],
    config: EmissaryTrainingConfig,
    *,
    validation_examples: Sequence[LabeledExample] = (),
) -> tuple[tuple[LabeledExample, ...], dict[str, Any]]:
    """Shuffle sorted classes and sorted per-class IDs with independent seeded RNGs.

    Round-robin over the fixed class order gives nested prefixes. For total N,
    each class gets N // K examples; the first N % K classes get one extra.
    A short class fails its assigned quota; budgets are never redistributed.
    """
    validate_partition_disjointness(examples, validation_examples)
    names = sorted(item.name for item in classes)
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("Selection requires at least two unique classes")
    groups: dict[str, list[LabeledExample]] = {name: [] for name in names}
    for example in examples:
        if example.label not in groups:
            raise ValueError(f"Unknown training label {example.label!r}")
        groups[example.label].append(example)
    if any(item.label not in groups for item in validation_examples):
        raise ValueError("Unknown validation label")
    random.Random(f"{config.selection_seed}:classes").shuffle(names)
    total = config.shots * len(names) if config.shot_unit == "per_class" else config.shots
    counts = {name: total // len(names) + (i < total % len(names))
              for i, name in enumerate(names)}
    for name in names:
        group = groups[name]
        if len(group) < counts[name]:
            raise ValueError(f"Insufficient fit-training support for {name!r}: "
                             f"requested {counts[name]}, available {len(group)}")
        group.sort(key=lambda item: item.sample_id)
        random.Random(f"{config.selection_seed}:examples:{name}").shuffle(group)
    selected = tuple(groups[names[i % len(names)]][i // len(names)] for i in range(total))
    metadata = {
        **asdict(config),
        "requested_total": total,
        "actual_total": len(selected),
        "per_class_counts": counts,
        "covered_class_count": sum(count > 0 for count in counts.values()),
        "class_count": len(names),
        "class_coverage": sum(count > 0 for count in counts.values()) / len(names),
        "class_order": names,
        "selected_examples": [
            {"sample_id": item.sample_id, "label": item.label, "text": item.text,
             "order": i, "content_sha256": sha256(item.text.encode("utf-8")).hexdigest()}
            for i, item in enumerate(selected)
        ],
    }
    return selected, metadata
