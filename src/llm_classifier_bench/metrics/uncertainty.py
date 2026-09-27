"""Conditional test-sampling uncertainty; never pool overlapping campaign seeds."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import EvaluationRecord, require_records


def stratified_quality_bootstrap(
    records: Sequence[EvaluationRecord],
    *,
    paired_records: Sequence[EvaluationRecord] | None = None,
    repetitions: int = 2000,
    seed: int = 20260925,
) -> dict:
    """Percentile 95% CIs, resampling within gold class with shared paired draws.

    Labels, fitted models, class balance and selection seed are held fixed.
    Paired effects are first minus second. A pair must have identical sample IDs,
    gold labels and input text (when recorded); ordering may differ. These are
    pointwise exploratory intervals, without multiplicity correction.
    """
    first = require_records(records)
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    labels = sorted({r.gold_label for r in first} | {r.predicted_label for r in first})
    label_index = {label: i for i, label in enumerate(labels)}
    gold = np.array([label_index[r.gold_label] for r in first])
    predicted = np.array([label_index[r.predicted_label] for r in first])
    other = None
    if paired_records is not None:
        second = require_records(paired_records)
        by_id = {r.sample_id: r for r in second}
        if set(by_id) != {r.sample_id for r in first}:
            raise ValueError("Paired sample IDs must match exactly")
        ordered = [by_id[r.sample_id] for r in first]
        if any(a.gold_label != b.gold_label or a.metadata.get("input") != b.metadata.get("input")
               for a, b in zip(first, ordered)):
            raise ValueError("Paired gold labels and input text must match")
        labels = sorted(set(labels) | {r.predicted_label for r in ordered})
        label_index = {label: i for i, label in enumerate(labels)}
        gold = np.array([label_index[r.gold_label] for r in first])
        predicted = np.array([label_index[r.predicted_label] for r in first])
        other = np.array([label_index[r.predicted_label] for r in ordered])

    def scores(indices, prediction):
        truth, pred = gold[indices], prediction[indices]
        tp = np.bincount(truth[truth == pred], minlength=len(labels))
        denom = np.bincount(truth, minlength=len(labels)) + np.bincount(pred, minlength=len(labels))
        f1 = np.divide(2 * tp, denom, out=np.zeros(len(labels)), where=denom > 0)
        return np.array([(truth == pred).mean(), f1.mean()])

    def effect(indices):
        result = scores(indices, predicted)
        return result if other is None else result - scores(indices, other)

    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(gold == value) for value in np.unique(gold)]
    draws = np.array([effect(np.concatenate([rng.choice(s, size=len(s), replace=True)
                                            for s in strata])) for _ in range(repetitions)])
    point = effect(np.arange(len(first)))
    intervals = np.quantile(draws, [.025, .975], axis=0)
    return {
        "method": "gold_stratified_percentile_bootstrap_v1",
        "confidence_level": .95,
        "repetitions": repetitions,
        "bootstrap_seed": seed,
        "n_samples": len(first),
        "paired_difference": other is not None,
        "conditioning": "fixed fitted models, label set, class balance and campaign seed",
        "multiplicity_adjustment": "none; pointwise exploratory intervals",
        "metrics": {name: {"estimate": float(point[i]), "lower": float(intervals[0, i]),
                           "upper": float(intervals[1, i])}
                    for i, name in enumerate(("accuracy", "macro_f1"))},
    }
