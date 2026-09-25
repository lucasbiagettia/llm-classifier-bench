"""Audit saved campaign probability metrics with independent NumPy formulas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from llm_classifier_bench.metrics import evaluate_jsonl
from llm_classifier_bench.core import PROBABILITY_SUM_TOLERANCE


METRICS = ("top_label_ece", "adaptive_ece", "multiclass_log_loss", "multiclass_brier_score")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_metrics(probabilities, gold, labels):
    """Independent NumPy formulas, with the repository's ten-bin conventions."""
    p = np.asarray(probabilities, dtype=float)
    indices = np.array([list(labels).index(label) for label in gold])
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == indices
    fixed = np.minimum((confidence * 10).astype(int), 9)
    groups = [np.flatnonzero(fixed == i) for i in range(10)]
    order = np.argsort(confidence, kind="stable")
    nominal = np.cumsum([len(b) for b in np.array_split(order, min(10, len(p)))])
    # Independent NumPy reference: move quantiles after all equal confidences,
    # collapse coincident boundaries, then split the ordered population.
    ends = np.unique(np.searchsorted(confidence[order], confidence[order[nominal - 1]], side="right"))
    adaptive = np.split(order, ends[:-1])

    def ece(bins):
        return float(sum(abs(correct[b].sum() - confidence[b].sum()) / len(p)
                         for b in bins if len(b)))

    return {
        "top_label_ece": ece(groups),
        "adaptive_ece": ece(adaptive),
        "multiclass_log_loss": float(-np.log(np.clip(p[np.arange(len(p)), indices], 1e-15, 1 - 1e-15)).mean()),
        "multiclass_brier_score": float(((p - np.eye(len(labels))[indices]) ** 2).sum(axis=1).mean()),
    }


def audit_run(run: Path):
    config = json.loads((run / "config.json").read_text())
    rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines() if line.strip()]
    labels = [c["name"] for c in config["dataset"]["classes"]]
    saved = json.loads((run / "metrics.json").read_text())
    recalculated = {m.name: m.value for m in evaluate_jsonl(run / "predictions.jsonl")}
    partitions = [config["dataset"][key] for key in
                  ("fit_train_sample_ids", "validation_sample_ids", "test_sample_ids")]
    checks = {
        "unique_partition_ids": all(len(ids) == len(set(ids)) for ids in partitions),
        "disjoint_partitions": all(not set(partitions[i]) & set(partitions[j])
                                   for i in range(3) for j in range(i + 1, 3)),
        "exact_test_ids_and_order": [r["sample_id"] for r in rows] == partitions[2],
        "known_labels": all(r["gold_label"] in labels and r["predicted_label"] in labels for r in rows),
        "saved_metrics_match": all(saved[name]["value"] == recalculated[name] for name in METRICS),
    }
    result = {
        "run_id": run.name,
        "source_sha256": {name: sha256(run / name) for name in ("config.json", "predictions.jsonl", "metrics.json")},
        "partition_sizes": dict(zip(("fit", "validation", "test"), map(len, partitions))),
        "class_names": labels,
        "original": {name: saved[name]["value"] for name in METRICS},
        "recalculated": {name: recalculated[name] for name in METRICS},
        "accuracy": recalculated["accuracy"],
        "checks": checks,
    }
    if all(r.get("probabilities") for r in rows):
        checks["complete_label_maps"] = all(set(r["probabilities"]) == set(labels) for r in rows)
        if not checks["complete_label_maps"]:
            raise ValueError(f"Incomplete probability maps: {run}")
        p = np.array([[r["probabilities"][label] for label in labels] for r in rows])
        checks["finite_in_range"] = bool(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all())
        checks["normalized"] = bool(np.allclose(p.sum(axis=1), 1, atol=PROBABILITY_SUM_TOLERANCE, rtol=0))
        checks["prediction_is_argmax"] = all(r["probabilities"][r["predicted_label"]] == max(r["probabilities"].values()) for r in rows)
        checks["confidence_matches_prediction"] = all(r["confidence"] is None or abs(r["confidence"] - r["probabilities"][r["predicted_label"]]) <= 1e-6 for r in rows)
        reference = reference_metrics(p, [r["gold_label"] for r in rows], labels)
        checks["independent_metrics_match"] = all(abs(reference[name] - recalculated[name]) < 1e-12 for name in METRICS)
        result.update(independent=reference, mean_confidence=float(p.max(axis=1).mean()),
                      max_normalization_error=float(abs(p.sum(axis=1) - 1).max()))
    else:
        checks["missing_distributions_unavailable"] = all(recalculated[name] is None for name in METRICS[2:])
        if all(r.get("confidence") is None and not r.get("probabilities") for r in rows):
            checks["missing_confidence_unavailable"] = all(recalculated[name] is None for name in METRICS)
    cs = sorted({r.get("raw_response", {}).get("selected_c") for r in rows
                 if r.get("raw_response", {}).get("selected_c") is not None})
    if cs:
        result["selected_c_values"] = cs
    if not all(checks.values()):
        raise ValueError(f"Audit failed for {run}: {checks}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = sorted(p.parent for p in (args.campaign / "runs").glob("*/predictions.jsonl"))
    if not runs:
        parser.error("No saved predictions found")
    report = {"campaign": str(args.campaign), "n_bins": 10, "runs": [audit_run(run) for run in runs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Audited {len(runs)} runs; saved {args.output}")


if __name__ == "__main__":
    main()
