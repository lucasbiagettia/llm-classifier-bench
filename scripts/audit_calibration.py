"""Audit saved campaign probabilities; optionally replay MiniLM C selection offline."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from llm_classifier_bench.metrics import evaluate_jsonl


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
    adaptive = np.array_split(np.argsort(confidence, kind="stable"), min(10, len(p)))

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
        checks["normalized"] = bool(np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0))
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


def replay(campaign: Path, train_arrow: Path, test_arrow: Path, model_path: Path):
    """Use exact saved sample order, cached source rows, and local model weights."""
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    import torch

    torch.set_num_threads(4)
    encoder = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    sources = {"train": Dataset.from_file(str(train_arrow)), "test": Dataset.from_file(str(test_arrow))}
    output = {
        "model_snapshot": model_path.name,
        "model_files_sha256": {str(p.relative_to(model_path)): sha256(p) for p in sorted(model_path.rglob("*")) if p.is_file()},
        "source_sha256": {"train_arrow": sha256(train_arrow), "test_arrow": sha256(test_arrow)},
        "device": "cpu", "torch_threads": 4,
        "library_versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "torch", "sentence-transformers", "transformers", "datasets")},
        "runs": [],
    }
    for k in (5, 10):
        run = next((campaign / "runs").glob(f"*__n{k:02d}__sentence-transformer-logreg"))
        config = json.loads((run / "config.json").read_text())
        saved = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
        settings = config["classifier"]["training"]
        partitions = {}
        for key in ("fit_train", "validation", "test"):
            ids = config["dataset"][f"{key}_sample_ids"]
            rows = [sources[sample_id.split(":")[1]][int(sample_id.split(":")[2])] for sample_id in ids]
            if key == "test":
                if any(r["text"] != s["input"] or r["category"] != s["gold_label"] for r, s in zip(rows, saved, strict=True)):
                    raise ValueError("Cached test rows differ from original artifacts")
            # Match the original classifier's single-example test encoding.
            embeddings = encoder.encode([r["text"] for r in rows], batch_size=1 if key == "test" else settings["embedding_batch_size"], show_progress_bar=False, convert_to_numpy=True)
            partitions[key] = (embeddings, [r["category"] for r in rows])
        candidates = []
        best_score, selected = -1, None
        for c in settings["c_values"]:
            model = LogisticRegression(C=c, max_iter=settings["max_iter"], random_state=settings["seed"])
            model.fit(*partitions["fit_train"])
            validation_accuracy = float(model.score(*partitions["validation"]))
            if validation_accuracy > best_score:
                best_score, selected = validation_accuracy, c
            p = model.predict_proba(partitions["test"][0])
            val_p = model.predict_proba(partitions["validation"][0])
            row = {"c": c, "validation_accuracy": validation_accuracy,
                   "validation_metrics": reference_metrics(val_p, partitions["validation"][1], model.classes_),
                   "test_accuracy": float(model.score(*partitions["test"])),
                   "test_mean_confidence": float(p.max(axis=1).mean()),
                   "test_metrics": reference_metrics(p, partitions["test"][1], model.classes_),
                   "coefficient_l2_norm": float(np.linalg.norm(model.coef_)),
                   "iterations": model.n_iter_.tolist()}
            if c == saved[0]["raw_response"]["selected_c"]:
                old = np.array([[r["probabilities"][str(label)] for label in model.classes_] for r in saved])
                row["max_probability_difference_from_saved"] = float(abs(p - old).max())
                row["predicted_labels_match_saved"] = list(model.predict(partitions["test"][0])) == [r["predicted_label"] for r in saved]
            candidates.append(row)
        output["runs"].append({"run_id": run.name, "selected_c": selected,
                               "saved_selected_c": saved[0]["raw_response"]["selected_c"], "candidates": candidates})
        print(f"Replayed {run.name}: C={selected}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("artifacts/benchmark_runs/20260804T014841Z"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-arrow", type=Path)
    parser.add_argument("--test-arrow", type=Path)
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    replay_args = (args.train_arrow, args.test_arrow, args.model_path)
    if any(replay_args) and not all(replay_args):
        parser.error("Replay requires --train-arrow, --test-arrow and --model-path")
    runs = sorted(p.parent for p in (args.campaign / "runs").glob("*/predictions.jsonl"))
    if not runs:
        parser.error("No saved predictions found")
    report = {"campaign": str(args.campaign), "n_bins": 10, "runs": [audit_run(run) for run in runs]}
    if all(replay_args):
        report["replay"] = replay(args.campaign, *replay_args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Audited {len(runs)} runs; saved {args.output}")


if __name__ == "__main__":
    main()
