"""Audit v2 evidence and measure missing supervised budgets; offline plan by default."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unicodedata

from llm_classifier_bench.budgets import MatchedBudgetConfig, select_matched_pools
from llm_classifier_bench.class_definitions import load_class_definition_profile
from llm_classifier_bench.classifiers import (
    BertClassifier, SentenceTransformerLogisticClassifier, TfidfLogisticClassifier,
)
from llm_classifier_bench.classifiers.emissary import (
    EmissaryClassifier, EmissaryClient, serialize_classification_dataset,
    _redact_download_urls,
)
from llm_classifier_bench.config import (
    BertTrainingConfig, EmissaryTrainingConfig, SentenceTransformerTrainingConfig,
    TfidfTrainingConfig,
)
from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.datasets import DatasetBundle
from llm_classifier_bench.datasets.selection import (
    LabeledSelectionConfig, select_labeled_examples, validate_partition_disjointness,
)
from llm_classifier_bench.measurement import MeasurementConfig
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark, split_train_validation
from run_banking77_scaling_benchmark_v2 import StaticDataset, sample_per_class


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("tfidf", "sentence-transformer", "bert", "emissary-qwen")
COUNTS = (5, 10, 15, 20)
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ids(examples):
    return [e.sample_id for e in examples]


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def audit_partitions(fit, reserved, test):
    validate_partition_disjointness(fit, reserved, test)
    # Also catch copies with differences only in case, Unicode or whitespace.
    seen = set()
    for partition in (fit, reserved, test):
        content = {normalized_text(e.text) for e in partition}
        require(not seen.intersection(content), "Partitions contain normalized text overlap")
        seen.update(content)
    return {"fit": len(fit), "reserved_validation": len(reserved), "test": len(test),
            "overlapping_ids": 0, "overlapping_exact_texts": 0,
            "overlapping_normalized_texts": 0}


def load_source(directory, manifest):
    splits = {}
    for split in ("train", "test"):
        path = directory / f"{split}.csv"
        require(sha256(path) == manifest["source"]["files"][path.name]["sha256"],
                f"Frozen source changed: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            splits[split] = tuple(LabeledExample(f"banking77:{split}:{i}", row["text"], row["category"])
                                  for i, row in enumerate(csv.DictReader(stream)))
    return splits


def build_bundle(source, classes):
    names = [c.name for c in classes]
    return DatasetBundle("banking77", tuple(classes),
        sample_per_class(source["train"], names, per_class=125, seed=42, purpose="train"),
        sample_per_class(source["test"], names, per_class=40, seed=42, purpose="test"))


def pools(bundle, budget):
    fit, reserved = split_train_validation(bundle.train, validation_fraction=.2, seed=42)
    selected, validation, metadata = select_matched_pools(
        fit, reserved, bundle.classes, MatchedBudgetConfig(budget, "per_class", 42, 0))
    require(not validation, "Unexpected consumed validation labels")
    return selected, reserved, metadata


def provider_payload(selected, classes, budget):
    # The adapter orders the shared pool again. Hash the actual upload order.
    ordered, _ = select_labeled_examples(selected, classes, LabeledSelectionConfig(budget, "per_class", 42))
    require(set(ids(ordered)) == set(ids(selected)), "Provider changed the shared training pool")
    return serialize_classification_dataset(ordered, classes)


def verify_predictions(path, test):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require([r["sample_id"] for r in rows] == ids(test), f"Test coverage/order changed: {path}")
    require(all((r["input"], r["gold_label"]) == (e.text, e.label) for r, e in zip(rows, test)),
            f"Test text/label changed: {path}")


def verify_selection(saved, selected):
    rows = saved["selection"]["selected_examples"]
    expected = [(e.sample_id, e.text, e.label) for e in selected]
    require([(r["sample_id"], r["text"], r["label"]) for r in rows] == expected,
            "Saved training selection differs from reconstructed source")


def prepare_plan(args):
    """Read-only audit, including the historical submitted Qwen payload identity."""
    manifest = read_json(args.reference_manifest)
    results = read_json(args.reference_results)
    fingerprints = {}

    def record(path):
        fingerprints[str(path)] = sha256(path)
        return read_json(path)

    record(args.reference_manifest)
    record(args.reference_results)
    source = load_source(args.source_dir, manifest)
    for split in ("train", "test"):
        fingerprints[str(args.source_dir / f"{split}.csv")] = sha256(args.source_dir / f"{split}.csv")
    definitions_path = Path(manifest["definitions"]["path"])
    require(sha256(definitions_path) == manifest["definitions"]["sha256"], "Definitions changed")
    fingerprints[str(definitions_path)] = sha256(definitions_path)
    profile = load_class_definition_profile(definitions_path)
    definitions = {c.name: c for c in profile.profile.classes}
    audit, references, conditions = [], [], {}
    for k in COUNTS:
        classes = tuple(definitions[n] for n in manifest["label_sets"][str(k)])
        bundle = build_bundle(source, classes)
        by_budget = {b: pools(bundle, b)[0] for b in (20, 50, 100)}
        require(by_budget[20] == by_budget[50][:20*k] == by_budget[100][:20*k], "20-shot pool not nested")
        require(by_budget[50] == by_budget[100][:50*k], "50-shot pool not nested")
        reserved = pools(bundle, 100)[1]
        audit.append({"k": k, **audit_partitions(by_budget[100], reserved, bundle.test)})
        # Check against the entire official test split, not only the selected K labels.
        audit_partitions(by_budget[100], reserved, source["test"])
        conditions[str(k)] = {"classes": [asdict(c) for c in classes],
            "test_sample_ids": ids(bundle.test), "reserved_validation_sample_ids": ids(reserved),
            "training_sample_ids": {str(b): ids(examples) for b, examples in by_budget.items()}}
        for method in METHODS:
            budget = 100 if method == "emissary-qwen" else 50
            candidates = [r for r in results["conditions"] if r["method"] == method and r["k"] == k and r["fit_per_class"] == budget]
            require(len(candidates) == 1, f"Missing/duplicate historical condition: {method}, K={k}")
            reference = candidates[0]
            require(reference["status"] == "completed" and reference["fit_per_class"] == budget,
                    "Historical reference is incomplete or uses a different budget")
            evidence = Path(reference["evidence"])
            run = evidence / "runs/new"
            config = record(run / "config.json")
            fit = record(run / "fit_metadata.json")
            selection = record(run / "labeled_budget.json")
            require(record(run / "status.json")["status"] == "completed", "Historical run incomplete")
            require(config["dataset"]["classes"] == conditions[str(k)]["classes"], "Historical definitions differ")
            require(config["dataset"]["test_sample_ids"] == ids(bundle.test), "Historical test differs")
            require(config["dataset"]["fit_train_sample_ids"] == ids(by_budget[budget]), "Historical fit IDs differ")
            require(config["dataset"]["validation_sample_ids"] == [], "Historical validation was consumed")
            require(fit["training_examples_used"] == budget*k and fit["validation_examples_used"] == 0,
                    "Historical label consumption differs")
            verify_selection(selection, by_budget[budget])
            predictions = evidence / "predictions.jsonl"
            verify_predictions(predictions, bundle.test)
            fingerprints[str(predictions)] = sha256(predictions)
            if config.get("pricing_path"):
                record(Path(config["pricing_path"]))
            item = {"method": method, "k": k, "budget": budget, "evidence": str(evidence),
                    "config": config["classifier"], "measurement": config["measurement"],
                    "pricing_path": config.get("pricing_path")}
            if method == "emissary-qwen":
                verify_selection(fit, by_budget[budget])
                payload = serialize_classification_dataset(by_budget[budget], classes)
                require(hashlib.sha256(payload).hexdigest() == fit["dataset_payload_sha256"],
                        "Historical Qwen submitted payload hash mismatch")
                uploads = [Path(root) / f"k{k}" / "train.jsonl" for root in manifest.get("raw_evidence_roots", [])]
                uploads = [p for p in uploads if p.is_file() and sha256(p) == fit["dataset_payload_sha256"]]
                require(bool(uploads), "Original historical Qwen upload is missing or changed")
                for path in uploads:
                    fingerprints[str(path)] = sha256(path)
                remote = fit["training_response"]
                require(remote["base_model"] == "Qwen3-4B-Base" and not remote.get("test_dataset")
                        and not remote.get("base_fine_tuned_model"), "Unexpected historical Qwen training inputs")
                item.update(project_id=fit["project_id"], parameters=remote["hyper_parameters"],
                            historical_dataset_sha256=fit["dataset_payload_sha256"])
            references.append(item)
    cells = [{"id": f"{method}__s42__k{k}__b{b}", "method": method, "k": k, "budget": b,
              "test_examples": 40*k}
             for method in METHODS for b in ((20,) if method == "emissary-qwen" else (20, 100)) for k in COUNTS]
    return {"schema_version": 1, "seed": 42, "budget_unit": "per_class", "cells": cells,
            "source_dir": str(args.source_dir), "source": manifest["source"],
            "conditions": conditions, "references": references, "input_sha256": fingerprints,
            "audit": {"status": "passed", "historical_supervised_conditions_checked": len(references),
                      "conditions": audit, "checked_against_entire_official_test": True,
                      "normalization": "NFKC + casefold + collapsed whitespace",
                      "scope": "Benchmark fit/reserved/test data and recorded Qwen payload; not base-model pretraining or provider internals"},
            "environment": ENVIRONMENT, "validation_labels_consumed": 0,
            "new_training_jobs": len(COUNTS), "new_deployments": len(COUNTS),
            "new_predictions": sum(c["test_examples"] for c in cells),
            "retries": 0, "warmup": 0, "local_timeout_s": args.local_timeout,
            "remote_timeout_s": args.remote_timeout}


class RecordedTrainingClient(EmissaryClient):
    """Record each mutating request before sending it; never silently resubmit."""

    def __init__(self, directory, parameters, expected_payload_sha256, **kwargs):
        super().__init__(**kwargs)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.parameters = parameters
        self.expected_payload_sha256 = expected_payload_sha256

    def once(self, operation, payload, call):
        with (self.directory / f"{operation}_intent.json").open("x") as stream:
            json.dump(payload, stream, indent=2)
        response = call()
        write_json(self.directory / f"{operation}_response.json", _redact_download_urls(response))
        return response

    def upload_dataset(self, **kwargs):
        digest = hashlib.sha256(kwargs["content"]).hexdigest()
        require(digest == self.expected_payload_sha256, "Upload differs from audited 20-shot payload")
        (self.directory / "train.jsonl").write_bytes(kwargs["content"])
        payload = {k: v for k, v in kwargs.items() if k != "content"}
        return self.once("dataset", {**payload, "sha256": digest},
                         lambda: super(RecordedTrainingClient, self).upload_dataset(**kwargs))

    def create_training_job(self, **kwargs):
        require(all(self.parameters.get(k) == v for k, v in kwargs["parameters"].items()),
                "Qwen training parameters differ from historical reference")
        kwargs["parameters"] = self.parameters
        return self.once("training", kwargs,
                         lambda: super(RecordedTrainingClient, self).create_training_job(**kwargs))

    def create_deployment(self, **kwargs):
        return self.once("deployment", kwargs,
                         lambda: super(RecordedTrainingClient, self).create_deployment(**kwargs))


def classifier_for(cell, reference, output, payload_sha256):
    method = cell["method"]
    config = reference["config"]
    training = dict(config["training"])
    for key in ("c_values", "ngram_range"):
        if key in training:
            training[key] = tuple(training[key])
    if method == "tfidf":
        return TfidfLogisticClassifier(training=TfidfTrainingConfig(**training))
    if method == "bert":
        return BertClassifier(model=config["model"], training=BertTrainingConfig(**training))
    if method == "sentence-transformer":
        return SentenceTransformerLogisticClassifier(model=config["model"],
            training=SentenceTransformerTrainingConfig(**training))
    parameters = reference["parameters"]
    # A fresh job always starts from the pretrained base. Historical IDs are not continued.
    training = EmissaryTrainingConfig(shots=20, shot_unit="per_class", selection_seed=42,
        mechanism="project_fine_tuning", project_id=reference["project_id"], base_model="Qwen3-4B-Base",
        **{k: parameters[k] for k in ("num_train_epochs", "learning_rate", "per_device_train_batch_size",
                                      "per_device_eval_batch_size", "dynamic_loss")},
        training_timeout_s=7200, deployment_timeout_s=1800)
    return EmissaryClassifier(training=training,
        client=RecordedTrainingClient(output / "remote", parameters, payload_sha256, read_timeout_s=120),
        experiment_name=f"budget20-k{cell['k']}-{time.time_ns()}")


def worker(plan, cell, root):
    for path, digest in plan["input_sha256"].items():
        require(sha256(path) == digest, f"Frozen input changed: {path}")
    output = root / "cells" / cell["id"]
    output.mkdir(parents=True, exist_ok=False)
    condition = plan["conditions"][str(cell["k"])]
    source = load_source(Path(plan["source_dir"]), plan)
    bundle = build_bundle(source, [ClassDefinition(**c) for c in condition["classes"]])
    selected, reserved, _ = pools(bundle, cell["budget"])
    audit_partitions(selected, reserved, source["test"])
    require(ids(selected) == condition["training_sample_ids"][str(cell["budget"])], "Fit pool changed")
    require(ids(bundle.test) == condition["test_sample_ids"], "Test pool changed")
    payload = provider_payload(selected, bundle.classes, cell["budget"])
    reference = next(r for r in plan["references"] if (r["method"], r["k"]) == (cell["method"], cell["k"]))
    classifier = classifier_for(cell, reference, output, hashlib.sha256(payload).hexdigest())
    write_json(output / "condition.json", {**cell, **condition, "fit_sample_ids": ids(selected)})
    result = run_benchmark(StaticDataset(bundle), classifier, BenchmarkRunConfig(
        output_root=output, run_id="run", validation_fraction=.2, split_seed=42,
        matched_budget=MatchedBudgetConfig(cell["budget"], "per_class", 42, 0),
        measurement=MeasurementConfig(**reference["measurement"]),
        pricing_path=Path(reference["pricing_path"]) if reference["pricing_path"] else None,
        metadata={"budget_extension": True, "cell": cell, "historical_reference": reference["evidence"],
                  "source": plan["source"], "manifest_sha256": sha256(root / "manifest.json")}))
    require(read_json(result.status_path)["status"] == "completed", "Run did not complete")
    verify_predictions(result.predictions_path, bundle.test)


def run_process(command, *, env, log, timeout_s, progress):
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    start = time.monotonic()
    try:
        while True:
            try:
                return process.wait(timeout=min(15, max(.01, timeout_s - (time.monotonic() - start)))), False
            except subprocess.TimeoutExpired:
                progress()
                if time.monotonic() - start >= timeout_s:
                    return None, True
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def execute(plan, args):
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    with (root / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = root / "manifest.json"
        if frozen.exists():
            require(read_json(frozen) == plan, "Existing execution has a different plan; inspect it before proceeding")
        else:
            write_json(frozen, plan)
            write_json(root / "provenance.json", {"python": sys.version,
                "code_sha256": {str(p): sha256(p) for folder in (ROOT / "src", ROOT / "scripts")
                                for p in folder.rglob("*.py") if "local" not in p.parts}})
        write_json(root / "audit.json", plan["audit"])
        state_path = root / "summary.json"
        state = read_json(state_path) if state_path.exists() else {}
        env = {**os.environ, **plan["environment"], "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
        selected = [c for c in plan["cells"] if not args.only or c["method"] in args.only]
        # Check both cached models before starting any cell; no downloads or training here.
        for ref in plan["references"]:
            if ref["method"] in {c["method"] for c in selected} & {"bert", "sentence-transformer"}:
                require(Path(ref["config"]["model"]).is_dir(), f"Missing historical model snapshot: {ref['config']['model']}")
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        for cell in selected:
            key = cell["id"]
            output = root / "cells" / key
            if key in state or output.exists():
                print(f"SKIP {key}: previous attempt retained (no automatic retry)", flush=True)
                continue
            state[key] = {"status": "running", "cell": cell}
            write_json(state_path, state)
            start = time.monotonic()
            print(f"START {key}", flush=True)

            def progress():
                path = output / "run/status.json"
                try:
                    status = read_json(path) if path.exists() else {}
                except json.JSONDecodeError:
                    status = {}  # A status write can be observed between truncate and flush.
                predictions = output / "run/predictions.jsonl"
                count = len(predictions.read_text().splitlines()) if predictions.exists() else 0
                print(f"PROGRESS {key}: {status.get('stage', 'loading')} {count}/{cell['test_examples']} "
                      f"elapsed={time.monotonic()-start:.0f}s", flush=True)

            try:
                with (logs / f"{key}.log").open("x") as log:
                    code, timeout = run_process([sys.executable, "-u", str(Path(__file__).resolve()),
                        "--execute", "--output-root", str(root), "--worker", key], env=env, log=log,
                        timeout_s=plan["remote_timeout_s"] if cell["method"] == "emissary-qwen" else plan["local_timeout_s"],
                        progress=progress)
                status_path = output / "run/status.json"
                status = read_json(status_path) if status_path.exists() else {}
                complete = code == 0 and not timeout and status.get("status") == "completed"
                state[key] = {"cell": cell, "status": "completed" if complete else "failed",
                              "returncode": code, "timeout": timeout, "evidence": str(output / "run")}
            except BaseException as exc:
                state[key] = {"cell": cell, "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                              "error": str(exc)}
                if not isinstance(exc, Exception):
                    raise
            finally:
                state[key]["wall_seconds"] = time.monotonic() - start
                write_json(state_path, state)
            print(f"END {key}: {state[key]['status']}", flush=True)
        return 0 if all(state.get(c["id"], {}).get("status") == "completed" for c in selected) else 2


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Interrupted by signal {signum}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Train and measure; otherwise audit/plan offline")
    parser.add_argument("--only", nargs="+", choices=METHODS)
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/v2_source"))
    parser.add_argument("--reference-manifest", type=Path, default=Path("reports/v2/manifest.json"))
    parser.add_argument("--reference-results", type=Path, default=Path("reports/v2/results.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/v2_budget_extension"))
    parser.add_argument("--local-timeout", type=int, default=1800, help="Seconds per local cell, including BERT 100/class")
    parser.add_argument("--remote-timeout", type=int, default=14400, help="Seconds for train, deploy and evaluation per Qwen cell")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    require(args.local_timeout > 0 and args.remote_timeout > 0, "Timeouts must be positive")
    if args.worker:
        require(args.execute, "Worker requires --execute")
        plan = read_json(args.output_root / "manifest.json")
        worker(plan, next(c for c in plan["cells"] if c["id"] == args.worker), args.output_root)
        return 0
    plan = prepare_plan(args)
    selected = [c for c in plan["cells"] if not args.only or c["method"] in args.only]
    print(f"AUDIT PASSED: {len(plan['references'])} historical supervised conditions; no ID/exact/normalized text overlap.")
    print(f"PLAN: {len(selected)} new conditions, {sum(c['test_examples'] for c in selected)} predictions, "
          f"{sum(c['method'] == 'emissary-qwen' for c in selected)} new Qwen training jobs; budgets per class.")
    if not args.execute:
        # Keep inspection separate from the live manifest and results.
        write_json(args.output_root / "plan.json", plan)
        write_json(args.output_root / "audit.json", plan["audit"])
        print(f"Offline only. Plan: {args.output_root / 'plan.json'}")
        return 0
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    return execute(plan, args)


if __name__ == "__main__":
    raise SystemExit(main())
