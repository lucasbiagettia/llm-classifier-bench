"""Continue hosted inputs while retaining historical predictions and separate ledgers."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from llm_classifier_bench.classifiers.base import Prediction
from llm_classifier_bench.metrics.evaluator import DEFAULT_METRICS, evaluate_jsonl, write_results_json
from llm_classifier_bench.recovery import pending_examples
from llm_classifier_bench.runner import (
    BenchmarkRunResult, _validate_predictions, _write_run_config, run_benchmark,
)


def request_digest(bundle, classifier, example):
    if hasattr(classifier, "_encoded"):  # Jev's frozen serialization.
        encoded = classifier._encoded(classifier._payload(bundle.classes, example.text))
    else:  # OpenAI Decisions' frozen serialization; no prepare/network operation.
        previous = classifier._classes
        try:
            classifier._classes = bundle.classes
            encoded = json.dumps(classifier._payload(example.as_input()), ensure_ascii=False,
                                 sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        finally:
            classifier._classes = previous
    return hashlib.sha256(encoded).hexdigest()


def validated_pending(bundle, classifier, source):
    """Check saved requests, including refusals captured by the older strict adapter."""
    source = Path(source)
    pending, old = pending_examples(bundle.test, [source / "predictions.jsonl"])
    expected = {e.sample_id: e for e in bundle.test}
    ledger = source / "usage.jsonl"
    if ledger.exists() and getattr(classifier, "refusal_policy", None) == "null":
        entries = [json.loads(line) for line in ledger.read_text().splitlines()]
        for attempt in entries:
            if not (attempt.get("kind") == "api_attempt" and attempt.get("phase") == "evaluation"
                    and attempt.get("decision_answer_type") == "refusal" and attempt.get("http_status") == 200
                    and attempt.get("error_type") == "OpenAIDecisionsRefusalError"):
                continue
            key = attempt["sample_id"]
            if key in old or key not in expected:
                raise ValueError("Duplicate or unexpected historical refusal")
            if attempt.get("endpoint") != classifier.endpoint:
                raise ValueError("Persisted refusal endpoint differs")
            observations = [e for e in entries if e.get("kind") == "inference_call"
                            and e.get("observation_index") == attempt["observation_index"]
                            and e.get("sample_ids") == [key]]
            if len(observations) != 1:
                raise ValueError("Historical refusal lacks a unique single-example timing")
            observation = observations[0]
            example = expected[key]
            old[key] = {"dataset": bundle.name, "classifier": classifier.name, "sample_id": key,
                        "input": example.text, "gold_label": example.label, "predicted_label": None,
                        "prediction_status": "refusal", "correct": False, "confidence": None,
                        "probabilities": None, "latency_ms": observation["elapsed_ms"],
                        "adapter_latency_ms": observation["elapsed_ms"], "model": attempt["model"],
                        "request_id": attempt.get("request_id"), "cost_usd": None,
                        "cost_reconciled": False, "usage_event_ids": [attempt["event_id"]],
                        "raw_response": {"refusal": True, "request_sha256": attempt["request_sha256"],
                                         "recovered_from_usage": str(ledger.resolve())}}
    for key, row in old.items():
        example = expected[key]
        digest = request_digest(bundle, classifier, example)
        if row["raw_response"].get("request_sha256") != digest:
            raise ValueError("Persisted hosted request differs from current frozen request")
        prediction = Prediction(key, row["predicted_label"], row["confidence"], row["probabilities"] or None,
                                row["adapter_latency_ms"], model=row["model"], raw_response=row["raw_response"])
        _validate_predictions(bundle, [prediction], examples=[example])
        if row["model"] != classifier.model:
            raise ValueError("Persisted hosted resolved model differs")
    return tuple(e for e in bundle.test if e.sample_id not in old), old


def continue_hosted(bundle, classifier, config, source, static_dataset):
    """Quality covers the full cohort; timing/cost evidence stays in its original run."""
    source = Path(source)
    pending, old = validated_pending(bundle, classifier, source)
    root = Path(config.output_root) / config.run_id
    root.mkdir(parents=True, exist_ok=False)
    def write(name, data):
        (root / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    _write_run_config(root / "config.json", run_id=config.run_id, dataset=bundle,
                      classifier=classifier, config=config, fit_train=bundle.train,
                      validation=bundle.validation, class_definitions_metadata={"source": "dataset_bundle", "profile": None})
    write("recovery.json", {"source_run": str(source.resolve()), "reused_examples": len(old),
          "pending_examples": len(pending), "inference_policy": classifier.inference_metadata(),
          "measurement_note": "Full-cohort quality only here. Costs, failed attempts and latency remain in source_run and continuations/pending; never treat historical usage as new spending."})
    def save_predictions(rows):
        (root / "predictions.jsonl").write_text("".join(
            json.dumps(rows[e.sample_id], ensure_ascii=False) + "\n" for e in bundle.test if e.sample_id in rows))
    save_predictions(old)
    write("status.json", {"status": "running", "run_id": config.run_id})
    succeeded = not pending
    try:
        if pending:
            run_benchmark(static_dataset(replace(bundle, test=pending)), classifier,
                          replace(config, output_root=root / "continuations", run_id="pending"))
        succeeded = True
    finally:
        _, combined = pending_examples(bundle.test, [root / "predictions.jsonl",
                                                   root / "continuations/pending/predictions.jsonl"])
        save_predictions(combined)
        complete = succeeded and len(combined) == len(bundle.test)
        write("status.json", {"status": "completed" if complete else "failed", "run_id": config.run_id,
                              "persisted_predictions": len(combined), "example_count": len(combined)})
        write("coverage.json", {"status": "completed" if complete else "failed",
                                "planned_examples": len(bundle.test),
                                "planned_sample_ids": [e.sample_id for e in bundle.test],
                                "null_predictions": sum(r["predicted_label"] is None for r in combined.values())})
        if complete:
            quality = DEFAULT_METRICS[:6]
            write_results_json(root / "metrics.json", evaluate_jsonl(root / "predictions.jsonl", metrics=quality))
    return BenchmarkRunResult(config.run_id, root, root / "config.json", root / "predictions.jsonl",
                              root / "status.json", root / "metrics.json", len(bundle.test))
