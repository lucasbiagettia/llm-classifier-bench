"""Resume Decisions through immutable normal runner segments, retaining refusals."""
import json

from llm_classifier_bench.classifiers.base import Prediction
from llm_classifier_bench.metrics.base import MetricResult
from llm_classifier_bench.metrics.evaluator import (
    DEFAULT_METRICS, evaluate_records, evaluation_record_from_mapping, results_as_dict,
)
from llm_classifier_bench.recovery import pending_examples
from llm_classifier_bench.runner import _validate_predictions
from run_budget_extension import read_json, write_json, require, ids, sha256


def segment_prefix(campaign_id, k):
    return f"{campaign_id}__seed42__n{k:02d}__openai-decisions-zero-shot__budget0-per_class"


def inspect_cell(root, campaign_id, cell, bundle, plan):
    """Read and validate every segment before deciding which requests remain."""
    pending = tuple(bundle.test)
    predictions, refusals, runs, costs, fingerprints = {}, {}, [], [], {}
    reference = plan["conditions"][str(cell["k"])]
    for run in sorted((root / "runs").glob(segment_prefix(campaign_id, cell["k"]) + "*")):
        config = read_json(run / "config.json")
        status = read_json(run / "status.json")
        if status["status"] == "dry_run":
            continue
        require(status["status"] in ("completed", "failed"), f"Unfinished segment; inspect before resuming: {run}")
        require(config["run_metadata"]["cell_id"] == cell["id"], "Resume cell identity changed")
        require(config["classifier"]["model"] == plan["decisions"]["requested_model"], "Resume model changed")
        require(config["dataset"]["classes"] == reference["classes"], "Resume definitions changed")
        require(config["dataset"]["test_sample_ids"] == ids(pending), "Resume segment test order/coverage changed")
        require(config["measurement"] == reference["measurement"], "Resume measurement changed")
        require(config["matched_budget"] == {"examples": 0, "unit": "per_class", "seed": 42,
                                             "validation_examples": 0}, "Resume budget changed")
        require(not config["dataset"]["fit_train_sample_ids"] and not config["dataset"]["validation_sample_ids"],
                "Resume unexpectedly consumed labels")
        metadata = read_json(run / "measurement.json")["runtime"]["classifier"]
        for key in ("endpoint", "requested_model", "timeout_s", "transport_max_retries"):
            require(metadata[key] == plan["decisions"][key], f"Resume transport changed: {key}")
        require(read_json(run / "pricing.json") == read_json(plan["pricing_path"]), "Resume pricing changed")
        remaining, saved = pending_examples(pending, [run / "predictions.jsonl"])
        require(list(saved) == ids(pending[:len(saved)]), "Persisted predictions are not a contiguous prefix")
        outputs = [Prediction(r["sample_id"], r["predicted_label"], r["confidence"], r["probabilities"],
                              r["adapter_latency_ms"]) for r in saved.values()]
        _validate_predictions(bundle, outputs, examples=pending[:len(saved)])
        ledger = [json.loads(line) for line in (run / "usage.jsonl").read_text().splitlines() if line.strip()]
        attempts = [e for e in ledger if e["kind"] == "api_attempt" and e["phase"] == "evaluation"]
        succeeded = [e for e in attempts if e["status"] == "success"]
        require([e["sample_id"] for e in succeeded] == list(saved), "Usage/predictions disagree; refusing ambiguous replay")
        report = read_json(run / "cost_report.json")
        for sample_id, row in saved.items():
            cost = report["per_successful_sample"][sample_id]
            predictions[sample_id] = {**row, "source_run_dir": str(run), "cost_usd": cost["cost_usd"],
                                      "cost_kind": cost["cost_kind"], "cost_reconciled": True}
        failed = [e for e in attempts if e["status"] == "failed"]
        if status["status"] == "failed":
            legacy_refusal = (status.get("error_type") == "ValueError"
                              and status.get("error_message") == "OpenAI Decisions refused classification")
            is_refusal = status.get("error_type") == "OpenAIDecisionsRefusalError" or legacy_refusal
            is_unavailable = (status.get("error_type") == "OpenAIDecisionsAPIError"
                              and status.get("error_message") == "OpenAI Decisions HTTP 503")
            require(is_refusal or is_unavailable,
                    f"Only recorded refusal or HTTP 503 failures can be resumed; inspect {run}")
            require(len(failed) == 1 and remaining and attempts == succeeded + failed,
                    "Missing or ambiguous failure evidence")
            event = failed[0]
            example = remaining[0]
            require(event["sample_id"] == example.sample_id
                    and event.get("provider") == "openai" and event.get("endpoint") == plan["decisions"]["endpoint"],
                    "Failure evidence does not match the pending example")
            if is_unavailable:
                require(event.get("http_status") == 503 and event.get("response_received") is True
                        and event.get("error_type") == "OpenAIDecisionsAPIError"
                        and event.get("decision_answer_type") is None,
                        "HTTP 503 failure is not corroborated by the usage ledger")
                # An explicit resume retries this still-pending input. Preserve the
                # failed segment and its cost report, including unknown usage.
            else:
                require(event.get("http_status") == 200
                        and (event.get("decision_answer_type") == "refusal" or legacy_refusal),
                        "Refusal evidence does not match the pending example")
                refusals[example.sample_id] = {"sample_id": example.sample_id, "input": example.text,
                    "gold_label": example.label, "status": "refused", "predicted_label": None,
                    "confidence": None, "probabilities": None, "request_id": event.get("request_id"),
                    "model": event.get("model"), "request_sha256": event.get("request_sha256"),
                    "usage": event.get("usage"), "source_run_dir": str(run), "usage_event_id": event["event_id"]}
                remaining = remaining[1:]
        else:
            require(not remaining and not failed, "Completed segment lacks full successful coverage")
        pending = remaining
        runs.append(str(run))
        costs.append(report)
        for path in run.iterdir():
            if path.is_file():
                fingerprints[str(path)] = sha256(path)
    return {"pending": pending, "predictions": predictions, "refusals": refusals,
            "runs": runs, "costs": costs, "evidence_sha256": fingerprints}


def next_segment_id(root, base):
    if not (root / "runs" / base).exists():
        return base
    index = 1
    while (root / "runs" / f"{base}__part{index:04d}").exists():
        index += 1
    return f"{base}__part{index:04d}"


def write_cell_artifacts(root, cell, bundle, state):
    """Keep successes and refusals distinct; never score only the accepted subset."""
    directory = root / "cells" / cell["id"]
    directory.mkdir(parents=True, exist_ok=True)
    good, refused = state["predictions"], state["refusals"]
    complete = not state["pending"]
    status = ("completed_with_refusals" if refused else "completed") if complete else "incomplete"
    coverage = {"status": status, "planned_examples": len(bundle.test), "planned_sample_ids": ids(bundle.test), "successful_examples": len(good),
                "refused_examples": len(refused), "pending_examples": len(state["pending"]),
                "attempted_examples": len(good) + len(refused), "runs": state["runs"],
                "refusal_policy": "record_and_continue_without_retry", "evidence_sha256": state["evidence_sha256"]}
    write_json(directory / "coverage.json", coverage)
    ordered = [good[e.sample_id] for e in bundle.test if e.sample_id in good]
    outcomes = [{**good[e.sample_id], "status": "classified"} if e.sample_id in good else refused[e.sample_id]
                for e in bundle.test if e.sample_id in good or e.sample_id in refused]
    for name, records in (("predictions.jsonl", ordered), ("outcomes.jsonl", outcomes),
                          ("refusals.jsonl", [r for r in outcomes if r["status"] == "refused"])):
        temporary = directory / (name + ".tmp")
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records))
        temporary.replace(directory / name)
    metadata = {"planned_examples": len(bundle.test), "successful_examples": len(good),
                "refused_examples": len(refused), "pending_examples": len(state["pending"])}
    if complete and not refused:
        metrics = results_as_dict(evaluate_records([evaluation_record_from_mapping(r) for r in ordered]))
    else:
        metrics = {m.name: MetricResult.unavailable(name=m.name, reason="Full-cohort predictions unavailable; refusals/pending samples are retained",
                                                   metadata=metadata).as_dict() for m in DEFAULT_METRICS}
    # Every segment uses the existing cost ledger, including its refused attempt.
    reports = state["costs"]
    total = sum(r["total_cost_usd"] for r in reports) if reports and all(r["total_cost_usd"] is not None for r in reports) else None
    for name, value in (("total_cost_usd", total), ("cost_per_1000_usd", total * 1000 / len(good) if total is not None and good else None)):
        metrics[name] = MetricResult(name, value, {**metadata, "scope": "all recorded attempts including refusals",
                                                  "source_runs": state["runs"]}).as_dict()
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "cost_report.json", {**metadata, "total_cost_usd": total,
        "known_cost_subtotal_usd": sum(r["known_cost_subtotal_usd"] for r in reports),
        "scope": "sum of existing segment cost reports including refusals", "source_runs": state["runs"]})
    return directory, coverage, metrics
