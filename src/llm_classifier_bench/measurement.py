"""Persist inference-call observations and aggregate them without running models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns
from typing import Any, Callable, Sequence

from .metrics.operational import MIN_P99_OBSERVATIONS, percentile


@dataclass(frozen=True, slots=True)
class MeasurementConfig:
    batch_size: int = 1
    warmup_examples: int = 0
    client_location: str | None = None
    cache_condition: str = "uncontrolled; provider cache hits not observed"
    hardware_profile: str | None = None

    def __post_init__(self):
        for name, minimum in (("batch_size", 1), ("warmup_examples", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not self.cache_condition.strip():
            raise ValueError("cache_condition cannot be empty")
        if self.hardware_profile is not None and (not isinstance(self.hardware_profile, str) or not self.hardware_profile.strip()):
            raise ValueError("hardware_profile cannot be empty")


def runtime_metadata(classifier) -> dict[str, Any]:
    """Inspect local environment only; do not import/load torch or query an API."""
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "sentence-transformers", "openai", "requests"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    cpu = platform.processor() or None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    torch = sys.modules.get("torch")
    hook = getattr(classifier, "inference_metadata", None)
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu_model": cpu, "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM")},
        "torch_threads": torch.get_num_threads() if torch is not None else None,
        "gpu_devices": ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                        if torch is not None and torch.cuda.is_initialized() else []),
        "library_versions": versions,
        "classifier": hook() if callable(hook) else {
            "backend": "unknown", "batch_execution": "unknown",
            "transport_retries": "unknown; injected/custom classifier",
        },
    }


def _latency_summary(rows) -> dict[str, Any]:
    values = [r["elapsed_ms"] for r in rows]
    return {
        "observation_count": len(values),
        "mean_ms": sum(values) / len(values) if values else None,
        "p50_ms": percentile(values, .5) if values else None,
        "p95_ms": percentile(values, .95) if values else None,
        "p99_ms": percentile(values, .99) if len(values) >= MIN_P99_OBSERVATIONS else None,
        "p99_minimum_observations": MIN_P99_OBSERVATIONS,
        "p99_unavailable_reason": None if len(values) >= MIN_P99_OBSERVATIONS else "insufficient observations",
    }


def aggregate_timings(rows: Sequence[dict], metadata: dict) -> dict:
    """Call latency is never confused with elapsed/batch_size or server latency."""
    previous_end = 0
    for row in rows:
        if not math.isfinite(row["elapsed_ms"]) or row["elapsed_ms"] < 0:
            raise ValueError("Timing durations must be finite and nonnegative")
        if row["end_offset_ns"] < row["start_offset_ns"]:
            raise ValueError("Timing end must follow start")
        if row["start_offset_ns"] < previous_end:
            raise ValueError("Sequential observations must not overlap")
        previous_end = row["end_offset_ns"]
        if not math.isclose(row["elapsed_ms"], (row["end_offset_ns"] - row["start_offset_ns"]) / 1e6,
                            abs_tol=1e-9, rel_tol=1e-12):
            raise ValueError("Timing duration does not match recorded clock offsets")
        if row["status"] not in {"success", "exception", "invalid_output"}:
            raise ValueError("Unknown timing status")
        if row["kind"] == "inference" and (type(row["batch_size"]) is not int or row["batch_size"] < 1):
            raise ValueError("Inference batch_size must be positive")
        if row["kind"] == "inference" and row["batch_size"] != len(row["sample_ids"]):
            raise ValueError("Inference batch_size does not match sample IDs")
    measured = [r for r in rows if r["kind"] == "inference" and r["phase"] == "evaluation"]
    successes = [r for r in measured if r["status"] == "success"]
    failures = [r for r in measured if r["status"] != "success"]
    attempted = sum(r["batch_size"] for r in measured)
    if attempted > metadata["planned_test_examples"]:
        raise ValueError("More attempted examples than planned")
    completed = sum(r["batch_size"] for r in successes)
    span_ms = ((max(r["end_offset_ns"] for r in measured) - min(r["start_offset_ns"] for r in measured)) / 1e6
               if measured else 0.0)
    active_ms = sum(r["elapsed_ms"] for r in measured)
    return {
        "schema_version": 1, "measurement": metadata,
        "preparation": [r for r in rows if r["kind"] == "stage"],
        "warmup": _latency_summary([r for r in rows if r.get("phase") == "warmup"]),
        "warmup_attempted_examples": sum(r["batch_size"] for r in rows if r.get("phase") == "warmup"),
        "warmup_failed_calls": sum(r.get("phase") == "warmup" and r["status"] != "success" for r in rows),
        "evaluation": {
            "phase_label": "after_explicit_warmup" if metadata["config"]["warmup_examples"] else "unwarmed; cold-start may be included",
            "attempted_examples": attempted, "successful_examples": completed,
            "unattempted_examples": metadata["planned_test_examples"] - attempted,
            "failed_calls": len(failures), "failed_call_fraction": len(failures) / len(measured) if measured else None,
            "unsuccessful_example_fraction": (attempted - completed) / attempted if attempted else None,
            "successful_call_latency": _latency_summary(successes),
            "failed_call_latency": _latency_summary(failures),
            "successful_call_latency_by_batch_size": {
                str(size): _latency_summary([r for r in successes if r["batch_size"] == size])
                for size in sorted({r["batch_size"] for r in successes})
            },
            "window_wall_ms": span_ms, "active_call_ms": active_ms,
            "throughput_successful_examples_per_second": completed * 1000 / span_ms if span_ms else None,
            "active_call_throughput_successful_examples_per_second": completed * 1000 / active_ms if active_ms else None,
            "throughput_denominator": "first evaluation call start to last call end, including failures and inter-call validation/logging",
            "amortized_successful_call_ms_per_example": sum(r["elapsed_ms"] for r in successes) / completed if completed else None,
        },
    }


def regenerate_report(run_dir: Path) -> dict:
    metadata = json.loads((run_dir / "measurement.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "timings.jsonl").read_text().splitlines() if line.strip()]
    return aggregate_timings(rows, metadata)


class TimingRecorder:
    """Small runner-owned recorder; no retries, scheduling, or model abstraction."""

    def __init__(self, run_dir: Path, config: MeasurementConfig, classifier, planned: int, observation_sink=None):
        self.run_dir, self.config, self.classifier = run_dir, config, classifier
        self.origin = perf_counter_ns()
        self.rows: list[dict] = []
        self.observation_sink = observation_sink
        self.metadata = {
            "schema_version": 1, "config": asdict(config), "planned_test_examples": planned,
            "classifier_name": classifier.name, "model": getattr(classifier, "model", None),
            "concurrency": 1, "application_retries": 0,
            "boundary": "classifier.predict(inputs) entry to normalized Prediction list return; includes adapter preprocessing and postprocessing",
            "excluded": ["dataset loading", "classifier construction before runner", "prepare", "fit including model loading", "warmup", "runner output validation and artifact writing"],
            "construction_time_ms": None,
            "construction_time_reason": "classifier is constructed by caller before run_benchmark",
            "clock": "perf_counter_ns; process-local monotonic offsets",
            "run_status": "running", "runtime": runtime_metadata(classifier),
        }
        (run_dir / "timings.jsonl").touch(exist_ok=False)
        self._save_metadata()

    def _save_metadata(self):
        (self.run_dir / "measurement.json").write_text(json.dumps(self.metadata, indent=2, allow_nan=False) + "\n")

    def measure(self, operation: Callable, *, kind: str, phase: str,
                sample_ids: Sequence[str] = (), validate: Callable | None = None):
        row = {"observation_index": len(self.rows), "kind": kind, "phase": phase,
               "sample_ids": list(sample_ids), "batch_size": len(sample_ids),
               "application_attempt": 1, "status": "success"}
        ended = None
        started = perf_counter_ns()
        try:
            result = operation()
            ended = perf_counter_ns()
            if validate is not None:
                try:
                    validate(result)
                except Exception:
                    row["status"] = "invalid_output"
                    raise
            return result, row
        except BaseException as exc:
            if row["status"] == "success":
                row["status"] = "exception"
            row["error_type"] = type(exc).__name__
            # Do not persist exception messages: provider errors can contain input/URLs.
            raise
        finally:
            if ended is None:
                ended = perf_counter_ns()
            row.update(start_offset_ns=started - self.origin, end_offset_ns=ended - self.origin,
                       elapsed_ms=(ended - started) / 1e6)
            if kind == "inference":
                row["amortized_ms_per_example"] = row["elapsed_ms"] / len(sample_ids)
            self.rows.append(row)
            with (self.run_dir / "timings.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            if self.observation_sink is not None:
                self.observation_sink(row)

    def finish(self, status: str):
        self.metadata["run_status"] = status
        self.metadata["runtime"] = runtime_metadata(self.classifier)
        self._save_metadata()
        report = regenerate_report(self.run_dir)
        (self.run_dir / "operational_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def add_measurement_arguments(parser):
    parser.add_argument("--inference-batch-size", type=int, default=1,
                        help="Inputs per predict call; existing adapters loop sequentially internally.")
    parser.add_argument("--warmup-examples", type=int, default=0,
                        help="Extra inference on fit examples before evaluation; hosted calls may be billed.")
    parser.add_argument("--client-location", default=None)
    parser.add_argument("--cache-condition", default=MeasurementConfig().cache_condition)
    parser.add_argument("--hardware-profile", default=None,
                        help="Operator-declared identity of the whole measured allocation; matches a measured_hardware rate card.")


def measurement_from_args(args):
    return MeasurementConfig(batch_size=args.inference_batch_size, warmup_examples=args.warmup_examples,
                             client_location=args.client_location, cache_condition=args.cache_condition,
                             hardware_profile=args.hardware_profile)
