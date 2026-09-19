"""Inference usage ledger and explicit, replayable monetary estimates."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Mapping


def amount(value) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("Money and runtime must be finite nonnegative numbers")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid monetary value") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("Money and runtime must be finite nonnegative numbers")
    return result


def load_pricing(path: Path | None) -> dict:
    card = json.loads(Path(path).read_text()) if path is not None else {
        "schema_version": 1, "currency": "USD", "api_rates": [], "local_hardware": None,
    }
    if card.get("schema_version") != 1 or card.get("currency") != "USD":
        raise ValueError("Pricing requires schema_version=1 and currency=USD")
    seen = set()
    for rate in card.get("api_rates", []):
        for key in ("models", "service_tiers"):
            if not isinstance(rate.get(key), list) or not rate[key] or any(
                not isinstance(value, str) or not value.strip() for value in rate[key]
            ):
                raise ValueError(f"Rate {key} must be a nonempty list of names")
        for model in rate["models"]:
            for tier in rate["service_tiers"]:
                key = (rate["provider"], model, tier)
                if key in seen:
                    raise ValueError(f"Ambiguous price for {key}")
                seen.add(key)
        for key in ("input_usd_per_million", "cached_input_usd_per_million", "output_usd_per_million"):
            amount(rate[key])
        if rate.get("cache_write_usd_per_million") is not None:
            amount(rate["cache_write_usd_per_million"])
    local = card.get("local_hardware")
    if local:
        amount(local["usd_per_hour"])
        if local.get("device") not in {"cpu", "cuda", "mps"} or not local.get("resource"):
            raise ValueError("Local rate requires device and resource description")
        if local.get("interpretation") not in {"cloud_equivalent", "measured_hardware"}:
            raise ValueError("Local estimates require cloud_equivalent or measured_hardware interpretation")
        if local["interpretation"] == "measured_hardware" and (
            not isinstance(local.get("hardware_profile"), str) or not local["hardware_profile"].strip()
        ):
            raise ValueError("A measured_hardware rate requires a hardware_profile")
    for rate in [*card.get("api_rates", []), *([local] if local else [])]:
        if not rate.get("source") or not rate.get("effective_date"):
            raise ValueError("Each rate needs a source and effective_date")
        date.fromisoformat(rate["effective_date"])
    return card


def _numeric_usage(value):
    """Retain numeric billing units without copying response/input text or URLs."""
    if isinstance(value, Mapping):
        return {str(k): _numeric_usage(v) for k, v in value.items()
                if v is None or isinstance(v, (Mapping, int, float))}
    if isinstance(value, float) and not (-float("inf") < value < float("inf")):
        return None
    return value


def capture_response(record: dict, response: Mapping) -> None:
    record["response_received"] = True
    if not isinstance(response, Mapping):
        return
    for source, target in (("id", "request_id"), ("model", "model"), ("service_tier", "service_tier")):
        if isinstance(response.get(source), str):
            record[target] = response[source]
    usage = response.get("usage")
    record["usage"] = _numeric_usage(usage) if isinstance(usage, Mapping) else None


@contextmanager
def capture_api_usage(sink, *, provider: str, sample_id: str, requested_model: str,
                      transport_attempts_complete: bool):
    """Record a response even when label parsing fails; unknown charges stay unknown."""
    record = {
        "kind": "api_attempt", "provider": provider, "sample_id": sample_id,
        "requested_model": requested_model, "model": None, "request_id": None,
        "service_tier": None, "usage": None, "observed_charge": None,
        "transport_attempts_complete": transport_attempts_complete,
        "response_received": False, "status": "success",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        yield record
    except Exception as exc:
        record["status"] = "failed"
        record["error_type"] = type(exc).__name__
        body = getattr(exc, "body", None)
        if isinstance(body, Mapping):
            capture_response(record, body)
        request_id = getattr(exc, "request_id", None)
        if isinstance(request_id, str):
            record["request_id"] = request_id
        raise
    finally:
        if sink is not None:
            sink(record)


def _count(value) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("missing or invalid token counts")
    return value


def local_hardware_rate(card: dict, device: str | None, hardware_profile: str | None = None) -> dict:
    rate = card.get("local_hardware")
    if not rate or not device or device.split(":")[0] != rate["device"]:
        raise ValueError("matching hardware rate/device unavailable")
    if rate["interpretation"] == "measured_hardware" and (
        not hardware_profile or hardware_profile != rate.get("hardware_profile")
    ):
        raise ValueError("measured hardware profile does not match the selected rate")
    return rate


def price_entry(entry: dict, card: dict, device: str | None, hardware_profile: str | None = None) -> dict:
    result = {"event_id": entry["event_id"], "phase": entry["phase"],
              "sample_ids": entry.get("sample_ids", [entry.get("sample_id")]),
              "kind": "unavailable", "cost_usd": None, "reason": None}
    try:
        charge = entry.get("observed_charge")
        if charge is not None:
            if charge.get("currency") != "USD" or not charge.get("source"):
                raise ValueError("observed charge requires USD and provenance")
            cost = amount(charge["amount"])
            result.update(kind="observed", source=charge["source"])
        elif entry["kind"] == "local_call":
            rate = local_hardware_rate(card, device, hardware_profile)
            cost = amount(entry["elapsed_ms"]) / Decimal(3600000) * amount(rate["usd_per_hour"])
            result.update(kind="estimated", rate=rate,
                          basis=f"active inference wall time; {rate['interpretation']}")
        elif entry.get("provider") == "openai":
            rate = next((r for r in card.get("api_rates", []) if
                         r["provider"] == "openai" and entry.get("model") in r["models"]
                         and entry.get("service_tier") in r["service_tiers"]), None)
            if rate is None:
                raise ValueError("exact model/service-tier rate unavailable")
            usage = entry.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            input_tokens = _count(usage.get("prompt_tokens"))
            cached = _count(details.get("cached_tokens"))
            output = _count(usage.get("completion_tokens"))
            writes = _count(0 if details.get("cache_write_tokens") is None else details["cache_write_tokens"])
            if cached + writes > input_tokens:
                raise ValueError("cache tokens exceed total input")
            if writes and rate.get("cache_write_usd_per_million") is None:
                raise ValueError("cache-write rate unavailable")
            if (details.get("audio_tokens") or 0) or ((usage.get("completion_tokens_details") or {}).get("audio_tokens") or 0):
                raise ValueError("text rate cannot price audio usage")
            completion_details = usage.get("completion_tokens_details") or {}
            reasoning = completion_details.get("reasoning_tokens")
            if reasoning is not None and _count(reasoning) > output:
                raise ValueError("reasoning tokens exceed total output")
            if usage.get("total_tokens") is not None and _count(usage["total_tokens"]) != input_tokens + output:
                raise ValueError("total tokens do not match input plus output")
            # Completion tokens already include reasoning tokens: never add twice.
            cost = (Decimal(input_tokens - cached - writes) * amount(rate["input_usd_per_million"])
                    + Decimal(cached) * amount(rate["cached_input_usd_per_million"])
                    + Decimal(writes) * amount(rate.get("cache_write_usd_per_million") or 0)
                    + Decimal(output) * amount(rate["output_usd_per_million"])) / Decimal(1000000)
            result.update(kind="estimated", rate=rate, basis="reported tokens at selected rate card",
                          billing_units={"uncached_input_tokens": input_tokens - cached - writes,
                                         "cached_input_tokens": cached, "cache_write_tokens": writes,
                                         "output_tokens_including_reasoning": output})
        else:
            raise ValueError("provider price or billing units unavailable")
        result["cost_usd"] = float(cost)
        result["cost_usd_decimal"] = str(cost)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        result.update(kind="unavailable", reason=str(exc))
    return result


def build_cost_report(entries: list[dict], card: dict, metadata: dict) -> dict:
    ids = [e["event_id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate usage event IDs would double-count spending")
    if any(e["phase"] not in {"warmup", "evaluation"} or e["kind"] not in {
        "inference_call", "api_attempt", "local_call", "unavailable_call"
    } for e in entries):
        raise ValueError("Unknown usage kind or phase")
    calls = [e for e in entries if e["kind"] == "inference_call"]
    attempts = [e for e in entries if e["kind"] in {"api_attempt", "local_call", "unavailable_call"}]
    device = metadata.get("runtime", {}).get("classifier", {}).get("device")
    profile = metadata.get("config", {}).get("hardware_profile")
    priced = [price_entry(e, card, device, profile) for e in attempts]

    def summarize(phase=None):
        selected = [(e, p) for e, p in zip(attempts, priced) if phase is None or e["phase"] == phase]
        known = sum((Decimal(p["cost_usd_decimal"]) for e, p in selected if p["cost_usd"] is not None), Decimal(0))
        complete = all(p["cost_usd"] is not None and e.get("transport_attempts_complete", True) for e, p in selected)
        phase_calls = [e for e in calls if phase is None or e["phase"] == phase]
        missing_calls = any(not any(
            e.get("observation_index") == call["observation_index"] for e, _ in selected
        ) for call in phase_calls)
        orphan_attempts = any(not any(
            e.get("observation_index") == call["observation_index"] for call in phase_calls
        ) for e, _ in selected)
        complete = (complete and not missing_calls and not orphan_attempts
                    and metadata.get("run_status") in {"completed", "failed"})
        kinds = {p["kind"] for e, p in selected}
        return {"total_cost_usd": float(known) if complete else None,
                "known_cost_subtotal_usd": float(known), "coverage_complete": complete,
                "cost_kind": (next(iter(kinds)) if len(kinds) == 1 else "mixed") if complete and kinds else
                             ("no_attempts" if complete and not selected and not phase_calls else "unavailable"),
                "missing_call_usage": missing_calls, "unmatched_attempts": orphan_attempts,
                "unpriced_entries": sum(p["cost_usd"] is None for e, p in selected),
                "hidden_retry_coverage_unknown": any(not e.get("transport_attempts_complete", True) for e, p in selected)}

    evaluation = [e for e in calls if e["phase"] == "evaluation"]
    # One denominator entry per logical example, even if its API attempts repeat.
    attempted_ids = {i for e in evaluation for i in e["sample_ids"]}
    successful_ids = {i for e in evaluation if e["status"] == "success" for i in e["sample_ids"]}
    total = summarize()
    count = len(successful_ids)
    per_sample = {}
    for sample_id in successful_ids:
        selected = [(e, p) for e, p in zip(attempts, priced)
                    if e["phase"] == "evaluation" and sample_id in p["sample_ids"]]
        available = bool(selected) and all(p["cost_usd"] is not None and e.get("transport_attempts_complete", True) for e, p in selected)
        value = sum((Decimal(p["cost_usd_decimal"]) / len(p["sample_ids"]) for e, p in selected if p["cost_usd"] is not None), Decimal(0))
        kinds = {p["kind"] for _, p in selected}
        per_sample[sample_id] = {"cost_usd": float(value) if available else None,
                                "cost_kind": (next(iter(kinds)) if len(kinds) == 1 else "mixed") if available else "unavailable",
                                "event_ids": [p["event_id"] for _, p in selected],
                                "allocation": "equal allocation within local batch; API attempts assigned by sample ID"}
    return {
        "schema_version": 1, "currency": "USD", "pricing": card,
        "run_status": metadata.get("run_status"),
        "backend": metadata.get("runtime", {}).get("classifier", {}).get("backend"),
        "pricing_mode": "selected rate card; estimates are not historical invoices",
        "scope": "recorded inference attempts including warmup and failures; preparation excluded",
        **total, "evaluation": summarize("evaluation"), "warmup": summarize("warmup"),
        "attempted_examples": len(attempted_ids), "successful_examples": count,
        "failure_rate": (len(attempted_ids - successful_ids) / len(attempted_ids)) if attempted_ids else None,
        "cost_per_1000_successful_examples_usd": total["total_cost_usd"] * 1000 / count
            if total["total_cost_usd"] is not None and count else None,
        "observed_charges_subtotal_usd": float(sum((Decimal(p["cost_usd_decimal"]) for p in priced if p["kind"] == "observed"), Decimal(0))),
        "estimated_cost_subtotal_usd": float(sum((Decimal(p["cost_usd_decimal"]) for p in priced if p["kind"] == "estimated"), Decimal(0))),
        "entries": priced, "per_successful_sample": per_sample,
    }


def recalculate_costs(run_dir: Path, pricing_path: Path | None = None, *,
                      deployment_hours: float | None = None,
                      volumes: tuple[int, ...] = (1000, 10000, 100000)) -> dict:
    card = load_pricing(pricing_path or run_dir / "pricing.json")
    entries = [json.loads(line) for line in (run_dir / "usage.jsonl").read_text().splitlines() if line.strip()]
    metadata = json.loads((run_dir / "measurement.json").read_text())
    report = build_cost_report(entries, card, metadata)
    report["usage_sha256"] = hashlib.sha256((run_dir / "usage.jsonl").read_bytes()).hexdigest()
    report["pricing_sha256"] = hashlib.sha256((pricing_path or run_dir / "pricing.json").read_bytes()).hexdigest()
    report["measurement_sha256"] = hashlib.sha256((run_dir / "measurement.json").read_bytes()).hexdigest()
    from llm_classifier_bench.self_hosted_costs import self_hosted_cost_report
    report["self_hosted"] = self_hosted_cost_report(
        run_dir, card, deployment_hours=deployment_hours, volumes=volumes,
    )
    return report


class CostRecorder:
    def __init__(self, run_dir: Path, classifier, pricing_path: Path | None):
        self.run_dir, self.classifier = run_dir, classifier
        self.card = load_pricing(pricing_path)
        self.phase = "evaluation"
        self.observation_index = None
        self.pending_attempts = []
        self.entries: list[dict] = []
        (run_dir / "usage.jsonl").touch(exist_ok=False)
        (run_dir / "pricing.json").write_text(json.dumps(self.card, indent=2, allow_nan=False) + "\n")
        hook = getattr(classifier, "set_usage_sink", None)
        if callable(hook):
            hook(self.record_attempt)

    def _append(self, payload):
        record = {**payload, "event_id": f"usage-{len(self.entries):06d}"}
        self.entries.append(record)
        with (self.run_dir / "usage.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def record_attempt(self, record):
        payload = {**record, "phase": self.phase, "observation_index": self.observation_index}
        self.pending_attempts.append(payload)
        self._append(payload)

    def record_timing(self, row):
        if row["kind"] != "inference":
            return
        self._append({**row, "kind": "inference_call"})
        hook = getattr(self.classifier, "inference_metadata", None)
        metadata = hook() if callable(hook) else {}
        if metadata.get("backend") == "local":
            self._append({**row, "kind": "local_call"})
        else:
            captured = {e["sample_id"] for e in self.pending_attempts}
            missing = set(row["sample_ids"]) - captured
            # A failed sequential batch may stop before contacting the provider for
            # its remaining inputs. Its captured attempts are the charged work.
            if not captured or (row["status"] == "success" and missing):
                self._append({**row, "kind": "unavailable_call"})
        self.pending_attempts = []

    def finish(self):
        report = recalculate_costs(self.run_dir)
        (self.run_dir / "cost_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report

    def close(self):
        hook = getattr(self.classifier, "set_usage_sink", None)
        if callable(hook):
            hook(None)
