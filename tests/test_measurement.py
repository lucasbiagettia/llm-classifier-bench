from __future__ import annotations

import pytest

from llm_classifier_bench.measurement import aggregate_timings


def observation(index, size=1):
    return {"kind": "inference", "phase": "evaluation", "status": "success",
            "start_offset_ns": index * 10_000_000, "end_offset_ns": index * 10_000_000 + 5_000_000,
            "elapsed_ms": 5.0, "batch_size": size,
            "sample_ids": [f"sample-{index}-{i}" for i in range(size)]}


def test_p99_gate_counts_calls_not_examples():
    metadata = {"config": {"warmup_examples": 0}, "planned_test_examples": 4000}
    rows = [observation(i, size=4) for i in range(1000)]
    small = aggregate_timings(rows[:250], metadata)["evaluation"]
    assert small["successful_examples"] == 1000
    assert small["successful_call_latency"]["p99_ms"] is None
    large = aggregate_timings(rows, metadata)["evaluation"]
    assert large["successful_call_latency"]["p99_ms"] == 5.0


def test_remainder_batch_keeps_separate_observation_population():
    metadata = {"config": {"warmup_examples": 0}, "planned_test_examples": 6}
    measured = aggregate_timings([observation(0, 4), observation(1, 2)], metadata)["evaluation"]
    assert measured["successful_examples"] == 6
    assert measured["successful_call_latency_by_batch_size"]["4"]["observation_count"] == 1
    assert measured["successful_call_latency_by_batch_size"]["2"]["observation_count"] == 1
    assert measured["throughput_successful_examples_per_second"] == 400


@pytest.mark.parametrize("changes", [{"elapsed_ms": -1}, {"elapsed_ms": float("nan")},
                                    {"elapsed_ms": 6}, {"batch_size": 2}, {"status": "unknown"}])
def test_corrupt_observations_are_rejected(changes):
    with pytest.raises(ValueError):
        aggregate_timings([{**observation(0), **changes}],
                          {"config": {"warmup_examples": 0}, "planned_test_examples": 1})
