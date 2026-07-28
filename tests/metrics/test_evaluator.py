from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_classifier_bench.metrics import (
    evaluate_jsonl,
    load_evaluation_records,
    results_as_dict,
)


def write_artifact(path: Path) -> None:
    rows = (
        {
            "sample_id": "1",
            "gold_label": "A",
            "predicted_label": "A",
            "confidence": 0.8,
            "probabilities": {"A": 0.8, "B": 0.2},
            "latency_ms": 10.0,
        },
        {
            "sample_id": "2",
            "gold_label": "B",
            "predicted_label": "B",
            "confidence": 0.7,
            "probabilities": {"A": 0.3, "B": 0.7},
            "latency_ms": 20.0,
        },
    )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_evaluate_saved_jsonl_without_rerunning_classifier(tmp_path: Path) -> None:
    artifact = tmp_path / "predictions.jsonl"
    write_artifact(artifact)

    records = load_evaluation_records(artifact)
    results = results_as_dict(evaluate_jsonl(artifact))

    assert len(records) == 2
    assert results["accuracy"]["value"] == pytest.approx(1.0)
    assert results["macro_f1"]["value"] == pytest.approx(1.0)
    assert results["mean_latency_ms"]["value"] == pytest.approx(15.0)
    assert results["total_cost_usd"]["value"] is None
    assert results["total_cost_usd"]["available"] is False


def test_invalid_jsonl_reports_line_number(tmp_path: Path) -> None:
    artifact = tmp_path / "broken.jsonl"
    artifact.write_text('{"sample_id": "1"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"broken\.jsonl:1"):
        load_evaluation_records(artifact)
