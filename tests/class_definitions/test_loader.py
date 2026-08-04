from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_classifier_bench.class_definitions.loader import load_class_definition_profile


def _write_profile(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "dataset": "tiny_news",
        "profile": "canonical_llm_enriched_v1",
        "review_status": "approved",
        "generation": {
            "method": "llm",
            "uses_train_examples": False,
            "uses_test_examples": False,
        },
        "classes": [
            {"canonical_name": "Sports", "description": "News about sports."},
            {"canonical_name": "World", "description": "News about world affairs."},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loader_fingerprints_and_reorders_to_dataset_canonical_order(tmp_path: Path) -> None:
    path = _write_profile(tmp_path / "profile.json")
    loaded = load_class_definition_profile(path)

    definitions = loaded.definitions_for(
        dataset_name="tiny_news",
        canonical_names=("World", "Sports"),
    )

    assert [item.name for item in definitions] == ["World", "Sports"]
    assert definitions[0].description == "News about world affairs."
    assert len(loaded.sha256) == 64
    assert loaded.benchmark_metadata()["profile"] == "canonical_llm_enriched_v1"


def test_loader_rejects_profile_with_different_label_inventory(tmp_path: Path) -> None:
    path = _write_profile(tmp_path / "profile.json")
    loaded = load_class_definition_profile(path)

    with pytest.raises(ValueError, match="does not match dataset canonical labels"):
        loaded.definitions_for(
            dataset_name="tiny_news",
            canonical_names=("World", "Business"),
        )
