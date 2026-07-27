from __future__ import annotations

import os
from dataclasses import replace

import pytest

from llm_classifier_bench.datasets import (
    AG_NEWS_SPEC,
    BANKING77_SPEC,
    HuggingFaceClassificationDataset,
)


pytestmark = pytest.mark.integration


def _network_enabled_or_skip() -> None:
    if os.getenv("RUN_HF_INTEGRATION") != "1":
        pytest.skip("Set RUN_HF_INTEGRATION=1 to download Hugging Face datasets")


@pytest.mark.parametrize(
    ("spec", "expected_class_count"),
    [
        (AG_NEWS_SPEC, 4),
        (BANKING77_SPEC, 77),
    ],
)
def test_real_huggingface_dataset_contract(spec, expected_class_count: int) -> None:
    _network_enabled_or_skip()
    small_spec = replace(
        spec,
        train_split="train[:8]",
        test_split="test[:8]",
    )

    bundle = HuggingFaceClassificationDataset(small_spec).load()

    assert len(bundle.classes) == expected_class_count
    assert len(bundle.train) == 8
    assert len(bundle.test) == 8
    assert all(example.label in bundle.class_names for example in bundle.train)
    assert all(example.label in bundle.class_names for example in bundle.test)
