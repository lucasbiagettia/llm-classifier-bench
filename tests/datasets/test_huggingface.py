from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from llm_classifier_bench.datasets import (
    ClassificationDataset,
    HFDatasetSpec,
    HuggingFaceClassificationDataset,
)


@dataclass
class FakeClassLabel:
    names: list[str]


class FakeSplit(list[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], label_names: list[str]) -> None:
        super().__init__(rows)
        self.features = {"label": FakeClassLabel(label_names)}


class FakeLoader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeSplit:
        self.calls.append(kwargs)
        label_names = ["World", "Sports", "Business", "Sci/Tech"]
        if kwargs["split"] == "train":
            return FakeSplit(
                [
                    {"text": "Markets rose after the earnings report.", "label": 2},
                    {"text": "The striker scored twice in the final.", "label": 1},
                ],
                label_names,
            )
        return FakeSplit(
            [
                {"text": "Researchers announced a new processor.", "label": 3},
                {"text": "Leaders met to discuss the peace agreement.", "label": 0},
            ],
            label_names,
        )


def build_dataset(loader: FakeLoader) -> HuggingFaceClassificationDataset:
    return HuggingFaceClassificationDataset(
        HFDatasetSpec(
            name="ag_news_test",
            path="example/ag_news",
            label_descriptions={
                "World": "World events.",
                "Sports": "Sports events.",
                "Business": "Business events.",
                "Sci/Tech": "Science and technology.",
            },
        ),
        loader=loader,
    )


def test_huggingface_adapter_normalizes_splits_and_classes() -> None:
    loader = FakeLoader()
    dataset = build_dataset(loader)

    bundle = dataset.load()

    assert isinstance(dataset, ClassificationDataset)
    assert bundle.name == "ag_news_test"
    assert bundle.class_names == ("World", "Sports", "Business", "Sci/Tech")
    assert [example.label for example in bundle.train] == ["Business", "Sports"]
    assert [example.label for example in bundle.test] == ["Sci/Tech", "World"]
    assert bundle.test[0].sample_id == "ag_news_test:test:0"
    assert bundle.classes[2].description == "Business events."

    assert loader.calls == [
        {"path": "example/ag_news", "split": "train"},
        {"path": "example/ag_news", "split": "test"},
    ]


def test_data_files_are_forwarded_to_generic_loader() -> None:
    calls: list[dict[str, Any]] = []

    def loader(**kwargs: Any) -> list[dict[str, str]]:
        calls.append(kwargs)
        return [{"text": "Where is my card?", "category": "card_arrival"}]

    data_files = {
        "train": "https://example.test/train.csv",
        "test": "https://example.test/test.csv",
    }
    dataset = HuggingFaceClassificationDataset(
        HFDatasetSpec(
            name="banking_test",
            path="csv",
            label_column="category",
            data_files=data_files,
            label_names=("card_arrival", "card_not_working"),
        ),
        loader=loader,
    )

    bundle = dataset.load()

    assert bundle.train[0].label == "card_arrival"
    assert calls == [
        {"path": "csv", "split": "train", "data_files": data_files},
        {"path": "csv", "split": "test", "data_files": data_files},
    ]


def test_bundle_inputs_do_not_expose_gold_labels() -> None:
    bundle = build_dataset(FakeLoader()).load()

    inputs = bundle.inputs("test")

    assert [item.sample_id for item in inputs] == [
        "ag_news_test:test:0",
        "ag_news_test:test:1",
    ]
    assert [item.text for item in inputs] == [
        "Researchers announced a new processor.",
        "Leaders met to discuss the peace agreement.",
    ]
    assert bundle.gold_labels("test") == ("Sci/Tech", "World")
    assert not hasattr(inputs[0], "label")


def test_generated_descriptions_humanize_label_names() -> None:
    loader = FakeLoader()
    dataset = HuggingFaceClassificationDataset(
        HFDatasetSpec(
            name="generated_descriptions",
            path="example/data",
            label_description_template="Intent about {readable_label}.",
        ),
        loader=loader,
    )

    bundle = dataset.load()

    sci_tech = next(item for item in bundle.classes if item.name == "Sci/Tech")
    assert sci_tech.description == "Intent about Sci or Tech."


def test_adapter_rejects_out_of_range_integer_label() -> None:
    class InvalidLoader(FakeLoader):
        def __call__(self, **kwargs: Any) -> FakeSplit:
            return FakeSplit([{"text": "Invalid row", "label": 99}], ["a", "b"])

    dataset = HuggingFaceClassificationDataset(
        HFDatasetSpec(name="invalid", path="example/invalid"),
        loader=InvalidLoader(),
    )

    with pytest.raises(ValueError, match="outside"):
        dataset.load()