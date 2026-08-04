from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest

from llm_classifier_bench.classifiers.base import Prediction
from llm_classifier_bench.core import ClassDefinition, ClassificationInput, LabeledExample
from llm_classifier_bench.datasets.base import DatasetBundle
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark


@dataclass
class FakeDataset:
    bundle: DatasetBundle

    @property
    def name(self) -> str:
        return self.bundle.name

    def load(self) -> DatasetBundle:
        return self.bundle


class FakeClassifier:
    def __init__(self) -> None:
        self.fit_sample_ids: list[str] = []
        self.validation_sample_ids: list[str] = []
        self.prepared_classes: list[str] = []

    @property
    def name(self) -> str:
        return "fake_classifier"

    def prepare(self, classes: Sequence[ClassDefinition]) -> None:
        self.prepared_classes = [item.name for item in classes]

    def fit(
        self,
        examples: Sequence[LabeledExample],
        *,
        validation_examples: Sequence[LabeledExample] = (),
    ) -> None:
        self.fit_sample_ids = [example.sample_id for example in examples]
        self.validation_sample_ids = [example.sample_id for example in validation_examples]

    def predict(self, examples: Sequence[ClassificationInput]) -> list[Prediction]:
        predictions: list[Prediction] = []
        for example in examples:
            label = "Sports" if "goal" in example.text else "World"
            probabilities = (
                {"World": 0.1, "Sports": 0.9}
                if label == "Sports"
                else {"World": 0.8, "Sports": 0.2}
            )
            predictions.append(
                Prediction(
                    sample_id=example.sample_id,
                    predicted_label=label,
                    confidence=probabilities[label],
                    probabilities=probabilities,
                    latency_ms=1.5,
                    model="fake-v1",
                    request_id=f"request-{example.sample_id}",
                    raw_response={"fixture": True},
                )
            )
        return predictions


class WrongOrderClassifier(FakeClassifier):
    @property
    def name(self) -> str:
        return "wrong_order"

    def predict(self, examples: Sequence[ClassificationInput]) -> list[Prediction]:
        predictions = super().predict(examples)
        return list(reversed(predictions))


class ZeroShotMetadataClassifier(FakeClassifier):
    supervision_regime = "zero_shot"
    training_examples_used = 0
    validation_examples_used = 0
    reasoning_effort = "minimal"

    @property
    def name(self) -> str:
        return "zero_shot_metadata"


def build_bundle() -> DatasetBundle:
    return DatasetBundle(
        name="tiny_news",
        classes=(
            ClassDefinition(name="World", description="World news."),
            ClassDefinition(name="Sports", description="Sports news."),
        ),
        train=(
            LabeledExample(sample_id="train-1", text="world leaders met", label="World"),
            LabeledExample(sample_id="train-2", text="the team scored a goal", label="Sports"),
        ),
        test=(
            LabeledExample(sample_id="test-1", text="the striker scored a goal", label="Sports"),
            LabeledExample(sample_id="test-2", text="leaders signed a treaty", label="World"),
        ),
        metadata={"source": "unit-test"},
    )


def test_runner_fits_predicts_persists_and_evaluates(tmp_path: Path) -> None:
    dataset = FakeDataset(build_bundle())
    classifier = FakeClassifier()

    result = run_benchmark(
        dataset,
        classifier,
        BenchmarkRunConfig(
            output_root=tmp_path,
            run_id="tiny-run",
            metadata={"code_version": "test", "seed": 1},
        ),
    )

    assert classifier.prepared_classes == ["World", "Sports"]
    assert classifier.fit_sample_ids == ["train-1", "train-2"]
    assert classifier.validation_sample_ids == []
    assert result.example_count == 2
    assert result.predictions_path.exists()
    assert result.metrics_path is not None and result.metrics_path.exists()

    config_payload = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config_payload["dataset"]["test_sample_ids"] == ["test-1", "test-2"]
    assert config_payload["dataset"]["fit_train_sample_ids"] == ["train-1", "train-2"]
    assert config_payload["dataset"]["validation_sample_ids"] == []
    assert config_payload["run_metadata"] == {"code_version": "test", "seed": 1}

    prediction_rows = [
        json.loads(line)
        for line in result.predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_id"] for row in prediction_rows] == ["test-1", "test-2"]
    assert [row["gold_label"] for row in prediction_rows] == ["Sports", "World"]
    assert [row["predicted_label"] for row in prediction_rows] == ["Sports", "World"]
    assert all(row["correct"] for row in prediction_rows)

    status_payload = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert status_payload["status"] == "completed"
    assert status_payload["stage"] == "completed"


def test_runner_records_failure_stage_and_re_raises(tmp_path: Path) -> None:
    dataset = FakeDataset(build_bundle())

    with pytest.raises(ValueError, match="preserve input order/sample_id"):
        run_benchmark(
            dataset,
            WrongOrderClassifier(),
            BenchmarkRunConfig(
                output_root=tmp_path,
                run_id="failed-run",
                evaluate=False,
            ),
        )

    status_path = tmp_path / "failed-run" / "status.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_payload["status"] == "failed"
    assert status_payload["stage"] == "predicting"
    assert status_payload["error_type"] == "ValueError"


def test_runner_refuses_to_overwrite_an_existing_run(tmp_path: Path) -> None:
    existing = tmp_path / "same-run"
    existing.mkdir()

    with pytest.raises(FileExistsError):
        run_benchmark(
            FakeDataset(build_bundle()),
            FakeClassifier(),
            BenchmarkRunConfig(
                output_root=tmp_path,
                run_id="same-run",
                evaluate=False,
            ),
        )


def test_split_train_validation_is_stratified_and_deterministic() -> None:
    from llm_classifier_bench.runner import split_train_validation

    examples = tuple(
        LabeledExample(f"world-{index}", f"world {index}", "World")
        for index in range(5)
    ) + tuple(
        LabeledExample(f"sports-{index}", f"sports {index}", "Sports")
        for index in range(5)
    )

    train_a, validation_a = split_train_validation(
        examples,
        validation_fraction=0.2,
        seed=42,
    )
    train_b, validation_b = split_train_validation(
        examples,
        validation_fraction=0.2,
        seed=42,
    )

    assert [item.sample_id for item in train_a] == [item.sample_id for item in train_b]
    assert [item.sample_id for item in validation_a] == [item.sample_id for item in validation_b]
    assert {item.label for item in validation_a} == {"World", "Sports"}
    assert len(validation_a) == 2
    assert len(train_a) == 8


def test_runner_persists_optional_classifier_experiment_metadata(tmp_path: Path) -> None:
    result = run_benchmark(
        FakeDataset(build_bundle()),
        ZeroShotMetadataClassifier(),
        BenchmarkRunConfig(
            output_root=tmp_path,
            run_id="metadata-run",
            evaluate=False,
        ),
    )

    config_payload = json.loads(result.config_path.read_text(encoding="utf-8"))
    classifier_metadata = config_payload["classifier"]
    assert classifier_metadata["supervision_regime"] == "zero_shot"
    assert classifier_metadata["training_examples_used"] == 0
    assert classifier_metadata["validation_examples_used"] == 0
    assert classifier_metadata["reasoning_effort"] == "minimal"


def test_runner_uses_versioned_class_definitions_when_configured(tmp_path: Path) -> None:
    profile_path = tmp_path / "definitions.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "tiny_news",
                "profile": "canonical_llm_enriched_v1",
                "review_status": "approved",
                "generation": {
                    "method": "llm",
                    "uses_train_examples": False,
                    "uses_validation_examples": False,
                    "uses_test_examples": False,
                },
                "classes": [
                    {
                        "canonical_name": "World",
                        "description": "Enriched description for world affairs.",
                    },
                    {
                        "canonical_name": "Sports",
                        "description": "Enriched description for sports events.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = FakeDataset(build_bundle())
    classifier = FakeClassifier()
    result = run_benchmark(
        dataset,
        classifier,
        BenchmarkRunConfig(
            output_root=tmp_path,
            run_id="definitions-run",
            evaluate=False,
            class_definitions_path=profile_path,
        ),
    )

    config_payload = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config_payload["class_definitions"]["source"] == "versioned_profile"
    assert config_payload["class_definitions"]["profile"] == "canonical_llm_enriched_v1"
    assert len(config_payload["class_definitions"]["sha256"]) == 64
    assert config_payload["dataset"]["classes"] == [
        {
            "name": "World",
            "description": "Enriched description for world affairs.",
        },
        {
            "name": "Sports",
            "description": "Enriched description for sports events.",
        },
    ]
