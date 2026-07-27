from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from llm_classifier_bench.classifiers.base import ClassificationInput, ClassDefinition, Classifier
from llm_classifier_bench.classifiers.emissary import (
    EmissaryClassifier,
    EmissaryResponseError,
    Experiment,
    parse_classification_response,
)


def sample_response() -> dict[str, Any]:
    return {
        "id": "classify-test-123",
        "model": "ex-test/0.0.0",
        "data": [
            {
                "index": 0,
                "probs": {
                    "sports": 0.09,
                    "finance": 0.86,
                    "cooking": 0.05,
                },
            }
        ],
        "created": 1780085360,
    }


def test_parse_classification_response_builds_prediction() -> None:
    response = sample_response()

    prediction = parse_classification_response(
        response,
        sample_id="sample-1",
        latency_ms=42.5,
    )

    assert prediction.sample_id == "sample-1"
    assert prediction.predicted_label == "finance"
    assert prediction.confidence == pytest.approx(0.86)
    assert prediction.probabilities == {
        "sports": 0.09,
        "finance": 0.86,
        "cooking": 0.05,
    }
    assert prediction.latency_ms == pytest.approx(42.5)
    assert prediction.model == "ex-test/0.0.0"
    assert prediction.request_id == "classify-test-123"
    assert prediction.raw_response == response


def test_parse_rejects_probabilities_that_do_not_sum_to_one() -> None:
    response = sample_response()
    response["data"][0]["probs"] = {"sports": 0.8, "finance": 0.8}

    with pytest.raises(EmissaryResponseError, match="do not sum to 1"):
        parse_classification_response(
            response,
            sample_id="sample-1",
            latency_ms=10.0,
        )


def test_parse_rejects_missing_probabilities() -> None:
    response = sample_response()
    del response["data"][0]["probs"]

    with pytest.raises(EmissaryResponseError, match="no non-empty 'probs'"):
        parse_classification_response(
            response,
            sample_id="sample-1",
            latency_ms=10.0,
        )


class StubClient:
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = dict(response)
        self.calls: list[dict[str, str]] = []

    def classify(
        self,
        *,
        model_id: str,
        text: str,
        data_format: str = "probs",
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "model_id": model_id,
                "text": text,
                "data_format": data_format,
            }
        )
        return dict(self.response)


class CreatingStubClient(StubClient):
    def __init__(self, response: Mapping[str, Any]) -> None:
        super().__init__(response)
        self.created_with: dict[str, Any] | None = None

    def create_experiment(
        self,
        *,
        name: str,
        classes: list[ClassDefinition],
        mode: str = "routing",
    ) -> Experiment:
        self.created_with = {
            "name": name,
            "classes": classes,
            "mode": mode,
        }
        return Experiment(experiment_id="ex-created", latest_version="0.0.0")


def test_classifier_predict_preserves_input_order() -> None:
    client = StubClient(sample_response())
    classifier = EmissaryClassifier(
        client=client,  # type: ignore[arg-type]
        model_id="ex-test/0.0.0",
    )

    inputs = [
        ClassificationInput(sample_id="a", text="First input"),
        ClassificationInput(sample_id="b", text="Second input"),
    ]

    predictions = classifier.predict(inputs)

    assert [prediction.sample_id for prediction in predictions] == ["a", "b"]
    assert [call["text"] for call in client.calls] == ["First input", "Second input"]
    assert all(call["data_format"] == "probs" for call in client.calls)


def test_classifier_create_uses_experiment_model_version() -> None:
    client = CreatingStubClient(sample_response())
    classes = [
        ClassDefinition("sports", "Sports and competitions"),
        ClassDefinition("finance", "Money, banking, and investments"),
    ]

    classifier = EmissaryClassifier.create(
        client=client,  # type: ignore[arg-type]
        experiment_name="unit-test",
        classes=classes,
    )

    assert classifier.model_id == "ex-created/0.0.0"
    assert client.created_with == {
        "name": "unit-test",
        "classes": classes,
        "mode": "routing",
    }


def test_emissary_classifier_satisfies_classifier_protocol() -> None:
    client = StubClient(sample_response())
    classifier = EmissaryClassifier(
        client=client,  # type: ignore[arg-type]
        model_id="ex-test/0.0.0",
    )

    assert isinstance(classifier, Classifier)
