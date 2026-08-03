from __future__ import annotations

import json
from types import SimpleNamespace

from llm_classifier_bench.classifiers import Classifier, OpenAIClassifier
from llm_classifier_bench.core import ClassDefinition, ClassificationInput


class StubCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp-123",
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps({"label": "Sports"}))
                )
            ],
            model_dump=lambda: {"id": "resp-123", "model": kwargs["model"]},
        )


class StubClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=StubCompletions())


def test_openai_classifier_is_zero_shot_and_normalizes_prediction() -> None:
    client = StubClient()
    classifier = OpenAIClassifier(
        model="test-model",
        reasoning_effort="minimal",
        client=client,
    )
    classes = (
        ClassDefinition("World", "World news."),
        ClassDefinition("Sports", "Sports news."),
    )

    classifier.prepare(classes)
    classifier.fit((), validation_examples=())
    prediction = classifier.predict(
        [ClassificationInput("sample-1", "The striker scored the winning goal.")]
    )[0]

    assert isinstance(classifier, Classifier)
    assert prediction.sample_id == "sample-1"
    assert prediction.predicted_label == "Sports"
    assert prediction.confidence is None
    assert prediction.probabilities is None
    assert prediction.model == "test-model"
    assert prediction.request_id == "resp-123"
    assert classifier.supervision_regime == "zero_shot"
    assert classifier.training_examples_used == 0
    assert classifier.validation_examples_used == 0

    call = client.chat.completions.calls[0]
    assert call["reasoning_effort"] == "minimal"
    assert call["response_format"]["json_schema"]["schema"]["properties"]["label"]["enum"] == [
        "World",
        "Sports",
    ]
