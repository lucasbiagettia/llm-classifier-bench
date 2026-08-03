from __future__ import annotations

from typing import Sequence

from llm_classifier_bench.classifiers import (
    Classifier,
    SentenceTransformerLogisticClassifier,
)
from llm_classifier_bench.config import SentenceTransformerTrainingConfig
from llm_classifier_bench.core import ClassDefinition, ClassificationInput, LabeledExample


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return [[float(len(text)), 1.0] for text in texts]


class FakeLogisticRegression:
    def __init__(self, *, C: float, max_iter: int, random_state: int) -> None:
        self.C = C
        self.classes_: list[str] = []

    def fit(self, embeddings, labels: Sequence[str]):
        self.classes_ = sorted(set(labels))
        return self

    def score(self, embeddings, labels) -> float:
        return 0.9 if self.C == 1.0 else 0.5

    def predict_proba(self, embeddings):
        return [[0.25, 0.75] for _ in embeddings]


def test_sentence_transformer_uses_validation_to_select_logistic_c(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_classifier_bench.classifiers.sentence_transformer._load_logistic_regression",
        lambda: FakeLogisticRegression,
    )
    classifier = SentenceTransformerLogisticClassifier(
        model="fake-encoder",
        encoder=FakeEncoder(),
        training=SentenceTransformerTrainingConfig(c_values=(0.1, 1.0, 10.0)),
    )
    classes = (
        ClassDefinition("World", "World news."),
        ClassDefinition("Sports", "Sports news."),
    )
    train = (
        LabeledExample("train-1", "leaders met", "World"),
        LabeledExample("train-2", "team scored", "Sports"),
    )
    validation = (
        LabeledExample("val-1", "treaty signed", "World"),
        LabeledExample("val-2", "goal scored", "Sports"),
    )

    classifier.prepare(classes)
    classifier.fit(train, validation_examples=validation)
    prediction = classifier.predict([ClassificationInput("test-1", "another goal")])[0]

    assert isinstance(classifier, Classifier)
    assert classifier.selected_c == 1.0
    assert prediction.predicted_label == "World" or prediction.predicted_label == "Sports"
    assert prediction.probabilities is not None
    assert set(prediction.probabilities) == {"World", "Sports"}
    assert sum(prediction.probabilities.values()) == 1.0
