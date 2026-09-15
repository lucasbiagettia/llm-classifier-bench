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
    # Configured order is World/Sports; sklearn's columns are Sports/World.
    assert prediction.predicted_label == "World"
    assert prediction.confidence == 0.75
    assert prediction.probabilities is not None
    assert prediction.probabilities == {"Sports": 0.25, "World": 0.75}
    assert sum(prediction.probabilities.values()) == 1.0
    metadata = classifier.fitted_metadata()
    assert metadata["selected_c"] == 1.0
    assert metadata["probability_label_order"] == ["Sports", "World"]
    assert metadata["candidate_scores"] == [
        {"c": 0.1, "validation_accuracy": 0.5},
        {"c": 1.0, "validation_accuracy": 0.9},
        {"c": 10.0, "validation_accuracy": 0.5},
    ]


def test_validation_accuracy_ties_keep_first_configured_c(monkeypatch) -> None:
    monkeypatch.setattr(FakeLogisticRegression, "score", lambda self, x, y: 0.97)
    monkeypatch.setattr(
        "llm_classifier_bench.classifiers.sentence_transformer._load_logistic_regression",
        lambda: FakeLogisticRegression,
    )
    classifier = SentenceTransformerLogisticClassifier(
        encoder=FakeEncoder(),
        training=SentenceTransformerTrainingConfig(c_values=(0.1, 1.0, 10.0)),
    )
    classifier.prepare((ClassDefinition("A", "A"), ClassDefinition("B", "B")))
    examples = (LabeledExample("1", "one", "A"), LabeledExample("2", "two", "B"))
    classifier.fit(examples, validation_examples=(LabeledExample("3", "three", "A"),))
    assert classifier.selected_c == 0.1
    assert classifier.fitted_metadata()["tie_breaking"] == "first_candidate_in_configured_order"
    assert [item["validation_accuracy"] for item in classifier.fitted_metadata()["candidate_scores"]] == [0.97] * 3
