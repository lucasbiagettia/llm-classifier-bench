from __future__ import annotations

import pytest

from llm_classifier_bench.classifiers import BertClassifier, Classifier
from llm_classifier_bench.core import ClassDefinition, LabeledExample


def test_bert_prepare_builds_closed_label_space_without_loading_model() -> None:
    classifier = BertClassifier(model="fake-model")
    classifier.prepare(
        (
            ClassDefinition("World", "World news."),
            ClassDefinition("Sports", "Sports news."),
        )
    )

    assert isinstance(classifier, Classifier)
    assert classifier._label_to_id == {"World": 0, "Sports": 1}
    assert classifier._id_to_label == {0: "World", 1: "Sports"}


def test_bert_fit_requires_prepare_before_loading_huggingface() -> None:
    classifier = BertClassifier(model="fake-model")

    with pytest.raises(RuntimeError, match="prepare"):
        classifier.fit(
            [LabeledExample("train-1", "text", "World")],
            validation_examples=(),
        )
