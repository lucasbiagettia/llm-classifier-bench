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


def test_explicit_cpu_wins_even_when_cuda_is_reported_available(monkeypatch):
    from types import SimpleNamespace
    class ReachedModelLoad(Exception): pass
    chosen = []
    class Model:
        def to(self, device):
            chosen.append(device)
            raise ReachedModelLoad()
    torch = SimpleNamespace(device=lambda value: value,
                            cuda=SimpleNamespace(is_available=lambda: True))
    tokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: object())
    model = SimpleNamespace(from_pretrained=lambda *a, **kw: Model())
    monkeypatch.setattr("llm_classifier_bench.classifiers.bert._load_transformer_stack", lambda: (torch, tokenizer, model))
    monkeypatch.setattr("llm_classifier_bench.classifiers.bert._set_seed", lambda *a: None)
    classifier = BertClassifier(device="cpu")
    classifier.prepare([ClassDefinition("A", "A"), ClassDefinition("B", "B")])
    with pytest.raises(ReachedModelLoad):
        classifier.fit([LabeledExample("a", "one", "A"), LabeledExample("b", "two", "B")])
    assert chosen == ["cpu"]
    assert classifier.inference_metadata()["requested_device"] == "cpu"
