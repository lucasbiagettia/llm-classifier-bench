from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from llm_classifier_bench.classifiers import Classifier, TfidfLogisticClassifier
from llm_classifier_bench.config import TfidfTrainingConfig
from llm_classifier_bench.core import ClassDefinition, ClassificationInput, LabeledExample


CLASSES = (ClassDefinition("zulu", "Zulu"), ClassDefinition("alpha", "Alpha"))
TRAIN = (
    LabeledExample("t0", "team goal goal", "zulu"),
    LabeledExample("t1", "team wins match", "zulu"),
    LabeledExample("t2", "leaders treaty treaty", "alpha"),
    LabeledExample("t3", "leaders sign agreement", "alpha"),
)
VALIDATION = (LabeledExample("v0", "team validationonly", "zulu"),
              LabeledExample("v1", "treaty validationonly", "alpha"))


def fitted(training=None, validation=VALIDATION):
    classifier = TfidfLogisticClassifier(training=training)
    classifier.prepare(CLASSES)
    classifier.fit(TRAIN, validation_examples=validation)
    return classifier


def test_sparse_features_no_vocabulary_or_idf_leakage(monkeypatch):
    original_fit = LogisticRegression.fit
    original_proba = LogisticRegression.predict_proba
    def fit(self, features, labels):
        assert issparse(features)
        assert features.shape[0] == len(TRAIN)
        return original_fit(self, features, labels)
    def predict_proba(self, features):
        assert issparse(features)
        return original_proba(self, features)
    monkeypatch.setattr(LogisticRegression, "fit", fit)
    monkeypatch.setattr(LogisticRegression, "predict_proba", predict_proba)
    classifier = fitted()
    reference = TfidfVectorizer(ngram_range=(1, 2)).fit([e.text for e in TRAIN])
    assert classifier._vectorizer.vocabulary_ == reference.vocabulary_
    np.testing.assert_array_equal(classifier._vectorizer.idf_, reference.idf_)
    before = classifier._classifier.coef_.copy()
    classifier.predict([ClassificationInput("test", "testonly")])
    assert "validationonly" not in classifier._vectorizer.vocabulary_
    assert "testonly" not in classifier._vectorizer.vocabulary_
    assert classifier._vectorizer.transform(["testonly"]).nnz == 0
    np.testing.assert_array_equal(classifier._classifier.coef_, before)
    np.testing.assert_array_equal(classifier._vectorizer.idf_, reference.idf_)


def test_probability_mapping_order_normalization_confidence_and_reproducibility():
    classifier = fitted()
    assert isinstance(classifier, Classifier)
    assert list(classifier._classifier.classes_) == ["alpha", "zulu"]
    inputs = [ClassificationInput("second", "team goal"),
              ClassificationInput("first", "leaders treaty"),
              ClassificationInput("unknown", "unseenword")]
    predictions = classifier.predict(inputs)
    repeated = fitted().predict(inputs)
    assert [p.sample_id for p in predictions] == [e.sample_id for e in inputs]
    expected = classifier._classifier.predict_proba(classifier._vectorizer.transform([e.text for e in inputs]))
    for index, (prediction, other) in enumerate(zip(predictions, repeated)):
        assert set(prediction.probabilities) == {"zulu", "alpha"}
        assert all(np.isfinite(v) and 0 <= v <= 1 for v in prediction.probabilities.values())
        assert sum(prediction.probabilities.values()) == pytest.approx(1)
        assert prediction.probabilities["alpha"] == pytest.approx(expected[index, 0])
        assert prediction.probabilities["zulu"] == pytest.approx(expected[index, 1])
        assert prediction.predicted_label == max(prediction.probabilities, key=prediction.probabilities.get)
        assert prediction.confidence == prediction.probabilities[prediction.predicted_label]
        assert prediction.probabilities == other.probabilities
        assert prediction.latency_ms >= 0
    assert predictions[0].predicted_label == "zulu"
    assert predictions[1].predicted_label == "alpha"


def test_validation_selection_matches_independent_real_estimators():
    # Noisy fixed fixture produces different validation scores across the C grid.
    rng = np.random.default_rng(19)
    words = np.array(["apple", "banana", "cherry", "date", "elderberry"])
    def examples(prefix, count):
        return tuple(LabeledExample(f"{prefix}{i}", " ".join(rng.choice(words, 4)),
                                    str(rng.integers(2))) for i in range(count))
    train, validation = examples("train", 16), examples("val", 12)
    training = TfidfTrainingConfig(c_values=(1000.0, 1.0, 0.001))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    features = vectorizer.fit_transform([e.text for e in train])
    val_features = vectorizer.transform([e.text for e in validation])
    candidates = [LogisticRegression(C=c, max_iter=training.max_iter, random_state=training.seed)
                  .fit(features, [e.label for e in train]) for c in training.c_values]
    scores = [m.score(val_features, [e.label for e in validation]) for m in candidates]
    assert len(set(scores)) > 1
    best = candidates[int(np.argmax(scores))]
    assert best.C != training.c_values[0]
    classifier = TfidfLogisticClassifier(training=training)
    classifier.prepare([ClassDefinition("1", "One"), ClassDefinition("0", "Zero")])
    classifier.fit(train, validation_examples=validation)
    assert classifier.selected_c == best.C
    np.testing.assert_allclose(classifier._classifier.coef_, best.coef_)
    assert [s["validation_accuracy"] for s in classifier.fitted_metadata()["candidate_scores"]] == scores


def test_validation_ties_retain_first_configured_candidate():
    classifier = fitted(TfidfTrainingConfig(c_values=(10.0, 1.0, 0.1)))
    scores = classifier.fitted_metadata()["candidate_scores"]
    assert len({s["validation_accuracy"] for s in scores}) == 1
    assert classifier.selected_c == 10.0


def test_no_validation_fits_only_fixed_fallback_and_never_scores(monkeypatch):
    calls = []
    original = LogisticRegression.fit
    def fit(self, features, labels):
        calls.append(self.C)
        return original(self, features, labels)
    def score(*args, **kwargs):
        pytest.fail("No training/test scoring allowed for fallback selection")
    monkeypatch.setattr(LogisticRegression, "fit", fit)
    monkeypatch.setattr(LogisticRegression, "score", score)
    classifier = fitted(TfidfTrainingConfig(fallback_c=2.5), validation=())
    assert calls == [2.5]
    assert classifier.selected_c == 2.5
    assert classifier.fitted_metadata()["selection_metric"] == "fixed_fallback"


@pytest.mark.parametrize("train,validation,message", [
    ((), (), "training example"),
    (TRAIN[:2], (), "Missing training classes"),
    (TRAIN + (LabeledExample("bad", "text", "unknown"),), (), "unknown labels"),
    (TRAIN, (LabeledExample("bad", "text", "unknown"),), "unknown labels"),
    ((LabeledExample("a", "!", "alpha"), LabeledExample("z", "?", "zulu")), (), "vocabulary"),
])
def test_failed_refit_clears_old_model(train, validation, message):
    classifier = fitted()
    with pytest.raises(ValueError, match=message):
        classifier.fit(train, validation_examples=validation)
    with pytest.raises(RuntimeError, match="fit"):
        classifier.predict([ClassificationInput("test", "team")])
    assert classifier.selected_c is None
    assert classifier.fitted_metadata() == {}


def test_lifecycle_and_prepare_reset():
    classifier = TfidfLogisticClassifier()
    with pytest.raises(RuntimeError, match="prepare"):
        classifier.fit(TRAIN)
    with pytest.raises(RuntimeError):
        classifier.predict([])
    classifier = fitted()
    classifier.prepare([ClassDefinition("other", "Other"), ClassDefinition("new", "New")])
    with pytest.raises(RuntimeError):
        classifier.predict([ClassificationInput("test", "team")])
    assert classifier.selected_c is None
    for classes in (CLASSES[:1], (CLASSES[0], CLASSES[0])):
        with pytest.raises(ValueError):
            classifier.prepare(classes)
        with pytest.raises(RuntimeError):
            classifier.fit(TRAIN)


@pytest.mark.parametrize("kwargs", [
    {"c_values": ()}, {"c_values": (float("nan"),)}, {"c_values": (0,)},
    {"fallback_c": float("inf")}, {"fallback_c": -1}, {"max_iter": 0},
    {"ngram_range": (2, 1)}, {"ngram_range": (1.0, 2)}, {"min_df": 0},
    {"max_features": 0}, {"seed": -1}, {"seed": 2**32}, {"lowercase": "yes"},
])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        TfidfTrainingConfig(**kwargs)
