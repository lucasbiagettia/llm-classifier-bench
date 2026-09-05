"""Sparse TF-IDF features and supervised logistic regression."""

from __future__ import annotations

from importlib.metadata import version
from time import perf_counter
from typing import Any, Sequence

from llm_classifier_bench.classifiers.base import Prediction
from llm_classifier_bench.config import TfidfTrainingConfig
from llm_classifier_bench.core import ClassificationInput, ClassDefinition, LabeledExample


class TfidfLogisticClassifier:
    """Fit features/weights on training only; select C on validation accuracy.

    Ties retain the first configured candidate. With no validation, only
    ``fallback_c`` is fitted. Neither path refits on train plus validation.
    """

    supervision_regime = "supervised"
    model = "tfidf-logistic-regression"

    def __init__(
        self, *, training: TfidfTrainingConfig | None = None,
        classifier_name: str = "tfidf-logreg",
    ) -> None:
        if not classifier_name.strip():
            raise ValueError("classifier_name cannot be empty")
        self.training = training or TfidfTrainingConfig()
        self._name = classifier_name
        self._class_names: tuple[str, ...] = ()
        self._reset_fit()

    @property
    def name(self) -> str:
        return self._name

    def _reset_fit(self) -> None:
        self._vectorizer: Any | None = None
        self._classifier: Any | None = None
        self.selected_c: float | None = None
        self.training_examples_used = 0
        self.validation_examples_used = 0
        self._fit_metadata: dict[str, Any] = {}

    def prepare(self, classes: Sequence[ClassDefinition]) -> None:
        self._reset_fit()
        self._class_names = ()
        names = tuple(item.name for item in classes)
        if len(names) < 2:
            raise ValueError("At least two classes are required")
        if len(names) != len(set(names)):
            raise ValueError("Class names must be unique")
        self._class_names = names

    def fit(
        self, examples: Sequence[LabeledExample], *,
        validation_examples: Sequence[LabeledExample] = (),
    ) -> None:
        self._reset_fit()
        if not self._class_names:
            raise RuntimeError("Call prepare(classes) before fit()")
        train, validation = tuple(examples), tuple(validation_examples)
        if not train:
            raise ValueError("At least one training example is required")
        labels = set(self._class_names)
        unknown = {item.label for item in train + validation} - labels
        if unknown:
            raise ValueError(f"Data contains unknown labels: {sorted(unknown)}")
        missing = labels - {item.label for item in train}
        if missing:
            raise ValueError(f"Missing training classes: {sorted(missing)}")

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        vectorizer_settings = {
            "analyzer": "word", "ngram_range": self.training.ngram_range,
            "lowercase": self.training.lowercase, "stop_words": None,
            "min_df": self.training.min_df, "max_df": 1.0,
            "max_features": self.training.max_features,
            "sublinear_tf": self.training.sublinear_tf, "norm": "l2",
            "use_idf": True, "smooth_idf": True,
            "token_pattern": r"(?u)\b\w\w+\b", "strip_accents": None,
        }
        vectorizer = TfidfVectorizer(**vectorizer_settings)
        try:
            train_features = vectorizer.fit_transform([item.text for item in train])
        except ValueError as exc:
            raise ValueError(f"Could not fit TF-IDF vocabulary: {exc}") from exc
        train_labels = [item.label for item in train]
        validation_features = (
            vectorizer.transform([item.text for item in validation]) if validation else None
        )
        validation_labels = [item.label for item in validation]
        candidates = self.training.c_values if validation else (self.training.fallback_c,)
        best_score = float("-inf")
        best_classifier = None
        scores = []
        for c_value in candidates:
            candidate = LogisticRegression(
                C=c_value, max_iter=self.training.max_iter,
                random_state=self.training.seed, solver="lbfgs",
            )
            candidate.fit(train_features, train_labels)
            score = (float(candidate.score(validation_features, validation_labels))
                     if validation else None)
            scores.append({"c": c_value, "validation_accuracy": score})
            if best_classifier is None or (score is not None and score > best_score):
                best_classifier = candidate
                best_score = score if score is not None else best_score

        # Publish fitted state only after all candidates have completed successfully.
        self._fit_metadata = {
            "supervision_regime": self.supervision_regime,
            "training_examples_used": len(train),
            "validation_examples_used": len(validation),
            "vectorizer": vectorizer_settings,
            "vocabulary_size": len(vectorizer.vocabulary_),
            "c_candidates": list(self.training.c_values),
            "fallback_c": self.training.fallback_c,
            "selected_c": best_classifier.C,
            "selection_metric": "validation_accuracy" if validation else "fixed_fallback",
            "tie_breaking": "first_candidate_in_configured_order",
            "candidate_scores": scores,
            "seed": self.training.seed,
            "solver": "lbfgs",
            "library_versions": {name: version(name) for name in ("scikit-learn", "numpy", "scipy")},
        }
        self._vectorizer, self._classifier = vectorizer, best_classifier
        self.selected_c = best_classifier.C
        self.training_examples_used = len(train)
        self.validation_examples_used = len(validation)

    def fitted_metadata(self) -> dict[str, Any]:
        """JSON-compatible post-fit information for the runner's optional hook."""
        return dict(self._fit_metadata)

    def predict(self, examples: Sequence[ClassificationInput]) -> list[Prediction]:
        if self._vectorizer is None or self._classifier is None:
            raise RuntimeError("Call prepare(classes) and fit(...) before predict()")
        predictions = []
        for example in examples:
            started_at = perf_counter()
            features = self._vectorizer.transform([example.text])
            values = self._classifier.predict_proba(features)[0]
            latency_ms = (perf_counter() - started_at) * 1_000
            probabilities = {
                str(label): float(value)
                for label, value in zip(self._classifier.classes_, values, strict=True)
            }
            label = max(probabilities, key=probabilities.__getitem__)
            predictions.append(Prediction(
                sample_id=example.sample_id, predicted_label=label,
                probabilities=probabilities, confidence=probabilities[label],
                latency_ms=latency_ms, model=self.model,
                raw_response={"selected_c": self.selected_c},
            ))
        return predictions


__all__ = ["TfidfLogisticClassifier"]
