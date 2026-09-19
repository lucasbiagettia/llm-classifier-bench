"""Frozen SentenceTransformer embeddings with supervised logistic regression."""

from __future__ import annotations

from importlib.metadata import version
from time import perf_counter
from typing import Any, Sequence

from llm_classifier_bench.classifiers.base import Prediction
from llm_classifier_bench.preparation import preparation_stage
from llm_classifier_bench.config import (
    DEFAULT_SENTENCE_TRANSFORMER_MODEL,
    SentenceTransformerTrainingConfig,
)
from llm_classifier_bench.core import ClassificationInput, ClassDefinition, LabeledExample


class SentenceTransformerLogisticClassifier:
    """Frozen sentence embeddings followed by multinomial logistic regression.

    The embedding model is never fine-tuned. Validation examples are used only to
    choose the logistic-regression ``C`` value from the configured candidate grid.
    The final held-out dataset test split is never seen during training or selection.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        training: SentenceTransformerTrainingConfig | None = None,
        classifier_name: str = "sentence-transformer-logreg",
        encoder: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        if not classifier_name.strip():
            raise ValueError("classifier_name cannot be empty")

        self.model = model
        self.training = training or SentenceTransformerTrainingConfig()
        self._name = classifier_name
        self._encoder = encoder
        self._classifier: Any | None = None
        self._classes: tuple[ClassDefinition, ...] = ()
        self._class_names: tuple[str, ...] = ()
        self.selected_c: float | None = None
        self._fit_metadata: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def prepare(self, classes: Sequence[ClassDefinition]) -> None:
        frozen = tuple(classes)
        if len(frozen) < 2:
            raise ValueError("At least two classes are required")
        names = tuple(item.name for item in frozen)
        if len(names) != len(set(names)):
            raise ValueError("Class names must be unique")
        self._classes = frozen
        self._class_names = names

    def fit(
        self,
        examples: Sequence[LabeledExample],
        *,
        validation_examples: Sequence[LabeledExample] = (),
    ) -> None:
        if not self._class_names:
            raise RuntimeError("Call prepare(classes) before fit()")
        train = tuple(examples)
        validation = tuple(validation_examples)
        if not train:
            raise ValueError("At least one training example is required")

        unknown = sorted({item.label for item in train + validation} - set(self._class_names))
        if unknown:
            raise ValueError(f"Training data contains unknown labels: {unknown}")

        with preparation_stage(self, "encoder.load", "load",
                               cache="reused_in_memory" if self._encoder is not None else "disk_cache_hit_unknown",
                               model=self.model, original_build_cost="unavailable") as evidence:
            encoder = self._encoder or _load_sentence_transformer(self.model)
            evidence["encoder_type"] = f"{type(encoder).__module__}.{type(encoder).__qualname__}"
            module = getattr(encoder, "_modules", {}).get("0")
            encoder_config = getattr(getattr(module, "auto_model", None), "config", None)
            evidence["resolved_revision"] = getattr(encoder_config, "_commit_hash", None)
        self._encoder = encoder

        with preparation_stage(self, "encoder.fit_train", "features", cache="computed_this_run", partition="fit_train"):
            train_embeddings = encoder.encode(
                [item.text for item in train],
                batch_size=self.training.embedding_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        train_labels = [item.label for item in train]

        validation_embeddings = None
        validation_labels: list[str] = []
        if validation:
            with preparation_stage(self, "encoder.validation", "features", cache="computed_this_run", partition="validation"):
                validation_embeddings = encoder.encode(
                    [item.text for item in validation],
                    batch_size=self.training.embedding_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            validation_labels = [item.label for item in validation]

        LogisticRegression = _load_logistic_regression()
        best_classifier: Any | None = None
        best_score = float("-inf")
        best_c: float | None = None
        candidate_scores = []

        for candidate_index, c_value in enumerate(self.training.c_values):
            candidate = LogisticRegression(
                C=c_value,
                max_iter=self.training.max_iter,
                random_state=self.training.seed,
            )
            with preparation_stage(self, f"lr.fit.{candidate_index}", "fit", c=c_value):
                candidate.fit(train_embeddings, train_labels)

            if validation_embeddings is None:
                score = 0.0
            else:
                with preparation_stage(self, f"lr.validate.{candidate_index}", "selection", c=c_value):
                    score = float(candidate.score(validation_embeddings, validation_labels))

            candidate_scores.append({
                "c": c_value,
                "validation_accuracy": score if validation else None,
            })

            if score > best_score:
                best_score = score
                best_classifier = candidate
                best_c = c_value
                best_index = candidate_index

        if best_classifier is None or best_c is None:  # defensive
            raise RuntimeError("Could not fit logistic regression")

        with preparation_stage(self, "lr.select", "selection",
                               selected_configuration={"c": best_c}, final_refit_performed=False,
                               selected_fit_names=[f"lr.fit.{best_index}"]):
            pass
        self._classifier = best_classifier
        self.selected_c = best_c
        self._fit_metadata = {
            "selected_c": best_c,
            "final_refit_performed": False,
            "candidate_scores": candidate_scores,
            "selection_metric": "validation_accuracy" if validation else "first_candidate_no_validation",
            "tie_breaking": "first_candidate_in_configured_order",
            "training_examples_used": len(train),
            "validation_examples_used": len(validation),
            "probability_label_order": [str(label) for label in best_classifier.classes_],
            "library_versions": {name: version(name) for name in ("scikit-learn", "numpy", "scipy")},
        }

    def fitted_metadata(self) -> dict[str, Any]:
        """Persist candidate scores through the runner's existing metadata hook."""
        return dict(self._fit_metadata)

    def set_preparation_sink(self, sink):
        self._preparation_sink = sink

    def inference_metadata(self) -> dict[str, Any]:
        device = getattr(self._encoder, "device", None)
        return {"backend": "local", "device": str(device) if device is not None else None,
                "max_input_tokens": getattr(self._encoder, "max_seq_length", None),
                "transport_max_retries": 0, "batch_execution": "sequential_single_example"}

    def predict(self, examples: Sequence[ClassificationInput]) -> list[Prediction]:
        if self._encoder is None or self._classifier is None:
            raise RuntimeError("Call prepare(classes) and fit(...) before predict()")

        predictions: list[Prediction] = []
        for example in examples:
            started_at = perf_counter()
            embedding = self._encoder.encode(
                [example.text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            probability_values = self._classifier.predict_proba(embedding)[0]
            latency_ms = (perf_counter() - started_at) * 1_000

            learned_labels = [str(label) for label in self._classifier.classes_]
            probabilities = {
                label: float(probability)
                for label, probability in zip(learned_labels, probability_values, strict=True)
            }
            predicted_label = max(probabilities, key=probabilities.__getitem__)

            # The runner requires a complete probability map over the benchmark class
            # space. A supervised run must therefore have seen every selected class.
            missing_labels = set(self._class_names) - set(probabilities)
            if missing_labels:
                raise ValueError(
                    "Training split did not expose all configured classes: "
                    + ", ".join(sorted(missing_labels))
                )

            predictions.append(
                Prediction(
                    sample_id=example.sample_id,
                    predicted_label=predicted_label,
                    confidence=probabilities[predicted_label],
                    probabilities=probabilities,
                    latency_ms=latency_ms,
                    model=self.model,
                    request_id=None,
                    raw_response={"selected_c": self.selected_c},
                )
            )

        return predictions


def _load_sentence_transformer(model: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is not installed. Run pip install -r requirements.txt."
        ) from exc
    return SentenceTransformer(model)


def _load_logistic_regression() -> Any:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is not installed. Run pip install -r requirements.txt."
        ) from exc
    return LogisticRegression


__all__ = ["SentenceTransformerLogisticClassifier"]
