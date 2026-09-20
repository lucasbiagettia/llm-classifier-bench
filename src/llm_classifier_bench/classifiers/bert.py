"""Supervised BERT-family sequence classifier using Hugging Face Transformers."""

from __future__ import annotations

import copy
import random
from time import perf_counter
from typing import Any, Sequence

from llm_classifier_bench.classifiers.base import Prediction
from llm_classifier_bench.preparation import preparation_stage
from llm_classifier_bench.config import BertTrainingConfig, DEFAULT_BERT_MODEL
from llm_classifier_bench.core import ClassificationInput, ClassDefinition, LabeledExample


class BertClassifier:
    """Fine-tuned transformer sequence classifier.

    ``fit`` trains on the provided training split. When validation examples are
    available, the best epoch by validation loss is restored before final inference.
    The benchmark's held-out test split is never consumed here.
    """

    requires_full_training_coverage = True

    def __init__(
        self,
        *,
        model: str = DEFAULT_BERT_MODEL,
        training: BertTrainingConfig | None = None,
        classifier_name: str = "bert-finetuned",
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        if not classifier_name.strip():
            raise ValueError("classifier_name cannot be empty")

        self.model = model
        self.training = training or BertTrainingConfig()
        self._name = classifier_name
        self._classes: tuple[ClassDefinition, ...] = ()
        self._class_names: tuple[str, ...] = ()
        self._label_to_id: dict[str, int] = {}
        self._id_to_label: dict[int, str] = {}
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None
        self.supervision_regime = "supervised"
        self.training_examples_used = 0
        self.validation_examples_used = 0

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
        self._label_to_id = {label: index for index, label in enumerate(names)}
        self._id_to_label = {index: label for label, index in self._label_to_id.items()}

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
        missing_train_labels = sorted(set(self._class_names) - {item.label for item in train})
        if missing_train_labels:
            raise ValueError(
                "Training split must contain every configured class: "
                + ", ".join(missing_train_labels)
            )

        self._fit_metadata = {}
        torch, AutoTokenizer, AutoModelForSequenceClassification = _load_transformer_stack()
        _set_seed(torch, self.training.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        sync = lambda: torch.cuda.synchronize(device) if str(device).startswith("cuda") else None
        with preparation_stage(self, "bert.load", "load", synchronize=sync,
                               model=self.model, cache="disk_cache_hit_unknown") as evidence:
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model,
                num_labels=len(self._class_names),
                label2id=self._label_to_id,
                id2label=self._id_to_label,
            )
            model.to(device)
            evidence["resolved_revision"] = getattr(getattr(model, "config", None), "_commit_hash", None)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.training.learning_rate,
            weight_decay=self.training.weight_decay,
        )

        best_state: dict[str, Any] | None = None
        best_validation_loss = float("inf")
        selected_epoch = self.training.epochs - 1

        for _epoch in range(self.training.epochs):
            with preparation_stage(self, f"bert.epoch.{_epoch}", "fit", synchronize=sync, epoch=_epoch):
                model.train()
                batches = _batched(train, self.training.batch_size, shuffle=True, seed=self.training.seed + _epoch)
                for batch_index, batch in enumerate(batches):
                    with preparation_stage(self, f"bert.tokenize.{_epoch}.{batch_index}", "tokenization", epoch=_epoch, partition="fit_train"):
                        encoded = tokenizer(
                            [item.text for item in batch],
                            padding=True,
                            truncation=True,
                            max_length=self.training.max_length,
                            return_tensors="pt",
                        )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    encoded["labels"] = torch.tensor(
                        [self._label_to_id[item.label] for item in batch],
                        dtype=torch.long,
                        device=device,
                    )

                    optimizer.zero_grad(set_to_none=True)
                    output = model(**encoded)
                    output.loss.backward()
                    optimizer.step()
            if validation:
                with preparation_stage(self, f"bert.validate.{_epoch}", "selection", synchronize=sync, epoch=_epoch):
                    validation_loss = _mean_validation_loss(
                        torch=torch,
                        model=model,
                        tokenizer=tokenizer,
                        examples=validation,
                        label_to_id=self._label_to_id,
                        batch_size=self.training.batch_size,
                        max_length=self.training.max_length,
                        device=device,
                        preparation_owner=self,
                    )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    selected_epoch = _epoch
                    with preparation_stage(self, f"bert.save.{_epoch}", "selection", synchronize=sync, epoch=_epoch):
                        best_state = {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        }

        if best_state is not None:
            with preparation_stage(self, "bert.restore", "restore", synchronize=sync, selected_epoch=selected_epoch):
                model.load_state_dict(best_state)
                model.to(device)

        with preparation_stage(self, "bert.select", "selection",
                               selected_configuration={"epoch": selected_epoch + 1}, final_refit_performed=False,
                               selected_fit_names=[f"bert.epoch.{e}" for e in range(selected_epoch+1)],
                               selection_basis="cumulative training prefix; epochs are not independent candidates"):
            pass
        self.training_examples_used = len(train)
        self.validation_examples_used = len(validation)
        self._fit_metadata = {"training_examples_used": len(train), "validation_examples_used": len(validation),
                              "selected_epoch": selected_epoch+1, "epochs_run": self.training.epochs,
                              "final_refit_performed": False,
                              "selection_metric": "validation_loss" if validation else "last_epoch"}
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def fitted_metadata(self):
        return dict(getattr(self, "_fit_metadata", {}))

    def set_preparation_sink(self, sink):
        self._preparation_sink = sink

    def inference_metadata(self) -> dict[str, Any]:
        return {"backend": "local", "device": str(self._device) if self._device is not None else None,
                "max_input_tokens": self.training.max_length, "truncation": True,
                "transport_max_retries": 0, "batch_execution": "sequential_single_example"}

    def predict(self, examples: Sequence[ClassificationInput]) -> list[Prediction]:
        if self._tokenizer is None or self._model is None or self._device is None:
            raise RuntimeError("Call prepare(classes) and fit(...) before predict()")

        torch, _, _ = _load_transformer_stack()
        predictions: list[Prediction] = []

        for example in examples:
            started_at = perf_counter()
            encoded = self._tokenizer(
                example.text,
                truncation=True,
                max_length=self.training.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self._model(**encoded).logits[0]
                probability_values = torch.softmax(logits, dim=-1).detach().cpu().tolist()
            latency_ms = (perf_counter() - started_at) * 1_000

            probabilities = {
                self._id_to_label[index]: float(probability)
                for index, probability in enumerate(probability_values)
            }
            predicted_label = max(probabilities, key=probabilities.__getitem__)

            predictions.append(
                Prediction(
                    sample_id=example.sample_id,
                    predicted_label=predicted_label,
                    confidence=probabilities[predicted_label],
                    probabilities=probabilities,
                    latency_ms=latency_ms,
                    model=self.model,
                    request_id=None,
                    raw_response={"logits": [float(value) for value in logits.detach().cpu().tolist()]},
                )
            )

        return predictions


def _load_transformer_stack() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch and transformers are required. Run pip install -r requirements.txt."
        ) from exc
    return torch, AutoTokenizer, AutoModelForSequenceClassification


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batched(
    examples: Sequence[LabeledExample],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> list[tuple[LabeledExample, ...]]:
    indices = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    return [
        tuple(examples[index] for index in indices[start : start + batch_size])
        for start in range(0, len(indices), batch_size)
    ]


def _mean_validation_loss(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    examples: Sequence[LabeledExample],
    label_to_id: dict[str, int],
    batch_size: int,
    max_length: int,
    device: Any,
    preparation_owner=None,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in _batched(examples, batch_size, shuffle=False, seed=0):
            with preparation_stage(preparation_owner, "bert.validation_tokenize", "tokenization", partition="validation"):
                encoded = tokenizer(
                    [item.text for item in batch],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            encoded["labels"] = torch.tensor(
                [label_to_id[item.label] for item in batch],
                dtype=torch.long,
                device=device,
            )
            losses.append(float(model(**encoded).loss.detach().cpu().item()))
    model.train()
    return sum(losses) / len(losses)


__all__ = ["BertClassifier"]
