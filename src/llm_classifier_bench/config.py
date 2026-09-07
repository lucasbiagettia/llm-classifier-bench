"""Central defaults for benchmark classifiers and train/validation splitting."""

from __future__ import annotations

import math
from dataclasses import dataclass


# OpenAI's cheapest GPT-5 model is a useful low-cost generative classification
# baseline. Pin the snapshot for reproducible benchmark runs; change it here when a
# different model is intentionally selected.
DEFAULT_OPENAI_MODEL = "gpt-5-nano"
# GPT-5 family supports configurable reasoning effort. Classification is a
# latency-sensitive closed-set task, so keep reasoning at the minimum level.
DEFAULT_OPENAI_REASONING_EFFORT = "minimal"

# Independent from the benchmark OpenAI classifier. It may point to the same
# cheap snapshot today, but changing the ontology generator must not change the
# classifier configuration.
DEFAULT_CLASS_DEFINITION_GENERATOR_MODEL = "gpt-5-nano"
DEFAULT_CLASS_DEFINITION_GENERATOR_REASONING_EFFORT = "minimal"

# Canonical Hugging Face checkpoints for the two supervised baselines.
DEFAULT_BERT_MODEL = "google-bert/bert-base-uncased"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_SPLIT_SEED = 42


@dataclass(frozen=True, slots=True)
class EmissaryTrainingConfig:
    """Labeled-example budget and explicit Emissary preparation mechanism."""

    shots: int = 0
    shot_unit: str | None = None
    selection_seed: int = 42
    selection_policy: str = "balanced_round_robin_v1"
    mechanism: str | None = None
    project_id: str | None = None
    base_model: str | None = None
    dataset_id: str | None = None
    dataset_sha256: str | None = None
    training_job_id: str | None = None
    checkpoint: int | None = None
    deployment_id: str | None = None
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 1
    dynamic_loss: bool = False
    dataset_timeout_s: float = 300.0
    training_timeout_s: float = 3_600.0
    deployment_timeout_s: float = 1_800.0
    poll_interval_s: float = 10.0
    inactive_timeout_s: int = 300

    def __post_init__(self) -> None:
        if type(self.shots) is not int or self.shots < 0:
            raise ValueError("shots must be a nonnegative integer")
        if self.shot_unit not in (None, "total", "per_class"):
            raise ValueError("shot_unit must be total or per_class")
        if self.shots and self.shot_unit is None:
            raise ValueError("Nonzero shots require an explicit shot_unit")
        if type(self.selection_seed) is not int:
            raise ValueError("selection_seed must be an integer")
        if self.selection_policy != "balanced_round_robin_v1":
            raise ValueError("Unsupported selection_policy; use balanced_round_robin_v1")
        if self.mechanism not in (None, "project_fine_tuning"):
            raise ValueError("mechanism must be project_fine_tuning when provided")
        if self.shots == 0 and self.mechanism is not None:
            raise ValueError("Zero-shot conditions do not use a training mechanism")
        provider_fields = (
            self.project_id,
            self.base_model,
            self.dataset_id,
            self.dataset_sha256,
            self.training_job_id,
            self.checkpoint,
            self.deployment_id,
        )
        if self.mechanism is None and any(value is not None for value in provider_fields):
            raise ValueError(
                "Project fields require mechanism=project_fine_tuning explicitly"
            )
        if self.mechanism == "project_fine_tuning":
            if self.shots == 0:
                raise ValueError("project_fine_tuning requires nonzero shots")
            for name in ("project_id", "base_model"):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"project_fine_tuning requires {name}")
        for name in (
            "project_id",
            "base_model",
            "dataset_id",
            "dataset_sha256",
            "training_job_id",
            "deployment_id",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string when provided")
        if self.deployment_id is not None and self.training_job_id is None:
            raise ValueError("deployment_id continuation requires training_job_id")
        if self.training_job_id is not None and self.dataset_id is None:
            raise ValueError("training_job_id continuation requires dataset_id")
        continuation_ids = (self.dataset_id, self.training_job_id, self.deployment_id)
        if any(continuation_ids):
            if (
                self.dataset_sha256 is None
                or len(self.dataset_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.dataset_sha256
                )
            ):
                raise ValueError(
                    "Continuation IDs require the lowercase dataset_sha256 from "
                    "the original fit metadata"
                )
        if self.checkpoint is not None and (
            type(self.checkpoint) is not int or self.checkpoint < 0
        ):
            raise ValueError("checkpoint must be a nonnegative integer")
        for name in (
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "inactive_timeout_s",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive")
        if type(self.dynamic_loss) is not bool:
            raise ValueError("dynamic_loss must be boolean")
        for name in (
            "dataset_timeout_s",
            "training_timeout_s",
            "deployment_timeout_s",
            "poll_interval_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")

    def require_live_support(self) -> None:
        if self.shots and self.mechanism != "project_fine_tuning":
            raise NotImplementedError(
                "Nonzero Emissary shots require the explicit project_fine_tuning "
                "mechanism. It trains a separate base model rather than the routing "
                "experiment; see docs/emissary_contract.md."
            )

    @property
    def uses_project_fine_tuning(self) -> bool:
        return self.shots > 0 and self.mechanism == "project_fine_tuning"

    def training_parameters(self) -> dict[str, int | float | bool]:
        return {
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "dynamic_loss": self.dynamic_loss,
        }


@dataclass(frozen=True, slots=True)
class BertTrainingConfig:
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_length: int = 256
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.max_length < 8:
            raise ValueError("max_length must be at least 8")


@dataclass(frozen=True, slots=True)
class SentenceTransformerTrainingConfig:
    embedding_batch_size: int = 64
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0)
    max_iter: int = 2_000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be at least 1")
        if not self.c_values or any(value <= 0 for value in self.c_values):
            raise ValueError("c_values must contain positive values")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1")


@dataclass(frozen=True, slots=True)
class TfidfTrainingConfig:
    """Sparse lexical baseline; grid order breaks validation-accuracy ties."""

    ngram_range: tuple[int, int] = (1, 2)
    lowercase: bool = True
    min_df: int = 1
    max_features: int | None = None
    sublinear_tf: bool = False
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0)
    fallback_c: float = 1.0
    max_iter: int = 2_000
    seed: int = 42

    def __post_init__(self) -> None:
        if (not isinstance(self.ngram_range, tuple) or len(self.ngram_range) != 2
                or any(type(n) is not int for n in self.ngram_range)
                or not 1 <= self.ngram_range[0] <= self.ngram_range[1]):
            raise ValueError("ngram_range must be an ordered pair of positive integers")
        for name in ("min_df", "max_iter"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_features is not None and (
            type(self.max_features) is not int or self.max_features < 1
        ):
            raise ValueError("max_features must be None or a positive integer")
        for name in ("lowercase", "sublinear_tf"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not isinstance(self.c_values, tuple) or not self.c_values:
            raise ValueError("c_values must be a nonempty tuple")
        for value in (*self.c_values, self.fallback_c):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError("c_values and fallback_c must be finite and positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")


__all__ = [
    "EmissaryTrainingConfig",
    "BertTrainingConfig",
    "DEFAULT_CLASS_DEFINITION_GENERATOR_MODEL",
    "DEFAULT_CLASS_DEFINITION_GENERATOR_REASONING_EFFORT",
    "DEFAULT_BERT_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_OPENAI_REASONING_EFFORT",
    "DEFAULT_SENTENCE_TRANSFORMER_MODEL",
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_VALIDATION_FRACTION",
    "SentenceTransformerTrainingConfig",
    "TfidfTrainingConfig",
]
