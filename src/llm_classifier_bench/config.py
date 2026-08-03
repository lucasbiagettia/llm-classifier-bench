"""Central defaults for benchmark classifiers and train/validation splitting."""

from __future__ import annotations

from dataclasses import dataclass


# OpenAI's cheapest GPT-5 model is a useful low-cost generative classification
# baseline. Pin the snapshot for reproducible benchmark runs; change it here when a
# different model is intentionally selected.
DEFAULT_OPENAI_MODEL = "gpt-5-nano"
# GPT-5 family supports configurable reasoning effort. Classification is a
# latency-sensitive closed-set task, so keep reasoning at the minimum level.
DEFAULT_OPENAI_REASONING_EFFORT = "minimal"

# Canonical Hugging Face checkpoints for the two supervised baselines.
DEFAULT_BERT_MODEL = "google-bert/bert-base-uncased"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_VALIDATION_FRACTION = 0.10
DEFAULT_SPLIT_SEED = 42


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


__all__ = [
    "BertTrainingConfig",
    "DEFAULT_BERT_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_OPENAI_REASONING_EFFORT",
    "DEFAULT_SENTENCE_TRANSFORMER_MODEL",
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_VALIDATION_FRACTION",
    "SentenceTransformerTrainingConfig",
]
