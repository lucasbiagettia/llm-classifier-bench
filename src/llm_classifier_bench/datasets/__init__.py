"""Dataset contracts, Hugging Face adapter, and initial registry."""

from .base import ClassificationDataset, DatasetBundle, SplitName
from .huggingface import HFDatasetSpec, HuggingFaceClassificationDataset
from .registry import (
    AG_NEWS_SPEC,
    BANKING77_SPEC,
    DATASET_REGISTRY,
    get_dataset,
)

__all__ = [
    "ClassificationDataset",
    "DatasetBundle",
    "SplitName",
    "HFDatasetSpec",
    "HuggingFaceClassificationDataset",
    "AG_NEWS_SPEC",
    "BANKING77_SPEC",
    "DATASET_REGISTRY",
    "get_dataset",
]
