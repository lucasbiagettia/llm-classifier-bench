"""Dataset contracts, Hugging Face adapter, and initial registry."""

from .base import ClassificationDataset, DatasetBundle, SplitName
from .huggingface import HFDatasetSpec, HuggingFaceClassificationDataset
from .hard_tasks import AnliDataset, ImplicitHateDataset
from .registry import (
    AG_NEWS_SPEC,
    BANKING77_SPEC,
    DATASET_REGISTRY,
    get_dataset,
)

__all__ = [
    "ClassificationDataset",
    "AnliDataset",
    "ImplicitHateDataset",
    "DatasetBundle",
    "SplitName",
    "HFDatasetSpec",
    "HuggingFaceClassificationDataset",
    "AG_NEWS_SPEC",
    "BANKING77_SPEC",
    "DATASET_REGISTRY",
    "get_dataset",
]
