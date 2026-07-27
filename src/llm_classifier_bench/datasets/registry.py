"""Small explicit registry of benchmark datasets."""

from __future__ import annotations

from llm_classifier_bench.datasets.huggingface import (
    HFDatasetSpec,
    HuggingFaceClassificationDataset,
)


BANKING77_SPEC = HFDatasetSpec(
    name="banking77",
    path="PolyAI/banking77",
    label_description_template=(
        "Banking customer-support intent about {readable_label}."
    ),
)

AG_NEWS_SPEC = HFDatasetSpec(
    name="ag_news",
    path="fancyzhx/ag_news",
    label_descriptions={
        "World": "News about international affairs, governments, conflicts, and world events.",
        "Sports": "News about sports, athletes, teams, matches, and competitions.",
        "Business": "News about companies, markets, finance, trade, and the economy.",
        "Sci/Tech": "News about science, technology, computing, research, and engineering.",
    },
)

DATASET_REGISTRY: dict[str, HFDatasetSpec] = {
    BANKING77_SPEC.name: BANKING77_SPEC,
    AG_NEWS_SPEC.name: AG_NEWS_SPEC,
}


def get_dataset(name: str) -> HuggingFaceClassificationDataset:
    """Build a dataset adapter from its stable registry name."""

    try:
        spec = DATASET_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset {name!r}. Available: {available}") from exc

    return HuggingFaceClassificationDataset(spec)
