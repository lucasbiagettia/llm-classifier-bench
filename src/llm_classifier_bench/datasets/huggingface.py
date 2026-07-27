"""Generic adapter for text-classification datasets on Hugging Face Hub."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.datasets.base import DatasetBundle


DatasetLoader = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class HFDatasetSpec:
    """Declarative mapping from a Hugging Face dataset to our common contract."""

    name: str
    path: str
    text_column: str = "text"
    label_column: str = "label"
    train_split: str = "train"
    test_split: str = "test"
    config_name: str | None = None
    revision: str | None = None
    label_names: tuple[str, ...] | None = None
    label_descriptions: Mapping[str, str] = field(default_factory=dict)
    label_description_template: str = "Category: {readable_label}."

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "path",
            "text_column",
            "label_column",
            "train_split",
            "test_split",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"HFDatasetSpec.{field_name} cannot be empty")

        if self.label_names is not None:
            if len(self.label_names) < 2:
                raise ValueError("label_names must contain at least two labels")
            if len(self.label_names) != len(set(self.label_names)):
                raise ValueError("label_names must be unique")


class HuggingFaceClassificationDataset:
    """Load any compatible Hugging Face dataset through an ``HFDatasetSpec``."""

    def __init__(
        self,
        spec: HFDatasetSpec,
        *,
        loader: DatasetLoader | None = None,
    ) -> None:
        self.spec = spec
        self._loader = loader or _load_dataset

    @property
    def name(self) -> str:
        return self.spec.name

    def load(self) -> DatasetBundle:
        train_raw = self._load_split(self.spec.train_split)
        test_raw = self._load_split(self.spec.test_split)
        label_names = self._resolve_label_names(train_raw, test_raw)

        classes = tuple(
            ClassDefinition(
                name=label_name,
                description=self._description_for(label_name),
            )
            for label_name in label_names
        )

        return DatasetBundle(
            name=self.name,
            classes=classes,
            train=self._normalize_split(train_raw, "train", label_names),
            test=self._normalize_split(test_raw, "test", label_names),
            metadata={
                "source": "huggingface",
                "path": self.spec.path,
                "config_name": self.spec.config_name,
                "revision": self.spec.revision,
                "train_split": self.spec.train_split,
                "test_split": self.spec.test_split,
                "text_column": self.spec.text_column,
                "label_column": self.spec.label_column,
            },
        )

    def _load_split(self, split: str) -> Any:
        kwargs: dict[str, Any] = {
            "path": self.spec.path,
            "split": split,
        }
        if self.spec.config_name is not None:
            kwargs["name"] = self.spec.config_name
        if self.spec.revision is not None:
            kwargs["revision"] = self.spec.revision
        return self._loader(**kwargs)

    def _resolve_label_names(self, train_raw: Any, test_raw: Any) -> tuple[str, ...]:
        if self.spec.label_names is not None:
            return self.spec.label_names

        for split in (train_raw, test_raw):
            features = getattr(split, "features", None)
            if features is None:
                continue

            label_feature = features.get(self.spec.label_column)
            names = getattr(label_feature, "names", None)
            if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
                normalized_names = tuple(str(name) for name in names)
                if len(normalized_names) >= 2:
                    return normalized_names

        string_labels: set[str] = set()
        for split in (train_raw, test_raw):
            for row in split:
                raw_label = row[self.spec.label_column]
                if not isinstance(raw_label, str):
                    raise ValueError(
                        "Could not infer label names. Provide HFDatasetSpec.label_names "
                        "or use a Hugging Face ClassLabel feature."
                    )
                string_labels.add(raw_label)

        if len(string_labels) < 2:
            raise ValueError("Dataset must expose at least two labels")
        return tuple(sorted(string_labels))

    def _normalize_split(
        self,
        raw_split: Any,
        split_name: str,
        label_names: tuple[str, ...],
    ) -> tuple[LabeledExample, ...]:
        examples: list[LabeledExample] = []

        for index, row in enumerate(raw_split):
            if self.spec.text_column not in row:
                raise ValueError(
                    f"Missing text column {self.spec.text_column!r} in {split_name} row"
                )
            if self.spec.label_column not in row:
                raise ValueError(
                    f"Missing label column {self.spec.label_column!r} in {split_name} row"
                )

            raw_text = row[self.spec.text_column]
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError(
                    f"Invalid text at {self.name}/{split_name}/{index}: {raw_text!r}"
                )

            raw_label = row[self.spec.label_column]
            label = _normalize_label(raw_label, label_names)

            examples.append(
                LabeledExample(
                    sample_id=f"{self.name}:{split_name}:{index}",
                    text=raw_text,
                    label=label,
                    metadata={
                        "dataset": self.name,
                        "source": self.spec.path,
                        "split": split_name,
                        "row_index": index,
                        "raw_label": raw_label,
                    },
                )
            )

        return tuple(examples)

    def _description_for(self, label_name: str) -> str:
        explicit = self.spec.label_descriptions.get(label_name)
        if explicit is not None:
            return explicit

        readable_label = label_name.replace("_", " ").replace("/", " or ")
        return self.spec.label_description_template.format(
            label=label_name,
            readable_label=readable_label,
        )


def _normalize_label(raw_label: Any, label_names: tuple[str, ...]) -> str:
    if isinstance(raw_label, bool):
        raise ValueError("Boolean labels are not supported")

    if isinstance(raw_label, int):
        if raw_label < 0 or raw_label >= len(label_names):
            raise ValueError(
                f"Label index {raw_label} is outside [0, {len(label_names) - 1}]"
            )
        return label_names[raw_label]

    if isinstance(raw_label, str):
        if raw_label not in label_names:
            raise ValueError(f"Unknown string label: {raw_label!r}")
        return raw_label

    raise ValueError(f"Unsupported label value: {raw_label!r}")


def _load_dataset(**kwargs: Any) -> Any:
    """Import lazily so unit tests do not need network or Hugging Face installed."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise RuntimeError(
            "Hugging Face datasets is not installed. Run pip install -r requirements.txt."
        ) from exc

    return load_dataset(**kwargs)
