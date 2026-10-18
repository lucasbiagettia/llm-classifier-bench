"""Pinned, single-label hard tasks; explanations never enter example.text."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import random

from llm_classifier_bench.core import ClassDefinition, LabeledExample
from .base import DatasetBundle
from .huggingface import _load_dataset

IMPLICIT_HATE_REVISION = "57d34fad1cf7d4d9e71774ea17c4c1a7f57af8d4"
ANLI_REVISION = "8e4813d81f46d313dac7892e1c28076917cfcdf9"
HATE_DESCRIPTIONS = {
    "white_grievance": "Implicit hate expressing frustration at a minority group's perceived privilege or portraying white people as unfairly victimized.",
    "incitement": "Implicit hate promoting hate groups or ideologies, encouraging solidarity or flaunting in-group power.",
    "inferiority": "Implicit hate implying that a demographic group or its members have less worth or are inferior to others.",
    "irony": "Implicit hate using sarcasm, humor or satire to demean a demographic group or its members.",
    "stereotypical": "Implicit hate associating a demographic group with negative attributes through stereotypes, euphemisms or metaphors.",
    "threatening": "Implicit hate indirectly committing to or advocating harm to a group's body, well-being, reputation or liberty.",
}
ANLI_DESCRIPTIONS = {
    "entailment": "Given the premise, the hypothesis must be true. Judge the hypothesis only against information supported by the premise.",
    "neutral": "Given the premise, the hypothesis could be true or false: the premise does not establish or rule it out.",
    "contradiction": "Given the premise, the hypothesis must be false: it conflicts with information in the premise.",
}


def content_id(text):
    return sha256(text.encode("utf-8")).hexdigest()


def _text(row, column):
    value = row[column]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or empty {column}")
    return value


def _group_rows(name, split, rows):
    """One scoring unit per exact input; keep every annotation together."""
    groups = defaultdict(list)
    for text, label, annotation in rows:
        groups[text].append((label, annotation))
    examples, conflicts = [], 0
    for text, values in sorted(groups.items(), key=lambda item: content_id(item[0])):
        labels = {v[0] for v in values}
        if len(labels) != 1:
            conflicts += 1
            continue
        examples.append(LabeledExample(
            f"{name}:{content_id(text)}", text, values[0][0],
            {"dataset": name, "source": "huggingface", "split": split,
             "annotations": [v[1] for v in values]},
        ))
    return tuple(examples), {"rows": sum(len(v) for v in groups.values()),
                             "groups": len(groups), "conflicting_groups_excluded": conflicts,
                             "retained_examples": len(examples)}


class ImplicitHateDataset:
    name = "implicit_hate"

    def __init__(self, *, loader=None, annotations_path: Path | None = None, split_seed=42):
        self.loader = loader or _load_dataset
        self.annotations_path = annotations_path
        self.split_seed = split_seed

    def load(self):
        raw = self.loader(path="csv", split="train", data_files={"train":
            f"https://huggingface.co/datasets/SALT-NLP/ImplicitHate/resolve/{IMPLICIT_HATE_REVISION}/implicit_hate.csv"})
        supplemental = defaultdict(list)
        if self.annotations_path is not None:
            with self.annotations_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    supplemental[_text(row, "post")].append(dict(row))
        rows, excluded_other = [], 0
        matched_posts = set()
        for index, row in enumerate(raw):
            label = row["implicit_class"]
            if label == "other":
                excluded_other += 1
                continue
            if label not in HATE_DESCRIPTIONS:
                raise ValueError(f"Unknown primary implicit_class: {label!r}")
            text = _text(row, "post")
            annotation = {"row_index": index, **dict(row)}
            if text in supplemental:
                annotation["supplemental_annotations"] = supplemental[text]
                matched_posts.add(text)
            rows.append((text, label, annotation))
        examples, audit = _group_rows(self.name, "unpartitioned", rows)
        partitions = {key: [] for key in ("train", "validation", "test")}
        for label in HATE_DESCRIPTIONS:
            group = sorted((e for e in examples if e.label == label), key=lambda e: e.sample_id)
            random.Random(f"{self.split_seed}:implicit-hate-split:{label}").shuffle(group)
            a, b = int(.6 * len(group)), int(.8 * len(group))
            for split, subset in zip(partitions, (group[:a], group[a:b], group[b:])):
                partitions[split].extend(replace(e, metadata={**e.metadata, "split": split}) for e in subset)
        return DatasetBundle(
            self.name, tuple(ClassDefinition(k, v) for k, v in HATE_DESCRIPTIONS.items()),
            tuple(partitions["train"]), tuple(partitions["test"]),
            metadata={"path": "SALT-NLP/ImplicitHate", "revision": IMPLICIT_HATE_REVISION,
                      "split_policy": "synthetic 60/20/20 stratified exact-input groups; not official splits",
                      "split_seed": self.split_seed, "excluded_primary_other_rows": excluded_other,
                      "secondary_label_policy": "metadata only; never additional classification targets",
                      "group_audit": audit,
                      "supplemental_annotations": {
                          "sha256": sha256(self.annotations_path.read_bytes()).hexdigest() if self.annotations_path else None,
                          "exact_matched_posts": len(matched_posts),
                          "unmatched_posts": len(set(supplemental) - matched_posts),
                          "join": "exact post equality only; preserve all matching rows; no inferred explanations"}},
            validation=tuple(partitions["validation"]),
        )


class AnliDataset:
    name = "anli_r3"

    def __init__(self, *, loader=None):
        self.loader = loader or _load_dataset

    def load(self):
        partitions, audit = {}, {}
        for split in ("train", "dev", "test"):
            raw = self.loader(path="parquet", split=split, data_files={split:
                f"https://huggingface.co/datasets/facebook/anli/resolve/{ANLI_REVISION}/plain_text/{split}_r3-00000-of-00001.parquet"})
            rows = []
            for index, row in enumerate(raw):
                label = row["label"]
                if type(label) is not int or label not in range(3):
                    raise ValueError(f"Invalid ANLI label: {label!r}")
                text = f"Premise: {_text(row, 'premise')}\nHypothesis: {_text(row, 'hypothesis')}"
                rows.append((text, tuple(ANLI_DESCRIPTIONS)[label],
                             {"row_index": index, "official_split": f"{split}_r3", **dict(row)}))
            partitions[split], audit[split] = _group_rows(self.name, f"{split}_r3", rows)
        # Keep the official holdout; remove duplicate inputs only from earlier
        # training/development candidates and retain their annotations together.
        owners = {}
        exclusions = {"train": [], "dev": []}
        for split in ("test", "dev", "train"):
            retained = []
            for example in partitions[split]:
                if example.text in owners:
                    owner_split, index = owners[example.text]
                    owner = partitions[owner_split][index]
                    partitions[owner_split][index] = replace(owner, metadata={**owner.metadata,
                        "annotations": [*owner.metadata["annotations"], *example.metadata["annotations"]]})
                    exclusions[split].append(example.sample_id)
                else:
                    owners[example.text] = (split, len(retained))
                    retained.append(example)
            partitions[split] = retained
        return DatasetBundle(
            self.name, tuple(ClassDefinition(k, v) for k, v in ANLI_DESCRIPTIONS.items()),
            tuple(partitions["train"]), tuple(partitions["test"]),
            metadata={"path": "facebook/anli", "revision": ANLI_REVISION, "round": "R3",
                      "train_split": "train_r3", "validation_split": "dev_r3", "test_split": "test_r3",
                      "input_format": "Premise: {premise}\\nHypothesis: {hypothesis}", "group_audit": audit,
                      "cross_split_duplicate_policy": "test > dev > train; exclude exact duplicate inputs from lower-priority candidates; merge annotations into existing owner; unique examples never move",
                      "cross_split_excluded_ids": exclusions},
            validation=tuple(partitions["dev"]),
        )


def explanation_coverage(examples):
    counts = Counter()
    for example in examples:
        fields = set()
        for annotation in example.metadata.get("annotations", []):
            for row in (annotation, *annotation.get("supplemental_annotations", [])):
                fields.update(k for k in ("reason", "target", "implied_statement")
                              if isinstance(row.get(k), str) and row[k].strip())
        counts.update(fields)
    return {"examples": len(examples), **{k: counts[k] for k in ("reason", "target", "implied_statement")}}
