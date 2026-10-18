"""Hard-task normalization, grouped annotation preservation and leakage checks."""
from dataclasses import replace

import pytest

from llm_classifier_bench.datasets import AnliDataset, ImplicitHateDataset, get_dataset
from llm_classifier_bench.datasets.hard_tasks import HATE_DESCRIPTIONS, explanation_coverage
from llm_classifier_bench.datasets.selection import validate_partition_disjointness


def hate_rows():
    return [{"post": f"post {label} {i}", "implicit_class": label,
             "extra_implicit_class": "irony", "implied_statement": "SECRET EXPLANATION"}
            for label in HATE_DESCRIPTIONS for i in range(15)]


def test_hate_primary_filter_split_and_grouped_explanations(tmp_path):
    rows = hate_rows()
    rows += [dict(rows[0], target="SECRET TARGET"),
             {"post": "excluded other", "implicit_class": "other"}]
    annotations = tmp_path / "stage3.tsv"
    annotations.write_text("post\ttarget\timplied_statement\npost white_grievance 0\tgroup\tSENSITIVE\nno match\tgroup\tmissing\n")
    dataset = ImplicitHateDataset(loader=lambda **_: rows, annotations_path=annotations)
    bundle = dataset.load()
    assert bundle.class_names == tuple(HATE_DESCRIPTIONS)
    assert bundle.metadata["excluded_primary_other_rows"] == 1
    assert bundle.metadata["supplemental_annotations"]["exact_matched_posts"] == 1
    assert bundle.metadata["supplemental_annotations"]["unmatched_posts"] == 1
    validate_partition_disjointness(bundle.train, bundle.validation, bundle.test)
    examples = bundle.train + bundle.validation + bundle.test
    group = next(e for e in examples if e.text == rows[0]["post"])
    assert len(group.metadata["annotations"]) == 2
    assert group.metadata["annotations"][0]["supplemental_annotations"][0]["implied_statement"] == "SENSITIVE"
    assert explanation_coverage(examples)["implied_statement"] == 90
    assert all("SECRET" not in e.as_input().text for e in examples)
    assert all(set(e.as_input().metadata) <= {"dataset", "source", "split", "row_index"} for e in examples)
    assert dataset.load() == bundle
    shuffled = ImplicitHateDataset(loader=lambda **_: list(reversed(rows))).load()
    assert [e.sample_id for e in shuffled.test] == [e.sample_id for e in bundle.test]


def test_conflicting_primary_annotations_excluded_as_a_whole():
    rows = hate_rows()
    rows.append(dict(rows[0], implicit_class="threatening"))
    bundle = ImplicitHateDataset(loader=lambda **_: rows).load()
    assert bundle.metadata["group_audit"]["conflicting_groups_excluded"] == 1
    assert rows[0]["post"] not in {e.text for e in bundle.train + bundle.validation + bundle.test}


def test_unknown_hate_label_fails():
    with pytest.raises(ValueError, match="Unknown primary"):
        ImplicitHateDataset(loader=lambda **_: [{"post": "text", "implicit_class": "typo"}]).load()


def anli_row(uid, premise, label=0, reason="SECRET REASON"):
    return dict(uid=uid, premise=premise, hypothesis="The claim", label=label, reason=reason)


def test_anli_official_splits_numeric_labels_and_duplicate_priority():
    calls = []
    def loader(**kwargs):
        calls.append(kwargs)
        return {"train": [anli_row("train", "training", 0), anli_row("dup", "heldout", 2)],
                "dev": [anli_row("dev", "development", 1)],
                "test": [anli_row("test", "heldout", 2)]}[kwargs["split"]]
    bundle = AnliDataset(loader=loader).load()
    assert bundle.class_names == ("entailment", "neutral", "contradiction")
    assert [e.label for e in bundle.train] == ["entailment"]
    assert [e.label for e in bundle.validation] == ["neutral"]
    assert bundle.test[0].text == "Premise: heldout\nHypothesis: The claim"
    assert len(bundle.test[0].metadata["annotations"]) == 2
    assert len(bundle.metadata["cross_split_excluded_ids"]["train"]) == 1
    assert {a["official_split"] for a in bundle.test[0].metadata["annotations"]} == {"train_r3", "test_r3"}
    assert [c["split"] for c in calls] == ["train", "dev", "test"]
    assert all("r3-" in next(iter(c["data_files"].values())) for c in calls)
    assert "reason" not in bundle.test[0].as_input().metadata
    assert "SECRET" not in bundle.test[0].as_input().text
    validate_partition_disjointness(bundle.train, bundle.validation, bundle.test)


@pytest.mark.parametrize("label", [-1, 3, True, "entailment"])
def test_invalid_anli_label(label):
    with pytest.raises(ValueError, match="Invalid ANLI label"):
        AnliDataset(loader=lambda **_: [anli_row("id", "text", label)]).load()


@pytest.mark.parametrize("name,cls", [("implicit_hate", ImplicitHateDataset), ("anli_r3", AnliDataset)])
def test_registry(name, cls):
    assert isinstance(get_dataset(name), cls)
