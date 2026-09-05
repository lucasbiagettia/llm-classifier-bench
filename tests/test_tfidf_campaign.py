"""Exercise both maintained campaign CLIs with local data and real sklearn."""
from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
import sys

import pytest

from llm_classifier_bench.class_definitions import ClassDefinitionProfile
from llm_classifier_bench.classifiers import TfidfLogisticClassifier
from llm_classifier_bench.config import TfidfTrainingConfig
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark


@pytest.mark.parametrize("script", ["run_banking77_scaling_benchmark", "run_banking77_scaling_benchmark_v2"])
def test_tfidf_only_campaign_and_saved_configuration_replay(script, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    campaign = importlib.import_module(script)
    probe = importlib.import_module("probe_tfidf_classifier")
    bundle = probe.fixture_bundle()
    definitions = tmp_path / "definitions.json"
    ClassDefinitionProfile(dataset=bundle.name, profile="local_fixture", classes=bundle.classes,
                           review_status="approved").write_json(definitions)
    monkeypatch.setattr(campaign, "get_dataset", lambda name: campaign.StaticDataset(bundle))
    def forbidden(*args, **kwargs):
        pytest.fail("TF-IDF-only campaigns must not construct API/transformer classifiers")
    for name in ("BertClassifier", "OpenAIClassifier", "SentenceTransformerLogisticClassifier", "EmissaryClient"):
        monkeypatch.setattr(campaign, name, forbidden)
    monkeypatch.setattr(campaign.EmissaryClassifier, "create", forbidden)
    monkeypatch.setattr(campaign, "utc_campaign_id", lambda: "fixture-campaign")
    args = [script, "--classifiers", "tfidf", "--class-counts", "2", "--seeds", "17",
            "--train-per-class", "4", "--test-per-class", "1", "--validation-fraction", "0.25",
            "--definitions", str(definitions), "--output-root", str(tmp_path),
            "--tfidf-c-values", "10", "1", "--tfidf-fallback-c", "2",
            "--tfidf-ngram-range", "1", "1", "--tfidf-sublinear-tf"]
    if script.endswith("_v2"):
        args += ["--min-train-per-class", "4", "--min-test-per-class", "1", "--strict-support"]
    monkeypatch.setattr(sys, "argv", args)
    campaign.main()
    root = tmp_path / "fixture-campaign"
    manifest = json.loads((root / "campaign.json").read_text())
    assert manifest["classifiers"] == ["tfidf"]
    assert manifest["tfidf_training_by_seed"]["17"]["c_values"] == [10, 1]
    summary = json.loads((root / "summary.json").read_text())
    assert len(summary) == 1
    row = summary[0]
    assert row["classifier_key"] == "tfidf"
    assert row["status"] == "completed"
    assert row["supervision_regime"] == "supervised"
    assert row["training_examples_used"] == 6
    assert row["validation_examples_used"] == 2
    for metric in ("accuracy", "macro_f1", "multiclass_log_loss", "multiclass_brier_score", "top_label_ece"):
        assert row[metric] is not None
    with (root / "summary.csv").open() as stream:
        assert list(csv.DictReader(stream))[0]["classifier_key"] == "tfidf"
    run_dir = Path(row["run_dir"])
    config = json.loads((run_dir / "config.json").read_text())
    fit = json.loads((run_dir / "fit_metadata.json").read_text())
    assert fit["selected_c"] == row["selected_c"]
    assert fit["training_examples_used"] == config["dataset"]["fit_train_size"] == 6
    assert fit["validation_examples_used"] == config["dataset"]["validation_size"] == 2
    assert fit["selection_metric"] == "validation_accuracy"
    assert fit["vectorizer"]["ngram_range"] == [1, 1]
    assert fit["vectorizer"]["sublinear_tf"] is True
    assert fit["seed"] == 17
    assert set(fit["library_versions"]) == {"scikit-learn", "numpy", "scipy"}
    data = config["dataset"]
    assert set(data["fit_train_sample_ids"]).isdisjoint(data["validation_sample_ids"])
    assert set(data["test_sample_ids"]).isdisjoint(data["fit_train_sample_ids"] + data["validation_sample_ids"])
    saved_predictions = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()]
    assert [p["sample_id"] for p in saved_predictions] == data["test_sample_ids"]
    assert all(set(p["probabilities"]) == set(bundle.class_names) for p in saved_predictions)

    # Reconstruct exactly from saved settings and IDs; no new sampling or tuning rule.
    training = config["classifier"]["training"]
    for field in ("ngram_range", "c_values"):
        training[field] = tuple(training[field])
    replay = TfidfLogisticClassifier(training=TfidfTrainingConfig(**training))
    replay.prepare(bundle.classes)
    by_id = {e.sample_id: e for e in bundle.train + bundle.test}
    replay.fit([by_id[i] for i in data["fit_train_sample_ids"]],
               validation_examples=[by_id[i] for i in data["validation_sample_ids"]])
    predictions = replay.predict([by_id[i].as_input() for i in data["test_sample_ids"]])
    assert replay.selected_c == fit["selected_c"]
    assert [p.probabilities for p in predictions] == [p["probabilities"] for p in saved_predictions]


def test_runner_test_only_tokens_never_enter_vocabulary(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    probe = importlib.import_module("probe_tfidf_classifier")
    classifier = TfidfLogisticClassifier()
    result = run_benchmark(probe.StaticDataset(probe.fixture_bundle()), classifier,
                           BenchmarkRunConfig(output_root=tmp_path, validation_fraction=0.25))
    assert "unseentoken" not in classifier._vectorizer.vocabulary_
    assert json.loads(result.status_path.read_text())["status"] == "completed"
    assert result.metrics_path.is_file()
    assert (result.run_dir / "fit_metadata.json").is_file()
