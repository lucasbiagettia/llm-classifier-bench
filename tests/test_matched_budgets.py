"""Fairness, context preflight and campaign evidence without paid calls/downloads."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from llm_classifier_bench.budgets import MatchedBudgetConfig, select_matched_pools
from llm_classifier_bench.classifiers import (
    BertClassifier, EmissaryClassifier, OpenAIClassifier,
    SentenceTransformerLogisticClassifier, TfidfLogisticClassifier,
)
from llm_classifier_bench.config import EmissaryTrainingConfig
from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.datasets.base import DatasetBundle
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark, split_train_validation


class StaticDataset:
    def __init__(self, bundle):
        self.bundle = bundle
        self.name = bundle.name

    def load(self):
        return self.bundle


def fixture_bundle():
    classes = (ClassDefinition("World", "World news"), ClassDefinition("Sports", "Sports news"))
    return DatasetBundle("matched_fixture", classes,
        tuple(LabeledExample(f"train-{c.name}-{i}", f"{c.name} unique training text {i}", c.name)
              for c in classes for i in range(12)),
        tuple(LabeledExample(f"test-{c.name}", f"{c.name} held out text", c.name) for c in classes))


class RecordingClient:
    max_retries = 0
    timeout = 120

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id="fixture", model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"label":"Sports"}'))],
            model_dump=lambda: {"id": "fixture", "model": kwargs["model"],
                                "usage": {"prompt_tokens": 100, "completion_tokens": 4}})


class FixtureEncoder:
    device = "cpu"

    def encode(self, texts, **kwargs):
        return np.array([[float("World" in text), float("Sports" in text)] for text in texts])


def read(path):
    return json.loads(path.read_text())


def run(tmp_path, classifier, budget, *, dry_run=False, bundle=None):
    return run_benchmark(StaticDataset(bundle or fixture_bundle()), classifier,
        BenchmarkRunConfig(output_root=tmp_path, run_id=classifier.name,
            validation_fraction=.25, split_seed=17, matched_budget=budget, dry_run=dry_run))


def test_shared_pools_nested_reproducible_and_seeded():
    bundle = fixture_bundle()
    fit, val = split_train_validation(bundle.train, validation_fraction=.25, seed=17)
    small, v1, m1 = select_matched_pools(fit, val, bundle.classes, MatchedBudgetConfig(5, "total", 17, 2))
    large, v2, m2 = select_matched_pools(fit[::-1], val[::-1], bundle.classes, MatchedBudgetConfig(10, "total", 17, 2))
    assert small == large[:5]
    assert v1 == v2
    assert m1["selection"]["actual_total"] == 5
    assert set(e.sample_id for e in large).isdisjoint(e.sample_id for e in v1 + bundle.test)
    per_class, _, _ = select_matched_pools(fit, val, bundle.classes, MatchedBudgetConfig(5, "per_class", 17))
    assert per_class == large
    other, _, _ = select_matched_pools(fit, val, bundle.classes, MatchedBudgetConfig(5, "total", 18))
    assert other != small


def test_real_supervised_and_prompt_runs_share_ids_and_count_actual_exposure(tmp_path):
    client = RecordingClient()
    classifiers = [TfidfLogisticClassifier(), SentenceTransformerLogisticClassifier(encoder=FixtureEncoder()),
                   OpenAIClassifier(client=client, in_context=True, context_window_tokens=32000)]
    artifacts = []
    configs = []
    for classifier in classifiers:
        result = run(tmp_path, classifier, MatchedBudgetConfig(5, "total", 17, 2))
        assert read(result.status_path)["status"] == "completed"
        artifacts.append(read(result.run_dir / "labeled_budget.json"))
        configs.append(read(result.config_path))
    assert len({a["pool_sha256"] for a in artifacts}) == 1
    assert all(c["dataset"]["fit_train_sample_ids"] == configs[0]["dataset"]["fit_train_sample_ids"] for c in configs)
    assert all(c["dataset"]["test_sample_ids"] == configs[0]["dataset"]["test_sample_ids"] for c in configs)
    assert artifacts[0]["consumption"] == artifacts[1]["consumption"] == {
        "training_examples_used": 5, "validation_examples_used": 2,
        "context_examples_used": 0, "total_labeled_examples_used": 7,
    }
    assert artifacts[2]["consumption"] == {
        "training_examples_used": 0, "validation_examples_used": 0,
        "context_examples_used": 5, "total_labeled_examples_used": 5,
    }
    fit_info = read(result.run_dir / "fit_metadata.json")
    assert fit_info["context_sample_ids"] == configs[0]["dataset"]["fit_train_sample_ids"]
    for call in client.calls:
        assert len(call["messages"]) == 12
        assert call["max_completion_tokens"] == 4096
        for i, demo in enumerate(artifacts[2]["selection"]["selected_examples"]):
            assert call["messages"][1+2*i]["content"] == f"Text:\n{demo['text']}"
            assert json.loads(call["messages"][2+2*i]["content"]) == {"label": demo["label"]}
        prompt = json.dumps(call["messages"])
        assert all(e["text"] not in prompt for e in artifacts[2]["validation_selection"]["selected_examples"])
    assert all(p["probabilities"] == {} for p in map(json.loads, result.predictions_path.read_text().splitlines()))


@pytest.mark.parametrize("classifier", [TfidfLogisticClassifier, BertClassifier, SentenceTransformerLogisticClassifier,
    lambda: EmissaryClassifier(experiment_name="fixture", training=EmissaryTrainingConfig(shots=1, shot_unit="total"))])
def test_missing_coverage_unsupported_before_model_or_api_work(tmp_path, classifier, monkeypatch):
    method = classifier()
    monkeypatch.setattr(method, "prepare", lambda *args: pytest.fail("Must preflight before prepare"))
    result = run(tmp_path, method, MatchedBudgetConfig(1, "total", 17))
    assert read(result.status_path)["status"] == "unsupported"
    assert "insufficient_class_coverage" in read(result.status_path)["error_message"]
    assert read(result.run_dir / "labeled_budget.json")["selection"]["class_coverage"] == .5
    assert not result.predictions_path.exists()


@pytest.mark.parametrize("budget", [0, 1])
def test_openai_allows_zero_or_partial_coverage(tmp_path, budget):
    client = RecordingClient()
    result = run(tmp_path, OpenAIClassifier(client=client, in_context=True, context_window_tokens=32000),
                 MatchedBudgetConfig(budget, "total", 17))
    assert read(result.status_path)["status"] == "completed"
    assert len(client.calls[0]["messages"]) == 2 + 2*budget
    assert read(result.run_dir / "labeled_budget.json")["consumption"]["total_labeled_examples_used"] == budget


@pytest.mark.parametrize("window", [None, 5300])
def test_context_preflight_checks_whole_test_set_before_any_request(tmp_path, window):
    client = RecordingClient()
    bundle = fixture_bundle()
    from dataclasses import replace
    bundle = replace(bundle, test=(bundle.test[0], LabeledExample("long", "é"*6000, "Sports")))
    result = run(tmp_path, OpenAIClassifier(client=client, in_context=True, context_window_tokens=window),
                 MatchedBudgetConfig(2, "total", 17), bundle=bundle)
    assert read(result.status_path)["status"] == "unsupported"
    assert "context_" in read(result.status_path)["error_message"]
    assert client.calls == []
    if window:
        plan = read(result.run_dir / "context_plan.json")
        assert plan["requests"][-1]["estimated_total_tokens"] > window


def test_provider_context_rejection_is_unsupported_and_retains_usage_ledger(tmp_path):
    client = RecordingClient()
    class ContextError(Exception):
        code = "context_length_exceeded"
    def fail(**kwargs):
        raise ContextError("server context limit")
    client.create = fail
    result = run(tmp_path, OpenAIClassifier(client=client, in_context=True, context_window_tokens=32000),
                 MatchedBudgetConfig(2, "total", 17))
    assert read(result.status_path)["status"] == "unsupported"
    assert (result.run_dir / "usage.jsonl").is_file()
    assert read(result.run_dir / "cost_report.json")["failure_rate"] == 1


def test_emissary_plan_consumes_exact_shared_pool_without_extra_examples(tmp_path):
    classifier = EmissaryClassifier(experiment_name="fixture", training=EmissaryTrainingConfig(shots=5, shot_unit="total", selection_seed=17,
        mechanism="project_fine_tuning", project_id="fixture", base_model="fixture"))
    result = run(tmp_path, classifier, MatchedBudgetConfig(5, "total", 17, 2), dry_run=True)
    assert read(result.status_path)["status"] == "dry_run"
    plan = read(result.run_dir / "preparation_plan.json")
    pool = read(result.run_dir / "labeled_budget.json")
    assert {e["sample_id"] for e in plan["selection"]["selected_examples"]} == {
        e["sample_id"] for e in pool["selection"]["selected_examples"]}


@pytest.mark.parametrize("budget", [MatchedBudgetConfig(100, "total"), MatchedBudgetConfig(2, "total", validation_examples=100)])
def test_insufficient_source_support_keeps_config_and_requested_budget(tmp_path, budget):
    result = run(tmp_path, TfidfLogisticClassifier(), budget)
    assert read(result.status_path)["status"] == "unsupported"
    assert result.config_path.exists()
    assert read(result.run_dir / "labeled_budget.json")["budget"]["examples"] == budget.examples


@pytest.mark.parametrize("script", ["run_banking77_scaling_benchmark", "run_banking77_scaling_benchmark_v2"])
@pytest.mark.parametrize("dry_run", [True, False])
def test_campaign_matched_budgets_are_separate_with_honest_summary(script, dry_run, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    campaign = importlib.import_module(script)
    from llm_classifier_bench.class_definitions import ClassDefinitionProfile
    bundle = fixture_bundle()
    profile = tmp_path / "definitions.json"
    ClassDefinitionProfile(dataset=bundle.name, profile="fixture", classes=bundle.classes,
                           review_status="approved").write_json(profile)
    monkeypatch.setattr(campaign, "get_dataset", lambda _: StaticDataset(bundle))
    monkeypatch.setattr(campaign, "utc_campaign_id", lambda: "matched")
    args = [script, "--classifiers", "tfidf", "--class-counts", "2", "--seeds", "17",
            "--train-per-class", "12", "--test-per-class", "1", "--validation-fraction", ".25",
            "--definitions", str(profile), "--output-root", str(tmp_path),
            "--matched-budgets", "0", "1", "2", "5", "--budget-unit", "total", "--validation-budget", "2"]
    if script.endswith("_v2"):
        args += ["--min-train-per-class", "12", "--min-test-per-class", "1", "--strict-support"]
    if dry_run:
        args += ["--dry-run"]
        monkeypatch.setattr(campaign.TfidfLogisticClassifier, "fit", lambda *a, **kw: pytest.fail("Dry run fit"))
    monkeypatch.setattr(sys, "argv", args)
    campaign.main()
    rows = read(tmp_path / "matched/summary.json")
    assert [r["status"] for r in rows] == ["unsupported", "unsupported"] + ["dry_run" if dry_run else "completed"]*2
    assert {r["comparison_regime"] for r in rows} == {"matched_labeled_budget"}
    assert [r["class_coverage"] for r in rows] == [0, .5, 1, 1]
    assert rows[-1]["training_examples_used"] == (None if dry_run else 5)
    assert rows[-1]["total_labeled_examples_used"] == (None if dry_run else 7)
    assert read(tmp_path / "matched/campaign.json")["matched_budgets"] == [0, 1, 2, 5]


@pytest.mark.parametrize("script", ["run_banking77_scaling_benchmark", "run_banking77_scaling_benchmark_v2"])
def test_all_methods_dry_run_pairs_pools_and_overrides_emissary_seed(script, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    campaign = importlib.import_module(script)
    from llm_classifier_bench.class_definitions import ClassDefinitionProfile
    bundle = fixture_bundle()
    profile = tmp_path / "definitions.json"
    ClassDefinitionProfile(dataset=bundle.name, profile="fixture", classes=bundle.classes,
                           review_status="approved").write_json(profile)
    monkeypatch.setattr(campaign, "get_dataset", lambda _: StaticDataset(bundle))
    monkeypatch.setattr(campaign, "utc_campaign_id", lambda: "all-methods")
    def forbidden(*args, **kwargs):
        pytest.fail("Dry-run must not load models, fit or construct provider clients")
    for method in (TfidfLogisticClassifier, BertClassifier, SentenceTransformerLogisticClassifier, OpenAIClassifier, EmissaryClassifier):
        monkeypatch.setattr(method, "prepare", forbidden)
        monkeypatch.setattr(method, "fit", forbidden)
    monkeypatch.setattr("llm_classifier_bench.classifiers.openai._build_openai_client", forbidden)
    monkeypatch.setattr("llm_classifier_bench.classifiers.emissary.EmissaryClient", forbidden)
    args = [script, "--classifiers", "tfidf", "sentence-transformer", "bert", "openai", "emissary",
            "--class-counts", "2", "--seeds", "17", "18", "--train-per-class", "12", "--test-per-class", "1",
            "--definitions", str(profile), "--output-root", str(tmp_path),
            "--matched-budgets", "0", "1", "2", "5", "--budget-unit", "total", "--dry-run",
            "--openai-context-window-tokens", "32000", "--emissary-mechanism", "project_fine_tuning",
            "--emissary-project-id", "fixture", "--emissary-base-model", "fixture"]
    if script.endswith("_v2"):
        args += ["--min-train-per-class", "12", "--min-test-per-class", "1", "--strict-support"]
    monkeypatch.setattr(sys, "argv", args)
    campaign.main()
    rows = read(tmp_path / "all-methods/summary.json")
    assert len(rows) == 40
    for seed in (17, 18):
        for count in (0, 1, 2, 5):
            paired = [r for r in rows if r["seed"] == seed and r["labeled_budget"] == count]
            assert len({r["pool_sha256"] for r in paired}) == 1
            for row in paired:
                expected = "dry_run" if count >= 2 or row["classifier_key"] == "openai" or (count == 0 and row["classifier_key"] == "emissary") else "unsupported"
                assert row["status"] == expected
                config = read(Path(row["run_dir"]) / "config.json")
                assert config["matched_budget"]["seed"] == seed
                if row["classifier_key"] == "emissary":
                    assert config["classifier"]["training"]["selection_seed"] == seed


@pytest.mark.parametrize("script", ["run_banking77_scaling_benchmark", "run_banking77_scaling_benchmark_v2"])
def test_matched_emissary_guard_counts_all_seed_and_class_conditions(script, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    campaign = importlib.import_module(script)
    monkeypatch.setattr(campaign, "get_dataset", lambda *_: pytest.fail("Guard before dataset load"))
    monkeypatch.setattr(sys, "argv", [script, "--classifiers", "emissary", "--class-counts", "2", "3",
        "--seeds", "17", "18", "--matched-budgets", "5", "10", "--budget-unit", "total",
        "--emissary-mechanism", "project_fine_tuning", "--emissary-project-id", "fixture",
        "--emissary-base-model", "fixture", "--emissary-allow-unpriced-training",
        "--emissary-max-training-jobs", "2"])
    with pytest.raises(ValueError, match="at least 8"):
        campaign.main()


def test_matched_emissary_unconfigured_mechanism_is_explicitly_unsupported(tmp_path):
    classifier = EmissaryClassifier(experiment_name="fixture", training=EmissaryTrainingConfig(shots=5, shot_unit="total"))
    result = run(tmp_path, classifier, MatchedBudgetConfig(5, "total"))
    assert read(result.status_path)["status"] == "unsupported"
    assert "mechanism" in read(result.status_path)["error_message"]
    assert not (result.run_dir / "usage.jsonl").exists()
