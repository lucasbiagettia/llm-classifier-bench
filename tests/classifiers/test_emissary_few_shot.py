"""Offline evidence only: mocks do not establish live provider compatibility."""
from dataclasses import replace
from hashlib import sha256
import json
from unittest.mock import Mock

import pytest
import requests

from llm_classifier_bench.classifiers.emissary import (
    EmissaryAPIError, EmissaryClassifier, EmissaryClient, EmissaryResponseError,
)
from llm_classifier_bench.config import EmissaryTrainingConfig as Config
from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.datasets.base import DatasetBundle
from llm_classifier_bench.datasets.selection import select_labeled_examples, validate_partition_disjointness
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark


def fixture_bundle(class_count=3, support=110):
    classes = tuple(ClassDefinition(f"class-{i}", f"Frozen definition {i}") for i in range(class_count))
    return DatasetBundle(
        "banking77", classes,
        tuple(LabeledExample(f"train-{c.name}-{i}", f"Training {c.name} number {i}", c.name)
              for c in classes for i in range(support)),
        tuple(LabeledExample(f"test-{c.name}", f"Held out {c.name}", c.name) for c in classes),
    )


class StaticDataset:
    def __init__(self, bundle):
        self.bundle = bundle
        self.name = bundle.name

    def load(self):
        return self.bundle


@pytest.mark.parametrize("kwargs", [
    {"shots": -1}, {"shots": True}, {"shots": 1.5}, {"shots": 5},
    {"shot_unit": "auto"}, {"selection_seed": True}, {"selection_policy": "random"},
])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_nested_balanced_selection_and_fingerprints():
    bundle = fixture_bundle(20)
    previous = ()
    for budget in [0, 5, 7, 100]:
        config = Config(budget, "total", 19)
        selected, metadata = select_labeled_examples(bundle.train, bundle.classes, config)
        assert selected[:len(previous)] == previous
        assert select_labeled_examples(tuple(reversed(bundle.train)), tuple(reversed(bundle.classes)), config)[0] == selected
        assert len(selected) == len({x.sample_id for x in selected}) == budget
        counts = list(metadata["per_class_counts"].values())
        assert max(counts) - min(counts) <= 1
        assert metadata["covered_class_count"] == min(20, budget)
        for item, evidence in zip(selected, metadata["selected_examples"]):
            assert evidence["content_sha256"] == sha256(item.text.encode()).hexdigest()
        previous = selected
    assert select_labeled_examples(bundle.train, bundle.classes, Config(100, "total", 20))[0] != previous


def test_per_class_and_insufficient_support():
    bundle = fixture_bundle(3, 7)
    selected, metadata = select_labeled_examples(bundle.train, bundle.classes, Config(5, "per_class"))
    assert len(selected) == metadata["requested_total"] == 15
    assert set(metadata["per_class_counts"].values()) == {5}
    assert select_labeled_examples(bundle.train, bundle.classes, Config(15, "total"))[0] == selected
    for config in [Config(8, "per_class"), Config(22, "total")]:
        with pytest.raises(ValueError, match="Insufficient"):
            select_labeled_examples(bundle.train, bundle.classes, config)
    # No redistribution when the specifically allocated class has insufficient support.
    short_label = metadata["class_order"][0]
    short = tuple(e for e in bundle.train if e.label != short_label)
    with pytest.raises(ValueError, match="Insufficient"):
        select_labeled_examples(short, bundle.classes, Config(1, "total"))


@pytest.mark.parametrize("change", ["id", "text"])
def test_partition_leakage_rejected_before_any_http(change, tmp_path):
    bundle = fixture_bundle()
    first = bundle.train[0]
    test = replace(bundle.test[0], **({"sample_id": first.sample_id} if change == "id" else {"text": first.text}))
    bundle = replace(bundle, test=(test,))
    client = Mock()
    classifier = EmissaryClassifier(client=client, experiment_name="must-not-create")
    with pytest.raises(ValueError, match="overlapping"):
        run_benchmark(StaticDataset(bundle), classifier, BenchmarkRunConfig(output_root=tmp_path))
    assert not client.mock_calls


def test_validation_overlap_and_duplicate_ids():
    bundle = fixture_bundle()
    with pytest.raises(ValueError, match="overlapping"):
        select_labeled_examples(bundle.train, bundle.classes, Config(), validation_examples=bundle.train[:1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_partition_disjointness(bundle.train + bundle.train[:1])


@pytest.mark.parametrize("budget", [1, 5, 100, 1000000])
def test_every_nonzero_budget_is_explicitly_unsupported(budget):
    client = Mock()
    classifier = EmissaryClassifier(client=client, experiment_name="blocked", training=Config(budget, "total"))
    for operation in [lambda: classifier.prepare(fixture_bundle().classes), lambda: classifier.fit([]), lambda: classifier.predict([])]:
        with pytest.raises(NotImplementedError, match="project_fine_tuning"):
            operation()
    assert not client.mock_calls
    with pytest.raises(ValueError, match="reference model"):
        EmissaryClassifier(model_id="ex-reference/0.0.0", training=Config(budget, "total"))


def test_runner_dry_run_persists_only_fit_selection_and_failed_live_plan(tmp_path):
    bundle = fixture_bundle()
    configs = []
    selections = []
    for shots in [0, 5, 100]:
        classifier = EmissaryClassifier(experiment_name="offline", training=Config(shots, "total"))
        result = run_benchmark(StaticDataset(bundle), classifier, BenchmarkRunConfig(
            output_root=tmp_path, dry_run=True, run_id=f"shots-{shots}", validation_fraction=.2,
        ))
        config = json.loads(result.config_path.read_text())
        plan = json.loads((result.run_dir / "preparation_plan.json").read_text())
        selected = [e["sample_id"] for e in plan["selection"]["selected_examples"]]
        assert set(selected) <= set(config["dataset"]["fit_train_sample_ids"])
        assert not set(selected) & set(config["dataset"]["validation_sample_ids"] + config["dataset"]["test_sample_ids"])
        assert not result.predictions_path.exists()
        assert not (result.run_dir / "fit_metadata.json").exists()
        assert json.loads(result.status_path.read_text())["status"] == "dry_run"
        assert classifier.client is None
        assert plan["live_supported"] == (shots == 0)
        assert plan["cost"]["value_usd"] is None
        configs.append(config)
        selections.append(selected)
    assert selections[2][:5] == selections[1]
    assert all(c["dataset"]["classes"] == configs[0]["dataset"]["classes"] for c in configs)
    assert all(c["dataset"]["test_sample_ids"] == configs[0]["dataset"]["test_sample_ids"] for c in configs)
    with pytest.raises(NotImplementedError):
        run_benchmark(StaticDataset(bundle), classifier, BenchmarkRunConfig(output_root=tmp_path, run_id="blocked-live"))
    assert (tmp_path / "blocked-live" / "preparation_plan.json").exists()
    assert classifier.client is None


def http_client(responses):
    session = Mock(spec=requests.Session)
    session.headers = {}
    session.post.side_effect = [Mock(ok=True, json=Mock(return_value=r)) if isinstance(r, dict) else r for r in responses]
    return EmissaryClient(api_key="offline-fixture-key", session=session), session


def test_real_runner_mocked_http_payloads_post_fit_identity_and_unknown_cost(tmp_path):
    bundle = fixture_bundle(2, 4)
    response = {"id": "req-1", "model": "ex-fixture/1.2.3",
                "data": [{"probs": {"class-0": .8, "class-1": .2}}],
                "usage": {"provider_evidence": 12}}
    client, session = http_client([{"id": "ex-fixture", "latest_version": "1.2.3"}, response, response])
    classifier = EmissaryClassifier(client=client, experiment_name="fixture")
    result = run_benchmark(StaticDataset(bundle), classifier, BenchmarkRunConfig(output_root=tmp_path))
    calls = session.post.call_args_list
    assert len(calls) == 3
    assert session.headers["X-API-Key"] == "offline-fixture-key"
    assert calls[0].args[0].endswith("/v1/experiments")
    assert calls[0].kwargs["json"] == {"name": "fixture", "mode": "routing", "classes": [
        {"name": c.name, "description": c.description} for c in bundle.classes]}
    for call, example in zip(calls[1:], bundle.test):
        assert call.args[0].endswith("/v1/classification")
        assert call.kwargs["json"] == {"model": "ex-fixture/1.2.3", "input": example.text, "data_format": "probs"}
    config = json.loads(result.config_path.read_text())
    metadata = json.loads((result.run_dir / "fit_metadata.json").read_text())
    assert "model_id" not in config["classifier"]
    assert config["classifier"]["training"]["shots"] == 0
    assert metadata["model_id"] == "ex-fixture/1.2.3"
    assert metadata["model_version"] == "1.2.3"
    assert metadata["experiment_creation_ms"] >= 0
    assert metadata["provider_training_ms"] is None
    assert metadata["selection"]["actual_total"] == 0
    assert metadata["cost"]["available"] is False
    predictions = [json.loads(line) for line in result.predictions_path.read_text().splitlines()]
    assert predictions[0]["raw_response"]["usage"] == response["usage"]
    assert all(p["latency_ms"] >= 0 for p in predictions)
    assert json.loads(result.metrics_path.read_text())["total_cost_usd"]["available"] is False


def test_creation_timeout_no_retry_and_reset():
    client, session = http_client([{"id": "ex-one", "latest_version": "0.0.0"}, requests.Timeout("ambiguous")])
    classifier = EmissaryClassifier(client=client, experiment_name="fixture")
    classifier.prepare(fixture_bundle().classes)
    classifier.fit(fixture_bundle().train)
    with pytest.raises(EmissaryAPIError):
        classifier.prepare(fixture_bundle().classes)
    assert session.post.call_count == 2
    assert classifier.model_id is None
    assert classifier.fitted_metadata()["selection"] is None
    with pytest.raises(RuntimeError, match="prepare"):
        classifier.predict([])


def test_response_version_and_class_space_are_checked():
    for response in [
        {"model": "ex-one/2.0.0", "data": [{"probs": {"class-0": .5, "class-1": .5}}]},
        {"model": "ex-one/1.0.0", "data": [{"probs": {"wrong": 1.0}}]},
    ]:
        client, _ = http_client([response])
        classifier = EmissaryClassifier(client=client, model_id="ex-one/1.0.0")
        classifier.prepare(fixture_bundle(2).classes)
        with pytest.raises(EmissaryResponseError):
            classifier.predict([fixture_bundle().test[0].as_input()])


@pytest.mark.parametrize("model_id", ["ex-unversioned", "ex-one/latest", "ex-one/", "/0.0.0"])
def test_unpinned_reference_model_rejected(model_id):
    with pytest.raises(ValueError, match="pin"):
        EmissaryClassifier(model_id=model_id)


def test_invalid_eager_create_does_not_mutate_remote():
    client = Mock()
    with pytest.raises(ValueError, match="classifier_name"):
        EmissaryClassifier.create(client=client, experiment_name="fixture", classes=fixture_bundle().classes,
                                  classifier_name=" ")
    assert not client.mock_calls


def test_post_fit_metadata_survives_inference_failure(tmp_path):
    client, _ = http_client([{"id": "ex-fixture", "latest_version": "0.0.0"}, requests.Timeout("inference")])
    classifier = EmissaryClassifier(client=client, experiment_name="fixture")
    with pytest.raises(EmissaryAPIError):
        run_benchmark(StaticDataset(fixture_bundle()), classifier,
                      BenchmarkRunConfig(output_root=tmp_path, run_id="failure"))
    metadata = json.loads((tmp_path / "failure" / "fit_metadata.json").read_text())
    assert metadata["model_id"] == "ex-fixture/0.0.0"
    assert metadata["selection"]["actual_total"] == 0
    assert json.loads((tmp_path / "failure" / "status.json").read_text())["stage"] == "predicting"
