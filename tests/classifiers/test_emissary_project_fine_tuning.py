"""Offline contract tests for Emissary's public Projects SFT mechanism."""

from __future__ import annotations

import json
from hashlib import sha256
from unittest.mock import Mock

import pytest
import requests

from llm_classifier_bench.classifiers.emissary import (
    EmissaryAPIError,
    EmissaryClassifier,
    EmissaryClient,
    EmissaryPreparationError,
    EmissaryPreparationTimeout,
    EmissaryResponseError,
    serialize_classification_dataset,
)
from llm_classifier_bench.config import EmissaryTrainingConfig
from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.datasets.base import DatasetBundle
from llm_classifier_bench.datasets.selection import select_labeled_examples
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark


def fixture_bundle(class_count=2, support=4):
    classes = tuple(
        ClassDefinition(f"class-{index}", f"Frozen definition {index}")
        for index in range(class_count)
    )
    return DatasetBundle(
        "banking77",
        classes,
        tuple(
            LabeledExample(
                f"train-{item.name}-{index}",
                f"Training {item.name} number {index}",
                item.name,
            )
            for item in classes
            for index in range(support)
        ),
        tuple(
            LabeledExample(f"test-{item.name}", f"Held out {item.name}", item.name)
            for item in classes
        ),
    )


class StaticDataset:
    def __init__(self, bundle):
        self.bundle = bundle
        self.name = bundle.name

    def load(self):
        return self.bundle


def fine_config(**overrides):
    values = {
        "shots": 4,
        "shot_unit": "total",
        "selection_seed": 17,
        "mechanism": "project_fine_tuning",
        "project_id": "ms-fixture",
        "base_model": "Llama-3.2-1B-Instruct",
        "poll_interval_s": 0.01,
    }
    values.update(overrides)
    return EmissaryTrainingConfig(**values)


def response(payload):
    result = Mock(spec=requests.Response)
    result.ok = True
    result.json.return_value = payload
    return result


def http_client(*, gets=(), posts=()):
    session = Mock(spec=requests.Session)
    session.headers = {}
    session.get.side_effect = [
        item if isinstance(item, BaseException) else response(item) for item in gets
    ]
    session.post.side_effect = [
        item if isinstance(item, BaseException) else response(item) for item in posts
    ]
    return EmissaryClient(api_key="offline-fixture-key", session=session), session


def base_model(*, supports_classification=True):
    return {
        "name": "Llama-3.2-1B-Instruct",
        "support_task": (
            ["classification", "text-generation"]
            if supports_classification
            else ["text-generation"]
        ),
        "parameter_template": {"num_train_epochs": 3},
    }


def project_detail():
    return {"id": "ms-fixture", "name": "fixture-project"}


def continuation_hash(bundle):
    selected, _ = select_labeled_examples(
        bundle.train,
        bundle.classes,
        fine_config(),
    )
    return sha256(serialize_classification_dataset(selected, bundle.classes)).hexdigest()


def dataset_detail():
    return {
        "id": "ds-fixture",
        "name": "dataset",
        "is_uploaded": True,
        "is_profiled": True,
        "compatible_task_types": ["classification"],
        "dataset_download_url": "https://signed.example/secret",
    }


def training_detail(status="Success"):
    return {
        "id": "tr-fixture",
        "status": status,
        "task_type": "classification",
        "base_model": "Llama-3.2-1B-Instruct",
        "train_dataset": {"id": "ds-fixture", "name": "dataset"},
        "created_at": 100,
        "updated_at": 200,
    }


def deployment_detail(status="Deployed", *, checkpoint=2):
    return {
        "id": "dp-fixture",
        "name": "fixture-deployment",
        "status": status,
        "task_type": "classification",
        "base_model": "Llama-3.2-1B-Instruct",
        "training_id": "tr-fixture",
        "checkpoint": checkpoint,
        "metadata": {"labels": ["class-0", "class-1"]},
    }


def prediction(sample_number):
    return {
        "id": f"request-{sample_number}",
        "model": "dp-fixture",
        "data": [{"index": 0, "probs": {"class-0": 0.75, "class-1": 0.25}}],
    }


def test_documented_classification_dataset_schema_is_exact_and_deterministic():
    bundle = fixture_bundle(2, 2)
    content = serialize_classification_dataset(bundle.train[:2], bundle.classes)
    rows = [json.loads(line) for line in content.decode().splitlines()]
    assert rows == [
        {
            "prompt": bundle.train[0].text,
            "completion": {"class-0": 1, "class-1": 0},
        },
        {
            "prompt": bundle.train[1].text,
            "completion": {"class-0": 1, "class-1": 0},
        },
    ]
    assert content.endswith(b"\n")
    assert serialize_classification_dataset(bundle.train[:2], bundle.classes) == content


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"mechanism": "project_fine_tuning", "shots": 0}, "training mechanism"),
        ({"project_id": None}, "project_id"),
        ({"base_model": None}, "base_model"),
        ({"deployment_id": "dp-one", "training_job_id": None}, "training_job_id"),
        ({"training_job_id": "tr-one", "dataset_id": None}, "dataset_id"),
        ({"checkpoint": -1}, "checkpoint"),
        ({"training_timeout_s": 0}, "training_timeout_s"),
        ({"dynamic_loss": 1}, "dynamic_loss"),
        ({"mechanism": None, "project_id": "ms-fixture"}, "Project fields"),
    ],
)
def test_project_training_configuration_rejects_ambiguous_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        fine_config(**overrides)


def test_end_to_end_runner_uses_confirmed_project_payloads_and_pins_deployment(tmp_path):
    bundle = fixture_bundle(2, 4)
    client, session = http_client(
        gets=[
            project_detail(),
            base_model(),
            dataset_detail(),
            training_detail("Running"),
            training_detail("Success"),
            [
                {"checkpoint": 1, "model_download_url": "https://signed.example/one"},
                {"checkpoint": 2, "model_download_url": "https://signed.example/two"},
            ],
            deployment_detail(),
        ],
        posts=[
            {"id": "ds-fixture", "name": "dataset", "is_uploaded": True},
            {"id": "tr-fixture", "name": "training", "status": "Pending"},
            {"id": "dp-fixture", "name": "fixture-deployment", "status": "Pending"},
            prediction(1),
            prediction(2),
        ],
    )
    classifier = EmissaryClassifier(
        client=client,
        training=fine_config(),
        experiment_name="fixture-condition",
        classifier_name="emissary-project-sft",
        sleep_fn=lambda _: None,
    )
    result = run_benchmark(
        StaticDataset(bundle),
        classifier,
        BenchmarkRunConfig(output_root=tmp_path, validation_fraction=0.25),
    )

    get_calls = session.get.call_args_list
    assert get_calls[0].args[0].endswith("/v1/projects/ms-fixture")
    assert get_calls[1].args[0].endswith("/v1/models/Llama-3.2-1B-Instruct")
    assert get_calls[2].args[0].endswith(
        "/v1/projects/ms-fixture/datasets/ds-fixture"
    )
    assert get_calls[3].args[0].endswith(
        "/v1/projects/ms-fixture/training-jobs/tr-fixture"
    )
    assert get_calls[5].args[0].endswith(
        "/v1/projects/ms-fixture/training-jobs/tr-fixture/checkpoints"
    )
    assert get_calls[6].args[0].endswith(
        "/v1/projects/ms-fixture/deployments/dp-fixture"
    )

    post_calls = session.post.call_args_list
    upload = post_calls[0]
    assert upload.args[0].endswith("/v1/projects/ms-fixture/datasets")
    assert upload.kwargs["data"] == {"name": "fixture-condition-dataset"}
    filename, dataset_bytes, media_type = upload.kwargs["files"]["file"]
    assert filename == "fixture-condition.jsonl"
    assert media_type == "application/json"
    rows = [json.loads(line) for line in dataset_bytes.decode().splitlines()]
    assert len(rows) == 4
    assert all(set(row["completion"]) == {"class-0", "class-1"} for row in rows)
    assert all(sum(row["completion"].values()) == 1 for row in rows)

    training = post_calls[1]
    assert training.args[0].endswith(
        "/v1/projects/ms-fixture/training-jobs"
    )
    assert training.kwargs["json"] == {
        "base_model_option": "pre-trained",
        "base_model": "Llama-3.2-1B-Instruct",
        "train_dataset_id": "ds-fixture",
        "name": "fixture-condition-training",
        "task_type": "classification",
        "parameters": {
            "learning_rate": 0.0002,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 1,
            "dynamic_loss": False,
        },
    }
    deployment = post_calls[2]
    assert deployment.kwargs["json"] == {
        "training_job_id": "tr-fixture",
        "checkpoint": 2,
        "name": "fixture-condition-deployment",
        "description": "llm-classifier-bench labeled-example condition",
        "inactive_timeout": 300,
    }
    for call, example in zip(post_calls[3:], bundle.test, strict=True):
        assert call.kwargs["json"] == {
            "model": "fixture-deployment",
            "input": example.text,
            "data_format": "probs",
        }

    metadata = json.loads((result.run_dir / "fit_metadata.json").read_text())
    assert metadata["status"] == "ready"
    assert metadata["mechanism"] == "project_fine_tuning"
    assert metadata["dataset_id"] == "ds-fixture"
    assert metadata["training_job_id"] == "tr-fixture"
    assert metadata["checkpoint"] == 2
    assert metadata["deployment_id"] == metadata["model_id"] == "dp-fixture"
    assert metadata["deployment_name"] == "fixture-deployment"
    assert metadata["selection"]["actual_total"] == 4
    assert metadata["provider_training_ms"] is None
    assert metadata["cost"]["available"] is False
    assert metadata["dataset_response"]["dataset_download_url"] == "<redacted>"
    assert all(
        item["model_download_url"] == "<redacted>"
        for item in metadata["checkpoint_responses"]
    )
    assert [item["status"] for item in metadata["training_status_history"]] == [
        "Running",
        "Success",
    ]
    metrics = json.loads(result.metrics_path.read_text())
    assert metrics["total_cost_usd"]["available"] is False


def test_base_model_support_is_checked_before_remote_mutation():
    bundle = fixture_bundle(2, 3)
    client, session = http_client(
        gets=[project_detail(), base_model(supports_classification=False)]
    )
    classifier = EmissaryClassifier(
        client=client,
        training=fine_config(),
        experiment_name="unsupported-base",
    )
    classifier.prepare(bundle.classes)
    with pytest.raises(ValueError, match="does not support classification"):
        classifier.fit(bundle.train)
    assert session.post.call_count == 0


def test_failed_training_persists_job_id_for_continuation(tmp_path):
    bundle = fixture_bundle(2, 3)
    client, _ = http_client(
        gets=[project_detail(), base_model(), dataset_detail(), training_detail("Failed")],
        posts=[
            {"id": "ds-fixture", "is_uploaded": True},
            {"id": "tr-fixture", "status": "Pending"},
        ],
    )
    classifier = EmissaryClassifier(
        client=client,
        training=fine_config(),
        experiment_name="failed-condition",
        sleep_fn=lambda _: None,
    )
    with pytest.raises(EmissaryPreparationError, match="Failed"):
        run_benchmark(
            StaticDataset(bundle),
            classifier,
            BenchmarkRunConfig(output_root=tmp_path, validation_fraction=0),
        )
    paths = list(tmp_path.glob("*/fit_metadata.json"))
    assert len(paths) == 1
    metadata = json.loads(paths[0].read_text())
    assert metadata["status"] == "failed"
    assert metadata["dataset_id"] == "ds-fixture"
    assert metadata["training_job_id"] == "tr-fixture"
    assert metadata["last_error"]["type"] == "EmissaryPreparationError"


def test_training_timeout_is_bounded_and_does_not_resubmit():
    bundle = fixture_bundle(2, 3)
    client, session = http_client(
        gets=[project_detail(), base_model(), dataset_detail(), training_detail("Pending")],
        posts=[
            {"id": "ds-fixture", "is_uploaded": True},
            {"id": "tr-fixture", "status": "Pending"},
        ],
    )
    times = iter([0.0, 0.0, 1.0])
    classifier = EmissaryClassifier(
        client=client,
        training=fine_config(training_timeout_s=0.5),
        experiment_name="timeout-condition",
        sleep_fn=lambda _: None,
        clock_fn=lambda: next(times),
    )
    classifier.prepare(bundle.classes)
    with pytest.raises(EmissaryPreparationTimeout, match="training job"):
        classifier.fit(bundle.train)
    assert session.post.call_count == 2
    assert classifier.fitted_metadata()["training_job_id"] == "tr-fixture"


def test_dataset_profiling_timeout_happens_before_training_submission():
    bundle = fixture_bundle(2, 3)
    unready = {
        "id": "ds-fixture",
        "is_uploaded": True,
        "is_profiled": False,
        "compatible_task_types": [],
    }
    client, session = http_client(
        gets=[project_detail(), base_model(), unready],
        posts=[{"id": "ds-fixture", "is_uploaded": True}],
    )
    times = iter([0.0, 1.0])
    classifier = EmissaryClassifier(
        client=client,
        training=fine_config(dataset_timeout_s=0.5),
        experiment_name="dataset-timeout",
        sleep_fn=lambda _: None,
        clock_fn=lambda: next(times),
    )
    classifier.prepare(bundle.classes)
    with pytest.raises(EmissaryPreparationTimeout, match="dataset"):
        classifier.fit(bundle.train)
    assert session.post.call_count == 1
    assert classifier.fitted_metadata()["dataset_id"] == "ds-fixture"


def test_deployment_terminal_failure_prevents_inference():
    bundle = fixture_bundle(2, 3)
    client, session = http_client(
        gets=[
            project_detail(),
            base_model(),
            dataset_detail(),
            training_detail(),
            [{"checkpoint": 2}],
            deployment_detail("Failed"),
        ],
        posts=[
            {"id": "ds-fixture", "is_uploaded": True},
            {"id": "tr-fixture", "status": "Pending"},
            {"id": "dp-fixture", "name": "fixture-deployment", "status": "Pending"},
        ],
    )
    classifier = EmissaryClassifier(
        client=client,
        training=fine_config(),
        experiment_name="deployment-failure",
        sleep_fn=lambda _: None,
    )
    classifier.prepare(bundle.classes)
    with pytest.raises(EmissaryPreparationError, match="Failed"):
        classifier.fit(bundle.train)
    assert session.post.call_count == 3
    with pytest.raises(RuntimeError, match="fit"):
        classifier.predict(bundle.inputs())


def test_ambiguous_training_submission_is_never_retried():
    bundle = fixture_bundle(2, 3)
    client, session = http_client(
        gets=[project_detail(), base_model(), dataset_detail()],
        posts=[
            {"id": "ds-fixture", "is_uploaded": True},
            requests.Timeout("ambiguous training submission"),
        ],
    )
    classifier = EmissaryClassifier(
        client=client,
        training=fine_config(),
        experiment_name="ambiguous-submission",
    )
    classifier.prepare(bundle.classes)
    with pytest.raises(EmissaryAPIError, match="ambiguous"):
        classifier.fit(bundle.train)
    assert session.post.call_count == 2
    metadata = classifier.fitted_metadata()
    assert metadata["dataset_id"] == "ds-fixture"
    assert metadata["training_job_id"] is None


def test_continuation_hash_must_match_current_selection_before_job_lookup():
    bundle = fixture_bundle(2, 3)
    client, session = http_client(gets=[project_detail(), base_model()])
    classifier = EmissaryClassifier(
        client=client,
        training=fine_config(
            dataset_id="ds-fixture",
            training_job_id="tr-fixture",
            dataset_sha256="0" * 64,
        ),
        experiment_name="wrong-selection",
    )
    classifier.prepare(bundle.classes)
    with pytest.raises(ValueError, match="dataset_sha256"):
        classifier.fit(bundle.train)
    assert session.post.call_count == 0
    assert session.get.call_count == 2


def test_existing_job_and_deployment_resume_without_non_idempotent_posts():
    bundle = fixture_bundle(2, 3)
    client, session = http_client(
        gets=[
            project_detail(),
            base_model(),
            dataset_detail(),
            training_detail(),
            [{"checkpoint": 2}],
            deployment_detail(),
        ],
        posts=[prediction(1)],
    )
    classifier = EmissaryClassifier(
        client=client,
        training=fine_config(
            dataset_id="ds-fixture",
            training_job_id="tr-fixture",
            checkpoint=2,
            deployment_id="dp-fixture",
            dataset_sha256=continuation_hash(bundle),
        ),
        experiment_name="continued-condition",
    )
    classifier.prepare(bundle.classes)
    classifier.fit(bundle.train)
    classifier.predict([bundle.test[0].as_input()])
    assert session.post.call_count == 1
    assert session.post.call_args.kwargs["json"]["model"] == "fixture-deployment"
    metadata = classifier.fitted_metadata()
    assert metadata["dataset_id"] == "ds-fixture"
    assert metadata["training_job_id"] == "tr-fixture"
    assert metadata["deployment_id"] == "dp-fixture"


def test_resumed_job_must_reference_the_pinned_dataset():
    bundle = fixture_bundle(2, 3)
    mismatched_job = training_detail()
    mismatched_job["train_dataset"] = {"id": "ds-other"}
    client, session = http_client(
        gets=[project_detail(), base_model(), dataset_detail(), mismatched_job]
    )
    classifier = EmissaryClassifier(
        client=client,
        training=fine_config(
            dataset_id="ds-fixture",
            training_job_id="tr-fixture",
            dataset_sha256=continuation_hash(bundle),
        ),
        experiment_name="mismatched-dataset",
    )
    classifier.prepare(bundle.classes)
    with pytest.raises(EmissaryResponseError, match="dataset"):
        classifier.fit(bundle.train)
    assert session.post.call_count == 0


def test_pinned_checkpoint_mismatch_fails_before_inference():
    bundle = fixture_bundle(2, 3)
    client, session = http_client(
        gets=[
            project_detail(),
            base_model(),
            dataset_detail(),
            training_detail(),
            [{"checkpoint": 2}],
            deployment_detail(checkpoint=1),
        ]
    )
    classifier = EmissaryClassifier(
        client=client,
        training=fine_config(
            dataset_id="ds-fixture",
            training_job_id="tr-fixture",
            checkpoint=2,
            deployment_id="dp-fixture",
            dataset_sha256=continuation_hash(bundle),
        ),
        experiment_name="mismatch-condition",
    )
    classifier.prepare(bundle.classes)
    with pytest.raises(EmissaryResponseError, match="checkpoint"):
        classifier.fit(bundle.train)
    assert session.post.call_count == 0


def test_documented_upload_size_limit_is_enforced_before_http():
    class HugeContent:
        def __bool__(self):
            return True

        def __len__(self):
            return 100_000_001

    client, session = http_client()
    with pytest.raises(ValueError, match="100 MB"):
        client.upload_dataset(
            project_id="ms-fixture",
            name="too-large",
            filename="too-large.jsonl",
            content=HugeContent(),  # type: ignore[arg-type]
        )
    assert session.post.call_count == 0
