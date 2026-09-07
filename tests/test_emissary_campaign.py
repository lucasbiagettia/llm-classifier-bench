"""Both CLI entry points use frozen definitions/test IDs across dry-run budgets."""
import importlib
import json
from pathlib import Path
import sys

import pytest

from llm_classifier_bench.class_definitions import ClassDefinitionProfile
from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.datasets.base import DatasetBundle


@pytest.mark.parametrize("script", ["run_banking77_scaling_benchmark", "run_banking77_scaling_benchmark_v2"])
def test_campaign_dry_runs_and_live_guard(script, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    campaign = importlib.import_module(script)
    classes = tuple(ClassDefinition(str(i), f"Frozen {i}") for i in range(3))
    bundle = DatasetBundle("banking77", classes,
        tuple(LabeledExample(f"train-{c.name}-{i}", f"Train {c.name} {i}", c.name) for c in classes for i in range(80)),
        tuple(LabeledExample(f"test-{c.name}", f"Test {c.name}", c.name) for c in classes))
    definitions = tmp_path / "definitions.json"
    ClassDefinitionProfile(dataset=bundle.name, profile="fixture", classes=classes,
                           review_status="approved").write_json(definitions)
    monkeypatch.setattr(campaign, "get_dataset", lambda _: campaign.StaticDataset(bundle))
    monkeypatch.setattr(campaign, "utc_campaign_id", lambda: "fixture-campaign")
    def forbidden(*args, **kwargs):
        pytest.fail("Dry runs and unsupported live conditions must not create clients")
    monkeypatch.setattr("llm_classifier_bench.classifiers.emissary.EmissaryClient", forbidden)
    monkeypatch.setattr(campaign, "EmissaryClient", forbidden)
    args = [script, "--classifiers", "emissary", "--class-counts", "2", "--seeds", "17",
            "--train-per-class", "70", "--test-per-class", "1", "--validation-fraction", ".2",
            "--definitions", str(definitions), "--output-root", str(tmp_path),
            "--emissary-shots", "0", "5", "100", "--emissary-shot-unit", "total"]
    if script.endswith("_v2"):
        args += ["--min-train-per-class", "70", "--min-test-per-class", "1", "--strict-support"]
    monkeypatch.setattr(sys, "argv", args + ["--dry-run"])
    campaign.main()
    root = tmp_path / "fixture-campaign"
    configs = [json.loads(p.read_text()) for p in sorted(root.glob("runs/*/config.json"))]
    plans = [json.loads(p.read_text()) for p in root.glob("runs/*/preparation_plan.json")]
    assert len(configs) == len(plans) == 3
    assert sorted(p["selection"]["actual_total"] for p in plans) == [0, 5, 100]
    assert all(c["dataset"]["classes"] == configs[0]["dataset"]["classes"] for c in configs)
    assert all(c["dataset"]["test_sample_ids"] == configs[0]["dataset"]["test_sample_ids"] for c in configs)
    assert all(c["class_definitions"] == configs[0]["class_definitions"] for c in configs)
    assert {r["status"] for r in json.loads((root / "summary.json").read_text())} == {"dry_run"}
    assert not list(root.glob("runs/*/predictions.jsonl"))
    # The entire mixed 0/5/100 live invocation fails before even loading data.
    monkeypatch.setattr(campaign, "get_dataset", forbidden)
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(NotImplementedError, match="Nonzero Emissary"):
        campaign.main()


@pytest.mark.parametrize("script", ["run_banking77_scaling_benchmark", "run_banking77_scaling_benchmark_v2"])
def test_campaign_explicit_project_sft_plan_and_unpriced_live_guards(
    script, tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    campaign = importlib.import_module(script)
    classes = tuple(ClassDefinition(str(index), f"Frozen {index}") for index in range(2))
    bundle = DatasetBundle(
        "banking77",
        classes,
        tuple(
            LabeledExample(f"train-{item.name}-{index}", f"Train {item.name} {index}", item.name)
            for item in classes
            for index in range(70)
        ),
        tuple(LabeledExample(f"test-{item.name}", f"Test {item.name}", item.name) for item in classes),
    )
    definitions = tmp_path / "definitions-project.json"
    ClassDefinitionProfile(
        dataset=bundle.name,
        profile="fixture-project",
        classes=classes,
        review_status="approved",
    ).write_json(definitions)
    monkeypatch.setattr(campaign, "get_dataset", lambda _: campaign.StaticDataset(bundle))
    monkeypatch.setattr(campaign, "utc_campaign_id", lambda: "project-plan")

    def forbidden(*args, **kwargs):
        pytest.fail("Project SFT dry-run and CLI guards must not construct a client")

    monkeypatch.setattr(campaign, "EmissaryClient", forbidden)
    args = [
        script,
        "--classifiers", "emissary",
        "--class-counts", "2",
        "--seeds", "17",
        "--train-per-class", "70",
        "--test-per-class", "1",
        "--validation-fraction", ".2",
        "--definitions", str(definitions),
        "--output-root", str(tmp_path),
        "--emissary-shots", "5",
        "--emissary-shot-unit", "total",
        "--emissary-mechanism", "project_fine_tuning",
        "--emissary-project-id", "ms-fixture",
        "--emissary-base-model", "Llama-3.2-1B-Instruct",
    ]
    if script.endswith("_v2"):
        args += [
            "--min-train-per-class", "70",
            "--min-test-per-class", "1",
            "--strict-support",
        ]

    monkeypatch.setattr(sys, "argv", args + ["--dry-run"])
    campaign.main()
    root = tmp_path / "project-plan"
    plan = json.loads(next(root.glob("runs/*/preparation_plan.json")).read_text())
    assert plan["live_supported"] is True
    assert plan["mechanism"] == "project_fine_tuning"
    assert plan["selection"]["actual_total"] == 5
    assert plan["selection"]["provider_dataset"]["schema"] == "classification_label_map"
    assert plan["planned_remote_operations"][0] == "GET /v1/projects/{project_id}"
    assert plan["planned_remote_operations"][1].endswith("Llama-3.2-1B-Instruct")
    assert any("training-jobs" in operation for operation in plan["planned_remote_operations"])
    assert "separately selected base model" in plan["scientific_comparability"]

    monkeypatch.setattr(campaign, "get_dataset", forbidden)
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="unpriced-training"):
        campaign.main()
    monkeypatch.setattr(sys, "argv", args + ["--emissary-allow-unpriced-training"])
    with pytest.raises(ValueError, match="max-training-jobs"):
        campaign.main()
