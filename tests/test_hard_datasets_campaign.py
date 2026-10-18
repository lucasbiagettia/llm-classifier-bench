"""Plan all methods with no network/models; replay frozen shared inputs locally."""
from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import sys

import pytest

from llm_classifier_bench.classifiers.base import Prediction
from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.datasets import DatasetBundle


@pytest.fixture
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("run_hard_datasets")
    def forbidden(*a, **kw):
        pytest.fail("Preparation must not perform network requests or train models")
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr(module, "run_benchmark", forbidden)
    return module


def fixture_bundle(train_per_class=30):
    classes = tuple(ClassDefinition(n, f"Definition {n}") for n in ("alpha", "beta", "gamma"))
    def examples(split, count):
        return tuple(LabeledExample(f"{split}:{c.name}:{i}", f"{c.name} {split} text {i}", c.name,
                     {"dataset": "anli_r3", "split": split, "annotations": [{"reason": "SECRET"}]})
                     for c in classes for i in range(count))
    return DatasetBundle("anli_r3", classes, examples("train", train_per_class), examples("test", 50),
                         validation=examples("dev", 10))


def prepare_fixture(campaign, monkeypatch, tmp_path, methods=None, extra=()):
    monkeypatch.setattr(campaign, "get_dataset", lambda _: campaign.StaticDataset(fixture_bundle()))
    args = campaign.parser().parse_args(["--datasets", "anli_r3", "--output-dir", str(tmp_path / "plan"),
                                        *(["--methods", *methods] if methods else []), *extra])
    plan = campaign.prepare(args, campaign.read_json(args.config))
    return args, plan


def test_all_methods_plan_without_models_or_api(campaign, monkeypatch, tmp_path):
    args, plan = prepare_fixture(campaign, monkeypatch, tmp_path)
    assert {c["method"] for c in plan["cells"]} == set(campaign.METHODS)
    assert plan["planned_api_calls"] == 8 * 120
    assert plan["planned_training_jobs"] == 2
    assert plan["estimated_priced_subtotal_usd"] > 0
    assert plan["total_cost_usd"] is None
    for cell in plan["cells"]:
        assert cell["test_examples"] == 120
        assert cell["fit_examples"] == (60 if cell["method"] in campaign.LOCAL | campaign.SFT else 0)
        assert cell["context_examples"] == (5 if cell["method"] == "openai-context" else 0)
        assert cell["validation_examples"] == 0
        assert bool(cell["blocker"]) == (cell["method"] in campaign.SFT)
    snapshot = campaign.load_snapshot(args.output_dir / "anli_r3.json")
    assert all("SECRET" not in e.text for e in snapshot.test)
    assert snapshot.test[0].metadata["annotations"][0]["reason"] == "SECRET"
    assert plan["datasets"]["anli_r3"]["selected_counts"]["train"] == {n: 20 for n in snapshot.class_names}


def test_reproducible_sampling_and_additional_validation(campaign):
    cfg = campaign.read_json(campaign.ROOT / "configs/hard_datasets_pilot.json")
    cfg["validation_per_class"] = 2
    first = campaign.select_bundle(fixture_bundle(), cfg)
    second = campaign.select_bundle(replace(fixture_bundle(), train=tuple(reversed(fixture_bundle().train))), cfg)
    assert first == second
    assert len(first.train) == 60 and len(first.validation) == 6 and len(first.test) == 120
    with pytest.raises(ValueError, match="Insufficient"):
        campaign.select_bundle(fixture_bundle(), {**cfg, "test_per_class": 100})


def test_changed_snapshot_stops_before_execution(campaign, monkeypatch, tmp_path):
    args, _ = prepare_fixture(campaign, monkeypatch, tmp_path, ["tfidf"])
    (args.output_dir / "anli_r3.json").write_text("changed")
    args.plan = args.output_dir / "plan.json"
    with pytest.raises(ValueError, match="Prepared file changed"):
        campaign.execute(args)


@pytest.mark.parametrize("flag,value,message", [("max_api_calls", 0, "max-api-calls"),
    ("max_training_jobs", 0, "max-training-jobs"), ("max_estimated_usd", 0, "max-estimated-usd")])
def test_budget_guards(campaign, monkeypatch, tmp_path, flag, value, message):
    args, _ = prepare_fixture(campaign, monkeypatch, tmp_path,
                             ["jev", "emissary-qwen"], ["--emissary-project-id", "project"])
    args.plan = args.output_dir / "plan.json"
    args.max_training_jobs = 1
    args.allow_unpriced = True
    setattr(args, flag, value)
    with pytest.raises(ValueError, match=message):
        campaign.execute(args)


def test_real_tfidf_fixture_replays_plan_and_official_dev(campaign, monkeypatch, tmp_path):
    from llm_classifier_bench.runner import run_benchmark
    args, plan = prepare_fixture(campaign, monkeypatch, tmp_path, ["tfidf"], ["--validation-per-class", "2"])
    monkeypatch.setattr(campaign, "run_benchmark", run_benchmark)
    args.plan = args.output_dir / "plan.json"
    output = campaign.execute(args)
    row, = campaign.read_json(output / "summary.json")
    assert row["status"] == "completed" and row["training_examples_used"] == 60
    assert row["validation_examples_used"] == 6
    config = campaign.read_json(Path(row["run_dir"]) / "config.json")
    assert config["dataset"]["test_sample_ids"] == plan["datasets"]["anli_r3"]["test_sample_ids"]
    assert all(i.startswith("dev:") for i in config["dataset"]["validation_sample_ids"])
    assert config["split"]["strategy"] == "provided_validation_partition"
    assert row["accuracy"] is not None and row["top_label_ece"] is not None
    assert row["latency_p95_ms"] is not None


def test_changed_plan_stops_before_execution(campaign, monkeypatch, tmp_path):
    args, _ = prepare_fixture(campaign, monkeypatch, tmp_path, ["tfidf"])
    args.plan = args.output_dir / "plan.json"
    args.plan.write_text(args.plan.read_text() + "\n")
    with pytest.raises(ValueError, match="Prepared plan changed"):
        campaign.execute(args)


def test_sft_validation_is_supplied_but_not_consumed(campaign, monkeypatch, tmp_path):
    args, plan = prepare_fixture(campaign, monkeypatch, tmp_path, ["emissary-qwen"],
        ["--emissary-project-id", "project", "--validation-per-class", "2"])
    cell, = plan["cells"]
    assert cell["validation_examples"] == 0
    assert cell["validation_examples_supplied"] == 6


def test_unpriced_requires_acknowledgement(campaign, monkeypatch, tmp_path):
    args, _ = prepare_fixture(campaign, monkeypatch, tmp_path, ["emissary"])
    args.plan = args.output_dir / "plan.json"
    with pytest.raises(ValueError, match="allow-unpriced"):
        campaign.execute(args)


def test_all_methods_receive_same_inputs_without_explanations(campaign, monkeypatch, tmp_path):
    from llm_classifier_bench.runner import run_benchmark
    args, plan = prepare_fixture(campaign, monkeypatch, tmp_path, extra=["--emissary-project-id", "project"])
    seen = []
    class Spy:
        name = "offline-spy"
        def prepare(self, classes): self.classes = classes
        def fit(self, examples, *, validation_examples=()): pass
        def predict(self, examples):
            seen.extend((e.sample_id, e.text, dict(e.metadata)) for e in examples)
            assert all("SECRET" not in e.text and "annotations" not in e.metadata for e in examples)
            return [Prediction(e.sample_id, self.classes[0].name, None, None, 0.0) for e in examples]
    monkeypatch.setattr(campaign, "build_classifier", lambda *a, **kw: Spy())
    monkeypatch.setattr(campaign, "run_benchmark", run_benchmark)
    args.plan = args.output_dir / "plan.json"
    args.max_training_jobs = 2
    args.allow_unpriced = True
    campaign.execute(args)
    n = 120
    assert len(seen) == n * len(campaign.METHODS)
    for start in range(0, len(seen), n):
        assert seen[start:start+n] == seen[:n]


def prepare_sweep(campaign, monkeypatch, tmp_path, methods=None):
    monkeypatch.setattr(campaign, "get_dataset", lambda _: campaign.StaticDataset(fixture_bundle(120)))
    args = campaign.parser().parse_args([
        "--datasets", "anli_r3", "--train-budgets", "100", "20", "50", "20",
        "--test-per-class", "2", "--emissary-project-id", "project",
        "--output-dir", str(tmp_path / "sweep"), *(["--methods", *methods] if methods else []),
    ])
    return args, campaign.prepare(args, campaign.read_json(args.config))


def test_sweep_expands_only_supervised_methods_and_preserves_nested_pools(campaign, monkeypatch, tmp_path):
    args, plan = prepare_sweep(campaign, monkeypatch, tmp_path)
    assert plan["configuration"]["train_budgets"] == [20, 50, 100]
    assert len(plan["cells"]) == 21
    assert plan["planned_training_jobs"] == 6
    assert plan["planned_api_calls"] == 12 * 6
    dataset = plan["datasets"]["anli_r3"]
    for method in campaign.METHODS:
        cells = [c for c in plan["cells"] if c["method"] == method]
        if method in campaign.LOCAL | campaign.SFT:
            assert [c["fit_per_class"] for c in cells] == [20, 50, 100]
            assert [c["fit_examples"] for c in cells] == [60, 150, 300]
        else:
            assert len(cells) == 1 and cells[0]["fit_examples"] == 0
        assert all(c["test_examples"] == 6 for c in cells)
    pools = dataset["fit_budgets"]
    assert set(pools["20"]["fit_sample_ids"]) < set(pools["50"]["fit_sample_ids"]) < set(pools["100"]["fit_sample_ids"])
    assert pools["50"]["counts"] == {n: 50 for n in fixture_bundle().class_names}
    # The sweep's smallest pool agrees with the original single-budget sampling.
    cfg = campaign.read_json(args.config)
    single = campaign.select_bundle(fixture_bundle(120), {**cfg, "test_per_class": 2})
    assert pools["20"]["fit_sample_ids"] == [e.sample_id for e in single.train]
    assert dataset["test_sample_ids"] == [e.sample_id for e in single.test]
    snapshot = campaign.load_snapshot(args.output_dir / "anli_r3.json")
    assert campaign.context_pool(campaign.fit_budget_bundle(snapshot, 20), cfg) == campaign.context_pool(single, cfg)


def test_sweep_execution_uses_each_fit_pool_and_keeps_context_fixed(campaign, monkeypatch, tmp_path):
    from llm_classifier_bench.runner import run_benchmark
    args, plan = prepare_sweep(campaign, monkeypatch, tmp_path)
    recorded = {}
    class Spy:
        def __init__(self, method, budget):
            self.name = f"{method}-{budget}"
        def prepare(self, classes): self.classes = classes
        def fit(self, examples, *, validation_examples=()):
            recorded[self.name] = {"fit": [e.sample_id for e in examples], "test": []}
        def predict(self, examples):
            recorded[self.name]["test"].extend(e.sample_id for e in examples)
            assert all("SECRET" not in e.text and "annotations" not in e.metadata for e in examples)
            return [Prediction(e.sample_id, self.classes[0].name, None, None, 0.) for e in examples]
    def build(method, cfg, **kwargs):
        return Spy(method, cfg["train_per_class"])
    monkeypatch.setattr(campaign, "build_classifier", build)
    monkeypatch.setattr(campaign, "run_benchmark", run_benchmark)
    args.plan = args.output_dir / "plan.json"
    args.max_training_jobs = 6
    args.allow_unpriced = True
    output = campaign.execute(args)
    rows = campaign.read_json(output / "summary.json")
    assert len(rows) == 21 and len({r["run_dir"] for r in rows}) == 21
    pools = plan["datasets"]["anli_r3"]["fit_budgets"]
    for method in campaign.LOCAL | campaign.SFT:
        for budget in (20, 50, 100):
            assert recorded[f"{method}-{budget}"]["fit"] == pools[str(budget)]["fit_sample_ids"]
    for record in recorded.values():
        assert record["test"] == plan["datasets"]["anli_r3"]["test_sample_ids"]
    for method in ("openai", "jev", "openai-decisions", "perplexity-decisions", "emissary"):
        assert recorded[f"{method}-20"]["fit"] == []
    assert len(recorded["openai-context-20"]["fit"]) == 5


def test_real_tfidf_sweep_fixture_writes_distinct_budget_runs(campaign, monkeypatch, tmp_path):
    from llm_classifier_bench.runner import run_benchmark
    args, plan = prepare_sweep(campaign, monkeypatch, tmp_path, ["tfidf"])
    monkeypatch.setattr(campaign, "run_benchmark", run_benchmark)
    args.plan = args.output_dir / "plan.json"
    output = campaign.execute(args)
    rows = campaign.read_json(output / "summary.json")
    assert [r["fit_per_class"] for r in rows] == [20, 50, 100]
    assert [r["training_examples_used"] for r in rows] == [60, 150, 300]
    assert all(r["status"] == "completed" for r in rows)
    for row in rows:
        config = campaign.read_json(Path(row["run_dir"]) / "config.json")
        assert config["dataset"]["test_sample_ids"] == plan["datasets"]["anli_r3"]["test_sample_ids"]
        assert config["dataset"]["fit_train_sample_ids"] == plan["datasets"]["anli_r3"]["fit_budgets"][str(row["fit_per_class"])]["fit_sample_ids"]


@pytest.mark.parametrize("budgets", [[0], [-1], [], [True], [20, 0]])
def test_invalid_sweep_budgets_rejected(campaign, monkeypatch, tmp_path, budgets):
    args = campaign.parser().parse_args(["--output-dir", str(tmp_path / "invalid")])
    cfg = campaign.read_json(args.config)
    cfg["train_budgets"] = budgets
    with pytest.raises(ValueError, match="train_budgets"):
        campaign.prepare(args, cfg)
    assert not args.output_dir.exists()


def test_single_and_sweep_cli_options_are_mutually_exclusive(campaign):
    with pytest.raises(SystemExit):
        campaign.parser().parse_args(["--train-per-class", "20", "--train-budgets", "20", "50", "100"])


def test_sft_shot_configuration_follows_each_sweep_budget(campaign):
    cfg = campaign.read_json(campaign.ROOT / "configs/hard_datasets_pilot.json")
    cfg["emissary"]["project_id"] = "project"
    for budget in (20, 50, 100):
        classifier = campaign.build_classifier("emissary-qwen", {**cfg, "train_per_class": budget},
                                               run_id="fixture", test_count=6)
        assert classifier.training.shots == budget
        assert classifier.training.shot_unit == "per_class"
