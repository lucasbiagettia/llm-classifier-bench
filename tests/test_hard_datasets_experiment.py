"""Reuse real plan structure and provision Emissary automatically, all offline."""
import importlib
from pathlib import Path

import pytest

from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.datasets import DatasetBundle


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("run_hard_datasets_experiment")
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **kw: None)
    for name in (*module.REQUIRED_CREDENTIALS, "EMISSARY_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)
    def forbidden(*a, **kw):
        pytest.fail("Offline launcher tests must not access the network")
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    def bundle(name, count):
        classes = tuple(ClassDefinition(f"label-{i}", f"Description {i}") for i in range(count))
        def examples(split, n):
            return tuple(LabeledExample(f"{name}:{split}:{c.name}:{i}", f"{name} {split} {c.name} text {i}", c.name,
                         {"annotations": [{"reason": "SECRET"}]}) for c in classes for i in range(n))
        return DatasetBundle(name, classes, examples("train", 110), examples("test", 45), validation=())
    campaign = module.campaign
    monkeypatch.setattr(campaign, "ImplicitHateDataset", lambda **kw: campaign.StaticDataset(bundle("implicit_hate", 6)))
    monkeypatch.setattr(campaign, "get_dataset", lambda name: campaign.StaticDataset(bundle(name, 3)))
    source = tmp_path / "source"
    a = campaign.parser().parse_args(["--train-budgets", "20", "50", "100", "--output-dir", str(source)])
    campaign.prepare(a, campaign.read_json(a.config))
    # Lazy provider constructors used for planning load .env; clear these again
    # so credential tests never depend on the developer's local configuration.
    for name in (*module.REQUIRED_CREDENTIALS, "EMISSARY_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(module, "DEFAULT_PREPARED_PLAN", source / "plan.json")
    # The launcher must use the prepared snapshots, not download/resample again.
    monkeypatch.setattr(campaign, "get_dataset", forbidden)
    monkeypatch.setattr(campaign, "ImplicitHateDataset", forbidden)
    calls, projects = [], []
    monkeypatch.setattr(campaign, "main", lambda argv: calls.append(argv))
    class Client:
        def create_project(self, name):
            projects.append(name)
            return {"id": "ms-created-fixture", "name": name}
    monkeypatch.setattr(module, "EmissaryClient", Client)
    return module, source, tmp_path / "run", calls, projects


def credentials(module, monkeypatch):
    for name in module.REQUIRED_CREDENTIALS:
        monkeypatch.setenv(name, "offline-fixture-key")


def test_auto_project_reuses_samples_and_executes_without_manual_id(launcher, monkeypatch):
    module, source, root, calls, projects = launcher
    credentials(module, monkeypatch)
    before = (source / "plan.json").read_bytes()
    module.main(["--allow-unpriced", "--output-dir", str(root)])
    assert len(projects) == 1
    assert len(calls) == 1 and calls[0][0] == "--execute"
    assert calls[0][calls[0].index("--max-api-calls") + 1] == "4320"
    assert calls[0][calls[0].index("--max-training-jobs") + 1] == "12"
    assert (source / "plan.json").read_bytes() == before
    plan = module.campaign.read_json(root / "plan.json")
    original = module.campaign.read_json(source / "plan.json")
    assert len(plan["cells"]) == 42 and plan["planned_new_projects"] == 1
    assert plan["configuration"]["emissary"]["project_id"] == "ms-created-fixture"
    assert not any(c["blocker"] for c in plan["cells"])
    assert plan["source_implementation_sha256"] == original["implementation_sha256"]
    for name in plan["datasets"]:
        assert plan["datasets"][name]["test_sample_ids"] == original["datasets"][name]["test_sample_ids"]
        assert plan["datasets"][name]["fit_budgets"] == original["datasets"][name]["fit_budgets"]
    for name, digest in plan["files_sha256"].items():
        assert module.campaign.digest(root / name) == digest
    assert module.campaign.digest(root / "plan.json") == (root / "plan.sha256").read_text().strip()
    assert module.campaign.read_json(root / "project_creation.json")["status"] == "created"


def test_existing_environment_project_does_not_create_a_new_one(launcher, monkeypatch):
    module, _, root, calls, projects = launcher
    credentials(module, monkeypatch)
    monkeypatch.setenv("EMISSARY_PROJECT_ID", "ms-existing")
    module.main(["--allow-unpriced", "--output-dir", str(root)])
    assert not projects and len(calls) == 1
    assert module.campaign.read_json(root / "plan.json")["configuration"]["emissary"]["project_id"] == "ms-existing"


def test_dry_run_no_credentials_no_remote_creation_no_execution(launcher):
    module, _, root, calls, projects = launcher
    module.main(["--dry-run", "--output-dir", str(root)])
    assert not projects and not calls
    plan = module.campaign.read_json(root / "plan.json")
    assert plan["planned_new_projects"] == 1
    assert sum(bool(c["blocker"]) for c in plan["cells"]) == 12


def test_reduced_test_preserves_nested_prefix_and_all_training_pools(launcher):
    module, source, root, calls, projects = launcher
    module.main(["--dry-run", "--test-per-class", "20", "--output-dir", str(root)])
    plan = module.campaign.read_json(root / "plan.json")
    original = module.campaign.read_json(source / "plan.json")
    assert plan["planned_api_calls"] == 2160
    assert not projects and not calls
    for name, k in (("implicit_hate", 6), ("anli_r3", 3)):
        info = plan["datasets"][name]
        assert len(info["test_sample_ids"]) == 20*k
        assert set(info["test_sample_ids"]) < set(original["datasets"][name]["test_sample_ids"])
        assert info["fit_budgets"] == original["datasets"][name]["fit_budgets"]
        assert set(info["selected_counts"]["test"].values()) == {20}


@pytest.mark.parametrize("extra", [[], ["--allow-unpriced", "--max-api-calls", "4319"],
    ["--allow-unpriced", "--max-training-jobs", "11"], ["--allow-unpriced", "--test-per-class", "0"],
    ["--allow-unpriced", "--max-estimated-usd", "nan"]])
def test_bad_caps_or_missing_acknowledgement_fail_before_work(launcher, monkeypatch, extra):
    module, _, root, calls, projects = launcher
    credentials(module, monkeypatch)
    with pytest.raises(SystemExit):
        module.main(["--output-dir", str(root), *extra])
    assert not calls and not projects and not root.exists()


def test_low_priced_estimate_guard_prevents_project_creation(launcher, monkeypatch):
    module, _, root, calls, projects = launcher
    credentials(module, monkeypatch)
    with pytest.raises(ValueError, match="Priced estimate"):
        module.main(["--allow-unpriced", "--output-dir", str(root), "--max-estimated-usd", "0"])
    assert not calls and not projects


def test_corrupt_source_rejected_before_any_remote_operation(launcher, monkeypatch):
    module, source, root, calls, projects = launcher
    credentials(module, monkeypatch)
    (source / "anli_r3.json").write_text("corrupt")
    with pytest.raises(ValueError, match="Prepared source file changed"):
        module.main(["--allow-unpriced", "--output-dir", str(root)])
    assert not calls and not projects and not root.exists()


def test_ambiguous_project_creation_is_recorded_and_never_retried(launcher, monkeypatch):
    module, _, root, calls, projects = launcher
    credentials(module, monkeypatch)
    class Client:
        def create_project(self, name):
            projects.append(name)
            raise TimeoutError("ambiguous")
    monkeypatch.setattr(module, "EmissaryClient", Client)
    with pytest.raises(TimeoutError):
        module.main(["--allow-unpriced", "--output-dir", str(root)])
    assert len(projects) == 1 and not calls
    assert module.campaign.read_json(root / "project_creation.json")["status"] == "failed_or_ambiguous"
    with pytest.raises(FileExistsError):
        module.provision_project(root, module.campaign.read_json(root / "plan.json"))
    assert len(projects) == 1


def test_insufficient_frozen_test_fails_without_resampling(launcher):
    module, _, root, calls, projects = launcher
    with pytest.raises(ValueError, match="test pool is too small"):
        module.main(["--dry-run", "--test-per-class", "41", "--output-dir", str(root)])
    assert not root.exists() and not calls and not projects


def test_missing_credentials_fail_before_work(launcher):
    module, _, root, calls, projects = launcher
    with pytest.raises(SystemExit):
        module.main(["--allow-unpriced", "--output-dir", str(root)])
    assert not root.exists() and not calls and not projects


def test_automatic_project_and_real_runner_complete_all_conditions_offline(launcher, monkeypatch):
    from llm_classifier_bench.classifiers.base import Prediction
    module, _, root, _, projects = launcher
    campaign = module.campaign
    credentials(module, monkeypatch)
    built_projects, outputs = [], []
    class Spy:
        name = "offline-spy"
        def plan_context(self, *args): return {"supported": True}
        def prepare(self, classes): self.classes = classes
        def fit(self, examples, *, validation_examples=()): pass
        def predict(self, examples):
            assert all("annotations" not in e.metadata and "SECRET" not in e.text for e in examples)
            return [Prediction(e.sample_id, self.classes[0].name, None, None, 0.) for e in examples]
    def build(method, cfg, **kwargs):
        if method in campaign.SFT:
            built_projects.append(cfg["emissary"]["project_id"])
        return Spy()
    monkeypatch.setattr(campaign, "build_classifier", build)
    def execute(argv):
        outputs.append(campaign.execute(campaign.parser().parse_args(argv)))
    monkeypatch.setattr(campaign, "main", execute)
    module.main(["--allow-unpriced", "--test-per-class", "1", "--output-dir", str(root)])
    assert len(projects) == 1 and built_projects == ["ms-created-fixture"] * 12
    rows = campaign.read_json(outputs[0] / "summary.json")
    assert len(rows) == 42 and all(r["status"] == "completed" for r in rows)
    assert {r["fit_per_class"] for r in rows} == {0, 20, 50, 100}
    assert {r["dataset"] for r in rows} == {"implicit_hate", "anli_r3"}


def test_resume_preserves_completed_tfidf_project_and_runs_only_pending(launcher, monkeypatch):
    module, _, original, _, projects = launcher
    campaign = module.campaign
    credentials(module, monkeypatch)
    real_build = campaign.build_classifier
    outputs = []
    class EncoderFailure:
        name = "sentence-transformer-logreg"
        def plan_context(self, *a): return {"supported": True}
        def prepare(self, classes): pass
        def fit(self, *a, **kw): raise RuntimeError("simulated incompatible CUDA")
    def first_build(method, cfg, **kwargs):
        return real_build(method, cfg, **kwargs) if method == "tfidf" else EncoderFailure()
    monkeypatch.setattr(campaign, "build_classifier", first_build)
    def execute(argv):
        outputs.append(campaign.execute(campaign.parser().parse_args(argv)))
    monkeypatch.setattr(campaign, "main", execute)
    with pytest.raises(RuntimeError, match="incompatible CUDA"):
        module.main(["--allow-unpriced", "--test-per-class", "1", "--output-dir", str(original)])
    before = next(original.glob("executions/*/summary.json")).read_bytes()
    assert len(projects) == 1
    from llm_classifier_bench.classifiers.base import Prediction
    class Spy:
        name = "offline-spy"
        def plan_context(self, *a): return {"supported": True}
        def prepare(self, classes): self.classes = classes
        def fit(self, *a, **kw): pass
        def predict(self, examples):
            return [Prediction(e.sample_id, self.classes[0].name, None, None, 0.) for e in examples]
    def pending_build(method, cfg, **kwargs):
        assert cfg["local_device"] == "cpu"
        return real_build(method, cfg, **kwargs) if method == "tfidf" else Spy()
    monkeypatch.setattr(campaign, "build_classifier", pending_build)
    continuation = original.parent / "continued"
    module.main(["--allow-unpriced", "--resume-from", str(original), "--test-per-class", "1",
                 "--output-dir", str(continuation)])
    assert len(projects) == 1, "Resume must not create another remote project"
    assert next(original.glob("executions/*/summary.json")).read_bytes() == before
    plan = campaign.read_json(continuation / "plan.json")
    assert len(plan["cells"]) == 39 and len(plan["all_cells"]) == 42
    assert len(plan["reused_results"]) == 3
    assert plan["configuration"]["emissary"]["project_id"] == "ms-created-fixture"
    assert plan["planned_new_projects"] == 0
    rows = campaign.read_json(outputs[-1] / "summary.json")
    assert len(rows) == 42 and sum(bool(r.get("reused")) for r in rows) == 3
    assert all(r["status"] == "completed" for r in rows)
    reused = plan["reused_results"][0]
    with (Path(reused["run_dir"]) / "predictions.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Completed evidence changed"):
        campaign.execute(campaign.parser().parse_args(["--plan", str(continuation / "plan.json")]))


def test_plain_replanning_drops_old_completion_bookkeeping(launcher):
    module, source, root, calls, projects = launcher
    plan = module.campaign.read_json(source / "plan.json")
    plan.update(reused_results=[{"stale": True}], reuse_files_sha256={"stale": "hash"}, resume_from="old")
    module.save_plan(source, plan)
    module.main(["--dry-run", "--output-dir", str(root)])
    result = module.campaign.read_json(root / "plan.json")
    assert len(result["cells"]) == 42
    assert not any(k in result for k in ("reused_results", "reuse_files_sha256", "resume_from"))
    assert not calls and not projects


def test_skip_perplexity_needs_no_key_and_persists_across_failed_resume(launcher, monkeypatch):
    module, _, root, calls, projects = launcher
    credentials(module, monkeypatch)
    monkeypatch.delenv("PERPLEXITY_API_KEY")
    original_build = module.campaign.build_classifier
    def build(method, *args, **kwargs):
        assert method != "perplexity-decisions", "Excluded provider must not even be planned"
        return original_build(method, *args, **kwargs)
    monkeypatch.setattr(module.campaign, "build_classifier", build)
    module.main(["--skip-perplexity", "--allow-unpriced", "--max-api-calls", "3960", "--output-dir", str(root)])
    plan = module.campaign.read_json(root / "plan.json")
    assert len(plan["all_cells"]) == 42 and len(plan["cells"]) == 40
    assert plan["excluded_methods"] == ["perplexity-decisions"]
    assert plan["planned_api_calls"] == 3960
    summary = root / "executions/fixture/summary.json"
    summary.parent.mkdir(parents=True)
    module.campaign.write_json(summary, [{"dataset": "implicit_hate", "classifier_key": "perplexity-decisions",
        "fit_per_class": 0, "status": "failed", "run_dir": str(root / "not-needed")}])
    continued = root.parent / "excluded_continued"
    module.main(["--resume-from", str(root), "--allow-unpriced", "--output-dir", str(continued)])
    resumed = module.campaign.read_json(continued / "plan.json")
    assert resumed["excluded_methods"] == ["perplexity-decisions"]
    assert len(resumed["cells"]) == 40 and len(resumed["all_cells"]) == 42
    assert len(projects) == 1 and len(calls) == 2


def test_resume_partial_jev_preserves_predictions_and_only_calls_missing_inputs(launcher, monkeypatch):
    import hashlib
    from llm_classifier_bench.classifiers.base import Prediction
    from llm_classifier_bench.classifiers.jev import JevClassifier
    module, _, original, _, projects = launcher
    campaign = module.campaign
    credentials(module, monkeypatch)
    real_build = campaign.build_classifier
    class Spy:
        name = "offline-spy"
        def plan_context(self, *a): return {"supported": True}
        def prepare(self, classes): self.classes = classes
        def fit(self, *a, **kw): pass
        def predict(self, examples):
            return [Prediction(e.sample_id, self.classes[0].name, None, None, 0.) for e in examples]
    monkeypatch.setattr(campaign, "build_classifier", lambda method, cfg, **kw:
                        real_build(method, cfg, **kw) if method == "jev" else Spy())
    calls = []
    def predict_one(self, example):
        calls.append(example.sample_id)
        if len(calls) == 2:
            raise ValueError("Jev probabilities do not sum to one")
        names = [c.name for c in self._classes]
        digest = hashlib.sha256(self._encoded(self._payload(self._classes, example.text))).hexdigest()
        return self._normalize({"model": self.model, "answers": {"classification": {
            "type": "choice", "choice": names[0],
            "probabilities": {n: .99 if i == 0 else 0 for i, n in enumerate(names)}}}},
            example.sample_id, 1., None, digest)
    monkeypatch.setattr(JevClassifier, "_predict_one", predict_one)
    outputs = []
    def execute(argv):
        outputs.append(campaign.execute(campaign.parser().parse_args(argv)))
    monkeypatch.setattr(campaign, "main", execute)
    with pytest.raises(ValueError, match="sum to one"):
        module.main(["--allow-unpriced", "--test-per-class", "1", "--output-dir", str(original)])
    source = next(original.glob("executions/*/runs/implicit_hate__jev"))
    old_predictions = (source / "predictions.jsonl").read_bytes()
    continued = original.parent / "continued_jev"
    module.main(["--allow-unpriced", "--resume-from", str(original), "--test-per-class", "1",
                 "--output-dir", str(continued)])
    assert len(projects) == 1
    assert (source / "predictions.jsonl").read_bytes() == old_predictions
    plan = campaign.read_json(continued / "plan.json")
    assert len(plan["reused_results"]) == 11 and len(plan["cells"]) == 31
    assert plan["cells"][0]["reused_predictions"] == 1
    assert plan["cells"][0]["cost"]["api_calls"] == 5
    assert len(calls) == 2 + 5 + 3  # Two original attempts, five remaining, three ANLI.
    assert calls.count(calls[0]) == 1
    assert calls.count(calls[1]) == 2  # Failed response was not persisted; request is repeated.
    rows = campaign.read_json(outputs[-1] / "summary.json")
    assert len(rows) == 42 and all(r["status"] == "completed" for r in rows)
    merged = outputs[-1] / "runs/implicit_hate__jev"
    assert len((merged / "predictions.jsonl").read_text().splitlines()) == 6
    assert len((merged / "continuations/pending/predictions.jsonl").read_text().splitlines()) == 5
    assert campaign.read_json(merged / "metrics.json")["accuracy"]["value"] is not None
    assert not (merged / "operational_report.json").exists()  # Independent measurement segments.
    # A second continuation accepts the consolidated full cohort and makes no calls.
    final = original.parent / "already_done"
    module.main(["--dry-run", "--resume-from", str(continued), "--test-per-class", "1",
                 "--output-dir", str(final)])
    assert not campaign.read_json(final / "plan.json")["cells"]


def test_exclude_failed_llama_and_preserve_qwen_across_resumes(launcher, monkeypatch):
    module, _, root, calls, projects = launcher
    credentials(module, monkeypatch)
    monkeypatch.delenv("PERPLEXITY_API_KEY")
    original_build = module.campaign.build_classifier
    def build(method, *args, **kwargs):
        assert method not in ("emissary-llama", "perplexity-decisions")
        return original_build(method, *args, **kwargs)
    monkeypatch.setattr(module.campaign, "build_classifier", build)
    module.main(["--skip-perplexity", "--skip-emissary-llama", "--allow-unpriced",
                 "--max-api-calls", "2880", "--max-training-jobs", "6", "--output-dir", str(root)])
    plan = module.campaign.read_json(root / "plan.json")
    assert len(plan["cells"]) == 34 and len(plan["all_cells"]) == 42
    assert plan["planned_training_jobs"] == 6 and plan["planned_api_calls"] == 2880
    assert sum(c["method"] == "emissary-qwen" for c in plan["cells"]) == 6
    assert sum(c["method"] == "emissary" for c in plan["cells"]) == 2
    summary = root / "executions/fixture/summary.json"
    summary.parent.mkdir(parents=True)
    module.campaign.write_json(summary, [{"dataset": "implicit_hate", "classifier_key": "emissary-llama",
        "fit_per_class": 20, "status": "failed", "run_dir": str(root / "failed-llama")}])
    continued = root.parent / "no_llama_continued"
    module.main(["--resume-from", str(root), "--allow-unpriced", "--output-dir", str(continued)])
    resumed = module.campaign.read_json(continued / "plan.json")
    assert set(resumed["excluded_methods"]) == {"perplexity-decisions", "emissary-llama"}
    assert len(resumed["cells"]) == 34 and len(resumed["all_cells"]) == 42
    assert len(projects) == 1 and len(calls) == 2
