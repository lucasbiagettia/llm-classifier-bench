"""Live launcher must validate configuration before any training or paid requests."""
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("run_hard_datasets_experiment")
    monkeypatch.setattr(module, "load_dotenv", lambda *a, **kw: None)
    for name in (*module.REQUIRED_CREDENTIALS, "EMISSARY_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    monkeypatch.setattr(module.campaign, "main", lambda argv: calls.append(argv))
    return module, calls


def credentials(module, monkeypatch):
    for name in module.REQUIRED_CREDENTIALS:
        monkeypatch.setenv(name, "offline-fixture-key")


def test_live_launcher_prepares_then_executes_all_methods(launcher, monkeypatch, tmp_path):
    module, calls = launcher
    credentials(module, monkeypatch)
    module.main(["--emissary-project-id", "project", "--allow-unpriced", "--output-dir", str(tmp_path)])
    assert len(calls) == 2
    prepare, execute = calls
    assert prepare[prepare.index("--train-budgets") + 1:prepare.index("--test-per-class")] == ["20", "50", "100"]
    assert prepare[prepare.index("--test-per-class") + 1] == "40"
    assert prepare[prepare.index("--methods") + 1:prepare.index("--train-budgets")] == list(module.campaign.METHODS)
    assert prepare[1:3] == ["implicit_hate", "anli_r3"]
    assert execute[0] == "--execute"
    assert execute[execute.index("--plan") + 1] == str(tmp_path / "plan.json")
    assert execute[execute.index("--max-api-calls") + 1] == "4320"
    assert execute[execute.index("--max-training-jobs") + 1] == "12"


def test_live_launcher_missing_project_stops_before_work(launcher, monkeypatch):
    module, calls = launcher
    credentials(module, monkeypatch)
    with pytest.raises(SystemExit):
        module.main(["--allow-unpriced"])
    assert not calls


def test_live_launcher_missing_credentials_stops_before_work(launcher):
    module, calls = launcher
    with pytest.raises(SystemExit):
        module.main(["--emissary-project-id", "project", "--allow-unpriced"])
    assert not calls


def test_live_launcher_reads_existing_environment_project(launcher, monkeypatch):
    module, calls = launcher
    credentials(module, monkeypatch)
    monkeypatch.setenv("EMISSARY_PROJECT_ID", "project-env")
    module.main(["--allow-unpriced", "--test-per-class", "20"])
    prepare = calls[0]
    assert prepare[prepare.index("--test-per-class") + 1] == "20"
    assert prepare[prepare.index("--emissary-project-id") + 1] == "project-env"


def test_launcher_dry_run_never_executes_or_requires_credentials(launcher):
    module, calls = launcher
    module.main(["--dry-run", "--test-per-class", "20"])
    assert len(calls) == 1
    assert "--execute" not in calls[0]


@pytest.mark.parametrize("extra", [[], ["--allow-unpriced", "--max-api-calls", "4319"],
    ["--allow-unpriced", "--max-training-jobs", "11"], ["--allow-unpriced", "--test-per-class", "0"],
    ["--allow-unpriced", "--max-estimated-usd", "nan"]])
def test_live_launcher_rejects_missing_acknowledgement_or_insufficient_limits(launcher, monkeypatch, extra):
    module, calls = launcher
    credentials(module, monkeypatch)
    with pytest.raises(SystemExit):
        module.main(["--emissary-project-id", "project", *extra])
    assert not calls
