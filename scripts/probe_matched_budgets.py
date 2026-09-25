"""Offline evidence for shared pools, real TF-IDF and recorded mock ICL requests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from probe_tfidf_classifier import fixture_bundle, StaticDataset
from llm_classifier_bench.budgets import MatchedBudgetConfig
from llm_classifier_bench.classifiers import OpenAIClassifier, TfidfLogisticClassifier
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark


class RecordingClient:
    """No transport: fixed label/usage are fixture values, not provider measurements."""
    max_retries = 0
    timeout = 120

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = {"id": f"fixture-{len(self.calls)}", "model": "fixture-openai",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 4}}
        return SimpleNamespace(**response,
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"label":"Sports"}'))],
            model_dump=lambda: response)


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/matched_budget_checks"))
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    bundle = fixture_bundle()
    rows = []
    previous_ids = []
    for budget in (0, 1, 2, 5):
        client = RecordingClient()
        classifiers = (TfidfLogisticClassifier(), OpenAIClassifier(
            model="fixture-openai", client=client, in_context=True, context_window_tokens=32000))
        paired = []
        for classifier in classifiers:
            result = run_benchmark(StaticDataset(bundle), classifier, BenchmarkRunConfig(
                output_root=args.output_root, run_id=f"{classifier.name}__budget{budget}",
                validation_fraction=.25, split_seed=17,
                matched_budget=MatchedBudgetConfig(budget, "total", 17, 2),
                metadata={"code_revision": revision, "working_tree_dirty": dirty,
                          "offline_fixture": True, "paid_requests": 0,
                          "limitation": "OpenAI label, usage and timings are mock values; TF-IDF is real local sklearn"}))
            status = load(result.status_path)
            pool = load(result.run_dir / "labeled_budget.json")
            paired.append(pool)
            expected = "unsupported" if classifier.name == "tfidf-logreg" and budget < 2 else "completed"
            assert status["status"] == expected, (classifier.name, status)
            rows.append({"classifier": classifier.name, "budget_total": budget,
                         "run_dir": str(result.run_dir), "status": status["status"],
                         "class_coverage": pool["selection"]["class_coverage"],
                         "pool_sha256": pool["pool_sha256"], "consumption": pool.get("consumption")})
        assert paired[0]["pool_sha256"] == paired[1]["pool_sha256"]
        ids = [e["sample_id"] for e in paired[0]["selection"]["selected_examples"]]
        assert ids[:len(previous_ids)] == previous_ids
        previous_ids = ids
        assert len(client.calls) == len(bundle.test)
        assert all(len(call["messages"]) == 2 + 2*budget for call in client.calls)
        consumption = paired[1]["consumption"]
        assert consumption["training_examples_used"] == consumption["validation_examples_used"] == 0
        assert consumption["context_examples_used"] == consumption["total_labeled_examples_used"] == budget
        (result.run_dir / "mock_requests.json").write_text(json.dumps(client.calls, indent=2) + "\n")
    report = {"code_revision": revision, "working_tree_dirty": dirty, "checks_passed": True,
              "offline_only": True, "paid_requests": 0, "model_downloads": 0,
              "checks": ["same fit/context and validation IDs across methods", "nested budget prefixes",
                         "class coverage and unsupported status", "prompt size and distinct-label accounting"],
              "limitation": "Synthetic fixture; real TF-IDF, mock OpenAI. No quality/provider-performance conclusion.",
              "rows": rows}
    (args.output_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output_root / "validation.json")


if __name__ == "__main__":
    main()
