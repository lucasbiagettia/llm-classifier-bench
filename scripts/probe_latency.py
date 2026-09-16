"""Small real TF-IDF timing run on a repeated local fixture; no network or API costs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import subprocess

from probe_tfidf_classifier import fixture_bundle, StaticDataset
from report_operational import render_report
from llm_classifier_bench.classifiers import TfidfLogisticClassifier
from llm_classifier_bench.measurement import (
    add_measurement_arguments, measurement_from_args, regenerate_report,
)
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/latency_smoke"))
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--pricing", type=Path, default=None, help="Versioned USD rate card; omitted prices stay unavailable")
    add_measurement_arguments(parser)
    parser.set_defaults(warmup_examples=5, client_location="local machine; no network",
                        cache_condition="model/vectorizer resident after fit and warmup; repeated fixture text; no response cache")
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("--examples must be >= 1")
    bundle = fixture_bundle()
    bundle = replace(bundle, test=tuple(replace(bundle.test[i % len(bundle.test)], sample_id=f"timing-{i}")
                                       for i in range(args.examples)))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout)
    result = run_benchmark(StaticDataset(bundle), TfidfLogisticClassifier(), BenchmarkRunConfig(
        output_root=args.output_root, run_id=args.run_id, validation_fraction=.25,
        measurement=measurement_from_args(args), pricing_path=args.pricing,
        metadata={"smoke": True, "repeated_fixture": True, "code_revision": revision,
                  "working_tree_dirty": dirty,
                  "limitation": "Repeated short fixture inputs test measurement plumbing, not representative latency/accuracy"},
    ))
    report = regenerate_report(result.run_dir)
    (result.run_dir / "operational_report.md").write_text(render_report(report, result.run_id))
    print(result.run_dir / "operational_report.md")


if __name__ == "__main__":
    main()
