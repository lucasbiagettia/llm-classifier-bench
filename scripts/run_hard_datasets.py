"""Prepare all-label pilots without inference; execute only a frozen plan on request."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import subprocess

from llm_classifier_bench.classifiers import (
    BertClassifier, EmissaryClassifier, JevClassifier, OpenAIClassifier,
    OpenAIDecisionsClassifier, PerplexityDecisionsClassifier,
    SentenceTransformerLogisticClassifier, TfidfLogisticClassifier,
)
from llm_classifier_bench.config import (
    BertTrainingConfig, EmissaryTrainingConfig, SentenceTransformerTrainingConfig,
    TfidfTrainingConfig,
)
from llm_classifier_bench.core import ClassDefinition, LabeledExample
from llm_classifier_bench.datasets import DatasetBundle, ImplicitHateDataset, get_dataset
from llm_classifier_bench.datasets.hard_tasks import content_id, explanation_coverage
from llm_classifier_bench.datasets.selection import (
    LabeledSelectionConfig, select_labeled_examples, validate_partition_disjointness,
)
from llm_classifier_bench.measurement import MeasurementConfig
from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark
from run_banking77_scaling_benchmark_v2 import (
    Condition, StaticDataset, summary_row, utc_campaign_id, write_summary_csv,
)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("tfidf", "sentence-transformer", "bert", "openai", "openai-context",
           "jev", "openai-decisions", "perplexity-decisions", "emissary",
           "emissary-qwen", "emissary-llama")
LOCAL = {"tfidf", "sentence-transformer", "bert"}
SFT = {"emissary-qwen", "emissary-llama"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def sample(examples, names, count, seed, split):
    selected = []
    for name in names:
        group = sorted((e for e in examples if e.label == name), key=lambda e: e.sample_id)
        random.Random(f"{seed}:{split}:{name}").shuffle(group)
        if len(group) < count:
            raise ValueError(f"Insufficient {split} support for {name}: {len(group)} < {count}")
        selected.extend(group[:count])
    return tuple(selected)


def select_bundle(bundle, cfg):
    # Audit the entire source before sampling: small pilots must not hide leakage.
    validate_partition_disjointness(bundle.train, bundle.validation or (), bundle.test)
    return replace(bundle,
        train=sample(bundle.train, bundle.class_names, cfg["train_per_class"], cfg["seed"], "train"),
        validation=sample(bundle.validation or (), bundle.class_names,
                          cfg["validation_per_class"], cfg["seed"], "validation"),
        test=sample(bundle.test, bundle.class_names, cfg["test_per_class"], cfg["seed"], "test"))


def context_pool(bundle, cfg):
    return select_labeled_examples(bundle.train, bundle.classes,
        LabeledSelectionConfig(cfg["context_total"], "total", cfg["seed"]),
        validation_examples=bundle.validation or ())[0]


def fit_budget_bundle(bundle, per_class):
    """Take nested prefixes of the frozen, independently shuffled class pools."""
    train = []
    for name in bundle.class_names:
        group = [e for e in bundle.train if e.label == name]
        if len(group) < per_class:
            raise ValueError(f"Insufficient frozen fit support for {name}: {len(group)} < {per_class}")
        train.extend(group[:per_class])
    return replace(bundle, train=tuple(train))


def build_classifier(method, cfg, *, run_id, test_count):
    seed, models = cfg["seed"], cfg["models"]
    if method == "tfidf":
        settings = {**cfg["tfidf"], "seed": seed}
        settings["c_values"] = tuple(settings["c_values"])
        return TfidfLogisticClassifier(training=TfidfTrainingConfig(**settings))
    if method == "sentence-transformer":
        settings = {**cfg[method], "seed": seed}
        settings["c_values"] = tuple(settings["c_values"])
        return SentenceTransformerLogisticClassifier(model=models[method],
            training=SentenceTransformerTrainingConfig(**settings))
    if method == "bert":
        return BertClassifier(model=models[method], training=BertTrainingConfig(**cfg[method], seed=seed))
    if method in ("openai", "openai-context"):
        return OpenAIClassifier(model=models["openai"], **cfg["openai"],
                                in_context=method == "openai-context")
    if method == "jev":
        return JevClassifier(model=models[method], max_requests=test_count, max_retries=0)
    if method == "openai-decisions":
        return OpenAIDecisionsClassifier(model=models[method], base_url=cfg["endpoints"][method])
    if method == "perplexity-decisions":
        return PerplexityDecisionsClassifier(model=models[method], base_url=cfg["endpoints"][method])
    if method in ("emissary", *SFT):
        training = (EmissaryTrainingConfig(shots=cfg["train_per_class"], shot_unit="per_class",
            selection_seed=seed, mechanism="project_fine_tuning", base_model=models[method], **cfg["emissary"])
            if method in SFT else EmissaryTrainingConfig())
        return EmissaryClassifier(training=training, experiment_name=run_id, classifier_name=method)
    raise ValueError(f"Unknown method {method}")


def planned_cost(method, classifier, bundle, cfg, rate_card):
    if method in LOCAL:
        return {"api_calls": 0, "cost_kind": "unavailable", "estimated_usd": None,
                "reason": "Local compute time and matching hardware rate unknown before execution"}
    # These are conservative byte proxies, not measured tokenizer counts or invoices.
    definitions = json.dumps([asdict(c) for c in bundle.classes], ensure_ascii=False)
    demos = context_pool(bundle, cfg) if method == "openai-context" else ()
    demo_bytes = len(json.dumps([{"text": e.text, "label": e.label} for e in demos], ensure_ascii=False).encode())
    inputs = [len(e.text.encode()) + len(definitions.encode()) + demo_bytes + 2048 for e in bundle.test]
    output = cfg["openai"]["completion_reserve_tokens"] if method in ("openai", "openai-context") else 0
    provider = {"openai-context": "openai", "openai-decisions": "openai", "perplexity-decisions": "perplexity"}.get(method, method)
    model = getattr(classifier, "model", None)
    endpoint = getattr(classifier, "endpoint", "https://api.openai.com/v1/chat/completions" if provider == "openai" else None)
    rate = next((r for r in rate_card["api_rates"] if r["provider"] == provider
                 and model in r["models"] and (not r.get("endpoints") or endpoint in r["endpoints"])), None)
    estimate = ((sum(inputs) * float(rate["input_usd_per_million"])
                 + len(inputs) * output * float(rate["output_usd_per_million"])) / 1e6 if rate else None)
    return {"api_calls": len(bundle.test), "input_tokens_proxy_total": sum(inputs),
            "max_input_tokens_proxy": max(inputs), "output_tokens_reserved_total": len(inputs) * output,
            "estimate_policy": "UTF-8 bytes of text, definitions, demonstrations plus 2048 framing per request; no cache discounts",
            "cost_kind": "estimated" if rate else "unavailable", "estimated_usd": estimate,
            "rate": rate, "reason": None if rate else "No matching inference tariff; SFT preparation also unpriced",
            "training_jobs": int(method in SFT),
            "preparation_cost_usd": None if method.startswith("emissary") else 0}


def prepare(args, cfg):
    cfg = json.loads(json.dumps(cfg))
    for key in ("seed", "train_per_class", "validation_per_class", "test_per_class", "context_total"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
        minimum = 0 if key in ("seed", "validation_per_class") else 1
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    budgets = getattr(args, "train_budgets", None)
    if budgets is None:
        budgets = ([args.train_per_class] if args.train_per_class is not None
                   else cfg.get("train_budgets", [cfg["train_per_class"]]))
    if (not isinstance(budgets, list) or not budgets
            or any(type(b) is not int or b < 1 for b in budgets)):
        raise ValueError("train_budgets must be a nonempty list of positive integers")
    cfg["train_budgets"] = sorted(set(budgets))
    # One maximum-size snapshot; smaller conditions share per-class prefixes.
    cfg["train_per_class"] = max(budgets)
    if args.emissary_project_id:
        cfg["emissary"]["project_id"] = args.emissary_project_id
    names = args.datasets or cfg["datasets"]
    methods = args.methods or list(METHODS)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    card = {"schema_version": 1, "currency": "USD", "api_rates": [], "local_hardware": None}
    for path in cfg["pricing"]:
        card["api_rates"].extend(read_json(ROOT / path)["api_rates"])
    write_json(root / "pricing.json", card)
    plan = {"schema_version": 1, "status": "prepared_without_training_or_inference", "configuration": cfg,
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
            "implementation_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (
                Path(__file__).resolve(), ROOT / "src/llm_classifier_bench/datasets/hard_tasks.py",
                ROOT / "src/llm_classifier_bench/datasets/base.py", ROOT / "src/llm_classifier_bench/runner.py")},
            "files_sha256": {"pricing.json": digest(root / "pricing.json")}, "datasets": {}, "cells": [],
            "measurement": asdict(MeasurementConfig()), "automatic_retries": 0,
            "failure_policy": "stop on first failure or refusal; preserve artifacts; no automatic resume",
            "api_call_scope": "inference only; resource creation and bounded polling excluded"}
    for name in dict.fromkeys(names):
        dataset = (ImplicitHateDataset(annotations_path=args.hate_annotations)
                   if name == "implicit_hate" else get_dataset(name))
        source = dataset.load()
        # Generic AG News has no official dev: validation=0 is still usable.
        bundle = select_bundle(source, cfg)
        snapshot = {"name": bundle.name, "classes": [asdict(c) for c in bundle.classes],
                    "metadata": dict(bundle.metadata), **{s: [asdict(e) for e in getattr(bundle, s) or ()]
                    for s in ("train", "validation", "test")}}
        filename = f"{name}.json"
        write_json(root / filename, snapshot)
        plan["files_sha256"][filename] = digest(root / filename)
        plan["datasets"][name] = {"snapshot": filename, "labels": list(bundle.class_names),
            "source": dict(source.metadata),
            "source_fingerprint_sha256": sha256(json.dumps({s: [(e.sample_id, e.label, content_id(e.text)) for e in getattr(source, s) or ()]
                                                       for s in ("train", "validation", "test")}, sort_keys=True).encode()).hexdigest(),
            "source_counts": {s: dict(Counter(e.label for e in getattr(source, s) or ())) for s in ("train", "validation", "test")},
            "selected_counts": {s: dict(Counter(e.label for e in getattr(bundle, s) or ())) for s in ("train", "validation", "test")},
            "source_explanation_coverage": {s: explanation_coverage(getattr(source, s) or ()) for s in ("train", "validation", "test")},
            "selected_explanation_coverage": {s: explanation_coverage(getattr(bundle, s) or ()) for s in ("train", "validation", "test")},
            "test_sample_ids": [e.sample_id for e in bundle.test],
            "fit_budgets": {str(b): {
                "fit_per_class": b,
                "fit_sample_ids": [e.sample_id for e in fit_budget_bundle(bundle, b).train],
                "counts": dict(Counter(e.label for e in fit_budget_bundle(bundle, b).train)),
            } for b in cfg["train_budgets"]},
            "input_characters": {"min": min(len(e.text) for e in bundle.test), "max": max(len(e.text) for e in bundle.test)}}
        for method in dict.fromkeys(methods):
            for budget in cfg["train_budgets"] if method in LOCAL | SFT else [0]:
                # Context mode stays fixed, drawn from the smallest shared pool.
                cell_bundle = fit_budget_bundle(bundle, budget or min(cfg["train_budgets"]))
                cell_cfg = {**cfg, "train_per_class": budget or min(cfg["train_budgets"])}
                missing_project = method in SFT and not cfg["emissary"].get("project_id")
                classifier = (None if missing_project else build_classifier(
                    method, cell_cfg, run_id=f"{name}-{method}-fit{budget}-plan", test_count=len(bundle.test)))
                preflight = getattr(classifier, "preflight", None)
                support = preflight(bundle.classes, bundle.inputs()) if preflight else {"supported": True}
                if method in ("openai", "openai-context"):
                    support = classifier.plan_context(bundle.classes,
                        context_pool(cell_bundle, cell_cfg) if method == "openai-context" else (), bundle.inputs())
                blocker = None if support["supported"] else support.get("reason", "context_limit_exceeded")
                if missing_project:
                    blocker = "Configure emissary.project_id or --emissary-project-id before preparation"
                plan["cells"].append({"dataset": name, "method": method, "blocker": blocker,
                    "regime": "supervised" if method in LOCAL | SFT else "in_context" if method == "openai-context" else "zero_shot",
                    "fit_per_class": budget,
                    "fit_examples": len(cell_bundle.train) if method in LOCAL | SFT else 0,
                    "validation_examples": len(bundle.validation) if method in LOCAL else 0,
                    "validation_examples_supplied": len(bundle.validation) if method in LOCAL | SFT else 0,
                    "context_examples": len(context_pool(cell_bundle, cell_cfg)) if method == "openai-context" else 0,
                    "test_examples": len(bundle.test), "preflight": support,
                    "cost": planned_cost(method, classifier, cell_bundle, cell_cfg, card)})
    plan["estimated_priced_subtotal_usd"] = sum(c["cost"]["estimated_usd"] or 0 for c in plan["cells"])
    plan["total_cost_usd"] = None
    plan["total_cost_reason"] = "Unpriced hosted preparation/inference and unmeasured local compute; subtotal is an estimate, not results"
    plan["planned_api_calls"] = sum(c["cost"]["api_calls"] for c in plan["cells"])
    plan["planned_training_jobs"] = sum(c["method"] in SFT for c in plan["cells"])
    write_json(root / "plan.json", plan)
    (root / "plan.sha256").write_text(digest(root / "plan.json") + "\n", encoding="utf-8")
    return plan


def load_snapshot(path):
    raw = read_json(path)
    return DatasetBundle(name=raw["name"], classes=tuple(ClassDefinition(**c) for c in raw["classes"]),
        train=tuple(LabeledExample(**e) for e in raw["train"]),
        test=tuple(LabeledExample(**e) for e in raw["test"]),
        validation=tuple(LabeledExample(**e) for e in raw["validation"]), metadata=raw["metadata"])


def execute(args):
    path = args.plan.resolve()
    plan, root = read_json(path), path.parent
    if digest(path) != (root / "plan.sha256").read_text(encoding="utf-8").strip():
        raise ValueError("Prepared plan changed; prepare a new plan to modify settings")
    for name, expected in plan["implementation_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Prepared implementation changed: {name}; prepare a new plan")
    for name, expected in plan["files_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Prepared file changed: {name}")
    blockers = [f"{c['dataset']}/{c['method']}: {c['blocker']}" for c in plan["cells"] if c["blocker"]]
    if blockers:
        raise ValueError("Blocked cells: " + "; ".join(blockers))
    if plan["planned_api_calls"] > args.max_api_calls:
        raise ValueError("Planned inference exceeds --max-api-calls")
    if plan["planned_training_jobs"] > args.max_training_jobs:
        raise ValueError("Planned SFT exceeds --max-training-jobs")
    if any(c["method"] not in LOCAL and (c["cost"]["estimated_usd"] is None or c["method"].startswith("emissary")) for c in plan["cells"]) and not args.allow_unpriced:
        raise ValueError("Hosted costs are incomplete; execution requires --allow-unpriced")
    if plan["estimated_priced_subtotal_usd"] > args.max_estimated_usd:
        raise ValueError("Priced subtotal exceeds --max-estimated-usd (estimate guard, not an invoice cap)")
    cfg = plan["configuration"]
    bundles = {name: load_snapshot(root / item["snapshot"]) for name, item in plan["datasets"].items()}
    for bundle in bundles.values():
        validate_partition_disjointness(bundle.train, bundle.validation, bundle.test)
    campaign_id = utc_campaign_id()
    output = root / "executions" / campaign_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "campaign.json", {**plan, "plan_sha256": digest(path)})
    rows = []
    for cell in plan["cells"]:
        name, method = cell["dataset"], cell["method"]
        budget = cell["fit_per_class"]
        bundle = fit_budget_bundle(bundles[name], budget or min(cfg["train_budgets"]))
        cell_cfg = {**cfg, "train_per_class": budget or min(cfg["train_budgets"])}
        # Zero-shot methods receive no labels. Demonstrations are a separate budget.
        if method not in LOCAL | SFT:
            bundle = replace(bundle, train=context_pool(bundle, cell_cfg) if method == "openai-context" else (), validation=())
        run_id = f"{name}__{method}"
        if budget and len(cfg["train_budgets"]) > 1:
            run_id += f"__fit{budget}"
        classifier = build_classifier(method, cell_cfg, run_id=f"{campaign_id}-{run_id}", test_count=len(bundle.test))
        result, error = None, None
        try:
            result = run_benchmark(StaticDataset(bundle), classifier, BenchmarkRunConfig(
                output_root=output / "runs", run_id=run_id, validation_fraction=0,
                split_seed=cfg["seed"], pricing_path=root / "pricing.json",
                measurement=MeasurementConfig(**plan["measurement"]),
                metadata={"prepared_plan_sha256": digest(path), "dataset": name, "method": method,
                          "fit_per_class": budget}))
        except BaseException as exc:
            error = exc
            raise
        finally:
            condition = Condition(cfg["seed"], len(bundle.classes), bundle.class_names, path,
                                  bundle, len(bundle.train), len(bundle.validation))
            row = summary_row(campaign_id=campaign_id, condition=condition, classifier_key=method,
                              classifier=classifier, result=result, error=error)
            row.update(dataset=name, fit_per_class=budget,
                       context_examples=cell["context_examples"], run_dir=str(output / "runs" / run_id))
            rows.append(row)
            write_json(output / "summary.json", rows)
            write_summary_csv(output / "summary.csv", rows)
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/hard_datasets_pilot.json")
    p.add_argument("--datasets", nargs="+", choices=["implicit_hate", "anli_r3", "ag_news"])
    p.add_argument("--methods", nargs="+", choices=METHODS)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/hard_datasets") / utc_campaign_id())
    p.add_argument("--dry-run", action="store_true", help="Default behavior; load/sample/plan only")
    p.add_argument("--execute", action="store_true", help="Execute an already prepared --plan")
    p.add_argument("--plan", type=Path)
    fit = p.add_mutually_exclusive_group()
    fit.add_argument("--train-per-class", type=int, help="Single fit budget per class (default 20)")
    fit.add_argument("--train-budgets", nargs="+", type=int,
                     help="Nested fit budgets per class, e.g. 20 50 100; zero-shot/context run once")
    for key in ("seed", "validation-per-class", "test-per-class", "context-total"):
        p.add_argument(f"--{key}", type=int)
    p.add_argument("--hate-annotations", type=Path, help="Optional official stage-3 TSV with exact post/target/implied_statement")
    p.add_argument("--emissary-project-id")
    p.add_argument("--max-api-calls", type=int, default=3000, help="Campaign inference attempt cap; no retries/warmup")
    p.add_argument("--max-training-jobs", type=int, default=0)
    p.add_argument("--max-estimated-usd", type=float, default=10, help="Priced estimate guard only; excludes unknown costs")
    p.add_argument("--allow-unpriced", action="store_true", help="Acknowledge unknown hosted charges before execution")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.dry_run and args.execute:
        p.error("--dry-run and --execute are mutually exclusive")
    for key in ("max_api_calls", "max_training_jobs", "max_estimated_usd"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            p.error(f"{key} must be finite and nonnegative")
    if args.execute:
        if args.plan is None:
            p.error("--execute requires a previously prepared --plan")
        if args.datasets or args.methods or any(getattr(args, k) is not None for k in
            ("seed", "train_per_class", "train_budgets", "validation_per_class", "test_per_class", "context_total", "emissary_project_id", "hate_annotations")):
            p.error("Execution uses frozen settings; prepare a new plan to change datasets, methods or budgets")
        print(execute(args))
    else:
        if args.plan:
            p.error("--plan is only used with --execute")
        plan = prepare(args, read_json(args.config))
        print(f"Prepared {len(plan['cells'])} cells; {plan['planned_api_calls']} inference calls; "
              f"priced estimate ${plan['estimated_priced_subtotal_usd']:.6f}; total cost unavailable.")
        print(args.output_dir / "plan.json")


if __name__ == "__main__":
    main()
