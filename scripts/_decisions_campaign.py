"""Decisions options shared by campaign entry points."""
from llm_classifier_bench.classifiers.openai_decisions import OpenAIDecisionsClassifier


def add_decisions_arguments(parser):
    parser.add_argument("--decisions-model", default=None, help="Overrides OPENAI_DECISIONS_MODEL (default gpt-6-luna)")
    parser.add_argument("--decisions-timeout-s", type=float, default=None,
                        help="Overrides OPENAI_DECISIONS_TIMEOUT_S (default 120)")


def decisions_options(args):
    return {"model": args.decisions_model, "timeout_s": args.decisions_timeout_s}


def decisions_manifest(args):
    return OpenAIDecisionsClassifier(**decisions_options(args)).inference_metadata()
