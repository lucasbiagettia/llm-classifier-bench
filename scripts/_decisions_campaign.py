"""Decisions options shared by campaign entry points."""
from llm_classifier_bench.classifiers.openai_decisions import OpenAIDecisionsClassifier
from llm_classifier_bench.classifiers.perplexity_decisions import PerplexityDecisionsClassifier


def add_decisions_arguments(parser):
    parser.add_argument("--decisions-model", default=None, help="Overrides OPENAI_DECISIONS_MODEL (default gpt-6-luna)")
    parser.add_argument("--decisions-timeout-s", type=float, default=None,
                        help="Overrides OPENAI_DECISIONS_TIMEOUT_S (default 120)")


def decisions_options(args):
    return {"model": args.decisions_model, "timeout_s": args.decisions_timeout_s}


def decisions_manifest(args):
    return OpenAIDecisionsClassifier(**decisions_options(args)).inference_metadata()


def add_perplexity_arguments(parser):
    parser.add_argument("--perplexity-model", default=None,
                        help="Overrides PERPLEXITY_DECISIONS_MODEL (default pplx-decider-v1.1-27b)")
    parser.add_argument("--perplexity-timeout-s", type=float, default=None,
                        help="Overrides PERPLEXITY_DECISIONS_TIMEOUT_S (default 30)")


def perplexity_options(args):
    return {"model": args.perplexity_model, "timeout_s": args.perplexity_timeout_s}


def perplexity_manifest(args):
    return PerplexityDecisionsClassifier(**perplexity_options(args)).inference_metadata()
