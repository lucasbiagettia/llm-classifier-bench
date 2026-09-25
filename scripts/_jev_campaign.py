"""Explicit Jev options shared by both campaign entry points."""
from llm_classifier_bench.classifiers.jev import DEFAULT_JEV_ENDPOINT, DEFAULT_JEV_MODEL


def add_jev_arguments(parser):
    parser.add_argument("--jev-model", default=DEFAULT_JEV_MODEL)
    parser.add_argument("--jev-endpoint", default=DEFAULT_JEV_ENDPOINT)
    parser.add_argument("--jev-timeout-s", type=float, default=30.0)
    parser.add_argument("--jev-max-retries", type=int, default=0)
    parser.add_argument("--jev-retry-backoff-s", type=float, default=1.0)
    parser.add_argument("--jev-max-requests", type=int, default=None,
                        help="Per-run attempt cap including warmup and retries; not a whole-campaign cap")
    parser.add_argument("--jev-context-window-tokens", type=int, default=32000)


def jev_options(args):
    return {"model": args.jev_model, "endpoint": args.jev_endpoint,
            "timeout_s": args.jev_timeout_s, "max_retries": args.jev_max_retries,
            "retry_backoff_s": args.jev_retry_backoff_s, "max_requests": args.jev_max_requests,
            "context_window_tokens": args.jev_context_window_tokens}
