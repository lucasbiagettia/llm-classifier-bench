# Technical Debt

## HSOLVED — Hardcoded Class Definitions

Class definitions were previously hardcoded in the dataset registry.

This created a methodological risk because class descriptions were not consistently derived from the source dataset metadata and were not produced through a systematic enrichment process. For zero-shot classifiers such as Emissary and generative LLM-based classifiers, these descriptions directly affect how the decision space is interpreted and can therefore materially influence benchmark performance.

### Resolution

Class-definition management has been separated from dataset loading and moved into a dedicated, reproducible preparation process.

The benchmark now follows these principles:

- Canonical class names are always preserved from the original dataset and are never  rewritten by the enrichment process.
- Official class descriptions are used when the source dataset provides them.
- When official descriptions are unavailable, descriptions can be generated through a reproducible LLM-based enrichment process.
- The LLM enrichment process receives only:
    - the dataset name;
    - a general description of the dataset task;
    - the complete canonical label inventory.
- The enrichment process does not receive training, validation, or test examples. This prevents labeled examples from leaking into zero-shot class definitions.
- All descriptions are generated offline, outside the benchmark execution cycle.
- Generated class definitions are persisted as versioned JSON artifacts and frozen before benchmark execution.
- The same frozen class-definition artifact is supplied to every zero-shot classifier.
- The benchmark runner loads and validates these artifacts but never generates descriptions dynamically.
- The dataset registry is responsible only for dataset loading and canonical label resolution, not for defining semantic descriptions.
- Benchmark runs persist the class-definition profile and associated metadata so the exact ontology used for a run can be reproduced.

Two definition profiles are supported:

- canonical_minimal_v1: deterministic descriptions derived mechanically from canonical labels, providing a low-intervention control condition.
- canonical_llm_enriched_v1: semantically enriched descriptions generated once using a fixed LLM, prompt, and generation procedure.

The LLM used to generate class descriptions is independent from any LLM later evaluated as a classifier. The generated definitions are treated as part of the frozen experimental dataset configuration rather than as part of the classifier.

This resolution removes manually hardcoded descriptions from the formal benchmark path while preserving reproducibility, zero-shot integrity, and fairness between classifiers.
## Incomplete Evaluation Metrics

The current metric suite is sufficient for smoke tests and initial comparisons, but it does not yet provide the complete set of diagnostics required for a formal benchmark.

Current metrics cover overall predictive performance, probabilistic calibration, latency, and cost. However, additional metrics will be necessary to better understand class-level behavior, performance under class imbalance, confidence-based decision making, and the statistical reliability of observed differences between classifiers.

### A future revision should add:

- Per-class precision, recall, and F1: expose which individual classes are easy or difficult for each classifier instead of relying only on aggregate metrics.
- Confusion matrix: identify systematic confusion patterns between semantically similar classes.
- Balanced accuracy: provide a performance measure that gives equal importance to each class and is more robust when datasets are imbalanced.
- Multiclass Matthews Correlation Coefficient (MCC): provide an additional global metric that remains informative under class imbalance and captures the quality of the complete classification outcome.
- Top-k accuracy: measure whether the gold label appears among the classifier's highest-probability alternatives, particularly useful as the number of candidate classes increases.
- Risk-coverage / selective accuracy: measure how predictive accuracy changes when the system only accepts predictions above a confidence threshold, allowing comparison of classifiers in scenarios where uncertain predictions may be rejected or escalated.
- Throughput: measure examples processed per second in addition to per-example latency, especially for local or batched classifiers.
- Training time and training cost: account for the operational cost of classifiers that require supervised training, fine-tuning, embedding generation, or other preprocessing, while zero-shot classifiers may require little or no training.
- Confidence intervals and bootstrap estimates: quantify uncertainty around metrics such as accuracy, macro-F1, calibration error, and other benchmark results, allowing observed differences between classifiers to be distinguished from sampling noise.

These additions should be implemented only where the underlying classifier output supports them. Metrics that require probabilities, confidence scores, training information, or cost data should be reported as unavailable rather than estimated or fabricated.