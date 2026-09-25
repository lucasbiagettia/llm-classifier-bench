# v1.3 — Add Jev to classifier comparisons and existing metrics

## Objective

Add TypeSafe AI's Jev as a classifier in release **v1.3**, and evaluate it with the benchmark's existing quality, calibration, latency and cost metrics on the existing datasets.

This is the scope previously discussed as “v3”. Dataset expansion belongs to a later release.

## Tasks and acceptance criteria

- [ ] Implement a Jev adapter using the documented Choice API and the existing classifier lifecycle. Preserve sample IDs, the exact configured label set and frozen label descriptions.
- [ ] Make credentials, model identifier, endpoint, timeouts and retry policy configurable. Save requested/resolved model versions and sanitized request/response metadata; pin a version when available.
- [ ] Normalize the chosen label and full label-to-probability mapping. Validate label alignment, finite values, normalization and consistency with the selected class; handle ties explicitly.
- [ ] Preserve the provider's raw `confidence` separately. For the benchmark's top-label confidence, use the probability assigned to the predicted class. Jev's native confidence describes the distribution's concentration and must not be substituted for that probability.
- [ ] Reuse accuracy, macro-F1, ECE, adaptive ECE, log loss and Brier score through the existing metric pipeline. Missing required outputs keep the affected metrics unavailable with a reason; invalid responses remain explicit failures.
- [ ] Support zero-shot evaluation and record zero labeled examples consumed. Declare supported supervision modes: any requested nonzero budget must either consume the exact shared context pool with separate label accounting or be explicitly unsupported. Do not present zero-shot results as matched nonzero-budget runs.
- [ ] Integrate Jev into the maintained campaign entry points and summaries, preserving the existing dataset splits, selected labels, test IDs and comparison regimes. Record provider label/context limits and unsupported configurations.
- [ ] Use the common inference timing boundary and usage ledger, including warmup, retries and failed attempts. Calculate inference cost per 1,000 valid classifications using a versioned, sourced rate card and observed usage; retain observed/estimated/unavailable conventions. Record preparation through the existing hooks without inventing training charges.
- [ ] Add offline request/response fixtures and regression tests for label order, probability handling, native-confidence differences, invalid/missing outputs, usage accounting and integration with both campaigns.
- [ ] Validate the real API with a small, authorized pilot once credentials and a spending cap are available. Save enough artifacts to reproduce the metric calculations without new inference, and distinguish simulated fixtures from live evidence.
- [ ] Include Jev in comparative tables/reports with the existing uncertainty and failure-coverage conventions. Reuse compatible baseline artifacts; document any configuration mismatch requiring a rerun.
- [ ] Update setup, execution and report-generation instructions and record the supported Jev configuration in the protocol for v1.3.

## Completion evidence

A tested adapter, reproducible offline fixtures, saved live-pilot predictions/usage and a short comparison report covering quality, calibration, latency and inference cost. Document limitations and unsupported cells. A particular model winning or appearing calibrated is not an acceptance criterion.

## Scope boundary

Use the existing datasets, metric definitions and reporting conventions. New datasets, new aggregate scores, Jev fine-tuning, hierarchical classification and a new training abstraction are outside this issue. Paid execution requires an explicit pilot budget; implementation and offline testing do not.

## Dependencies and references

- Existing matched-example, timing, cost and reporting infrastructure.
- Before live validation: TypeSafe API access, a confirmed model/version, current pricing and an authorized pilot cap.
- [Jev introduction](https://docs.typesafe.ai/introduction)
- [Choice request/response contract](https://docs.typesafe.ai/primitives/choice)
- [Native confidence semantics](https://docs.typesafe.ai/confidence)
- [Authentication and quick start](https://docs.typesafe.ai/introduction/quickstart)

Recheck provider documentation when implementing; API limits, versions and pricing can change.

