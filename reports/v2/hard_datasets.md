# Hard classification tasks — Implicit Hate and ANLI R3

**40/40 included conditions completed**, 20 per dataset; seed 42 and 40 held-out test examples/class (240 Implicit Hate, 120 ANLI R3). Supervised fit budgets are 20/50/100 per class; OpenAI in-context uses five demonstrations total and zero-shot methods use no fit labels. Validation labels consumed: zero. These fixed-label tasks test domain difficulty, not label-cardinality scaling.

| Method | Fit/class | Implicit Hate (6 classes) | ANLI R3 (3 classes) |
| --- | ---: | ---: | ---: |
| TF-IDF + LR | 20 | 40.42% | 30.83% |
| TF-IDF + LR | 50 | 48.75% | 30.00% |
| TF-IDF + LR | 100 | 45.00% | 30.00% |
| MiniLM + LR | 20 | 48.75% | 41.67% |
| MiniLM + LR | 50 | 55.83% | 35.00% |
| MiniLM + LR | 100 | 57.92% | 35.00% |
| BERT | 20 | 17.92% | 29.17% |
| BERT | 50 | 30.42% | 39.17% |
| BERT | 100 | 44.58% | 33.33% |
| OpenAI zero-shot | 0 | 30.83% | 49.17% |
| OpenAI in-context (5 total) | 0 | 37.50% | 45.83% |
| Jev zero-shot | 0 | 48.33% | 66.67% |
| GPT-6 Decisions zero-shot | 0 | 49.17% | 47.50% |
| Emissary routing zero-shot | 0 | 30.83% | 42.50% |
| Emissary Qwen SFT | 20 | 33.75% | 39.17% |
| Emissary Qwen SFT | 50 | 45.00% | 38.33% |
| Emissary Qwen SFT | 100 | 55.83% | 40.83% |
| Emissary Llama SFT | 20 | 44.58% | 34.17% |
| Emissary Llama SFT | 50 | 45.83% | 36.67% |
| Emissary Llama SFT | 100 | 51.25% | 37.50% |

Implicit Hate: MiniLM+LR at 100/class leads the observed accuracy (57.92%); Qwen at 100/class reaches 55.83%, while Llama reaches 44.58%, 45.83% and 51.25% at 20/50/100 per class. ANLI R3: Jev zero-shot leads the observed accuracy (66.67%). These are exploratory point estimates on small cohorts; Llama scores 34.17%, 36.67% and 37.50% on ANLI R3. Increasing fit labels does not consistently improve every method. Comparisons between supervised and zero-shot methods use different label budgets.

GPT-6 Decisions refused 18/240 Implicit Hate inputs (92.50% classification coverage); its 49.17% accuracy and 0.5070 macro-F1 include those refusals as errors on all 240 inputs. ANLI Decisions has 120/120 classification coverage. This full-denominator policy differs from the historical Banking77 accepted-only supplement above; those older results remain unchanged.

Jev Implicit Hate has three probability vectors renormalized for sums within 0.05 of one; original values and the correction policy remain in raw evidence. All other Jev vectors are unchanged and none are missing. This differs from the historical strict sum policy. Six Emissary Llama SFT conditions completed in the October 20, 2026 retry (Mexico City date), using the original frozen fit/test cohorts and byte-identical training uploads to Qwen at equal budgets. Earlier failed training attempts remain historical evidence. Perplexity remains excluded by user instruction; that exclusion is not a zero score.

Implicit Hate uses a synthetic stratified 60/20/20 split after grouping identical posts; ANLI uses official R3 train/dev/test boundaries, removing cross-split exact duplicates from lower-priority pools. Class descriptions are fixed assistant-authored rubrics, not independently human-reviewed. Per-example explanations stay outside classifier inputs. Local transformers use CPU, BERT two epochs, and Qwen/Llama SFT one epoch; BERT length 512 differs from the earlier Banking77 recipe. One seed, model-specific truncation and uncontrolled hosted hardware limit interpretation. No cross-provider cost or latency winner is claimed; Emissary prices remain unavailable.

## Dataset protocol

| Dataset | Source revision | Labels | Fit totals | Test | Split |
| --- | --- | ---: | --- | ---: | --- |
| Implicit Hate (6 classes) | `57d34fad1cf7d4d9e71774ea17c4c1a7f57af8d4` | 6 | 120/300/600 | 240 | synthetic 60/20/20 |
| ANLI R3 (3 classes) | `8e4813d81f46d313dac7892e1c28076917cfcdf9` | 3 | 60/150/300 | 120 | official R3 |

Implicit Hate excludes primary `other` and retains six fine labels. ANLI labels are entailment, neutral and contradiction; all methods see `Premise: …` followed by `Hypothesis: …`. The frozen summary includes exact labels/rubrics, source counts, explanation coverage, test IDs, fit-pool IDs, partition audits and source hashes. Selection followed the user-specified tasks, replacing the issue’s proposed SELU-first candidate search; no SELU review is claimed. Difficulty is motivated by implicit meaning and overlapping hate categories, and adversarial premise–hypothesis reasoning, rather than selecting an observed accuracy range. The fixed sampling preceded the user’s API runs; no test-driven resampling or tuning was performed. Implicit Hate license/access information remains unresolved in the public mirror, ANLI documents CC BY-NC 4.0; no raw corpus is committed. [Source and partition documentation](../../docs/hard_datasets.md).

## Full-cohort quality

Accuracy intervals are pointwise 95% gold-stratified bootstrap intervals (2,000 draws, seed 20260925), conditional on the fixed models, label set and campaign seed. They do not estimate training-seed variation. No multiplicity-adjusted significance claim is made. Missing probabilities leave calibration unavailable for the full cohort; they are never replaced by a fabricated vector.

| Dataset | Method | Fit/class | Accuracy [95% CI] | Macro-F1 | ECE | Adaptive ECE | Log loss | Brier |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Implicit Hate (6 classes) | TF-IDF + LR | 20 | 40.42% [35.00, 46.25] | 0.3895 | 0.1905 | 0.1905 | 1.6904 | 0.7968 |
| Implicit Hate (6 classes) | TF-IDF + LR | 50 | 48.75% [42.92, 55.00] | 0.4808 | 0.2487 | 0.2487 | 1.6045 | 0.7619 |
| Implicit Hate (6 classes) | TF-IDF + LR | 100 | 45.00% [39.17, 51.25] | 0.4419 | 0.1820 | 0.1959 | 1.5326 | 0.7304 |
| Implicit Hate (6 classes) | MiniLM + LR | 20 | 48.75% [42.92, 54.58] | 0.4677 | 0.1907 | 0.1907 | 1.4872 | 0.7155 |
| Implicit Hate (6 classes) | MiniLM + LR | 50 | 55.83% [49.58, 61.67] | 0.5492 | 0.1824 | 0.1824 | 1.3162 | 0.6401 |
| Implicit Hate (6 classes) | MiniLM + LR | 100 | 57.92% [52.08, 63.75] | 0.5705 | 0.1571 | 0.1803 | 1.2268 | 0.5947 |
| Implicit Hate (6 classes) | BERT | 20 | 17.92% [13.75, 22.08] | 0.1344 | 0.0705 | 0.0672 | 1.7813 | 0.8298 |
| Implicit Hate (6 classes) | BERT | 50 | 30.42% [25.83, 35.00] | 0.2373 | 0.0766 | 0.0907 | 1.7015 | 0.8007 |
| Implicit Hate (6 classes) | BERT | 100 | 44.58% [39.58, 50.00] | 0.3910 | 0.1757 | 0.1808 | 1.5669 | 0.7475 |
| Implicit Hate (6 classes) | OpenAI zero-shot | 0 | 30.83% [25.83, 36.25] | 0.2951 | — | — | — | — |
| Implicit Hate (6 classes) | OpenAI in-context (5 total) | 0 | 37.50% [32.08, 42.92] | 0.3387 | — | — | — | — |
| Implicit Hate (6 classes) | Jev zero-shot | 0 | 48.33% [42.50, 54.17] | 0.4610 | 0.1760 | 0.1760 | 2.1927 | 0.7066 |
| Implicit Hate (6 classes) | GPT-6 Decisions zero-shot | 0 | 49.17% [43.33, 55.42] | 0.5070 | — | — | — | — |
| Implicit Hate (6 classes) | Emissary routing zero-shot | 0 | 30.83% [26.25, 35.83] | 0.2708 | 0.1512 | 0.1506 | 1.8063 | 0.8324 |
| Implicit Hate (6 classes) | Emissary Qwen SFT | 20 | 33.75% [29.17, 38.33] | 0.2598 | 0.1887 | 0.1887 | 2.1753 | 0.8115 |
| Implicit Hate (6 classes) | Emissary Qwen SFT | 50 | 45.00% [39.58, 50.42] | 0.4329 | 0.0726 | 0.0837 | 1.3973 | 0.6741 |
| Implicit Hate (6 classes) | Emissary Qwen SFT | 100 | 55.83% [49.58, 61.67] | 0.5560 | 0.0732 | 0.0960 | 1.2311 | 0.5953 |
| ANLI R3 (3 classes) | TF-IDF + LR | 20 | 30.83% [23.33, 38.33] | 0.2794 | 0.0909 | 0.1657 | 1.0971 | 0.6654 |
| ANLI R3 (3 classes) | TF-IDF + LR | 50 | 30.00% [23.31, 36.67] | 0.2603 | 0.0765 | 0.1232 | 1.1034 | 0.6696 |
| ANLI R3 (3 classes) | TF-IDF + LR | 100 | 30.00% [22.50, 37.50] | 0.2853 | 0.0733 | 0.1881 | 1.1133 | 0.6770 |
| ANLI R3 (3 classes) | MiniLM + LR | 20 | 41.67% [33.33, 50.00] | 0.4089 | 0.0771 | 0.1824 | 1.1030 | 0.6670 |
| ANLI R3 (3 classes) | MiniLM + LR | 50 | 35.00% [26.67, 43.35] | 0.3428 | 0.0759 | 0.1150 | 1.1122 | 0.6715 |
| ANLI R3 (3 classes) | MiniLM + LR | 100 | 35.00% [26.65, 43.33] | 0.3467 | 0.1200 | 0.1152 | 1.1432 | 0.6879 |
| ANLI R3 (3 classes) | BERT | 20 | 29.17% [21.67, 36.67] | 0.2507 | 0.0818 | 0.1214 | 1.1074 | 0.6728 |
| ANLI R3 (3 classes) | BERT | 50 | 39.17% [31.67, 46.67] | 0.3490 | 0.0320 | 0.1413 | 1.0921 | 0.6622 |
| ANLI R3 (3 classes) | BERT | 100 | 33.33% [25.83, 40.83] | 0.2923 | 0.0688 | 0.0728 | 1.0946 | 0.6637 |
| ANLI R3 (3 classes) | OpenAI zero-shot | 0 | 49.17% [42.50, 56.67] | 0.4511 | — | — | — | — |
| ANLI R3 (3 classes) | OpenAI in-context (5 total) | 0 | 45.83% [38.33, 53.33] | 0.4290 | — | — | — | — |
| ANLI R3 (3 classes) | Jev zero-shot | 0 | 66.67% [58.33, 75.00] | 0.6714 | 0.1378 | 0.1319 | 1.6315 | 0.4934 |
| ANLI R3 (3 classes) | GPT-6 Decisions zero-shot | 0 | 47.50% [39.17, 55.83] | 0.4547 | 0.3263 | 0.3641 | 3.3858 | 0.7744 |
| ANLI R3 (3 classes) | Emissary routing zero-shot | 0 | 42.50% [35.83, 48.33] | 0.3556 | 0.3446 | 0.3416 | 1.3661 | 0.8249 |
| ANLI R3 (3 classes) | Emissary Qwen SFT | 20 | 39.17% [30.83, 47.50] | 0.3899 | 0.1441 | 0.1888 | 1.2015 | 0.7197 |
| ANLI R3 (3 classes) | Emissary Qwen SFT | 50 | 38.33% [31.65, 45.83] | 0.3354 | 0.2383 | 0.2383 | 1.2683 | 0.7609 |
| ANLI R3 (3 classes) | Emissary Qwen SFT | 100 | 40.83% [32.48, 50.00] | 0.4060 | 0.1652 | 0.2122 | 1.1728 | 0.7056 |
| Implicit Hate (6 classes) | Emissary Llama SFT | 20 | 44.58% [38.32, 50.43] | 0.4399 | 0.1011 | 0.1324 | 1.5240 | 0.7231 |
| Implicit Hate (6 classes) | Emissary Llama SFT | 50 | 45.83% [40.41, 51.67] | 0.4474 | 0.1457 | 0.1658 | 1.5275 | 0.7228 |
| Implicit Hate (6 classes) | Emissary Llama SFT | 100 | 51.25% [45.00, 57.50] | 0.5105 | 0.1596 | 0.1696 | 1.4405 | 0.6799 |
| ANLI R3 (3 classes) | Emissary Llama SFT | 20 | 34.17% [26.67, 41.67] | 0.2866 | 0.2248 | 0.2233 | 1.3438 | 0.7936 |
| ANLI R3 (3 classes) | Emissary Llama SFT | 50 | 36.67% [28.33, 45.00] | 0.3590 | 0.2885 | 0.2800 | 1.4366 | 0.8396 |
| ANLI R3 (3 classes) | Emissary Llama SFT | 100 | 37.50% [29.17, 46.67] | 0.3694 | 0.3363 | 0.3272 | 1.5770 | 0.8784 |

## Refusals and probability coverage

| Dataset | Method | Classified / planned | Refusals | Missing probability vectors |
| --- | --- | ---: | ---: | ---: |
| Implicit Hate (6 classes) | OpenAI zero-shot | 240/240 | 0 | 240 |
| Implicit Hate (6 classes) | OpenAI in-context (5 total) | 240/240 | 0 | 240 |
| Implicit Hate (6 classes) | Jev zero-shot | 240/240 | 0 | 0 |
| Implicit Hate (6 classes) | GPT-6 Decisions zero-shot | 222/240 | 18 | 18 |
| ANLI R3 (3 classes) | OpenAI zero-shot | 120/120 | 0 | 120 |
| ANLI R3 (3 classes) | OpenAI in-context (5 total) | 120/120 | 0 | 120 |
| ANLI R3 (3 classes) | Jev zero-shot | 120/120 | 0 | 0 |
| ANLI R3 (3 classes) | GPT-6 Decisions zero-shot | 120/120 | 0 | 0 |

## Operations and evidence

There are 7,200 planned evaluation outcomes, including 18 refusals. Those outcomes are not 7,200 independent input texts: every method uses the same test cohort within a dataset. Six Qwen and six Llama trainings succeeded, one per model/dataset/budget. Local methods have fixed recipes and consume the same nested fit pools at equal budgets. SFT selection and serialized upload hashes were checked against saved metadata; successful training/deployment IDs and provider hyperparameters are retained in the JSON summary. This verifies client-side evidence, not provider internals or base-model pretraining contamination.

Inference uses batch size one, no warmup and no automatic transport retries. Interrupted conditions were resumed at sample level, preserving earlier predictions. The failed Jev normalization input was requested again; the historical Decisions refusal was recovered from its usage ledger without another request. Cost/latency populations remain separated by segment. `operations` in the JSON retains original attempt counts, failed attempts, known inference subtotals and latency observations for each segment; preparation and training are excluded from inference prices. Unknown Emissary and local hardware prices remain unavailable. No total campaign bill or pooled resumed throughput is asserted.

The Llama failure was `tr-DfFCVXuyPg6bawDpqSKTFR`, Implicit Hate 20/class, status Failed at 0% progress, with no checkpoints. The provider detail API returned no cause and `/logs` returned 404. Llama was initially excluded; the later complete six-condition retry is now included. Failed training attempts are not scored as predictions. Llama starts fresh from Llama-3.2-1B-Instruct; Qwen starts from Qwen3-4B-Base. This does not isolate model size or instruction tuning. Retry plan/summary hashes and per-run evidence are retained in JSON.

## Reproduction

All model/API runs were performed by the user. This report only reads saved artifacts and recomputes metrics, integrity checks and uncertainty offline. It does not train or call providers.

Render from the committed summary (no credentials or raw datasets):

```bash
PYTHONPATH=src venv/bin/python scripts/build_hard_datasets_report.py --update-readme
```

Recompute from the locally retained raw campaign:

```bash
PYTHONPATH=src venv/bin/python scripts/build_hard_datasets_report.py \
  --campaign-root artifacts/hard_experiments/20261018T162808464373Z \
  --llama-campaign-root artifacts/hard_llama_retry/20261021T011632187851Z --update-readme
```

See [machine-readable results and SHA-256 evidence](hard_datasets_results.json), [flat quality table](hard_datasets_results.csv), and [execution instructions](../../docs/hard_datasets.md). The 64 Banking77 historical condition objects and four Decisions supplement conditions remain separate and unchanged.
