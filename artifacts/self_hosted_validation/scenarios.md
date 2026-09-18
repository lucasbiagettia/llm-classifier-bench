# Inference cost: local-fixture

recorded inference attempts including warmup and failures; preparation excluded

Run status: completed. Cost kind: **estimated**. Currency: USD.

| Quantity | Value |
| --- | ---: |
| Total inference cost | 3.114492155e-05 |
| Known subtotal (not a complete total when coverage is unknown) | 3.114492155e-05 |
| Observed charges subtotal | 0 |
| Estimated cost subtotal | 3.114492155e-05 |
| Cost / 1,000 successfully classified evaluation examples | 0.0001557246077 |
| Attempted evaluation examples | 200 |
| Successfully classified evaluation examples | 200 |
| Evaluation failure fraction | 0 |

Success means a valid classification, whether or not the label is correct.
A failed batch has no accepted outputs; its submitted examples count as unsuccessful.
Warmup costs enter the total, but warmup examples never enter the denominator.

## Coverage and assumptions

- Complete cost coverage: True.
- Unpriced entries: 0.
- Hidden retry coverage unknown: False.
- Pricing: selected rate card; estimates are not historical invoices.
- Usage SHA-256: `fb3c4862900fa674998135d4469a585d4b357eda17bb65fee3fc8b2f8c8a9618`.
- Selected pricing SHA-256: `cd04d9082e716c20aa86eb3c1053bb0c5f7ef1bf9ed3979537d05bf996b71290`.

- Measurement metadata SHA-256: `5ec8c3548fe7ba9edae98d44642d151754cb9ce957538b64cbe64b7b369b46f2`.

Full pricing assumptions and per-attempt evidence are in the JSON report.

## Self-hosted inference projections

Interpretation: **cloud_equivalent**; all projected costs are estimates.

- Measured valid classifications/hour: 1156110.598.
- Continuous USD / 1,000 valid classifications: 0.0001943218361.
- Evaluation calls: 200; failure fraction: 0.
- P50/P95 call latency ms: 1.905804 / 5.84088835.
- Batch size / concurrency: 1 / 1.
- Phase: after_explicit_warmup.
- Cache: model/vectorizer resident after fit and warmup; repeated fixture text; no response cache.
- Rate allocation: Illustrative AWS Lightsail Linux public-IPv4 2Xlarge: 8 vCPUs, 32 GB RAM.
- Rate source/date: https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-bundles.html / 2026-09-16.
- Rate assumptions: Illustrative flat hourly cloud equivalent derived from the September 16 snapshot: USD 164/month divided by an assumed 730 hours/month. Applied to active processing or user-specified allocation hours, including idle time. Not a quoted hourly offer, provider invoice or hardware performance equivalence. Excludes preparation, taxes, storage/network extras, billing minimums, rounding and autoscaling. No GPU rate supplied..
- Measured CPU / GPUs: Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz / [].
- Input characters: {"unit": "Unicode code points before preprocessing; not tokens", "count": 200, "min": 11, "max": 20, "p50": 16.0, "p95": 20.0}; token-length distribution unavailable.

Each volume scenario includes one measured warmup; preparation remains excluded.

| Valid classifications | Continuous hours | Continuous USD | Continuous USD/1,000 | Deployed USD/1,000 | Busy fraction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 0.0008686489706 | 0.0001951485359 | 0.0001951485359 | 5.391780822 | 3.619370711e-05 |
| 10000 | 0.008653371308 | 0.001944045061 | 0.0001944045061 | 0.5391780822 | 0.0003605571378 |
| 100000 | 0.08650059468 | 0.01943301031 | 0.0001943301031 | 0.05391780822 | 0.003604191445 |

Deployed allocation hours: 24.

Assumptions:

- Same measured hardware allocation, input distribution, batching, concurrency and cache conditions.
- Measured throughput is extrapolated, not theoretical maximum throughput or a latency SLA.
- Rate is USD/hour for the entire measured allocation, not USD/hour per GPU or per core.
- One measured warmup per scenario; model loading/training and other preparation excluded.
- Flat prorated hourly pricing; no minimum charges, rounding, traffic bursts, autoscaling or availability guarantees.
- All scenario volumes are valid classifications; predicted labels need not be correct.

FLOPs per prediction: unavailable. Optional diagnostic; no comparable operation count measured; peak TFLOPS are not a dollar conversion.
