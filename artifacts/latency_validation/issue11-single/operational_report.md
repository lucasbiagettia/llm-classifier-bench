# Operational report: issue11-single

Generated from `measurement.json` and `timings.jsonl`; no inference was repeated.

Run status: **completed**. Phase: after_explicit_warmup.

## Measurement conditions

| Condition | Value |
| --- | --- |
| Boundary | classifier.predict(inputs) entry to normalized Prediction list return; includes adapter preprocessing and postprocessing |
| Backend | local |
| Classifier / model | tfidf-logreg / tfidf-logistic-regression |
| Device | cpu |
| CPU | Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz |
| GPUs initialized in process | [] |
| Platform / Python | Linux-7.1.9-1-MANJARO-x86_64-with-glibc2.44 / 3.14.7 |
| Client location | local machine; no network |
| Batch size / concurrency | 1 / 1 |
| Batch execution | sequential_single_example |
| Cache conditions | model/vectorizer resident after fit and warmup; repeated fixture text; no response cache |
| Warmup calls / failures | 5 / 0 |
| Warmup examples | 5 |
| Application retries | 0 |

Transport retry/timeout settings, library versions, affinity and thread settings:

```json
{
  "classifier": {
    "backend": "local",
    "device": "cpu",
    "transport_max_retries": 0,
    "batch_execution": "sequential_single_example"
  },
  "library_versions": {
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit-learn": "1.9.0",
    "torch": "2.13.0",
    "transformers": "5.14.1",
    "sentence-transformers": "5.6.1",
    "openai": "2.52.0",
    "requests": "2.34.2"
  },
  "cpu_affinity": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
  ],
  "thread_environment": {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": null,
    "TOKENIZERS_PARALLELISM": null
  },
  "torch_threads": null
}
```

## Preparation (excluded from inference)

| Stage | Status | Wall ms |
| --- | --- | ---: |
| prepare | success | 0.007852 |
| fit | success | 2555.229443 |

Classifier construction occurred before the runner and is not timed here.
Model loading performed inside prepare/fit is included in those stage durations.

## Evaluation call latency

| Population | Calls | Mean ms | P50 ms | P95 ms | P99 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Successful | 200 | 1.214918 | 0.933287 | 2.746468 | unavailable |
| Failed or invalid | 0 | unavailable | unavailable | unavailable | unavailable |
| Successful, batch=1 | 200 | 1.214918 | 0.933287 | 2.746468 | unavailable |

P99 requires at least 1,000 calls in its population; this heuristic is not a confidence interval.

## Throughput and coverage

- Successful / attempted / unattempted examples: 200 / 200 / 0.
- Failed-call fraction: 0.000000.
- Successful examples per second, observed window: 705.996464.
- Successful examples per second, active calls only: 823.101116.
- Observed window / active call ms: 283.287538 / 242.983512.
- Amortized successful-call ms per example: 1.214918.

Throughput denominator: first evaluation call start to last call end, including failures and inter-call validation/logging.

Amortized time is a processing rate, not an individual request latency. Existing
adapters process examples sequentially even when predict receives a larger list.
Hosted calls include client/network/service time; local calls include preprocessing,
local compute and output materialization. These are client-observed boundaries, not
identical measures of server compute. Unknown provider cache, hardware and injected
transport-attempt timings remain unavailable.
