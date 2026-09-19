# Preparation investment and amortization

Complete preparation: True. Cost kind: **estimated**.

- Preparation wall milliseconds: 916.261975.
- Total incremental preparation USD: 5.717921001e-05.
- Cost basis: active prepare/fit wall time at whole-allocation rate; children are explanatory components.

Parent prepare/fit intervals are counted once. Stage components use exclusive time;
nested child time is subtracted from its parent before grouping. Warmup belongs to inference.

## Measured components

| Category | Exclusive ms |
| --- | ---: |
| parent | 894.864095 |
| features | 10.180573 |
| fit | 8.304565 |
| selection | 2.912742 |

## Selected configuration

- Configuration: `{"c": 0.1}`.
- Final refit performed: False.
- Selected fit exclusive ms: 3.209115.
- Selected fit cost estimate USD: 2.002644064e-07.
subset of total investment; excludes shared features, selection evaluation and restoration; never add to total.

## Amortization

Inference scenario: **deployed**.

| Valid predictions | Preparation USD | Inference scenario USD | Total USD/prediction | Availability |
| ---: | ---: | ---: | ---: | --- |
| 1000 | 5.717921001e-05 | 5.391780822 | 0.005391838001 | estimated projection |
| 10000 | 5.717921001e-05 | 5.391780822 | 0.0005391838001 | estimated projection |
| 100000 | 5.717921001e-05 | 5.391780822 | 5.391838001e-05 | estimated projection |

Same measured conditions/rates and failure mix; one inference warmup; preparation added once.

Preparation charges remain separate from inference charges. Unknown costs and infeasible
inference scenarios do not become zero. Stage estimates do not allocate an observed invoice.

Timing includes instrumentation overhead. Hardware, configuration, cache provenance,
rate assumptions, billing evidence and hashes are in the JSON report and source artifacts.
