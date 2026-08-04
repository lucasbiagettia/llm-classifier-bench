# Versioned class definitions

This patch separates **canonical dataset labels** from the semantic descriptions
shown to zero-shot classifiers.

The benchmark runner never calls the description-generating LLM. Generation is
an offline preparation step that produces a frozen JSON artifact.

## Architecture

```text
source dataset
    -> canonical labels
    -> offline generator
    -> versioned JSON profile
    -> freeze / review
    -> benchmark runner loads JSON
    -> identical ClassDefinition[]
       -> Emissary
       -> OpenAI
```

The LLM generator receives only:

- dataset name;
- optional dataset context supplied on the CLI;
- the complete canonical label inventory.

It does **not** receive train, validation, or test examples.

## Generate the deterministic control profile

```bash
PYTHONPATH=src python scripts/generate_class_definitions.py \
  --dataset banking77 \
  --mode minimal
```

Output:

```text
class_definitions_data/banking77/canonical_minimal_v1.json
```

## Generate the LLM-enriched profile

The generator model is configured independently from the model used later by
`OpenAIClassifier`.

```bash
PYTHONPATH=src python scripts/generate_class_definitions.py \
  --dataset banking77 \
  --mode llm-enriched \
  --dataset-context "Banking customer-support intent classification."
```

You can choose another cheap generator without changing the benchmark
classifier:

```bash
PYTHONPATH=src python scripts/generate_class_definitions.py \
  --dataset banking77 \
  --mode llm-enriched \
  --model YOUR_GENERATOR_MODEL \
  --dataset-context "Banking customer-support intent classification."
```

Generation is a single API preparation step. The resulting file is reused by
all later runs.

## Review and freeze

Generated profiles start with:

```json
"review_status": "unreviewed"
```

Review the descriptions only for objective semantic errors. Do not optimize
individual descriptions after seeing classifier results. If the generation
procedure needs correction, produce a new profile version such as
`canonical_llm_enriched_v2` rather than silently changing v1.

Once the chosen file is final, mark it `approved`, commit it, and do not modify
it during the benchmark campaign.

The runner stores the exact file SHA-256 in every run config.

## Use a frozen profile in the runner

```python
from pathlib import Path

from llm_classifier_bench.runner import BenchmarkRunConfig, run_benchmark

config = BenchmarkRunConfig(
    class_definitions_path=Path(
        "class_definitions_data/banking77/canonical_llm_enriched_v1.json"
    ),
)

result = run_benchmark(dataset, classifier, config)
```

Before `classifier.prepare()`, the runner verifies that the profile contains
exactly the dataset's canonical labels. Missing, renamed, duplicated, or extra
labels fail the run.

`config.json` records:

- profile name;
- profile path;
- SHA-256 of the exact file;
- review status;
- generator metadata;
- the exact final class descriptions used by the classifier.

## Formal benchmark recommendation

Keep both profiles:

```text
canonical_minimal_v1
canonical_llm_enriched_v1
```

Use the enriched profile as the fixed semantic condition and retain minimal as
a robustness/ablation condition. Every zero-shot classifier must receive the
same profile within a condition.
