# Emissary classification scaling on Banking77

**Exploratory technical brief for Tanmay Chopra**  
Campaign: `20260804T014841Z` | Seed: 42 | Candidate labels: 5, 10, 20, 25

## Executive take

The most important result in this pilot is that **Emissary loses more classification accuracy as the candidate label set grows than the other methods in this run**. On the full test sets, Emissary moves from **94.0% accuracy at 5 labels to 78.4% at 25 labels**, a decline of **15.6 percentage points**. GPT-5-nano declines 8.8 points, fine-tuned BERT 6.6 points, and MiniLM + logistic regression 2.4 points.

The nested design lets us run a cleaner check on the same 100 examples from the original 5 classes. In that fixed cohort, where the inputs and gold labels are unchanged and only additional candidate labels are introduced, Emissary falls from **94% to 78%**. GPT-5-nano falls from 94% to 87%, BERT from 99% to 95%, and MiniLM + LR finishes at the same 97% where it started. This suggests that label-space expansion itself is an important part of the Emissary degradation observed in this seed.

There are also two encouraging signals. First, **top-label calibration remains reasonably good**. Emissary top-label ECE stays between 0.036 and 0.067 across the four conditions. On the fixed cohort, mean confidence falls as accuracy falls, rather than remaining spuriously high. Second, Emissary is **substantially faster than GPT-5-nano in this benchmark environment**, with p50 latency around 185 to 205 ms versus roughly 771 to 817 ms for GPT-5-nano.

These are exploratory results from one seed and one nested class selection. They should be treated as a diagnostic signal, not a general conclusion about Emissary.

## 1. Scaling result

![Overall accuracy](emissary_brief_assets/overall_accuracy.png)

| Method | 5 labels | 10 labels | 20 labels | 25 labels | Change 5 to 25 |
|---|---:|---:|---:|---:|---:|
| Emissary, zero-shot | 94.0% | 84.0% | 79.3% | 78.4% | **-15.6 pp** |
| GPT-5-nano, zero-shot | 94.0% | 84.5% | 85.5% | 85.2% | -8.8 pp |
| BERT fine-tuned | 99.0% | 94.5% | 94.3% | 92.4% | -6.6 pp |
| MiniLM + LR | 97.0% | 97.0% | 97.0% | 94.6% | -2.4 pp |

The supervised systems have access to labeled Banking77 training data, while Emissary and GPT-5-nano are zero-shot. Their absolute scores are therefore not an equal-information comparison. The useful comparison here is the **shape of degradation as the decision space grows**.

## 2. Fixed-cohort test: same examples, more candidate labels

![Fixed cohort accuracy](emissary_brief_assets/fixed_cohort_accuracy.png)

| Method | K=5 | K=10 | K=20 | K=25 |
|---|---:|---:|---:|---:|
| Emissary | 94% | 91% | **79%** | **78%** |
| GPT-5-nano | 94% | 92% | 92% | 87% |
| BERT fine-tuned | 99% | 97% | 96% | 95% |
| MiniLM + LR | 97% | 98% | 98% | 97% |

This is the most informative diagnostic in the pilot. The 100 inputs are identical across all four columns. Emissary correctly classifies 94 of them at K=5. By K=25, **16 of those 94 previously correct examples have flipped to an incorrect label**, while none of the six original errors flip to correct.

The errors are also concentrated. Of those 16 newly lost examples, **13 are predicted as `order_physical_card` at K=25**. Eight of those 13 have `card_about_to_expire` as the gold label. `order_physical_card` first enters the candidate set at K=20, the same step where fixed-cohort Emissary accuracy drops sharply from 91% to 79%.

This pattern is consistent with **semantic distractor sensitivity**, where adding a plausible nearby intent changes decisions on examples that were previously stable. It does not establish the mechanism. The concentration could also depend on this particular class subset, label wording, or the current unreviewed class descriptions.

## 3. Calibration: a positive signal, with an important qualification

| Labels | Accuracy | Top-label ECE | Log loss | Multiclass Brier |
|---:|---:|---:|---:|---:|
| 5 | 94.0% | 0.036 | 0.262 | 0.109 |
| 10 | 84.0% | 0.067 | 0.678 | 0.283 |
| 20 | 79.3% | 0.055 | 0.877 | 0.317 |
| 25 | 78.4% | 0.061 | 0.948 | 0.322 |

The positive interpretation is that **top-label confidence remains fairly well calibrated even as accuracy falls**. On the fixed 100-example cohort, mean top confidence moves from 0.91 at K=5 to 0.76 at K=25, while accuracy moves from 0.94 to 0.78. The model becomes less confident as the task becomes harder, which is preferable to remaining highly confident while accuracy degrades.

The qualification is that the full probability distribution gets worse. Log loss and Brier score both rise materially with the number of labels. So the current result supports a narrow calibration claim about the predicted top label, not a claim that the whole probability distribution is stable.

## 4. Latency

![Latency](emissary_brief_assets/latency_p50.png)

| Method | p50 latency across K | p99 latency across K |
|---|---:|---:|
| Emissary | 185 to 205 ms | 240 to 277 ms |
| GPT-5-nano | 771 to 817 ms | 1,280 to 1,751 ms |
| BERT fine-tuned | 34 to 36 ms | 49 to 67 ms |
| MiniLM + LR | 8 to 9 ms | 14 to 20 ms |

In this setup, Emissary is roughly **4x faster at p50 than the GPT-5-nano API baseline**. That is operationally interesting for zero-shot classification. The local supervised models are much faster still. These latency numbers include different serving paths, network effects, and hardware, so they should be presented as observed end-to-end benchmark latency rather than pure model-runtime comparisons.

## 5. What I would investigate next

1. **Repeat across multiple seeds.** The current campaign uses one nested class path. Five or more seeds would show whether Emissary's steeper degradation is robust to class composition and example sampling.
2. **Audit semantic distractors.** Measure the rate at which previously correct examples flip when each new label block is introduced, and identify which added labels receive those flips. `order_physical_card` is a strong first case to inspect.
3. **Review and freeze class descriptions before publication.** The current definition profile is marked `unreviewed`. A wording issue could amplify label interference, particularly for semantically adjacent card intents.
4. **Separate zero-shot and supervised conclusions.** Emissary versus GPT-5-nano is the cleanest operational zero-shot comparison. BERT and MiniLM + LR are useful reference points for how supervised approaches scale, not like-for-like competitors.
5. **Keep calibration claims narrow.** Top-label ECE looks good. Proper scoring rules do not remain stable as K grows.

## Bottom line

The pilot identifies a real engineering question worth pursuing: **Emissary is competitive with GPT-5-nano at five labels and considerably faster in this setup, but in this seed it is more sensitive to expansion of the candidate label space.** The fixed-cohort analysis suggests that a meaningful part of the drop comes from newly introduced labels changing decisions on examples that Emissary previously classified correctly.

That is a useful result even if later seeds reduce the size of the effect. The next step is to determine whether this is a repeatable property of the classifier, a small set of problematic semantic distractors, or an artifact of this class subset and its current label descriptions.

### Experimental notes

Banking77; balanced test support of 20 examples per selected class; nested class subsets 5 ⊂ 10 ⊂ 20 ⊂ 25; seed 42. Emissary and GPT-5-nano use zero labeled training examples. BERT and MiniLM + LR are supervised, with fixed per-class data support. OpenAI probability-based calibration metrics are unavailable because the baseline did not expose a comparable probability distribution. Cost metrics were not available in this campaign.
