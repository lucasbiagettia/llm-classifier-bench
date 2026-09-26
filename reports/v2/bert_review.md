# Offline BERT review — 2026-09-25

Review performed after stopping all runs. No model loading, training, inference,
CUDA computation or provider call was used for this review. It uses source code,
the earlier recorded CUDA diagnostic and saved benchmark artifacts.

## CUDA finding

Hardware: GTX 1050, compute capability 6.1. Installed PyTorch reported
`2.13.0+cu130`, with kernels starting at sm_75; an earlier tiny CUDA computation
failed with `no kernel image is available for execution on the device`.
`torch.cuda.is_available()` nevertheless returned true.

This is an installed-binary/hardware incompatibility, not evidence that BERT
training is broken. There is a robustness gap in `BertClassifier.fit`: choosing
CUDA based only on `is_available()` can select this unusable GPU. The frozen
campaign avoids that path with `CUDA_VISIBLE_DEVICES=''`; saved runtime metadata
confirms CPU execution. No dependencies or device policy were changed during
this review. Any future GPU setup requires a separate compatibility check and
new latency/preparation measurements.

## Training and output review

The code clears gradients, computes supervised loss, calls backward and steps
AdamW on every batch. It sets training mode each epoch, evaluates validation
without gradients, saves the best state on CPU, restores it and uses eval mode
for final inference. The saved label maps and probability argmax agree with the
reported predictions; saved fit/validation sizes match the requested regimes.
No training-loop bug was established by this review; it is not a proof that every
possible bug is absent.

Illustrative saved seed-42 results (already evaluated before the pause):

| K | BERT 5/class accuracy | Optimizer steps, 5/class | BERT 100/class accuracy | Reference accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 5 | .3800 | 6 | .9500 | .9600 |
| 10 | .1700 | 12 | .9550 | .9600 |
| 20 | .1025 | 21 | .9100 | .9250 |
| 25 | .0840 | 24 | .9160 | .9100 |

The number of optimizer steps is `3 * ceil(5*K/16)`. Very weak 5-shot performance
is consistent with the tiny update budget, while the same implementation learns
at 100/class. This supports undertraining as a plausible explanation; it does
not establish it causally. Do not change epochs based on these test results.

Preparation time is also consistent with CPU work: seed-42 matched 100/class
runs took approximately 200, 444, 1,095 and 2,651 seconds at K=5/10/20/25.
Recorded singleton p50 latency varied across runs (roughly 34–101 ms across the
seed-42 conditions). Cache, input lengths, host contention and thermal state were
not controlled tightly enough to attribute this variability to model quality.

The interrupted seed-43 K=10 5-shot run stopped during fitting with a recorded
`KeyboardInterrupt`, and has zero predictions. All other completed artifacts
remain untouched. Restarting that cell tomorrow requires repeating its incomplete
training, not any previously completed predictions.
