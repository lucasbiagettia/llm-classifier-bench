# Recovery amendment 1 — historical scope

**Current scope:** execution is paused; OpenAI recovery now permits budget 0 only.
The original live recovery attempt stopped before any calls because dry-run
definitions already existed. The planner was corrected and verified twice offline.
See [the revised plan](protocol.md). The historical 16-cell discussion below does
not authorize resuming few-shot runs.

The first frozen execution encountered OpenAI token-per-minute rate limits in
16 cells. Original attempts, usage, failures and validated prediction prefixes
are immutable. Continuation reconstructs the original source condition, checks
the matched pool SHA-256 and every pending prompt SHA-256, and sends only IDs
without a validated saved prediction. No model, label, demonstration, test or
quality-metric setting changes.

Continuation uses paced singleton requests (no hidden retry), with start spacing
estimated from the saved context plan at a conservative 160,000 token/minute
target. The byte estimate divided by four is a scheduling heuristic, not billing
usage. A subsequent 429 remains a recorded failed attempt. Subsequent invocations
can again select only remaining IDs across all retained attempts.

Pacing is inside `predict`: continuation latency includes waiting and must be
reported separately from unpaced initial latency. Do not average the two phases
into a claim about intrinsic API latency or unthrottled throughput. Total usage
and cost include every attempt; unknown charge information remains unknown.
Quality is calculated only when the union of validated prediction IDs equals the
entire frozen test condition exactly, with no duplicate IDs. Report which cells
needed continuation and keep original failure rates visible.

This recovery applies to stateless OpenAI inference. It does not serialize trained
local models or resume Emissary training jobs. Local checkpoints and generic
interrupted-training resume remain outside this implementation. Jev invalid
probability distributions are not retried until a desired valid output appears;
those cells remain failed under the frozen probability-validity criterion.
