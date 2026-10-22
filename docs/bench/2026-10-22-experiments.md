# Common experiment records: 2026-10-22

This round unifies control comparisons, training and playback around the same reproducible experiment record and artifact store.
It preserves the existing interfaces and leaves controller formulas, rewards and manipulation success rates unchanged.
The implementation contract and examples are in [EXPERIMENTS.md](../EXPERIMENTS.md).

## Shared records and compatibility

Schema version 1 records environment, registered task and implementation identity, resolved algorithm settings, seeds by role, requested and consumed budgets, versions, metrics, lifecycle and artifact manifests.
Execution UUID and semantic input fingerprint are distinct.
MJB and full integration-state snapshots retain compiled assets, controls, applied forces, mocap and warm starts.
Artifact paths and bytes are validated, metadata writes are atomic, and historical policy/checkpoint bytes survive replacement of the named policy folder.

Comparisons own a parent and one child per method, including skipped and failed methods.
Their actual policy inputs are fixed to verified archived bytes to avoid a checkpoint replacement race.
Training owns a run parent and independent trial/rung segments, with a single writer for each record.
Resume checkpoint SHA affects input identity; actual weight hashes determine policy lineage rather than trial order or the latest attempted rung.
Live control/playback observations close on stop, replacement, reset, model change and shutdown without replacing controller objects or scheduling.
Unobserved contact/plugin metrics remain null with an explicit reason.

All 60 MCP tool definitions match the prior snapshot exactly.
Existing results and policy metadata retain old fields and gain additive references.
Scripts can query bounded history and a canonical experiment UUID inside the current world.
The same attachment recorder is used by script, MCP and viewer playback adapters.

Independent review corrected unsafe recorded policy paths, foreign-parent record writes, callable metadata, missing contact observations, release-related snapshots/storage failures, hidden controller dependencies, policy lineage, submission failures and checkpoint input identity.
The final independent review found no remaining concrete P1/P2 issues.

## Actual execution and video

The installed `robotforge mcp` command ran in an isolated home.
Headless Chrome rendered the real MuJoCo world and recorded the complete viewer flow as H.264, 1920x1080, 30fps, 28.13 seconds.
The video contains verification-only labels populated from actual returned experiment IDs and results.
It is a real browser capture, not generated animation.
The final MP4 was fully decoded without errors and its final frame was visually inspected.

The video is stored locally at `../../../robotforge-evidence/2026-10-22-experiments/live/experiment-validation.mp4`.
It is not committed as a repository binary.

- MCP compared Reach IK and PD from the same saved live state, preserving the MCP reset seed `None`.
  A fresh process reproduced both methods' physical metrics; wall time was excluded from this comparison.
- Bounded live IK control produced a closed record while preserving the existing attached-controller behavior.
  A fresh process restored its snapshot and advanced its actual recorded physics-step count.
  This real live replay was not bitwise identical: maximum qpos error was 3.875e-10, qvel 6.116e-10 and ctrl 0.
  Strict equality is reported as false; the separate 1e-8 absolute-tolerance check passes.
  Both the actual errors and the strict result remain visible in the evidence.
- A detached coordinator and ARS worker trained Reach with seed 0 and a requested budget of 64 steps.
  Actual consumption was 800 steps, preserving the existing whole-iteration budget semantics.
  Parent and segment records both completed and retained the policy and checkpoint artifacts.
- A fresh process reconstructed the recorded RunSpec and trained again.
  Policy SHA256 matched exactly: `cd8881fe0a926ba34856714278bf651cdd5b29d2bf9d80927442b2bce8065797`.
  This verifies a small deterministic NumPy training path, not optional SB3 runtimes or learning quality.
- Chrome clicked the actual Training Play button from pause, observed the policy toast and advancing physics time, then replayed the saved policy.
  A separate token-authenticated script ran the policy, stepped 0.12 simulated seconds, stopped it, queried the closed record and inspected common history.
  Playback retained both its training-run and matching policy-segment origins.
  The browser reported no JavaScript errors.

The successful run retained 11 terminal records across verification, comparison, control, training and replay.
Core tests separately restore complete integration state and verify 25 subsequent physics steps bit-for-bit; this stronger result is not generalized to every live controller replay.

## Checks and evidence

- Wheel build, Ruff lint/format and independent review passed.
- All 128 targeted cases passed together, including 98 newly added experiment cases.
- Standard installed CLI/MCP/HTTP/WebSocket/rendering smoke passed in 3.1 seconds.
- Complete default regression: 3170 passed, 127 skipped, zero failures/errors across all 3297 selected cases (599.6 seconds; two green shards).
  No default case was omitted; 63 slow/browser cases were deselected by the existing project default marker expression.
  Source, scripts and tests stayed unchanged during this run.
- Real Chrome gamepad regression: all 8 cases passed in 33.71 seconds.
- Built-in behavior comparison: all 280 qpos/qvel/ctrl hash markers in 14 scenarios match `bcf2cf7`.
- Source/test fingerprint: `b519a9eede156c606126cefdac4e1e5200a5bf0ba7c098a1e247d6c2dfd4ea36`.

Raw evidence is in `../../../robotforge-evidence/2026-10-22-experiments/`, including test collection/logs/XML, source hashes, worker outputs, experiment JSON and artifacts, real MCP/script replies, independent-process reproductions and the MP4.
The accompanying JSON contains the compact results.

## Limits

Restoring compiled snapshots requires the recorded MuJoCo version and compatible platform; editable source assets and plugin Python memory are not bundled.
Replay refuses known unsupported controller/callback dependencies and checks actual final-state errors.
Controller preparation before attachment, recomputed derived fields and interactive changes can prevent bitwise live replay.
Training and playback checks measure framework contracts and reproducibility, not learned task quality.

Native Windows, optional RL runtimes and current embedded MCP App behavior were not validated in this round.
mypy/pyright, Bandit and pip-audit are unavailable in the project environment; no dependency was installed for them.
No paid Actions workflow, release or deployment was started.
Prior manipulation holdouts remain untouched.
