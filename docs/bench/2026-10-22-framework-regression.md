# Framework regression and controller execution, 2026-10-22

This round establishes a reviewed framework baseline from common experiment records and migrates MCP `run_controller` onto the shared execution kernel.
Existing algorithms, rewards and manipulation success rates are unchanged.
Other legacy entry points retain admission protection for later, separate migrations.

## Baseline and compatibility

Seven fixed cases cover Balance comparison, Reach control and replay, NumPy ARS training, immutable policy lineage, complete integration-state restore and actual MCP controller dispatch.
The baseline was captured against core `aae92839f5a8c91f4e143d4c90e893fa808309a2` in an isolated worktree with only the three new harness files untracked.
Its intentional `git_dirty=true` describes the new harness on the old core, not the migrated implementation.
A fresh process independently repeated all seven cases.
The committed candidate passes against that unchanged baseline; source revision changes are informational and never bless changed physical outputs.
The baseline manifest SHA256 is `e6c82b5c364b47273a8c45ee85441f37b50e1676ecd7cff5bbf2f72d4f88852b`.

The runner validates case inventory, versioned inputs, runtime compatibility, finite available metrics, terminal records, contained artifact paths and content hashes.
It rejects missing evidence, corrupted artifacts and changed physics, and refuses unsupported loaded plugins or MuJoCo callbacks.
Capture requires an explicit flag and does not replace the committed baseline.
See [the baseline contract](../FRAMEWORK_REGRESSION.md) for commands and exit codes.

All 60 MCP tool definitions match the preceding snapshot exactly.
The controller response retains existing fields and adds an execution-record reference.
The old script controller flow and `move` remain compatible.
Fourteen built-in control scenarios retain all 280 qpos/qvel/ctrl hash markers against `aae9283`.

## Execution boundary

Controller preparation, both sides of a scheduled push, stepping, cancellation and reset share one lease and cumulative deadline.
Preparation uses a tracked snapshot worker without holding the live pose lock.
Repeated cancellation retains the lease until the real worker exits.
Cleanup stops only the controller owned by the call, preserving replacement and reset state and clearing velocity/force commands through the existing stop behavior.
Measurement completion and wall exhaustion preserve the existing attached-controller behavior.

Common execution records retain actual step counts, push timing, end reasons and linked controller origins.
Final arrays are marked unavailable after generation invalidation instead of recording replacement-world state.
Artifact failures preserve the original cancellation or applied result and expose recording failure separately.
Independent Python reviews approved both the harness and execution migration after these boundaries were corrected.

## Real verification and video

Installed MCP plus rendered Chrome exercised headless, playing, paused and mid-run viewer disconnection.
All four runs executed exactly 400 physics steps with the push at step 100.
Final qpos, qvel and ctrl SHA256 values agree exactly across all four runs; watching changes pacing only.
Actual MCP stop closed the execution as cancelled; actual reset closed it as invalidated, and a later controller request succeeded.
A real script attached a controller, advanced 60 steps and stopped it through the existing interface.
A real rover `move` advanced 0.161 m over 0.6 simulation seconds and stopped.
Chrome reported no JavaScript errors.
Transport task cancellation and repeated cancellation are covered by boundary tests; the video demonstrates actual MCP stop interruption.

The actual Chrome recording is 19.23 seconds, H.264, 1920x1080 at 30 fps.
Its verification labels are populated from observed results.
FFmpeg decoded the complete MP4 successfully and the final rendered frame was inspected.
Raw video: `../../../robotforge-evidence/2026-10-22-framework-regression/live/framework-validation.mp4`.
The MP4 is kept outside Git together with its experiment artifacts.

## Checks

| Check | Result |
|---|---|
| Wheel, Ruff lint/format, diff and secret-pattern audit | Passed |
| Final focused boundary and baseline selection | 79 passed, including 51 new cases |
| Complete default selection | 3221 passed, 127 skipped, 0 failures, 0 errors; 3348 selected, 1020.7 s |
| Real Chrome gamepad regression | 8 passed, 47.60 s |
| Installed CLI/MCP/HTTP/WebSocket/render smoke | Passed, 3.4 s |
| Common-record candidate and independent baseline repeat | Seven cases passed each |
| Previous-core behavior comparison | 14 scenarios, 280 identical markers |
| Live viewer-state comparison | Four modes, exact state hashes and step counts |

Source, scripts, tests and dependency files stayed unchanged throughout final verification.
Their inventory SHA256 is `68f1341b91f4277286e6e5ba78d3fd94f4cf6c949d0c5076b140ad270e00aa5a`.
The actual video and committed candidate identify the same active package bytes, `8b3a0326fa163502bf8012f23bd10392bb9c801f570395043b13ec92d0b47b43`.
The tested implementation commit is `62557d2533ea458687c3c93c0e765a9c71485cc1`; the following commit changes verification documentation only.
Machine-readable results are [saved alongside this report](2026-10-22-framework-regression.json).
Raw collection, two complete shard logs/XML files, focused checks, browser results, old-core/candidate records, source hashes and the MP4 are in `../../../robotforge-evidence/2026-10-22-framework-regression/`.

## Limits and next increment

Actual verification used macOS arm64, Python 3.12.13, MuJoCo 3.15.0 and NumPy 2.5.4.
Native Windows and optional RL backends were not rerun.
Mypy, pyright, Bandit and pip-audit were unavailable; no static typing or dependency audit pass is claimed.
Blocking native or plugin work remains cooperatively cancellable.
Different runtime baselines require separate reviewed profiles.
No dependencies, release, deployment or paid Actions were added.

Next, admit remaining pose/gripper and arm execution families one increment at a time against this baseline.
Multi-phase arm flows and wind/fast-only simulation require explicit budget and lifecycle preservation before migration.
Manipulation success-rate tuning remains paused.
