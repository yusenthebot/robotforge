# Task and algorithm contracts: 2026-10-22

This round unifies discovery, task identity, parameter validation and adapter selection across MCP, scripts, the viewer and training.
It does not tune control algorithms, rewards, drawer behavior or manipulation success.
Earlier runtime, mobile-baseline, execution-kernel and capability work is included in the same branch and retains its separate reports.

## Changes and compatibility

- Built-in, saved-world and plugin tasks use the shared registry and binder.
  Plugin schema metadata is additive; legacy factories retain unknown JSON options and copied defaults.
  NumPy Python settings remain supported.
- Control method schemas and validation derive from existing adapters.
  Training algorithms retain the existing common Algorithm/Hyper registry and validate selected-algorithm settings before allocating an environment.
- Registered identity is separate from implementation Task.name.
  Saved policies restore task parameters before explicit overrides, including wrappers of built-ins and reward tasks named `custom` internally.
- Script controller attachment shares execution admission, ownership and cancellation cleanup.
  Viewer discovery stays lazy and caches both task and algorithm registration changes.
  Existing mode menus remain available.
- All 60 existing MCP tool definitions match the prior snapshot exactly.
  Response additions preserve old fields.
- Independent review found and corrected wrong-robot saved-task attachment, NumPy boundary compatibility, specialized adapter checks, script worker cancellation ownership and adaptive contact comparison duration.

## Real execution

An installed `robotforge mcp` process ran with an isolated home.
A generated world plugin supplied a schema-bound Reach wrapper, a reward task without an explicit task_name and a small training algorithm using TrainContext.
Its code was written through the existing plugin-authoring path; imported plugin trust rules were not changed.

MCP, script-process and HTTP scene catalogs were compared for semantic equality.
Plugin IK/PD control and comparisons ran on the real compiled model.
Invalid task settings were refused before attachment.

A separate coordinator and worker trained each of two policy runs: the Reach wrapper with the plugin algorithm, and the reward task with built-in ARS.
Both completed and saved policies.
The arm policy restored the nondefault goal `[0.19, 0.09, 0.24]` and tolerance `0.035` in MCP, script and viewer WebSocket playback.
Script `run_controller` and legacy `run_policy` both ran, advancing 0.24 simulated seconds.
The reward task replayed under its registered identity, with `implementation_task: custom`.
These short runs verify contracts and playback, not policy quality.

Headless Chrome rendered the real viewer, read the shared catalog and clicked the Training Play button.
The Training Play check began while paused and observed the successful reply toast and time advancing past 0.3 seconds.
The viewer also accepted its `task_run` policy request through the real WebSocket path and restored the saved arm goal and tolerance.
Browser console errors were checked.
Plugin source metadata reload was verified through MCP; lazy reload/removal and two-viewer update handling have separate regression tests.

Fourteen built-in controller scenarios were replayed against `b0522a0`: reach IK, PD, Cartesian and impedance/OSC; cartpole PD/LQR/MPC/MPPI; stand hold/MPC/MPPI; and rover PD.
All 280 qpos/qvel/ctrl byte-hash markers matched.
The standard installed CLI/MCP/HTTP/WebSocket/rendering smoke passed in 3.5 seconds.

## Checks and evidence

- Wheel build, Ruff lint/format and JavaScript syntax passed.
- The 107 new contract tests passed together.
  They cover schema/default isolation, aliases, model/actuator ownership, old/new policy metadata, worker preflight/cleanup, residual guards, cancellation, reload and contact duration.
- Complete default selection: 3072 passed, 127 skipped, zero failures (3199 cases, 718.1 s; both shards green).
- The existing real Chrome gamepad suite passed 8 cases in 33.03 seconds.
- Source and tests stayed unchanged during the complete run; fingerprint `e41e1615fa47eeb596003f63d8f176bed894e3ad670e4cc70d70a5adb83f49f3`.
- Independent final review found no remaining concrete P1/P2 issues after the fixes.

Raw evidence lives in `../../../robotforge-evidence/2026-10-22-task-contract/`: complete test collection and shard logs/XML, source hashes, tool definitions, training records and saved policies, script output, before/after physics hashes and Chrome screenshot.
The local evidence directory is not committed.
Compact results are kept in the accompanying JSON.

Native Windows behavior and optional RL runtimes were not validated on this Mac.
mypy/pyright, Bandit and pip-audit are not installed in the project environment; no dependency was added for them.
This round did not trigger paid Actions runs or publish a release.
Existing mobile holdout results were not rerun or tuned.
