# Framework regression baseline

The fixed `framework-regression-v1` suite uses the common experiment store to compare framework behavior across source revisions.
It does not rank manipulation success or tune controllers, rewards or learned task quality.
The committed baseline is [framework-regression-v1.json](bench/framework-regression-v1.json).

## Run a candidate

```sh
uv run --frozen python scripts/framework_regression.py \
  --out /path/to/empty/candidate-directory \
  --baseline docs/bench/framework-regression-v1.json
```

The output contains `manifest.json`, `report.json`, terminal experiment records and verified artifacts.
Exit 0 means pass, 1 means regression or invalid evidence, and 2 means unsupported runtime.
The existing output directory must be empty; the runner never overwrites another run or updates a baseline during comparison.
Capture is a separate explicit action:

```sh
uv run --frozen python scripts/framework_regression.py \
  --out /path/to/empty/capture-directory --capture
```

Capture alone does not approve a new committed baseline.
Review input, runtime and physical output changes before deliberately replacing the baseline file.
Baseline JSON is inert comparison data; its recorded paths are never restored, executed or imported.

## What is compared

| Case | Evidence |
|---|---|
| Balance comparison | Seed 7, LQR/PD, identical start and push; physical metrics and final integration-state array hashes |
| Reach control | Fixed IK target and 25 physics steps; actual task settings and qpos/qvel/ctrl hashes |
| Reach replay | Restore the recorded attachment in a private simulation; strict equality and explicit replay tolerance |
| NumPy ARS | Seed 7, short Balance episodes; requested/actual steps, ranking protocol and policy array hashes |
| Policy lineage | Replacement with another seed preserves old archived policy bytes and matching segment origins |
| Snapshot restore | Complete integration state, applied forces and ten subsequent steps agree exactly |
| MCP controller | Actual MCP dispatcher, 60 tool definitions, 40 physics steps and push at step 20 |

Each case has a stable ID and a versioned input signature.
Comparison requires the exact case inventory, complete outputs and passing case status.
Physical arrays are identified by dtype, shape and array bytes, rather than NPZ container metadata.
Wall timings, UUIDs, output locations and source revision changes do not count as physical regressions.
Python, platform, NumPy, MuJoCo and plugin provenance must match; incompatible runtimes are unsupported, not passed.
The runner refuses already loaded plugins and active MuJoCo callbacks instead of silently producing an unaudited baseline.

`ExperimentRecord.fingerprint` includes code version and is deliberately not used as a cross-version equality test.
The report retains both source identities and explicitly reports source changes.
Every candidate record and every declared artifact is validated under the candidate's own output directory, including content SHA, terminal status, recorded outputs and input signature.
Missing/new/duplicate cases, invalid schema, unavailable/nonfinite metrics, altered input signatures, changed physics and corrupt evidence fail closed.
A source change cannot automatically bless changed physical outputs.

## Origin and scope

The initial baseline ran the new harness against the previous core at `aae92839f5a8c91f4e143d4c90e893fa808309a2` in an isolated worktree.
Only the three new harness files were untracked there, so its recorded `git_dirty=true` is intentional.
The recorded active source hash identifies that exact old-core-plus-harness capture.
A second fresh process repeated all seven cases before the baseline was copied unchanged into the repository.
The migrated candidate was run independently and compared against that old-core capture.

The MCP controller case exercises the newly migrated `run_controller` entry point.
Execution ownership, cumulative deadlines, repeated cancellation, reset/replacement and actuator cleanup have additional boundary tests.
Real installed MCP and Chrome validation also compares headless, playing, paused and disconnected viewing states with exact step counts and final-state hashes.
Those live checks and the actual test video are documented in the [verification report](bench/2026-10-22-framework-regression.md).

The default test suite, existing browser regressions and installed CLI smoke remain required checks for changes to execution adapters.
This small acceptance suite supplements them; it does not establish Windows or optional backend compatibility.
Remaining legacy adapters retain admission protection and should migrate in separate increments.

The following increment migrated MCP pose and gripper without changing this suite or its committed capture.
It also compares ten fixed headless MCP scenarios against the preceding `fac7d71` core and tests the new ownership boundaries.
Those additional captures supplement the seven-case baseline rather than silently changing its inventory.
Actual Chrome and script checks, including the test video, are recorded in [the pose/gripper report](bench/2026-10-22-pose-gripper.md).

The arm planning/multi-phase increment also retains this original seven-case capture unchanged.
Fourteen frozen old-core MCP recipes cover queued movement, carrier waiting, pick/hold/pour, handover, catch, approach/carry, fetch, direct operation, force stall, valve regrasp and visit.
Thirteen completed cases match exact step counts, public results and terminal qpos/qvel/ctrl; the remaining unreachable mounted request retains its refusal and physics while cleaning its empty controller.
These captures verify fixed outcomes and terminal state, not every intermediate trajectory or general success rates.
Actual MCP/Chrome checks compare initial integration state, public results, exact steps and terminal arrays across headless, playing, paused and disconnected modes for movement, handover and mobile operation.
See [the arm execution report](bench/2026-10-23-arm-execution.md) for measured results, evidence provenance, boundary tests and the actual test video.
