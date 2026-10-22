# Pose and gripper execution migration, 2026-10-22

MCP `pose` and `gripper` now share the execution kernel and recorded lifecycle with controller measurements.
The complete definitions of all 60 MCP tools and their existing result fields remain compatible; the execution reference is additive.
Control algorithms, queued behavior and manipulation success rates are unchanged.
Implementation and compatibility details are in [EXECUTION.md](../EXECUTION.md).

## What changed

The former controller-only record lifecycle is extracted into `experiments.execution.RecordedExecution`.
`control.execution.ControllerExecution` remains as its task-specific compatibility subclass.
All three adapters now share admission, cumulative wall deadlines, generation checks, actual step accounting, cancellation cleanup and record persistence.
Pose retains its 20-second simulation cap and gripper its five-second cap.
Snapshot and metadata preparation consume the original deadline, checked atomically with behavior creation under the session lock.

The existing behavior factories still select and configure the same controllers.
An already active ArmMotion is reused, and a gripper waypoint remains behind its queued work.
Normal completion, the five-second cap and wall exhaustion retain the attached controller.
The record exposes `behavior_done` separately from completion of the bounded request.
Final metric/artifact storage failure preserves an applied result and reports `execution.recording_error`.

Cancellation cleans only the exact still-owned endpoint and never a replacement or reset-world controller.
For a mounted arm it leaves a pre-existing moving carrier alone, and position servos retain their latest targets.
Humanoid stop intentionally preserves the existing zero-speed balancing controller and its pending/current pose.
Cancelling the owned pose request therefore does not erase that pose; later physics stepping may continue it.
This inherited stop behavior is explicit in tests and is not a new pose-abort algorithm.
Root-resource stop matching remains compatible: a valid endpoint on the same carrier interrupts the run, while cleanup still targets only its commanded endpoint.

Scripts retain their existing single lease, cumulative budgets and bool/None results.
Pose/gripper completion checks now reject a different controller identity before reading it, and read-only polling cannot adopt that replacement for cleanup.
Viewer arm jogging, demonstration replay and the remaining arm/manipulation adapters retain their existing execution paths.

## Unchanged baseline and old-core comparison

All seven cases of the committed `framework-regression-v1` baseline pass on the committed candidate.
The baseline JSON remains byte-for-byte unchanged, SHA256 `e6c82b5c364b47273a8c45ee85441f37b50e1676ecd7cff5bbf2f72d4f88852b`.
Its original core/harness capture provenance remains documented in [FRAMEWORK_REGRESSION.md](../FRAMEWORK_REGRESSION.md).

An additional fixed ten-case capture ran on clean previous core `fac7d71e3eb904c5ae19fab2257b30dba80918d2` before source edits.
The identical external harness then ran on the committed candidate.
Cases cover wave, squat, arms_up, bow, look_around and stand; gripper open/close; gripper behind an existing queue; and gripper on a moving carrier.
All legacy public results, actual step counts and final qpos/qvel/ctrl hashes agree exactly in every case.
All 60 complete tool definitions also agree exactly.
Both sides retain common experiment records, captured environments, final arrays and the same harness SHA.
These scoped captures supplement the canonical baseline without changing its inventory or blessing a new golden result.
Fourteen additional built-in control scenarios retain all 280 physics markers against `fac7d71`.

## Actual installed MCP, Chrome and video

The installed `robotforge mcp` process was driven by the official MCP client and rendered in actual Chrome.
Both pose and gripper were tested headless, while playing, while paused and through mid-run viewer disconnection.
All four pose runs use exactly 2260 steps and all four gripper runs exactly 260 steps.
For each tool, initial integration-state hashes and final qpos/qvel/ctrl hashes agree strictly across all modes.
The displayed `shown_live` values are false/true/false/true.

To make starting states comparable, the test pauses and resets before each call and uses the supported minimum viewer speed until ownership is observed, then restores normal speed.
The execution record verifies that no physics drift occurred before admission.
The extra verification control socket is also closed during disconnection, so no hidden watcher keeps live pacing active.

Actual stop interrupts pose and gripper measurements with cancelled records.
Actual reset invalidates an in-flight gripper, and a subsequent open request succeeds.
A real script exercises `Robot.pose`, `Sim.gripper` and `Gripper.open`, preserving bool and None results.
A gripper command on a continuously moving mounted-arm rover retains nonzero base wheel commands without taking ownership of its carrier.
Chrome reports no JavaScript errors.
Task cancellation, generation replacement, same-generation controller replacement and storage failures have separate boundary tests.

The actual test recording is 44.6 seconds, H.264, 1920x1080 at 30 fps, captured from 2648 real Chrome screencast frames.
Verification labels are populated from observed results.
FFmpeg decoded the complete MP4, and the pose and final rendered frames were visually inspected.
The recording was shown inline to the user and is retained outside Git at `../../../robotforge-evidence/2026-10-22-pose-gripper/live/pose-gripper-validation.mp4`.

## Checks and evidence

| Check | Result |
|---|---|
| Wheel, Ruff lint/format, diff and secret-pattern audit | Passed |
| Controller lifecycle extraction selection | 12 passed |
| Final focused execution/script/behavior/framework selection | 166 passed, including 27 new boundary cases |
| Complete default selection | 3248 passed, 127 skipped, zero failures/errors; 3375 selected, 720.5 s |
| Real Chrome gamepad regression | 8 passed, 34.46 s |
| Installed CLI/MCP/HTTP/WebSocket/render smoke | Passed, 3.7 s |
| Committed seven-case framework candidate | Passed |
| Previous-core entry compatibility | 10 cases and 60 complete tool definitions identical |
| Previous-core control behavior | 14 scenarios and 280 markers identical |
| Actual viewing-state comparison | Both tools, four modes each, exact initial/final state hashes and step counts |

Independent Python review approved after correcting two findings: expired snapshot preparation could still issue a command, and a final metric write failure could change a successful command into a failure.
Both now have regressions for both tools.

The implementation tested is `86c939b17277e050ba010a002acf21ca1b84cd76`; the following commit changes verification documentation only.
Source, tests, scripts and dependency files stayed unchanged throughout final verification.
Their complete inventory SHA256 is `51d09128bd2e4a020b447afcead9fd0cff10546fbee641a2a25d7c671cb90cb5`.
The real recording, committed framework candidate and committed ten-case capture identify the same active package source, `e73ed7e7e65482a115e312475356dd331aaf05723aac083d4e241bc66b7fcd74`.
The [machine-readable report](2026-10-22-pose-gripper.json) records actual counts, source/dependency provenance and evidence references.
Raw collection, both full shard logs/XML, focused results, old/new records, browser output, source hashes and the MP4 are in `../../../robotforge-evidence/2026-10-22-pose-gripper/`.

Actual verification is macOS arm64, Python 3.12.13, MuJoCo 3.15.0 and NumPy 2.5.4.
Native Windows and optional RL backends were not rerun.
Mypy, pyright, Bandit and pip-audit remain unavailable; no pass for those tools is claimed.
Blocking native/plugin work is still cooperatively cancellable, and snapshots do not capture arbitrary Python controller/RNG state.
No dependencies, release, deployment or paid Actions were added.
Next migrate arm planning and multi-phase adapters separately, preserving carrier settling, phase budgets and fast-only/wind behavior.
Individual manipulation success-rate tuning remains paused.
