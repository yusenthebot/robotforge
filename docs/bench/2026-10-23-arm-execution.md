# Arm planning and multi-phase execution validation

Date: 2026-10-23, macOS arm64.
Previous core: `6ac862e8bc41f01645ba9ca12d7690e66a5abb0a`.
Tested source SHA256: `a2f248df09fbbd4284b8591fc39fa3d5752f1d6e75521eba6abfefd66ce4102d`.

MCP `move_arm`, `pick_and_place` and `operate` now share recorded owned execution with the existing kernel.
Script arm planning uses tracked workers and one controller-family ledger under its original cumulative user budget.
No controller, planning objective, grasp heuristic, reward or task success criterion changed.

## Measured checks

| Check | Result |
| --- | --- |
| Complete default suite | 3285 passed, 127 skipped; all 3412 selected, 830.2 s |
| Focused execution/script boundaries | 117 passed, including 37 new cases |
| Existing canonical framework baseline | 7 passed; committed capture SHA unchanged |
| Previous-core entry comparison | 13 completed recipes exactly equal; one documented refusal cleanup difference |
| Public MCP definitions | All 60 full definitions identical |
| Actual Chrome gamepad regression | 8 passed |
| Wheel, Ruff, installed CLI/official MCP smoke | Passed |
| Independent Python review | Approved |

The original baseline SHA256 remains `e6c82b5c364b47273a8c45ee85441f37b50e1676ecd7cff5bbf2f72d4f88852b`.
Default collection was split into two complete shards without the landing helper's slow-test cutoff.
Every default-selected node ran; default slow/browser exclusions and optional-backend skips retain their previous meaning.
All runtime source, tests, scripts and dependency files stayed byte-identical during final validation.

## Fixed old-core captures

The external frozen harness ran against an isolated old-core checkout and then the candidate in serial bounded subprocesses.
Its SHA256 is `c3fa77d383ee7af4ecd6343e30fa1c62dc92fa20ff49953688ffff8f9e0b4659`.
Complete tool definitions, initial arrays, dispatcher step counts, public responses, final qpos/qvel/ctrl and controller slots were compared.
The 13 completed recipes cover fixed/queued movement, pick, hold, pour, handover, catch, approach/carry, fetch, drawer operation, expected force stall, valve regrasp and mobile visit.
These differential captures use the existing offline test fixtures, including a Panda collision stand-in without visual meshes.
The expected force stall remains unsuccessful at the same force and displacement; this migration does not improve that outcome.

The remaining moving-carrier recipe settles its base for the same 670 steps and then refuses the same unreachable target.
Error text, final physics and stopped-base state match.
The candidate removes the newly attached empty arm controller, while the old core left it idle.
This intentional cleanup difference is explicit and does not count as successful mounted-arm motion.
Terminal-state equality in these fixtures does not prove equality of every intermediate trajectory or general success rates.

## Actual application and video

The installed CLI ran an official MCP stdio client and a real Chrome viewer connected to MuJoCo.
The final video uses the complete cached MuJoCo Menagerie Panda model, with compiled mesh presence verified, rather than the offline collision stand-in.
Its handover uses 13,270 steps, whereas the separate fixture differential uses 14,110; these distinct model variants are not claimed to match each other.
Movement, handover and mobile operation were each run headlessly, playing, paused and with the viewer disconnected mid-run.
Initial full integration-state hash, public response, exact steps and final qpos/qvel/ctrl hashes match for every viewing mode.
Watching changes pacing only; a paused viewer still permits an explicit request to complete.
Stop during handover records cancelled; reset during mobile operation records invalidated; later script calls work.
An earlier recording with invisible fixture arms was retained separately and replaced after visual inspection.
The real script uses `Sim.move_arm`, `Robot.arm.move_to` and `Gripper.close` with their existing return types and budget.
Browser errors are empty.

[Actual test video](../../../robotforge-evidence/2026-10-23-arm-execution/live/arm-multiphase-validation.mp4) (141.80 s, 1920×1080, 30 fps).
The MP4 was fully decoded and representative frames were visually inspected.
It records the actual application with evidence labels, not a generated motion illustration.
Full hashes, phase records and video metadata are in [the JSON report](2026-10-23-arm-execution.json).
Raw evidence stays outside the repository at `../robotforge-evidence/2026-10-23-arm-execution/`.

## Boundaries and limitations

Original carrier wait and main/settle phase caps remain.
Headless/watched wall allowances sum the applicable legacy phase allowances, anchored to initial admission; snapshot, metadata and planning are included and later phases never renew them.
Legacy response duration still excludes carrier wait, while the execution record charges every actual step.
Settling retains old nested elapsed-time rounding; the strict cumulative sim/step limit still clips any extra batch.
Scripts retain their existing user-supplied cumulative limits.

The family ledger trusts explicit synchronous mutations and already-owned controller actors, follows hidden descendants, and preserves foreign replacements.
Reset restores owned temporary model settings before their old IDs become stale.
Repeated cancellation waits for actual native workers and child/process cleanup before releasing execution ownership.
Blocking native/plugin calls remain cooperative and live planning retains its existing session lock.
Snapshots do not serialize arbitrary Python controller closures or RNG runtime state.

Validation is local macOS arm64; native Windows and optional RL backends were not rerun.
`mypy`, `pyright`, `bandit` and `pip-audit` are unavailable in the project environment; their absence is recorded, not counted as a passing check.
No new dependencies, release, deployment, paid Actions or task success-rate tuning occurred.
