# Runtime boundaries and mobile manipulation, 2026-10-21

This round fixes known runtime boundaries and evaluates complete mobile manipulation jobs with real, pinned Menagerie models in MuJoCo.
Changes remain local and uncommitted.
Machine-readable results and provenance are in [2026-10-21-mobile-results.json](2026-10-21-mobile-results.json).

## Physical results

| Case group | Before | After | Remaining failure |
|---|---:|---:|---|
| Panda drawer, development seeds 8–11 | 0/4 | 4/4 | None in these four cases |
| Panda cross-room fetch, development seed 8 | 1/1 | 1/1 | None |
| Go2 arm drawer, development seed 8 | 1/1 | 1/1 | None |
| ANYmal + UR5e fetch, development seed 8 | 0/1 | 0/1 | Falls while approaching the object |
| Panda drawer, frozen holdout seeds 40–59 | Not rerun | 19/20 | Seed 50: gripper closes without holding the block |
| Panda cross-room fetch, frozen holdout seeds 40–42 | Not rerun | 3/3 | None in this small sample |
| ANYmal + UR5e fetch, frozen holdout seeds 40–42 | Not rerun | 3/3 | None in this small sample |

The fixed development set improved from 2/7 to 6/7.
The frozen holdout set completed 25/26 attempts, with zero refusals excluded from the denominator.
The older 7/20 Panda drawer result used seeds 8–27, so it is historical context, not a paired comparison with the new holdout.
The three ANYmal holdout successes do not establish general reliability; its known seed 8 fall remains reproducible.
The holdout was not used to tune parameters after its results were observed.

The final regression and holdout share source fingerprint `185fbf8de2868d40965b47e3a9fa498cd9a56b3b20c0841df409950f4200706a` on base revision `96763e058d87c8821d3b00a41c061f32ad9cc38d`.
The baseline uses unchanged behavior code from that revision plus the diagnostic benchmark wrapper and runtime-only fixes.
The final batches used two workers; the baseline used one.
Wall times include setup, planning and simulation and are not a controlled speed comparison.
For the held-out Panda drawers, median wall time was 38.6 seconds and maximum 62.0 seconds.

## What changed

- Opening an MJCF successfully detaches the previous editable world and import metadata; a failed compilation preserves the previous world.
- Script joint reads enumerate all hinge and slide joints directly, bypassing the observation display limit.
- Script runs reserve exclusive ownership of the simulation clock, yield during headless stepping, check wall deadlines, and clean up child processes, RPC clients and output pumps on cancellation.
- Script process groups are terminated on POSIX; Windows uses a gated child and a kill-on-close Job Object.
  Output drainage is bounded even when a detached descendant retains stdout.
- Omitted training algorithms reach the NumPy ARS fallback without optional RL packages; explicit PPO/SAC requests keep their existing semantics.
- Trusted world plugins load before coordinator planning; initialization errors become failed run statuses.
- Queued trials, active training and evaluations share one absolute deadline; interrupted evaluation is not ranked as a complete score, and saved unscored policies are labeled.
- Candidate standing poses now rehearse the complete grasp, arm fold and loaded departure at perturbed poses, including alternate grasp orientations.
  Wheeled robots apply the same check to their current pose.
  Each orientation starts with independent IK warm state.

Applying the current-pose perturbation envelope to walkers caused a Go2 regression: failure after 151.6 seconds instead of a previously successful fetch.
That extra current-pose check was restricted to wheeled robots; the final Go2 case succeeds again.
Newly searched walker poses still receive complete-path rehearsal.
This is why the original walker current-stance behavior is preserved.

## Evidence and limits

The local evidence folder is `../robotforge-evidence/2026-10-21/` relative to the repository root.
It contains per-case generated worlds, stage traces, final `qpos`/`qvel`/`ctrl`, raw JSONL, source patches, manifests and verification logs.
Final physics arrays are diagnostic snapshots, not resumable controller checkpoints.
The benchmark defaults personal plugins off, records custom model/plugin overrides, and retains all attempts in its summary.
See [MOBILE.md](../MOBILE.md) for repeatable commands and the frozen case lists.

Python code reviews found no remaining high-priority issue after a real inherited-stdout descendant regression was fixed.
Native Windows execution is still pending; Job Object behavior was checked against [Microsoft's documentation](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects), not certified by a Windows run.
Deliberately detached POSIX descendants are outside automatic process-group cleanup.
Already-running thread work, blocking plugin code and a single blocking environment step remain cooperative cancellation boundaries.
No new dependencies were added.
Wheel build, Ruff lint/format and targeted regression checks passed.
The complete default selection ran 3020 tests: 2891 passed, 127 skipped and two failed.
One old test expected omitted algorithms to require RL extras; it now tests explicit PPO, preserving that error contract.
The other failure came from a module-scoped test leaving `boom.py` behind when tests were reordered; it now cleans up in `finally`.
Both affected files then passed (188 passed, 3 skipped), and deliberately running the script test before the refusal cases passed all 87 tests.
The complete selection was not repeated after these test-only fixes.
The final test changes also passed Python review.

The installed CLI smoke passed: 60 MCP tools, live HTTP/WebSocket physics, rendered PNG, rover movement and a child script.
A separate real MCP session without PyTorch/SB3 omitted `algo`, selected ARS and finished training.
This tiny training run checks the route, not policy quality: it requested 64 steps and took 815 because ARS checks its step budget at iteration boundaries.
The step budget is not a strict cap; the shared wall-clock deadline is checked inside rollouts.
Actual Chrome loaded the saved before/after seed 8 and failed seed 50 physics states with zero JavaScript errors.
Visual inspection confirmed the successful seed 8 block on the destination table and seed 50 still at the drawer.
These are final-state inspections, not recorded trajectory replays.
Logs, JUnit XML, screenshots and a concise `verification.json` are in the local evidence folder.
Mypy/Pyright and Bandit/pip-audit were unavailable in this environment and were not run; the changed-file credential scan found no matches.

## Next work

1. Diagnose and recover from the held-out drawer miss using the actual object pose after contact, with bounded regrasp attempts and a newly frozen evaluation set.
2. Separate ANYmal approach stability from loaded-clearance planning before expanding claims about legged mobile manipulation.
3. Run native Windows verification, including descendant cleanup and the pre-existing conveyor test failure, before treating A8 as current.
4. Define and test strict versus whole-iteration training step budgets; the current ARS budget can overshoot a small request substantially.
