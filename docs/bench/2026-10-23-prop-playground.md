# Build and play with local props

This increment extends direct robot creation into a creative scene loop: pick a prop, preview its size/orientation, place, interact through real physics and undo the latest addition.
Build in the existing maps/terrain menu offers five server-owned presets: ball, wooden block, cylinder pin, fixed barrier and table.
Three sizes and quarter-turn buttons work on desktop and touch screens; the existing four robot starters remain available.
See [the play guide](../CREATIVE_PLAY.md) for controls, dimensions and motion/reset semantics.

## Implementation and compatibility

The browser sends a strict preset, size enum, integer quarter turn, finite bounded surface coordinates and current scene version.
Shape, dimensions, color, mass and fixed status remain server owned.
The catalog gains object_presets and successful additions gain an opaque undo receipt; existing world formats, MCP definitions and script APIs remain compatible.
Surface height translates to primitive center height or table base height as appropriate, while ground placements keep following terrain through their XY representation.
Dynamic object tops permit stacking; robot bodies remain excluded as support surfaces.
At most one temporary wireframe mesh shows the size envelope outside the pickable world.
Changing size disposes the previous geometry, rotation reuses it, and cancel/success removes it.
There is a small active-preview drawing cost, with no new rendering pass, light, shadow caster or texture asset.

Placement uses existing edit/execution ownership, private candidate compilation and saved/live collision preflight, checked again before commit.
Explicit shape distances cover fixed-fixed pairs omitted by ordinary contact generation.
Coincident boxes and cylinders can return zero through native CCD, so near-zero candidate queries use libccd with option flags restored in finally.
The committed simulation's collision settings remain unchanged.

Undo add is a one-use inverse edit, not a pop from the generic history stack.
The server retains one receipt bound to store identity, version/generation, exact entity entry and saved-file hash.
It removes only that entity and preserves the existing carry contract for other free-body poses, joint positions and control targets, including entities already in contact.
This is not a full integration-state restore.
Edits restart motion with zero velocities; uninterrupted dynamic state is not promised.
Reset, other edits, world/lesson changes and external saved-file changes invalidate the receipt.
Compilation failure preserves a valid receipt for retry.
WorldStore also detects disk edits made during compilation before writing its own result.
This is optimistic conflict detection, not an atomic transaction with arbitrary external editors.

## Verification status

Baseline commit: 86305fb489b680c1cc32709a0e0a1415d11b6f1f.
Frozen candidate runtime SHA256: ed392c58959f2d1d459ce0ed898835672bc0733d97948265d99c5bdecce67016.
131 focused core/store/carry/parser cases pass.
24 final actual Chrome cases pass: eight object cases, five existing creation cases, eight gamepad cases and three lighting cases.
The eight object cases include desktop/phone stacking/tabletop placement and imported fixed-robot support exclusion.
Independent Python boundary and frontend/source reviews approve the final implementation; the external evidence audit verifies all 6579 indexed files.
Wheel, Ruff and all 21 JavaScript parser cases pass.
The complete default selection passes: 3400 passed, 127 skipped, 82 deselected, zero failures/errors over 3527 selected cases in 1145.45 seconds.
All seven unchanged canonical framework cases and the actual installed CLI/viewer/MCP smoke pass, including 60 MCP tools and a real script process.
Optional mypy, pyright, Bandit and pip-audit are unavailable locally; native Windows and physical phone hardware remain unverified.

All 11 existing install/viewer budgets pass without relaxation: 3642 KB wheel, 163 MB installed, 42 packages, 0.70 s CLI start-to-frame, about 60 FPS at DPR 2, 2.50/7.50 ms standard/asset render samples, 132 MB server peak and 5.20% idle CPU.
The 7.50 ms complex-asset sample is retained; it is below the existing 12 ms limit and is not described as a speed improvement.
Import, physics-throughput and MCP timing budget sections are outside this partial budget run.

The initial native browser attempts caught test setup issues: localhost requests went through the system proxy, the native size picker did not advance as the harness assumed, and a collision probe clicked a robot surface that correctly refused placement before dispatch.
Localhost proxy bypass and an open ground point whose prop footprint overlaps the rover correct those probes.
Three visible size buttons replace the native picker and preserve keyboard button activation.
Phone stacking uses a close view rather than targeting a tiny block in a distant wide shot.
Final source review also caught fixed robot bases bypassing a dynamic-body-only support check.
The corrected rule excludes robot ownership independently of motion state, and a real imported fixed-body browser case guards it.
Earlier native-v1 and A1/B1/A2/B2 captures remain explicitly pre-fix evidence; accepted final captures use the corrected source.

## Accepted native evidence

The accepted desktop recording is 39.70 seconds at 1440 x 900; the phone recording is 40.90 seconds at 390 x 844, both encoded at 30 FPS from chronological Chrome screencast frames.
The encoding rate is separate from application rendering or physics throughput.
Both videos fully decode; raw screenshots and contact sheets extracted from the actual MP4 files were inspected.
Both retain creation, scale/rotation preview, loose-block stacking, tabletop block/arm, arm controls, occupied-table rejection and retry, one-use undo and retry, terrain/floor/map changes and reset.
Stacked block center height is 0.3375 m over a 0.27 m support, tabletop block center is 0.8175 m over 0.75 m, and the arm base is at 0.75 m.
The tested undo occurs while paused and preserves complete qpos/qvel/ctrl arrays exactly; the general edit contract still restarts velocities.

Desktop W drive moves the ball 0.889806 m and the pin 0.545388 m, with actual named rover/prop contacts recorded.
Phone native touch acquires both rover chassis bodies and moves the ball 0.522893 m and pin 0.571355 m.
Contact counts are nominal 20 ms samples containing a pair, rather than independent collisions.
Phone force-drag tips both rovers; this is shown as observed behavior, rather than wheel/joystick control.
The phone evidence is Chrome touch emulation on this Mac, with close diagnostic framing that crops some surroundings; physical handset performance is unverified.

An earlier final-phone attempt starts outside the viewport at CSS x = -12.79054, acquires no rover and fails the push observation.
Its 31.77-second partial video, state and screenshot remain in the evidence.
One camera-only retry fits both endpoints inside the viewport and checks the acquired body; it changes no physics, controller, drag strength or gesture distance.
The failed attempt is not counted as a product pass.

Final videos relative to the evidence root: native-final/desktop.mp4 and native-final-phone/phone.mp4.
Desktop SHA256: c00e7a7ef0e3e94fb7ce0257b99ed855aa1a258583d576fd198add3543592bb2.
Phone SHA256: 20bc1e0f86acd2d6363a733672fd2813a6c72f7fdde366b7b80f3ab38b4cacca.

## Matched loading and rendering

Final-A1, final-B1, final-A2 and final-B2 run serially with frozen empty and rover/prop worlds at desktop/phone sizes, DPR 1.
World and initial physics hashes, camera, viewport, Chrome version and drawing buffer match; paused state does not drift and the source hashes remain stable.
All 16 captures have no JavaScript or WebGL errors.
Scene descriptions differ only by the additive object preset catalog.
Shared materials and lighting from the preceding increments are retained.
The server is already running for navigation measurements, and fresh profiles do not flush machine-wide GPU caches.
Each render measurement contains 40 synchronous render and one-pixel readback samples, distinct from active physics throughput.

The payload increases by 9080 encoded HTTP bytes and one resource, and the wheel grows by 5511 bytes.
Corresponding render workloads remain unchanged: empty 2 calls/4 triangles and props 34 calls/14566 triangles, with five textures.
Two observations per side provide limited evidence; slower samples remain visible in the companion JSON and external report.

| View | Fresh frame before / after, ms | Warm frame before / after, ms | Render median before / after, ms | Render p95 before / after, ms |
| --- | --- | --- | --- | --- |
| Empty desktop | 176.35 / 178.90 | 85.50 / 102.95 | 1.40 / 1.35 | 2.05 / 2.20 |
| Empty phone | 187.40 / 198.80 | 84.90 / 86.30 | 0.75 / 0.80 | 1.20 / 1.60 |
| Props desktop | 323.30 / 301.45 | 181.95 / 218.10 | 2.20 / 2.075 | 2.95 / 2.80 |
| Props phone | 271.35 / 283.15 | 215.80 / 163.15 | 1.25 / 1.60 | 1.80 / 2.35 |

The table reports medians of the two observations on each side.
Performance is mixed: desktop warm navigation is about 20% slower between two-sample medians, while phone props warm navigation is lower.
Props-phone render median increases by 0.35 ms and p95 by 0.55 ms in these samples.
Paused RAF stays near 60 FPS, with one candidate empty-phone sample at 59.51 FPS.
No overall speed or memory-efficiency improvement is claimed.
Chrome-process-tree RSS sums are retained as non-unique resident-set measurements and must not be added to the overlapping harness process tree.

## Reproduction

```sh
uv run --frozen pytest -q tests/test_viewer_objects.py tests/test_viewer_creation.py tests/test_store.py tests/test_carry.py tests/test_viewer_js.py
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost uv run --frozen pytest -q -m browser tests/test_viewer_objects_browser.py tests/test_viewer_creation_browser.py tests/test_viewer_gamepad.py tests/test_viewer_lighting_browser.py
uv run --frozen pytest -q
uv run --frozen python scripts/framework_regression.py --baseline docs/bench/framework-regression-v1.json --out /tmp/robotforge-prop-framework
uv run --frozen python scripts/bench.py --only install viewer --save --out /tmp/robotforge-prop-budget
uv run --frozen python scripts/smoke.py
```

External raw evidence stays under ../robotforge-evidence/2026-10-23-prop-playground relative to the repository root.
The frozen comparison includes identical empty and rover/prop worlds at desktop and phone sizes.
No individual action-success tuning, new dependency, paid Actions, release or deployment is part of this increment.
The broader visual/playability goal remains active.

## Reproducibility and remaining scope

The companion JSON links a completed unified experiment record with immutable videos, manifests, source/tests, raw result logs and their SHA256 values.
The full raw frame set remains outside Git and is indexed by final-artifacts.json; it is not duplicated into the experiment store.
Candidate-source.patch reconstructs the implementation and tests on baseline 86305fb, and git apply --check passes in the detached baseline checkout.
To recreate that checkout after cleanup, use git worktree add --detach ../robotforge-prop-baseline 86305fb489b680c1cc32709a0e0a1415d11b6f1f.
The accepted external harness copies and their helper scripts are retained alongside the immutable record; use new output directories for repeat runs.
The current four-tab panel retains the existing three-column layout, so Modes wraps onto a second row; all four tabs remain usable in the recorded phone flow.
Further interface refinement, creative editing and wood temporal stability remain within the active broader goal.
