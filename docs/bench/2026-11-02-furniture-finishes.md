# Furniture surface finishes (2026-11-02)

Tables and bins now use the existing Surface control, alongside boxes, balls and cylinders.
Single and mixed selections support Original, Plastic, Wood and Steel, with one application and latest Undo.
The optional `surface_material` field stays separate from physical `material`; legacy defaults, payloads and arrangement format remain compatible.
Table tops, legs and aprons share the chosen finish while rubber feet keep their original material.
Bins retain their existing shell geometry and fixed/free/yaw transforms; Original restores sink/bowl/dish name-based defaults.
Existing authored colors, physics, movement admission and action success remain unchanged.
Applications preserve current poses and controls, reset motion/time and pause, following the existing finish-edit contract.

## Scope and source

The baseline is the retained PR77 tree, identical to merged `d879706876fce9b29f012aae836fea1ee8864716`.
The candidate is the isolated `robotforge-furniture-finishes` checkout, frozen at 814 source/test/script/manifest files in `candidate-source-v1.json`, SHA `044fb272cae1ccc72c1617a4206ef195e5128e52cb6c792c71123d2484fc3847`.
Evidence is retained at `/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-furniture-finishes`; the [JSON record](2026-11-02-furniture-finishes.json) pins exact inputs, receipts and outputs.
Every bounded execution rechecked the original 13 WIP files, retained accepted/rejected checkouts, the 812-file baseline and 78 original native-model inputs.
All owned server/browser/encoder jobs finished or were reaped; no model download or staging fallback occurred.
No shader, texture file, dependency, polling loop, physics setting or performance budget was added or relaxed.

## Verification

- 453 scoped compiler, handler, arrangement, group copy/move/carried-selection and material cases passed.
- 2 new complete Chrome journeys passed at 1280px English and 320px Chinese, including actual rendered materials, rubber feet, Original, no-op, Undo, reload and saved-arrangement insertion.
- 19 existing Chrome surface cases passed, retaining late replies, reset, disconnect and display-wait boundaries.
- 60 visual/physics and actual GPU surface cases passed, including the original asset shader checks.
- Total: 534 distinct scoped pytest passes, without counting the seven same-runtime framework comparisons or 24 fresh-installed compiler cases as pytest cases.
- The original viewer/WebSocket/MCP `smoke.py` passed from a fresh locked offline installation, along with installed carried-selection and furniture batch/Undo checks.
- All 430 product payload files match source, built wheel and fresh installation byte-for-byte; Ruff lint/format and diff checks pass.

This is scoped local macOS acceptance, not a full-suite, Windows or remote-CI result.
Bandit, pip-audit and mypy were unavailable in the existing interpreter; no new development dependency was installed.
After the user's unlock reply, both native desktop inventory and direct Chrome access still reported a locked Mac; native OS/browser zoom remains unverified.

## Original-model evidence

Actual Malo workshop and Go2 warehouse captures retain the original robots, terrain and sensors, with the same declared furniture fixtures and cameras as PR77.
Baseline and candidate authored-world bytes match exactly.
Each world has two baseline and eleven candidate native 1280 by 800 diagnostic frames, for 26 frames and 18 correlated phase-aligned full-RGBA comparisons.
Initial baseline/candidate hashes match for all 481 model arrays in both worlds.
During intentional finish edits, 47 specified physical, geometry and base-color arrays remain exact; visual material IDs are allowed to change.
The latest qpos stays within absolute 1e-12, controls stay exact, velocities/time reset and the simulation remains paused.
Original, Undo and final restored views match the corresponding complete framebuffers exactly; camera comparisons use the established absolute 1e-9 tolerance.
Diagnostic capture temporarily hides only the source-identified robot selection ring and restores it; ordinary UI samples retain normal overlays and sensor panels.

Root and independent ordinary-view inspection agree that Steel changes the bins visibly while preserving their authored gray/red hues.
Wood on the red tables is subtle or neutral at this distance, and mixed Plastic is close to the initial appearance.
These observations establish useful furniture finish customization, without establishing improved default appearance, polished-metal realism or a general visual-quality gain.
The renderer tests separately verify the actual Wood/Metal/Shell bindings and preserved rubber feet, rather than relying on saved metadata alone.

## Cost

The wheel is 4,151,883 bytes, 446 bytes above the 4,151,437-byte baseline, with 42,421 bytes remaining under the unchanged 4 MiB limit.
Locked installation is 164.0744 MB across 42 packages, with 52.5149 MB of listed platform wheel downloads.
All four original installation budgets and all seven viewer budgets in each fixed A-B-B-A run pass; unselected budget categories are recorded as `not run`.

| Run | Cold start s | FPS | Frame ms | Asset FPS | Asset frame ms | Dog peak MB | Idle CPU % |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 baseline | 0.9133 | 60.0491 | 1.5 | 60.2069 | 4.0 | 144.1 | 4.8 |
| B1 candidate | 0.9374 | 60.1009 | 1.6 | 60.2590 | 4.1 | 143.2 | 5.0 |
| B2 candidate | 0.9283 | 60.1069 | 1.5 | 60.1369 | 2.8 | 144.0 | 4.6 |
| A3 baseline | 0.9260 | 60.3408 | 1.6 | 59.9161 | 4.0 | 143.2 | 4.8 |

The dog-server memory column is not the separate warehouse peak; full per-run warehouse values remain in JSON.
These measurements establish original-budget compliance, not a general speedup or a native-model cold-start benchmark.
The native captures' navigation-to-ready observations exclude process/server startup and should not be substituted for the benchmark cold-start metric.

Initial HTTP resource paths are identical before/after in both native worlds, with encoded body size decreasing by 31 bytes each.
Malo initial Resource Timing transfer totals are 48,516,173/48,516,142 bytes; Go2 totals are 22,001,741/22,001,710 bytes.
Seven observed Apply-click-to-visible samples per world, including UI readiness and camera settling, span 0.7752–0.9084 s for Malo and 1.0875–1.2085 s for Go2.
These are finite workflow observations, not backend latency distributions or warm-load benchmarks.

The existing versioned whole-scene mesh reload remains expensive during repeated edits.
The ten-action candidate workflows, including Original and Undo, transfer 519,382,124 HTTP bytes for Malo and 227,723,372 for Go2, versus 48,549,475 and 22,035,043 for the read-only baselines.
Each candidate session has ten extra versioned mesh requests, with 44 total HTTP requests versus 34 in each baseline.
These are whole owned local sessions, including repeated recompiles, not initial-load regressions or WebSocket payload bandwidth measurements.
Reducing that repeated transfer is the next measured optimization target; review the retained rejected reuse work before choosing a cache design.

## Retained corrections and limits

The first scoped-test invocation referenced nonexistent `test_prop_looks.py`; it ran zero tests, and the corrected invocation passed all 453 selected cases.
The first report assembler expected `not_run` instead of the actual `not run` status string; the v2 parser correction changed no product, budget or test outcome.
Both failed receipts/logs are retained.
Before capture execution, independent review found late cancelled-future replies and incomplete setup-failure diagnostics in the new helper; both were fixed, repinned and re-reviewed before all four successful captures.
The capture/helper method hashes and package-install helper pins are recorded in JSON.

## Watchable evidence

Each full native UI sample retains its acquisition timestamps and hash, at most 5 Hz.
The four H.264 30fps containers preserve every acquired sample and timing, pass duration/frame-count preflight and complete decoding; no cropping, time cap or fabricated simulation motion was used.
Root inspected all four six-frame decoded contact sheets and four native selected-control images; this is discrete inspection, not continuous watching or smoothness certification.

| World | Baseline | Candidate | Acquired samples |
|---|---|---|---:|
| Malo | [6.467 s recording](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-furniture-finishes/malo-baseline-media-v1/workflow.mp4) | [38.167 s recording](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-furniture-finishes/malo-candidate-media-v1/workflow.mp4) | 5 / 19 |
| Go2 | [6.600 s recording](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-furniture-finishes/go2-baseline-media-v1/workflow.mp4) | [40.867 s recording](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-furniture-finishes/go2-candidate-media-v1/workflow.mp4) | 5 / 19 |

The broad visual/lightweight/play goal remains active, and single-action success tuning remains deferred.
