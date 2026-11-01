# Crate ordinary-view prerequisite (2026-11-02)

The current evidence does not establish a salient shared crate-joint defect, so no geometry candidate is admitted.
There is no product, material, shader, shadow, physics, interface, asset or dependency change.
The broader visual/lightweight/play goal remains active.

## What the source establishes

The shared `_crate` builder has 6mm side boards, 12mm corner posts and 1.5mm chamfers.
Side boards extend into the posts; top/bottom boards overlap both; `mesh.merge` concatenates them.
Warehouse `crate_*` objects and harbor floating `crate_1..3` use this builder.
Harbor quay `cargo_*` objects resolve the plain box builder and are excluded from the shared-target claim.

The external analyzer inspected 39 dimension cases without compiling a model.
It projects outward triangles on the six requested outer planes and uses exact rational clipping for strictly positive intersection area, with a separately declared 1e-12m plane and 1e-9 normal tolerance.
Boundary contact is excluded; positive numerical slivers are retained.
Pairwise area sums count multiplicity and are not union areas or visible-defect measurements.

| Case, size m | Positive triangle pairs | Pairwise area sum m² | Triangles | Array bytes |
| --- | ---: | ---: | ---: | ---: |
| Warehouse rack, 0.45/0.40/0.32 | 392 | 0.0176040023 | 2320 | 128752 |
| Warehouse pallet, 0.50/0.36/0.30 | 396 | 0.0172438880 | 2320 | 128752 |
| Harbor floating, 0.60/0.50/0.30 | 400 | 0.0199918575 | 2512 | 139344 |

These array bytes are NumPy mesh storage, not network transfer, GPU memory or package cost.
The 10mm/1mm schema-valid cases also expose existing out-of-envelope construction; this is retained as a separate source diagnostic, not a newly fixed or compiled-runtime result.

## Actual application evidence

All 809 source/scripts/tests/lock inputs are frozen at main `77172e6`; freeze SHA is `be101ce1bc54199db2856ddeb47ed058ff2435b78121ad3fe8f27c9408ae7cd0`.
The linked [JSON report](2026-11-02-crate-joint-screen.json) binds 25 methods/receipts/review inputs, five capture manifests and two film hashes.
Three fresh isolated homes/profiles/servers load the exact original Go2 warehouse, original Malo workshop, and a default harbor created through the public API with seed zero and its boat starter.
Go2 has 29 crate-named objects; harbor has three; the unchanged Malo workshop has none and serves as a non-target control.
Actual body/geom names, native geometry/material hashes, linked target shader hashes and sparse nearest-mesh visibility establish the target routes.

English desktop views use 1280×800 native pixels at DPR1, paused at zero.
Both ordinary views per world are warmed, then captured and repeated with exact full RGBA and camera/projection equality.
All 481 public model arrays and recorded state/world bytes stay identical within each run.
No fog normalization, crop, alternate material, diagnostic geometry, shadow toggle, download, staging or conversion is used.
The retained Malo cache adapter refreshes only cloned size/mtime bookkeeping after dependency payload hashes match.

Two additional Go2/harbor runs add a prospectively fixed 16-step camera translation, 5mm per step along world+x for both eye and target.
Actual full-viewport JPEG acquisition is serial and at most 5Hz; all 32 originals, acquisition clocks and actual camera states are retained.
The final view restoration passes the same exact static/runtime gates.
Across five runs, 20 native PNG/RGBA files form ten within-run repeat pairs.
All eight phase-aligned baseline/motion-helper comparisons are also exact; these comparisons are correlated with the repeats and use unchanged product sources.
All owned application, Chrome and media processes are reaped; source, original WIP, retained experimental trees, manifests and 78 native asset inputs stay exact.

## Observation and decision

Root inspected six full baseline PNGs, two decoded contact sheets with 12 sample cells, four native JPEG endpoints and two native decoded PNGs.
An independent reviewer inspected the four target baseline PNGs and ten native JPEG samples at indices 0/4/8/12/15.
Warehouse front seams contain fine discontinuous bright pixels; the ordinary side view is legible, and harbor silhouettes/corners remain continuous.
Neither inspection establishes a useful shared defect warranting a geometry candidate.
These discrete inspections do not exclude temporal flicker; JPEG samples cannot establish single-pixel causality, and continuous playback was not reviewed.
Do not infer z-fighting from the analytical overlap alone or retune shadows/materials to manufacture a positive result.

Both full-frame H.264 films pass complete decode and the unchanged 0.15s duration gate using the previously reviewed explicit 30fps encoder.
Every acquisition-end timestamp and JPEG hash is copied unchanged into staging; there are no arrival inversions, trimming or output frame caps.

| Actual baseline film | Seconds | Decoded frames | Raw samples |
| --- | ---: | ---: | ---: |
| [Go2 warehouse](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-crate-joints/go2-motion-media-v1/workflow.mp4) | 3.333333 | 100 | 16 |
| [Harbor floating crate](/Users/yusenthebot/dev/robotforge-evidence/2026-11-02-crate-joints/harbor-motion-media-v1/workflow.mp4) | 3.333333 | 100 | 16 |

## Scope and next step

No fresh pytest, package/install, loading/FPS budget, desktop zoom, Windows or remote CI pass is claimed; runtime inputs remain identical to the PR75 baseline.
All actual analysis/capture/media jobs use the pre-existing interpreter.
A planning agent accidentally invoked `uv run` for helper lint, creating an ignored local environment with 48 packages; it remains intact and was not used for actual verification.
Source review corrected the unexecuted media adapter's inner-timeout cleanup and start/end timestamp mismatch before its first run.
Push/merge are pending at report creation and require a separate verified delivery receipt.

Next, reproduce the current selection burden with a table, two tabletop items and one stacked item.
Surface-aware placement already exists; the missing flow is selecting a support together with everything it carries before group movement.
Add an explicit paused-state read-only selection suggestion, retain the complete 16-item limit and version/generation/world/selection guards, then reuse group move/copy/save and Undo.
This is a creation/play improvement; individual action success remains outside the development focus.
