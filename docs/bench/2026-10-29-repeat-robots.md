# Continuous robot placement

Template robots, curated library models and saved designs share the existing opt-in Keep placing control with props and terrain.
Choose a source once, enable Keep placing, choose each new valid point, then press Done.
Each robot still has a separate atomic compile/commit, fresh version/generation guards, unique name and independent selection.
The last addition alone remains undoable; default one-shot placement and existing APIs are retained.
Each model action resolves its current source ID, with no frozen-source or arbitrary-import batch contract.

Repeating creation selects the new robot without forced camera fit/focus.
Camera eye and aim remain; existing panel avoidance can still change projection when the selected robot's sensor views change.
The opt-in robot hint reserves stable vertical space so translated progress/next-point wrapping does not move the composition with unchanged panels.
A delayed new model's first zero-time pose preserves matching committed Undo; an actual rewind within the same model still invalidates it.

## Verification

Runtime `1f984522c19ad83bcdb4fb5bbab73a08eb2ea0cb94c38729429429a55af8e6f5`, 775 frozen source/script/test/dependency inputs, original 13 WIP files retained byte-exact.
685 scoped default viewer/gallery/placement cases, 356 complete current non-slow browser cases and 9 GPU cases yield 1050 distinct pytest passes.
The eighteen new cases cover desktop/Chinese touch, three sources, unchanged camera, fresh requests, no repeated Enter, occupancy retry, both actual reset-message orders, missing-generation cleanup, real 15 s display timeout, map/socket cancellation and same-version rewind.
Seven unchanged canonical cases, actual source/fresh offline installed 60-tool CLI/viewer/MCP/script smoke, Ruff/format/JS parsing and original 11 install/viewer budgets pass.
Optional mypy/pyright/bandit/pip-audit are unavailable; no new dependency or paid Actions run is added.
This is not the full regular/slow suite or cross-platform certification.

Six complete before/after native Chrome movies cover original Malo and Go2 world copies and an explicit saved red-arm fixture.
Both cohorts add the same two robots, select each independently, undo only the last and reload the persisted world.
The primary picker-entry count drops 2 to 1; each robot still compiles independently, and scripted counts are not human task-time measurements.
All six movies fully decode, and the root inspected 36 decoded samples.
Portrait uses desktop Chrome touch emulation at device DPR 3 / main DPR 2, not a physical phone.
Cached Go2 is a QA precondition, not an offline-product claim.

Twelve paired workflow main canvases and 24 default original Malo/Go2/Panda canvases are exact.
All stored workflow model arrays/scalars plus qpos/qvel/ctrl match, with only isolated QA path prefixes normalized, pointers resolved and relocated file bytes checked.
Sensor previews are closed through the same native switches before fixed-camera placement controls.
Canonical post-add/Undo screenshots use the same existing native Fit view before fixed observation to normalize old one-shot clipping changes.
Actual workflow transitions remain in the recordings; these controls do not certify every frame, locomotion, reconnects or separately stored full integration vectors.

Initial phone hint/camera failures, late-first-pose Undo failures, the mesh-interceptor mistake and preliminary camera-control mismatches remain retained with their exact commands.
The interrupted preparatory focused-v3 attempt is not acceptance evidence.
Final source and tests stay fixed throughout accepted jobs.

## Cost

Wheel 4,031,512 B, +309 B over the previous 4,031,203 B, with 417 source/wheel/installed product files byte-equal.
No new runtime asset, request, shader pass or dependency is introduced.

| Original world | Browser-cold main-ready ms before / candidate | Actual reload ms before / candidate |
|---|---:|---:|
| malo | 459.05 / 380.35 | 224.20 / 191.70 |
| go2 | 428.75 / 423.68 | 273.20 / 258.45 |

ABBA followed by BAAB uses four fresh package processes/profiles per cohort/world, actualDPR2/light, cold then reload, with server compile separate.
These are local GPU-finished main-ready and sampled Chrome-tree RSS observations, not universal speed, new-download, VRAM or leak claims.
Full receipts, helper versions, failed attempts, raw/decoded media, hashes and wheel remain at `../robotforge-evidence/2026-10-29-repeat-robots/`.
Final videos use `{before,candidate}-{template,library,design}-v3/workflow.mp4` under that directory.
Broad visual-quality, lightweight and creative-play goal remains active; no new material-aesthetic or individual action-success gain is claimed this round.
