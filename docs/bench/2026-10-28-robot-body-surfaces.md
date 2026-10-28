# Robot part surfaces — 2026-10-28

Robots → Part surface now lets a user choose Original/plastic/wood/steel on one
direct body/link, highlight its actual visible geometry, and apply/reset with
latest-only Undo. Templates, imported URDF/MJCF and library robots share the
same optional `body_surfaces` world field. Layout save/restore and reload retain
it; an empty mapping is omitted, preserving legacy serialized worlds.

This is explicit creative styling, not an inference of real materials. The local
Malo URDF and Go2 source provide colour information without reliable part-level
CMF. There is no global name/gray-material remapping or new shared shader.
Original rebuilds from actual source rather than an approximate inverse style.

## Compatibility and boundaries

The first slice supports unmounted rigid robots and opaque, untextured direct
visuals. It excludes child bodies and declared added-sensor dressing, refuses
mounted/flexible/transparent/native-textured cases, and validates names/counts.
Source geom/material colours, alpha and RF ray invisibility remain intact.
New materials are isolated; source shared materials are never mutated. Native
textures are not replaced. RGB appearance can change using existing renderer
expressions; this does not promise equal RGB pixels between renderers.

Version/generation/world digest and exact entry checks precede cached-only
preparation and atomic publication, with rollback. Applying or undoing carries
live free-body/robot pose and controls, resets velocities/time and pauses; it
does not change saved starts. No-op keeps state/history/Undo exactly. Another
edit/reset/map switch invalidates old Undo. An actual cross-type Undo AttributeError
found during tests is fixed by checking the receipt type before field access.
Existing MCP tool definitions/parameters remain unchanged (60 tools).

## Real verification

Frozen runtime `56078aa460b14c93b48c585893b846df46c77f0d77c089594dd24a482caf0013`; 767 source/test/script/dependency
final-stage inputs unchanged. Before the JS-only layout fix, complete default selection: 4037 passed,
128 skipped. Eight related browser modules:
57 passed/0 skipped;
Final viewer-default/static-reader selection: 663 passed/6 skipped;
two additional slow GPU modules: 9 passed/0 skipped.
These staged selections contain 4103 distinct pytest passes. All final selections pass.
The final JS runtime did not repeat the entire default suite: a hash bridge proves
all product Python, dependencies and other tests identical; only app.js and the
layout browser file changed. All viewer-default files and static readers reran;
not all browser/all slow or Windows certification. Seven unchanged canonical
framework cases pass using the actual Python3.12.13 binary required by the
existing baseline. Everyday tests/models run Python3.12.16; no baseline/version
gate is relaxed. Source and fresh offline installed CLI/viewer/MCP smoke,
Ruff/format, JS parsers and offline wheel pass. Optional type/security tools are
unavailable, not counted as passes; added/untracked-line audit is limited.

Trusted CDP mouse/touch/keyboard inputs in real Chrome on copied original Malo, Go2 and rover worlds
exercise steel→wood→same-setting no-op→Original→Undo, save layout, change map,
restore and reload. Every session keeps444 nonappearance model arrays exact,
all nonmaterial object names checked, unaffected material semantics exact and
pose/control state within1e-12; material IDs may shift after adding materials.
Original raw mesh hashes and user world bytes stay unchanged. Separate tests
cover actual ray, added-sensor housing/placement/arrays, failure rollback,
postpublication failure, reset races and WebGL loss/restore during display wait.

| Actual workflow | Full decoded duration | MP4 frames |
|---|---:|---:|
| malo | 19.900000 s | 597 |
| go2 | 12.233333 s | 367 |
| rover | 13.966667 s | 419 |

All full timestamped videos decode without errors; root inspected18 decoded
sample frames. Physics is paused, initial camera/probes instrumented and world
rebuilds can refit: workflow frames are not isolated same-camera comparisons,
human timings, locomotion/manipulation success or every-frame quality evidence.
Go2 uses320×812 Chinese touch emulation, not a physical phone.

Three untouched Malo/Go2/Panda worlds retain48 final fixed PNGs/24 main-3D-canvas byte-exact pairs
between separate baseline/candidate package roots, actual DPR1/2, two lighting
presets and ordinary/detail views, with explicit light media in both cohorts.
One full-window PNG has934different pixels, max channel delta6; no whole-window
identity claim. All raw screenshots remain. All model/state fields compare exactly except
separately retained transient `paths`/`mesh_pathadr`; camera eye/target tolerance
is1e-10. Projection, buffers, default materials, draw work, GPU resources/programs
agree. Identical relative mesh labels are not independently resolved asset proof.

Native Chrome zoom remains pending manual unlock: the Mac relocked after the long background run. The isolated headed process completed setup/model-state guard only; a managed-tab shortcut did not change zoom and is not counted as a pass. The async unlock request remains pending. Current CDP/browser rendering tests and feature workflows are separate evidence; no native OS zoom or physical monitor certification is claimed.

## Cost

Original11/11 install/viewer budgets pass once without threshold changes:
wheel3948453B (+6898B),
163.6596MB installed/42packages,
standard CLI frame0.5713s, standard/assets60.21/60.14FPS
atDPR2, 1.7/3.2ms draw,
134.5MB dog-server peak and4.0%idle CPU.
All416 source/wheel/fresh installed product files agree. The
original13 WIP files remain unchanged. No new dependency, asset, image texture
or shader pass; default startup adds one small JS module/request (30→31 HTTP
resources; cold encoded-resource bytes+6406 in the first paired samples).
HTTP totals exclude WebSocket frames: the only differing observed scene field
is additive `robot_surface`, compact-JSON2506B/Malo,911B/Go2,871B/Panda. These
are serialized metadata diagnostics, not measured wire bytes.

The prospective fixed ABBA+BAAB comparison uses separate package processes and
fresh Chrome profiles: four samples per code/world, each browser-cache-cold
navigation followed by actual warm reload. All individual raw samples, resource
bytes and Chrome RSS snapshots remain in the JSON/evidence; sample mean deltas:

| Original workspace | Browser-cache-cold | Warm reload |
|---|---:|---:|
| malo | +8.88 ms | +29.80 ms |
| go2 | -37.55 ms | -65.70 ms |

Models/OS caches are warm, servers already compiled and paused. Ready is matched
main scene/no busy or loading plus2RAF/gl.finish, not all sensor shader compilation,
remote download, CLI startup or human input latency. Small cohorts/order variance
do not establish universal speed or causality; RSS is a sample tree, not peak,
steady memory or VRAM. Ambient OS theme was not explicitly fixed in this load
cohort; separate final visual controls fix light. Standard budgets use different
worlds from these loads.

Current-code original/wood single fresh-process samples at fixedDPR2 show the
cost of opt-in styling and its visible selection outline. Each view has30
GPU-finished draws after10 warmups and120 RAF intervals; these repeated frames
are not independent trials, and process peak RSS includes model compilation.
Closed/editor resource, frame and all sample details remain in JSON.

| World/style | Closed draw median ms | Editor draw median ms | Programs closed/editor |
|---|---:|---:|---:|
| malo, original | 0.50 | 0.30 | 18 / 19 |
| malo, wood | 0.50 | 0.30 | 19 / 20 |
| go2, original | 0.70 | 0.60 | 11 / 12 |
| go2, wood | 1.35 | 1.50 | 11 / 12 |
| rover, original | 0.20 | 0.20 | 10 / 11 |
| rover, wood | 0.20 | 0.10 | 10 / 11 |

## Retained attempts and limits

Initial float32 hash serialization failures, wrong source-metallic sentinel test,
the real cross-Undo error, native select/reload test-driver assumptions, and
Malo-v1 names-map shape comparison failure are retained alongside final passes.
The first related-browser batch had50passes/1phone-English Restore native-click
delivery timeout. An instrumented exact case passed with connected target and
trusted click. The second original51-case batch again had50passes/1failure,
this time desktop Chinese confirmation/focus timeout. Their exact original event
causes remain unproven. A controlled delayed-world negative test then failed2/2:
mapreply had completed and the old scene was rendered while stale layout actions
were enabled. The final fix clears/cancels old catalogs at map submit and waits
for actual render readiness. Failed-request recovery and context loss/restore
also pass. Thirteen layout cases and the expanded saved-layout context case
passed before final regression. Both original failures and both controlled
negative failures remain.
The corrected native comparator separately checks actual nonmaterial names and
unaffected material semantics; it does not require material numeric IDs to stay
fixed when adding materials before sensors. Initial nonexistent test-filename
assertion and bare system-Python import failure are also retained. The first controls mixed light/dark OS theme and failed; a fresh matched-light
comparison corrected this uncontrolled input without product changes. A v2
background probe wrongly assumed Color rather than gradient Texture, retained
and corrected in v3. No passing
budget was repeated, no algorithm/success rate or performance threshold tuned.
Code and Python reviews approve their stated source/test scopes.

Evidence: `../robotforge-evidence/2026-10-28-robot-body-surfaces/`, including all
attempt logs/XML, package roots/bridge, raw frames/full videos, fixed controls,
cost/load cohorts, native zoom and immutable closure index. Media is local,
not bundled in the wheel or Git. The broad shared quality/lightweight/play goal
stays active. Physical phones/other drivers/Windows and individual action-success
tuning remain outside this slice. See [CREATIVE_PLAY.md](../CREATIVE_PLAY.md).
