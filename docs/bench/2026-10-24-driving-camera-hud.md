# Driving camera and shared HUD, 2026-10-24

Round71 supports the broad shared-material, lightweight-viewer and creative-play goal. **Current-source acceptance: PASS; original install/viewer budget11/11.** GitHub disposition is recorded separately. Single-action success tuning remains deferred.

## Changes and observed problems

A top-level camera button finds the selected robot and follows a mobile base; fixed arms are focused without motion. Neither action plays the simulation or sends control commands. Manual orbit, whole-scene view and creation/editing enter explicit free view. Continued keyboard/gamepad/screen-hold renewals keep that choice; explicit follow resumes tracking. Free cancels pending camera glide. Scene load/selection/disconnect clear stale tracking; loading disables the button. Existing controls, protocols, lesson tracking, physics and sensors stay compatible.

Real main-source phone input reproduced the old hold-renewal camera takeover. The complete native phone journey also exposed Undo-add covering the left direction after adding a barrier. Undo actions, keyboard hints and screen driving now share normal vertical layout. Empty space does not capture canvas input. A narrow fine-pointer test further exposed an older stale keyboard hint: creation cleared its mode but refreshed only screen driving. It now refreshes the shared driving state after clearing creation.320/390px EN/ZH native direction/Undo cases and720px keyboard/Undo cases cover the resulting states.

No dependency, texture, shader, geometry, render pass, per-frame raycast or backend API was added. The previous camera-only source and its failed9/11 budget remain immutable in `../2026-10-24-driving-camera/`; the new source fixes actual UI defects and receives its own single original-method budget. No old result was replaced or threshold relaxed.

## Verification and cost

Runtime `afabb4e3d7e86a2340472de4177ef8bebcdcb4d95d4c83cb0aa0947c7d06a28e`. 621 frozen source/test/script/lock/budget files unchanged. 1478 distinct pytest passes: 1280 focused-default(one skipped), 190 complete browser and 8 additional slow resource/sensor cases.19new browser cases are counted once. Seven unchanged canonical framework cases, source and fresh offline locked-wheel CLI/viewer/MCP smoke(60tools), Ruff/format and JavaScript parsing pass. All 394 packaged source files match. Full default suite not rerun; mypy/pyright/bandit/pip-audit unavailable and not installed.

Five compiled-model arrays/scalars/expanded payloads and frozen physics match main. All22fixed images are pixel-exact with matching actual projection, camera, resources and GL draw counts(60120each across40samples/view). Per-view times remain mixed. Fresh/warm navigation requests remain30 for wood worlds and29 for park; encoded bytes and all first-frame samples are retained in the JSON. Sequential observations establish neither a universal speedup nor the cause of earlier failed timing.

| Original budget measurement | New source |
|---|---:|
| Wheel | 3854686B (+756B vs main) |
| Installed/packages | 163.378MB / 42 |
| Dependency download | 52.5149MB |
| CLI first frame | 0.7997s |
| Standard/assets FPS | 60.1469 / 53.8779 |
| Standard/assets synchronized frame | 3.3 / 10.1ms |
| Server peak/idleCPU | 132.6MB / 4.4percent |

The single unchanged official budget has acceptance failures: `[]`. Other budget groups were not run. Previous failure remains failed; performance variation is not attributed to this HUD change or existing Blender jobs. No other app/job/proxy changed or stopped.

## Real workflow and media

Four original recordings cover desktop English and Chrome touch-emulated phone Chinese, before/after. Native inputs build a barrier, find the rover, drive/contact, manually orbit during sustained input, resume viewing, reverse/turn, save, change map, restore, reload and replay. Each has two drives with at least5distinct positive-force rover/barrier contact times. Restore matches its own saved variant; final pause/reset returns to the saved start. No backend stepping or pose writes occur while recording.

Each of three Find operations uses3native pointer-downs via the old drawer versus1through the new camera control. The old sustained-input probe resumes tracking; the new probe remains free. The phone common before/after route uses right turn because main's left was obscured after adding; the new phone recording additionally holds left. The original blocked-left phone attempt remains in the prior evidence, while six new native-browser regressions verify both turn/release and actual Undo. These counts are observed task actions, not a usability study or physical-device certification.

All four MP4s strictly decode, retain full raw timestamp spans and have six decoded sample frames inspected each(24total). Raw CDP arrival-order timestamp inversions are retained explicitly in the JSON. A complete timestamp/original-index permutation rebuilds the playback timeline without rewriting or omitting any raw JPEG or manifest row; sorted timestamps must be unique/strictly increasing.30fps CFR still resamples inputs, so every raw JPEG is not guaranteed in the encoded movie. Inversion cause is unproven. Original desktop encodings and the phone encoding failure remain. Sample review does not certify every frame or temporal smoothness. Physical phones/gamepads, Windows, every model and every task were not verified. Gamepad coverage uses simulated browser Gamepad API.

Original evidence failures remain: off-route native placement; an unawaited nested helper Promise; missing readiness wait; the real Undo obstruction; initial test camera aimed outside the intended placement; a removed test import; and the stale post-creation hint. Final collection awaits actual eye/target/projection stability, verifies the placed barrier within2cm and actual direction readiness. Product tests and budget were frozen only after the final source fix; no collection failure was relabelled as product success.

Actual videos under the raw directory: `workflow-laptop-baseline/workflow.mp4`, `workflow-laptop-final/workflow.mp4`, `workflow-phone-baseline/workflow.mp4`, `workflow-phone-final/workflow.mp4`. See [numeric report](2026-10-24-driving-camera-hud.json). Raw/source/helper/media/index/ExperimentStore evidence: `../robotforge-evidence/2026-10-24-driving-camera-hud/`; prior failed candidate hash index retained. Broad goal active.
