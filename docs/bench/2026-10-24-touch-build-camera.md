# Adjust the camera while building

This increment improves creative placement on phones and desktop: tap to place on release, drag to orbit, pinch to zoom/pan, and continue building without closing the placement bar.
It also fixes a visibility problem found in the actual recording: after adding a single small prop or undoing back to one, the camera remained distant while scene-sized fog hid the result completely.
The existing APIs, physics, controllers and material shaders remain unchanged.
Individual manipulation-success tuning remains deferred; the broad visual-quality and playability goal stays active.

## Interaction and visibility

Placement observes the same pointer stream as OrbitControls, without consuming camera events.
A primary, unmodified left-button tap can place only after release.
Moving more than eight CSS pixels, a second finger, mouse-button chords, cancellation or lost capture invalidate that gesture.
Returning to the starting position or lifting just one finger does not make it eligible again.
Queued placement rechecks the session, scene version, runtime generation, busy state and gesture epoch after camera listeners finish.
Reset, external edit, blur and a gesture begun while a reply is pending cannot submit stale points.
Enter and the explicit placement button cannot submit while a canvas pointer is held.
The existing 500 ms repeat guard still blocks duplicate placement while allowing camera input.
Size, orientation, repeat placement and latest-addition undo retain their existing semantics.

Above-water fog keeps its original scene-derived near/far lower bounds, extending them only when the camera-to-target distance requires it.
Moving closer restores the original range rather than accumulating expansion.
This mutates one stored Fog object on camera changes; it adds no render passes or geometry scans per frame.
Underwater attenuation remains separate, and the camera clipping planes are unchanged.
The empty-scene card now accounts for visible nonplane geometry, water and cloth instead of counting only robots.

Baseline: `6ab0a346f1779241c36034c61bb108ff5a7c889f`.
Candidate runtime SHA256: `a1d2c2ef95b4fe09e1c1b3abb0963fc1ebe841ae553d9d9be530f1abf1c4443e`.
Raw evidence is retained outside Git at `../robotforge-evidence/2026-10-24-touch-build-camera/`.

## Verification scope

Real Chrome tests cover native touch and mouse gestures, one-finger continuation after a pinch, repeat-placement controls, latest-only undo, stale requests and cancellation.
Actual rendered pixel checks wait for the current scene version and verify the first and surviving blue prop at the retained camera distance.
Additional real GPU checks cover a distant/grazing view, contraction of fog ranges, underwater camera movement and restoration, and empty/terrain onboarding transitions.

The phone recording uses Chrome touch emulation at 390 by 844, rather than a physical phone.
Its matched baseline and candidate start from the same empty editable world and camera, then use native input for orbit, pinch, two deliberate taps, Done and Undo.
Physics is paused at 0.1 seconds; this recording demonstrates editing and camera control, not a new dynamic task or manipulation result.
Only initial framing uses the automated camera hook; the final view is not reframed to hide the small-world visibility problem.
Timestamped raw frames, native pointer/touch events, requests and actual world states are retained.

Fixed gallery, kitchen and park comparisons reuse frozen world descriptions and paused integration states, identical cameras and a shared harness.
Navigation timings start on an already-running server; fresh browser profiles do not clear machine or driver caches.
Synchronous render timings include one-pixel GPU readback; paused RAF cadence does not measure active physics throughput.
Process-tree RSS can double-count shared pages and does not measure GPU memory.
Heavy browser, simulation, encoding and performance runs are serialized.

Windows, physical mobile devices, other GPU drivers and extreme zoom beyond the existing clipping range remain unverified.
Blur, unexpected lost capture and outside-canvas release have source guards but no separate new browser case in this increment.

## Matched recordings and cost

The old source adds one unintended pin on drag start and another during the pinch; the final source keeps the object count at zero throughout those gestures.
Two deliberate taps then add exactly two pins, retaining Large size and 90-degree rotation.
Done retains both; Undo removes only the second pin.
The first pin and remaining pin are visible in the final screenshots and actual raw frames, with the camera retained after editing.
No external CSS hides the onboarding card in this phone journey.

Both H264 phone videos fully decode at 390 by 844 and 30 FPS: baseline 7.800 seconds and final candidate 8.033 seconds.
Raw timestamped screencast JPEGs are resampled to 30 FPS with frame duplication; no candidate footage reaches the 20-second cap.
The candidate recording's 7.934-second wall time and 7.916-second raw-frame span are different measurements from its encoded duration.
The media files are `phone-before-v2/phone.mp4` and `phone-after-v3/phone.mp4` under the raw evidence folder.

In the new final A1/B1/A2/B2 comparisons, all nine fixed-view PNGs match byte for byte across all four runs.
Scene descriptions, paused initial/final physics, draw calls, triangles, geometries, textures and program counts match.
The fixed-scene diagnostic hides the robot-only onboarding card equally with logged external CSS, as in the previous harness; the phone workflow above does not.
There are no console/runtime/GL errors.

The wheel grows from 3,732,750 to 3,734,009 bytes (+1,259 bytes).
Every packaged runtime file matches the exact baseline archive or final source; only app.js, creation.js and i18n.js differ.
There are no new dependencies, requests, textures or rendering passes.
All three fixed worlds retain 28 HTTP resources; fresh navigation transfers 4,030 additional encoded bytes and warm navigation transfers the same bytes as before.

Descriptive medians from two runs per source, milliseconds:

| World | Fresh before / after | Warm before / after | Pooled synchronized draw median before / after |
| --- | ---: | ---: | ---: |
| Gallery | 325.00 / 314.15 | 197.05 / 188.50 | 3.80 / 4.00 |
| Kitchen | 294.65 / 318.15 | 190.85 / 186.45 | 2.30 / 2.30 |
| Park | 358.45 / 354.00 | 219.10 / 229.00 | 2.90 / 2.80 |

Draw medians pool 240 synchronous samples per world/source across three cameras and two runs.
Park draw p95 increases from 4.30 to 4.60 ms; gallery first-action two-RAF timing increases from 31.05 to 32.30 ms.
All fixed-view RAF medians remain about 60 FPS.
These small, mixed samples establish neither a general speed improvement nor a statistical performance guarantee.
All slow samples and RSS measurements remain in the accompanying machine report.

## Retained unsuccessful attempts

The old-source negative control fails immediately because touchStart places a pin.
The earlier candidate passes gesture checks but fails actual blue-pixel validation: one small pin becomes fully fogged.
Its recording, gray end-view and negative test remain available; final acceptance uses the corrected source.
An early recording target fell under the placement panel after camera movement and timed out; final matched recordings use the same two world targets projected through each actual camera.
A partial-finger test initially used the wrong CDP touchEnd interpretation; the corrected stream names the ended finger and keeps the other active, confirmed by native touch logs.
The first fog-specific fixture sampled the origin while the object used its default x=0.3 position; the final fixture explicitly places it at the sampled origin.
A local test attempt omitted NO_PROXY and timed out before a usable trace, then was terminated; the rerun excludes the local server from the proxy.
All these logs are retained rather than counted as final product failures or successful validation.

## Final regression and budgets

All 101 browser-marked cases, eight additional slow GPU asset/sensor cases and 300 focused default cases pass: 409 distinct tests, including ten new gesture/fog cases and 21 existing JavaScript parser checks.
The seven unchanged canonical framework cases pass.
Actual CLI/viewer/MCP smoke verifies 60 tools, rendering, rover movement and a real script subprocess.
The final wheel installed into a new temporary environment with the exact locked dependencies passes the same smoke in 6.3 seconds; environment creation/install plus smoke takes 20.37 seconds.
Wheel build, complete package/source byte correspondence, Ruff check/format, focused secret/debug audit and diff checks pass.
Independent source, Python and evidence review found no remaining blocking issues.
The complete default Python suite was not rerun for this viewer-only change.
Optional mypy, pyright, Bandit and pip-audit are unavailable locally and were not installed.

All 11 existing install/viewer budgets pass without relaxation: 3646.49 KB wheel, 163.165 MB install, 42 packages, 52.515 MB dependency download, 0.5009 s CLI-to-first-frame, 60.13/60.08 FPS standard/assets at DPR 2, 2.30/5.00 ms render, 132.3 MB server peak and 5.8 percent idle CPU.
Import, physics-throughput and MCP timing budgets are outside this partial run.
The machine report retains every measured value, source hash, test stage and limitation, and links the completed scoped experiment record and raw evidence index.
The broad project goal remains active; this record completes only the increment's verification.
