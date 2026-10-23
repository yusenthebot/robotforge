# Draw the posed scene once

This increment supports the shared visual-quality and playability goal by removing redundant startup rendering and making scene-load failure recoverable.
The renderer now waits for actual body poses and the resulting camera, fog and light fit before drawing a world.
The loading note remains visible until the normal main-view render returns; failed mesh loads expose a native English/Chinese Retry button.
Materials, geometry, physics, public tool definitions and control interfaces are unchanged.
There are no new dependencies, rendering passes or texture resources.

## Observed cause and choice

Baseline: `be5e4a289b85b239c03df520fdd5d732f7b49825`.
Final runtime SHA256: `5731b178e1732f7aabe5c58c431f9d49baa73d3a96b863a179aa2dee0c9964f8`.
Raw evidence: `../robotforge-evidence/2026-10-23-viewer-startup/`.

The previous material round retained first-use gallery/kitchen observations of 1792/1232 ms against an already-used baseline.
This round does not reproduce those outliers and does not establish a cold GPU-cache cause or claim to fix them.
The initial unique-GLSL-comment diagnostic changed source identity, not necessarily driver-cache state; its first nonce was not logged and cannot be claimed exactly replayable.

What profiling does reproduce is an unposed first world draw followed by another draw after fog/light/camera fitting.
The first unposed draw takes 85–91 ms in the initial fresh-profile observations, followed by a 71–92 ms posed draw.
Observed `getProgramInfoLog` calls account for 63–76% of the first draw's CPU wall time.
Turning off shader diagnostics alone would leave lazy uniform/attribute queries able to wait for the same work.
The mesh-complete-to-first-draw interval includes construction and event scheduling; the pre-debug GL span includes environment setup, so neither is an isolated PMREM timing.

The chosen change prevents the provisional draw rather than adding asynchronous compilation and its cancellation/disposal lifecycle.
The version check accepts a paused world at time zero.
WebSocket frame acknowledgements and reconnect continue while poses are gated during a newer mesh fetch.
Only the latest request can install a scene or expose its load error.
Picking, navigation raycasts, selection bounds and pose-dependent auxiliary passes obey the same readiness boundary.
Context loss invalidates the submitted-version marker until restoration and a successful normal draw.
Explicit Retry reconnects to obtain the latest scene rather than retrying an expired mesh version; there is no automatic mesh retry loop.

## Matched measurements

Final measurements are recorded in the accompanying JSON and external `final-startup-comparison.json`.
Four serial A1/B1/A2/B2 runs use the same observer, frozen gallery/kitchen/park worlds, exact initial/final physics, viewport, DPR and browser version.
Each world has a fresh browser profile followed by a warm navigation in the same tab/profile/server.
Machine and driver caches are not cold; each source has already run on this machine.

Medians of two observations per side, navigation to the first posed render's two-frame proxy (milliseconds):

| World | Fresh profile before / after | Warm navigation before / after |
|---|---:|---:|
| Gallery | 398.6 / 327.6 | 212.0 / 180.6 |
| Kitchen | 385.9 / 311.9 | 240.3 / 169.3 |
| Park | 422.3 / 343.0 | 287.2 / 211.1 |

The measured fresh-profile improvement is 71–79 ms and warm-navigation improvement is 31–76 ms.
All 114 source, fixture, state, resource-count, fixed-image and GL checks pass.
Candidate first world draws are posed in all 12 navigations; 11 of 12 baseline first draws are unposed, including all six fresh-profile observations.
The baseline gallery warm A2 navigation already draws its posed scene first.
In fresh-profile observations, program counts at first-world/first-posed draw change from 12/22 to 12/12 in the gallery, 14/26 to 14/14 in the kitchen and 14/27 to 15/15 in the park.
The remaining candidate posed draw itself is still synchronous: fresh-profile medians are about 88, 91 and 103 ms respectively.
This removes duplicate work; it does not eliminate first-draw blocking.

The observer identifies the actual main scene and camera on both sources, then records the first posed render's CPU return and a direct two-native-animation-frame proxy.
The proxy is not a GPU completion, operating-system presentation or first-action latency measurement.
Observed GL wrappers add instrumentation overhead but no new GL queries or readbacks during startup.
Post-startup measurements retain synchronous render/readback samples, animation-frame cadence, network resources, renderer counts and process-tree RSS.
RSS may double-count shared pages; animation-frame cadence in paused worlds is not active-physics throughput.
No overall memory-saving or universally faster steady-state-rendering claim is made.
For example, park near-view render median rises from 2.05 to 2.75 ms and pooled p95 from 3.3 to 3.5 ms; kitchen near-view median rises from 1.40 to 1.58 ms.
The earlier gate-1 diagnostic also retains a worse single warm-park animation-frame gap (83.4 to 100 ms).

Selection is cleared equally after the timed window and two fixed views per world are compared as PNG bytes.
Unmodified default views are also retained: the corrected first-pose selection bounds can change the ring without changing materials.
All six final fixed-view PNGs match exactly across A1/B1/A2/B2.
The wheel grows by 853 bytes, fresh HTTP encoded-resource bytes by 2779 bytes, and requests remain at 28.

## Verification and actual use

The complete browser-marked suite passes: 71 cases in 310.55 seconds.
Eight additional slow-marked asset/sensor GPU cases pass in 30.68 seconds.
Five new boundary cases cover held first poses at time zero, overlapping same-body-count scene loads, desktop HTTP failure/native Retry, phone network failure/native Retry and context restoration.
The old source fails the first-pose test as expected, with 21 premature world draws; its failure log is retained as a negative control.
An initial test-harness assertion window started before the delayed scene arrived and counted legitimate old-scene frames; the correction and original failed run are both retained.

All 283 focused default tests, seven unchanged canonical framework cases and real installed CLI/viewer/MCP smoke pass, including 60 tools and a real script process.
Together with the browser and slow GPU suites, 362 distinct selected cases pass; the five new cases are already included in the 71 browser cases.
Wheel build, Ruff and 21 JavaScript parser cases pass, and the source/test patch applies cleanly to the baseline.
All 11 unchanged install/viewer budgets pass: 3643 KB wheel, 163 MB installed, 42 packages, 52.5 MB dependency download, 0.50 s CLI-to-first-frame, about 60 FPS at DPR 2, 2.50/4.90 ms standard/asset render, 133 MB server peak and 5.80% idle CPU.
Import, physics-throughput and MCP timing budgets are outside this partial budget run.
The final isolated budget follows terminal completion of smoke; an earlier accidentally overlapping budget was interrupted and retained with the proxy-stalled smoke log, then smoke was rerun with explicit localhost proxy bypass.
Those interrupted attempts are not accepted measurements.
The complete default Python suite is not repeated for this viewer-only increment.
Optional mypy, pyright, Bandit and pip-audit are unavailable locally.
Physical phones, Windows and other GPU drivers remain unverified.
Independent source review checks readiness, supersession, failure recovery and the test assertion boundary.

Separate desktop English and phone Chinese recordings demonstrate one deliberately failed mesh request, the actual Retry button, a real prop placement, play/pause, reset to time zero and camera movement.
Each final clip is H264, 30 FPS, 20.033 seconds and 601 fully decoded frames; desktop is 1280 x 800 and phone is 390 x 844.
The earlier workflow harness incorrectly expected a second confirmation after a canvas click had already committed placement.
Its helper, log, edited world and explicit failure note remain; that failed attempt did not save its in-memory screencast frames.
The phone clip uses Chrome touch/viewport emulation on this Mac, not physical phone hardware.
The controlled network fault is explicitly labeled in the raw protocol.
Timestamped browser frames, UI actions, resulting physics and video decoding checks remain external.

## Reproduce and scope

Use `uv run --frozen`, the unchanged framework and budget references, and the saved harness/fixture hashes.
The baseline can be recreated with `git worktree add --detach ../robotforge-startup-baseline be5e4a289b85b239c03df520fdd5d732f7b49825`.
The external baseline archive is pinned to that commit and records its archive provenance separately from runtime Git metadata.
The candidate source/test patch applies cleanly to that baseline.
Run browser, simulation and performance batches serially.
The completed scoped experiment record and raw evidence index are linked by the accompanying JSON.
The broader goal remains active: improve shared materials and temporal stability, then clarify creative navigation and editing.
Individual manipulation success-rate tuning remains deferred.
