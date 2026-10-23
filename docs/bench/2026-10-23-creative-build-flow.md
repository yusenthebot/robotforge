# Build several props without reopening the picker

This increment improves creative play using the existing physics and prop library.
Choose a prop once, opt into Keep placing, and place successive copies with the same size and quarter-turn rotation.
Each success clears the previous point and waits for the committed scene, actual poses and normal main-view submission before accepting another placement.
Done or Escape exits; the existing Undo add removes only the latest committed entity.
The default single-placement behavior and single-placement robot picker remain available.

The Scene & build menu now includes four equal-width tabs in one row, native arrow/Home/End navigation, roving keyboard focus, associated tab panels and a safe fallback for obsolete saved tab names.
Tab navigation does not drive the selected robot.
Build links directly to the existing robot picker, while plain model files offer a Choose a map action to become an editable world.
No robot templates, materials, meshes, physics/controller settings, dependencies or rendering passes are added or tuned.

## Boundaries and compatibility

Baseline commit: `f3f6ad30d1758292f490a599ecb7441d6fa13ffe`.
Baseline runtime SHA256: `5731b178e1732f7aabe5c58c431f9d49baa73d3a96b863a179aa2dee0c9964f8`.
Candidate runtime SHA256: `cfec922a0d86b1f97b5ae5227a154c00dc408295edf71c9540e94399fdf4a047`.
Raw evidence: `../robotforge-evidence/2026-10-23-creative-build-flow/`.

Repeated building still commits one ordinary placement at a time under the existing edit/occupancy/admission locks.
Reset can change the integration state without changing scene version or simulation time, so the viewer state now carries the existing session generation.
Robot and prop placement requests accept an optional strict nonnegative integer generation and reject a mismatch at admission; the existing compile-to-commit generation check remains.
When that request field is supplied, the reply includes its commit generation.
Legacy callers omitting it retain their exact reply shape, and existing MCP tool definitions are unchanged.

The viewer retains generation updates during a pending edit and reconciles them against its own reply before rearming.
A newer generation cancels; an older observed generation waits.
Unknown initial metadata cannot authorize a repeat, and a missing commit generation cannot rearm it.
An independent placement-session counter prevents late asynchronous completions from restoring a canceled mode.
Local reset cancels immediately; external edits, observed reset, lesson transitions and disconnect withdraw the active mode.
Every new position is raycast against the current submitted scene and checked server-side.
Held Enter is suppressed, the old point is discarded after success, and the existing 500 ms click guard also protects repeated placement.

## Evidence protocol

Four serial A1/B1/A2/B2 runs retain the same frozen gallery/kitchen/park worlds and exact initial/final physics, 1280 x 800 viewport, DPR 1, fixed views and uniform startup observer.
Fresh browser profiles are followed by warm navigation; machine/driver shader caches are not claimed cold.
First-posed CPU return and its two-native-animation-frame proxy are separate from GPU or operating-system presentation and first-action latency.
Post-startup synchronous render/readback samples and process-tree RSS retain their existing measurement limitations.

Full pictures retain intended menu-label differences rather than requiring them to match.
A separate paired capture hides UI visibility equally on both sources after timing, preserving its layout and checking unchanged camera/projection/physics, to compare the same six 3D views.
The external visibility override is a diagnostic capture step, not an application change.
All 115 startup/source/state/resource checks and all 42 companion image checks pass.
The six companion canvas PNG pairs are byte-identical.
Full screenshot differences are retained: 8569–10520 pixels per view, confined to the changed header area, with unmasked difference images.
The first companion harness demanded bit-identical double-precision camera arrays across animation frames and failed on numerical settling.
That attempt, its helper and log remain; the final observer records arrays and uses a 1e-9 numeric tolerance, while still requiring exact physics and PNG bytes.
Observed maximum camera-array deltas are 8.53e-14 across the CSS step and 2.66e-15 between sources.

Navigation to the first posed render's two-frame proxy, medians of two observations per side (milliseconds):

| World | Fresh profile before / after | Warm navigation before / after |
|---|---:|---:|
| Gallery | 327.90 / 335.90 | 178.00 / 177.30 |
| Kitchen | 314.10 / 297.20 | 174.10 / 180.35 |
| Park | 354.30 / 359.85 | 206.80 / 221.35 |

Results are mixed; park warm navigation is about 14.6 ms slower.
Gallery far-view synchronous render median rises from 2.65 to 3.00 ms and pooled p95 from 4.1 to 4.2 ms; kitchen far p95 rises from 2.5 to 2.6 ms.
Some other medians improve, but this is not an overall rendering-speed or memory-saving claim.
Fresh HTTP encoded-resource bytes grow by 5211 bytes, with 28 requests on each source.
A1 was captured before implementation; its standalone helper did not log exact UTC start/finish boundaries, so those fields are explicitly unknown rather than reconstructed.

Matched before/after desktop and phone recordings perform the same task: four balls at the same locations, followed by selecting and canceling a rover in the existing picker.
The old flow reopens the menu and reselects the prop for every placement; the new flow selects once and opts into repeat.
Counts measure required actions in this specific task, not a user study; automation completion time is not a human usability metric.
Both sources place four balls at exactly the same recorded positions in each viewport.
Prop-picker accesses and Ball choices each change from four to one, removing three menu reopenings; the new workflow additionally uses an explicit opt-in checkbox and Done.
This does not imply every operation or every complete task becomes 75% shorter.
The direct Add robot action is one step from an already-open Build page, reusing the existing picker rather than duplicating its cards or placement logic.
A separate real playback verifies rover/ball interaction without changing controller or contact parameters.
Phone evidence uses Chrome touch/viewport emulation on this Mac, not a physical phone.
The four matched desktop/phone before/after clips use English UI and last 24.03–24.10 seconds each; EN/ZH behavior is covered by the browser suite.
The final 13.90-second desktop playback records 219 current-contact observations by body ownership, 0.69833 m rover displacement and 0.45289 m ball displacement over simulation time 0 to 3.308 seconds.
Contact observations are samples, not a count of separate collisions; transient contacts between samples can be missed.
All five canonical H264/30 FPS clips and the retained first contact-observer attempt decode cleanly, with 31 media checks passing and no application exceptions/console errors.
Desktop exports are 1280 x 800; phone exports are 390 x 844.
The final playback export was rebuilt from unchanged timestamped raw frames using scale-to-fit/padding because the first CDP JPEG was 1280 x 794 and the following 800 were 1280 x 800.
The original export and failed dimension check remain; there is no crop, simulation replay or altered action in this format repair.

## Verification and retained attempts

New browser cases cover repeated large rotated props in desktop/phone EN/ZH, fresh-point/held-Enter behavior, latest-only Undo, checkbox Escape, native tab keys and no teleop, obsolete storage fallback, zero-time external reset, overlapping commit/reply/reset, delayed initial state, local reset/disconnect/lesson, occupancy refusal/retry and actual plain-model-to-map creation.
Default creation, startup, gamepad, lesson, lighting, GPU assets and sensor regressions continue to use real Chrome.
Python cases verify stale and malformed generation admission, explicit commit generation and unchanged legacy reply shapes for robots and props.
The final source passes all 85 browser-marked cases in 361.54 seconds, eight additional slow GPU asset/sensor cases in 29.65 seconds, and 300 focused default cases in 13.20 seconds: 393 distinct selected cases.
The 14 new browser cases are included in the 85, and the six new Python cases are included in the 300.
The old source fails the tab-layout test as expected, with tab tops at 73 and 103 pixels; its negative-control failure remains separate from candidate results.
Seven unchanged canonical framework cases, installed CLI/viewer/MCP smoke with 60 tools and a real script process, wheel build, Ruff and all 21 JavaScript parser cases pass.
The complete default Python suite is not repeated for this viewer-focused increment.

All 11 existing install/viewer budgets pass without relaxation: 3645 KB wheel, 163 MB installed, 42 packages, 52.5 MB dependency download, 0.50 s CLI-to-frame, 60.2/60.1 FPS at DPR 2 for standard/asset views, 2.90/5.00 ms synchronous render, 133 MB server peak and 6.00% idle CPU.
Idle CPU is at the existing limit and is retained as measured.
Import, physics-throughput and MCP timing budgets are outside this partial budget run.
The wheel grows by 1422 bytes; the seven changed package members are the viewer placement/server code and five existing frontend files.
The source/test patch applies cleanly to the pinned baseline archive.

Initial test-harness mistakes are retained: an async generator passed to all(), a zero-time precondition that had not paused the world, reading the detached store after a lesson, and using low-level session model replacement instead of the actual open_model entry point.
The last correction uses a real saved MJCF through Playground.open_model, followed by the native map action.
The first playback contact observer filtered geom names and missed the rover's unnamed collision geoms.
Its motion video, zero-count observation and original helper are retained; the corrected observer uses actual geom-to-body ownership and records raw IDs/names/distances, without modifying simulation or controls.
Optional mypy, pyright, Bandit and pip-audit are unavailable locally.
Physical phone hardware, Windows and other GPU drivers remain unverified.
Source and Python reviewers approved the generation/rearm logic and actual regression boundaries.
The accompanying JSON records the final selected test results, unchanged canonical framework and budget references, actual smoke, source/review evidence, media hashes and immutable experiment record.
The evidence directory and report filename use the capture start date, October 23; final checks continue on October 24 in Asia/Shanghai, with raw UTC timestamps preserved.

## Scope and remaining work

This is repeated placement, not a general scene editor or multi-level undo history.
It still uses the existing model recompilation semantics: other entity poses/control targets are retained, velocities restart, and map/terrain changes keep their previous initialization behavior.
Touch or left-click in active placement commits an object; exit placement before dragging the camera on a phone.
Menu keyboard navigation and parameter controls are verified; choosing new surfaces still uses pointer/touch, so this does not claim a complete keyboard-only building workflow.
Materials and their remaining temporal stability limits remain a separate shared-visual task.
Run heavy browser, simulation and benchmark batches serially with `uv run --frozen`.
The broader lightweight visual-quality and playability goal stays active; individual manipulation success-rate tuning stays deferred.
