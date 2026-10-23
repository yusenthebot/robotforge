# Reversible editing of existing rigid props

This increment makes existing scene props editable through the viewer, including fixed barriers and previously saved objects.
The broad visual-quality, lightweight experience and playability goal remains active.
Baseline is `708531ef8a055d72ba7364c2ffb2db7420ddd311`.
Verified runtime is `0475f89353459a332be50f558f1b203ca2ccc329a1e9554a5eca3dc9e232d091`.
Functional checks pass; merge remains pending the unchanged idle CPU budget.
Raw evidence lives outside Git at `../robotforge-evidence/2026-10-24-object-editing/`.

## Interaction

Build lists existing spheres, boxes, cylinders, tables and bins by their exact saved names.
Users may choose a name by keyboard or enter an explicit scene-selection mode and tap a visible prop.
Ordinary physics dragging remains available outside that mode.
A temporary bounding box identifies the selection without modifying any shared surface material.
It is allocated on selection and disposed when selection ends.

Move/rotate reuses the existing placement marker, wireframe, qualified release gesture, camera controls, Enter and Escape.
Dimensions, mass, colour, material and identity stay unchanged.
The old prop remains visible until commit and its own geometry is excluded from placement raycasts.
Remove targets only the selected object, with no fuzzy-name matching or robot/terrain/task removal.
One-use Undo edit restores the previous descriptor, original list order and the edited free body's pre-edit actual pose.
Other objects keep their poses; this does not restore their historical state or a global history stack.
Reset keeps the newly saved layout.

This is upright repositioning rather than arbitrary six-axis transformation.
Free props lose their current roll and pitch when repositioned.
As with existing world edits, recompilation preserves other roots, joints and control targets but zeroes velocities and restarts motion.
Robots, mechanisms, generated complex props, scaling and multiselection remain outside this editing entry point.

## Transaction and compatibility

New viewer actions use strict request schemas with exact object names, model version, runtime generation and a saved-world byte digest.
The compiler also captures a digest of its actual input descriptor, carried through normal CLI loading and world reloads.
Metadata and edits compare the current file against the running model's descriptor, using the same captured bytes for parsing and hashing.
An external change between compilation and loading cannot turn an uncompiled descriptor into an accepted edit target.
This adds metadata without adding fields to world.yaml, MCP tools, script signatures or dependencies.

Preflight builds a private candidate, checks saved-state occupancy, then rechecks live occupancy under lesson/session locks immediately before the existing atomic world edit.
The target's old free pose is excluded from live carry so it cannot overwrite the candidate location.
Unrelated poses are carried exactly, including unrelated deep contacts.
Undo uses the same occupancy and support checks, and can be retried after an occupied location is cleared.
The existing execution lease rejects edits during active task/script control.
Version, generation, store identity and disk checks reject stale requests, including zero-time resets.

Support protection checks saved and live states.
Native contacts provide upward normals, including support on a tilted prop below its global bounding-box top.
Static pairs omitted by native contact generation use nearest points and primitive face normals, with an eight-millimetre conservative near-contact threshold.
This is a support guard, not a general force/stability solver.
Fixed bin walls and anonymous fixed visual meshes are assigned to the authoritative object using the compiler's exact naming conventions, rather than body zero or a guessed preset-like object name.
Saved task references prevent moving or removing a task's object, target, sort source/destination or start-state object.

## Iterations and evidence limits

Initial backend fixture errors placed the support block at its default x=0.3 instead of underneath the test load and incorrectly supplied loose-material settings to tables/bins; the fixtures were corrected rather than changing placement semantics.
An initial browser attempt failed before browser startup because localhost used the external proxy; subsequent commands explicitly bypass localhost proxies.
Real CLI loading exposed missing fingerprint initialization for already saved worlds; this was fixed at the CLI construction path.
A fixed visual mesh originally had no selection owner, and a bin's fixed wall aliases could be treated as separate objects; both gained explicit ownership mapping and regressions.
Independent review found and corrected the tilted-support blind spot, selection Escape cleanup and split-read descriptor/hash race.
Programmatically picking a tiny wall immediately after closing a phone panel used a still-changing camera projection; the test now waits for the existing panel projection to settle before computing its physical target point.
The projection may settle at a nonzero offset, so requiring disabled view offset was an incorrect harness assumption.

The first complete desktop capture held native W for five seconds: the rover advanced 2.028 m but had not reached the relocated barrier, so its zero-contact assertion correctly failed. A retained eight-second repeat reached actual contacts without changing geometry, controller or runtime source. The full journey remains primary evidence; a separate closer view is supplementary.

The first whole-app browser stage passes all 107 cases.
The following slow GPU stage initially passes seven cases and fails the sensor world-switch case; an isolated candidate repeat also fails while one baseline repeat passes.
An instrumented copy proves that the old `points > 0` wait accepts the hidden lidar's previous world (`first`, version 4) while the expected new version is 7.
Waiting for actual `second`, version 7, produces the correct shown lidar without another user action.
The test now binds world/version before its existing points condition and keeps the original shown assertion; sensor runtime code is unchanged.
All eight corrected slow GPU cases pass together.
The failed suite/XML, original test and inputs, diagnostic source/observations and retained browser result remain available.
The completed browser stage does not select this slow-only module, so it is retained rather than repeated; default/framework/budget/smoke stages use the updated test snapshot.

Independent evidence review also found that the original journey exporter retains sent WebSocket requests and received-byte totals but omits the original received reply payloads.
Those original manifests remain intact, and separately scoped native receipt replays retain raw network events and check one same-connection reply per edit request, exact targets, version/generation, Undo tokens and refusal-state invariance.
The supplement is an independent run and is not retroactively presented as raw receipt evidence from the older videos.

## Verification status: needs work

Complete default suite: 3447 passed, 127 skipped, 126 deselected in 1125.05 seconds.
All 107 browser-marked cases and eight additional slow GPU asset/sensor cases pass: 3562 distinct passed cases across the three selections.
The unchanged seven-case framework baseline passes with source-bound manifests and verified record/artifact hashes.
Ruff, formatting, JavaScript parser checks, package/source equality and source/Python/evidence reviews pass.
Actual source CLI/viewer/MCP/script smoke passes in 3.3 s; a fresh offline locked-dependency wheel install passes the same smoke in 5.6 s, including all 60 MCP tools.
The separate source smoke was initially launched without the localhost proxy bypass and failed before answering through that proxy; the original log is retained and the explicit NO_PROXY repeat passes.
Optional mypy/pyright/bandit/pip-audit tooling is unavailable and was not installed.

Ten of eleven existing install/viewer budgets pass, but the unchanged 6.0% idle server CPU budget fails at 6.2% in both the initial run and the one predeclared full repeat.
This is not a fully green verification and the change remains a draft pending that guard.
The failure is retained, the budget is not relaxed, and no passing CPU upper bound or cost-free feature is claimed.
The scoped experiment record is marked failed for this unresolved budget, while the broad product goal remains active.
Record identity and immutable artifact references are in the companion JSON.

## Fixed-scene and install costs

Four serial A1/B1/A2/B2 processes reuse the same frozen gallery, kitchen, park and wood-prop snapshots at 1280x800, DPR 1.
All 17 full 3D-canvas views in each of four cross-source image pairings match exactly, with unchanged geometry, physics, camera inputs and GPU resources when no edit is selected.
The changed panels and temporary selection helper are exercised separately in the native journeys.
The complete scene payload changes: the sole allowed field is the new `object_edit` metadata.
The full payloads and their hashes remain in the external evidence, rather than silently claiming complete payload equality.

Matched wheels grow from 3,734,783 to 3,741,656 bytes (+6,873 B).
Every installed package member matches its selected source byte for byte.
Fresh HTTP encoded resources increase by 8,448 B and warm navigation by 979 B, still 28 resources in every measured scene.
No dependencies, texture resources or render passes are added.
Aggregate WebSocket bytes vary with runtime message timing and are retained per navigation; they are not a deterministic metadata-only delta.
Canonical JSON sizes for the added metadata are 108/1679/324/224 B for gallery/kitchen/park/wood props, with 0/18/2/1 editable objects respectively; these are descriptor sizes, not HTTP or measured WebSocket totals.

The following are medians of two measured cohorts per source, not statistical bounds or a general speed claim.
Fresh profiles run against an already-running local server and do not erase machine/driver caches.
The first-frame measure is the existing instrumented two-frame proxy, not a GPU presentation timestamp or first successful edit.

| World | Fresh first-frame proxy, before/after ms | Warm navigation, before/after ms |
|---|---:|---:|
| Gallery | 343.90 / 322.25 | 195.65 / 201.05 |
| Kitchen | 308.80 / 307.10 | 194.15 / 189.10 |
| Park | 354.95 / 362.05 | 234.90 / 220.20 |
| Wood props | 201.95 / 191.15 | 110.10 / 117.70 |

Draw timings are also mixed: the cohort median of per-run 40-draw medians rises from 2.375 to 3.300 ms for gallery near and 2.150 to 3.350 ms for gallery low, while park low changes from 2.275 to 2.150 ms and kitchen near remains 1.800 ms.
Each draw includes synchronous one-pixel GPU readback; separate one-second RAF traces and all median/p95/raw samples remain available.
RSS sums Chrome/process trees, may count shared pages repeatedly and is not GPU VRAM.

The first existing install/viewer budget run passes ten checks but measures idle server CPU at 6.2%, above its unchanged 6.0% limit.
A predeclared four-process A1/B1/A2/B2 diagnostic reuses one saved standing-dog world and the original five-second CPU-share method: baseline 5.8%/6.4%, candidate 7.4%/6.0%.
The direction differs by pair, and these samples neither prove a stable regression nor a 6.0% upper bound.
One fresh complete budget repeat is declared before that diagnostic, regardless of its result; the failed first run, all four CPU samples and the repeat are retained without changing the runtime or budget.
The repeat also measures 6.2%, so the guard remains failed and merge is deferred.

One separately instrumented A/B/B/A diagnostic then observes four new servers for 30 seconds each, retaining all six five-second slices and actual wall intervals.
Its original report-on-exit wrapper fails to write its report after normal SIGTERM; that helper/log is retained, and a signal-triggered report taken after the measured window completes the bounded batch.
In measured windows all four runs have zero scene messages and zero model-version changes, consistent with source review finding no recurring editor metadata path in the idle loop.
Baseline nominal 30-second CPU shares are 6.900%/6.900%, candidate 7.133%/6.833%; GC elapsed wall durations are 22.904/20.577 ms versus 24.265/23.471 ms, not CPU times.
The instrumented first-five-second samples are A1 5.6%, B1 7.0%, B2 6.6% and A2 6.2%; they are separate observations, not the original official samples.
Diagnostic advance-call counts and simulation times are cumulative child-session totals through report time, not deltas restricted to the measured window.
These wrappers add equal observation work to both sources, use the same coarse `ps` CPU counter and do not replace the original uninstrumented five-second budget result.
They do not establish a causal editor regression, exonerate its cost or produce a passing budget.
The next investigation must address measurement reliability and actual idle cost with the original failure preserved; no physics/controller changes or GC disabling were made to fit this increment.

## Actual recordings and receipt supplement

The baseline's 2.400-second desktop and 2.533-second phone clips show the missing saved-object edit affordance; they do not fabricate an equivalent old workflow.
Original full native candidate journeys retain all 14 desktop events (23.966667 s, 719 decoded frames) and 11 phone events (15.000000 s, 450 frames).
The desktop journey moves the old fixed wall, cancels and undoes edits, removes the earlier ball while retaining the later pin, refuses a loaded table and occupied spot, retries placement, selects the fixed visual mesh, drives into the relocated barrier and resets.
Actual native W advances the rover 2.430 m with 39 sampled contact frames; reset returns its XY to (-2,0) exactly while retaining the edited wall.
The phone journey uses native CDP touch at 390x844 in Chinese and does not add or verify a touch driving interface.

Independent receipt supplements replay the same full event sequences and frozen initial world/state, with a clearer desktop drive camera.
They retain original CDP network events and all WebSocket sent/received frames, binding each of seven edit-family requests to one reply on the same connection.
Undo tokens, exact names, version/generation transitions, refusal reasons and unchanged refused/cancelled state are checked against actual replies and observations.
The final evidence summary has 1203 passing scoped checks, including 100 desktop and 97 phone receipt assertions and 34 additional raw-event/media bindings.
Independent read-only acceptance verifies the new raw payloads and hashes; the original formal cohorts and journeys remain byte-exact in the archived preceding summary.

| Recording | Duration / decoded frames | Full video |
|---|---:|---|
| Receipt desktop, English 1280x800 | 24.133333 s / 724 | [Desktop](../../../robotforge-evidence/2026-10-24-object-editing/receipts-v3-candidate-desktop/workflow.mp4) |
| Receipt phone touch emulation, Chinese 390x844 | 15.066667 s / 452 | [Phone](../../../robotforge-evidence/2026-10-24-object-editing/receipts-v3-candidate-phone/workflow.mp4) |
| Supplementary visible contact replay | 24.233333 s / 727 | [Full contact workflow](../../../robotforge-evidence/2026-10-24-object-editing/contact-v2-candidate-desktop/workflow.mp4) |
| Additional contact excerpt | 9.033333 s / 271 | [Drive excerpt](../../../robotforge-evidence/2026-10-24-object-editing/contact-v2-candidate-desktop/drive-contact-excerpt.mp4) |

All full workflows use timestamped raw browser frames without the earlier helper's 20-second cap, H264 at 30 FPS, validated durations/counts and complete error-free decoding.
Raw frames, unsuccessful attempts, helper/source snapshots, exact receipts and decoded samples are retained outside Git.
The coordinator inspected actual screenshots and MP4-decoded approach/contact frames; selected stills cannot establish dynamic quality or replace viewing the full videos.
These are local Mac Chrome/Metal checks and touch emulation, not physical-phone, Windows or other-driver validation.
Arbitrary six-axis editing, global history and the previous wood colour-map temporal regression remain outside this increment; the broad product goal stays active.
