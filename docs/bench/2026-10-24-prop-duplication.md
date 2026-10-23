# Reuse configured props, 2026-10-24

Users can duplicate a configured rigid prop, arrange a course variation, save it, explore another map and restore it to drive again.
This improves creative play while retaining the accepted material/rendering baseline.
The broad visual quality, playability and lightweight-loading goal stays active; individual manipulation success tuning remains deferred.

## Behavior and compatibility

Select an existing sphere, box, cylinder, table or bin in Build and press Duplicate (复制).
The server copies the full authored ObjectEntry: shape, size, color including RGBA, material, optional mass and fixed setting.
Only a bounded unique name, placement and heading change; the name avoids robot, object and terrain namespaces.
The viewer supports90-degree rotation, source-surface stacking, repeated placement with a fresh point, Escape/Done and the existing one-use Undo edit.
Authored full-turn headings are normalized for copies; existing move behavior remains unchanged.
A successful copy waits for matching scene metadata, poses and rendered version before another placement.
Requests retain strict version/generation/world-digest and execution-owner checks; delayed replies after an external reset cancel repetition without resubmitting.
The selected source stays unchanged, so its task references and supported objects do not prevent copying it elsewhere.
Task bindings, live tilt, velocity and controller state are not cloned into the new object.

Both saved and actual occupied space are checked, including fixed bin walls.
Undo removes only the newest copy and refuses while it supports another object.
Successful editing preserves unrelated poses and control targets but resets all velocities and simulation time through the existing recompilation path.
The inherited file-publication-before-runtime-install boundary is unchanged; this round does not add a full post-publication rollback or crash-atomic transaction guarantee.
MCP and script signatures, dependency lock, renderer, materials, physical definitions and control algorithms remain unchanged.

## Real app workflow

Accepted main7597f4b6334f48bdd460bc63ea746cff22a26005 and candidate runtimeb6a2e8891de433c55e899b0ce9cf0886bdbc9a1c9848322724b8af3c03d347f6 start with the same rover and configured green wooden wall.
The baseline has no native Duplicate control; adding a starter barrier produces different settings.
Baseline script/agent APIs can still reproduce a configured object; this comparison concerns the native creative workflow.
Candidate native mouse/touch and keyboard input copies twice, rotates, undoes, copies again, saves a named variation, loads Empty floor, restores and reloads the page.
All three authored walls preserve complete source settings; the final headings are15/105/15degrees.
Actual WebSocket receipts retain three successful duplicate replies plus undo, save and restore for each candidate journey.
Copy request/reply samples are88.4–92.5ms desktop and88.6–99.9ms phone-sized; these are local single-run replies, not full-scene rendering latency guarantees.

| Recording | Duration | Fully decoded frames | Contact-drive path | Positive-force samples |
| --- | ---: | ---: | ---: | ---: |
| Baseline desktop English | 10.97s | 329 | 1.710m | 5 |
| Candidate desktop English | 17.80s | 534 | 1.663m | 5 |
| Candidate phone-sized Chinese | 17.93s | 538 | 1.652m | 5 |

A bounded native-keyboard driver observes actual rover poses, steers toward the copied barrier and records MuJoCo contacts with exact rover/target ownership and normal force>0.01N at five distinct simulation times.
The harness never sets live poses or advances the viewer session itself.
It then pauses, confirms acknowledged pause, resets through the native R key and checks the unchanged authored layout, advanced generation, zero time and actual rover starting position.
The source and starter/copy dimensions/headings differ, so these observations establish real interaction, not superior dynamics or continuous contact.
The earlier restored-course straight drive separately measures0.515m desktop and0.514m phone-sized.
A fresh interpreter reopens the desktop recording's actual saved world, verifies identical checkpoint bytes/ID/hash, changes to the warehouse, restores the same three configured walls and drives0.570m.

The phone-sized run uses emulated native touch for build/layout controls and keyboard driving, not a physical phone or touch-only drive test.
A read-only QA camera adjustment makes the contact segment visible; the product camera implementation is unchanged.
Initial close-camera and narrow-phone framing recordings remain retained under framing-v1 and framing-v2-phone.
Every final video encodes the full retained timestamp span without a duration cap and fully decodes as H.264.
Six decoded stills per final video were inspected; this sampled review is not a claim of watching every frame or certifying temporal quality.

## Tests and costs

775 scoped default cases pass with one skipped; all128 browser and eight additional slow asset/sensor cases pass:911 distinct pytest cases.
The40 new Python and eight new real Chrome cases cover authored/live properties, naming, source task/support cases, surface origins, collision refusal, inverse constraints, forged input, compile-time invalidation and bilingual viewport workflows.
The unchanged seven-case framework baseline, source CLI/viewer/60-tool MCP/script smoke, fresh offline locked-wheel smoke, Ruff, formatting,22JavaScript parses and exact wheel/source equality pass.
Independent Python and frontend reviewers approve the scoped source changes; evidence acceptance is separately recorded.
The complete default suite was not rerun. Optional mypy, pyright, bandit and pip-audit remain unavailable and were not installed.

Exactly one unchanged full install/viewer budget run passes all11 measured limits.
No threshold, duration or aggregation rule changed, and no budget was repeated until green.

| Metric | Observed candidate |
| --- | ---: |
| Wheel | 3,757,720B; +777B |
| Installed size / packages | 163.2393MB /42 |
| Dependency download | 52.5149MB, unchanged |
| CLI first frame | 0.4921s |
| Standard / asset FPS, DPR2 | 60.105 /60.207 |
| Standard / asset draw | 3.0 /4.9ms |
| Dog-server peak RSS / idle CPU | 132.8MB /5.4% |

This CPU sample passes the existing6.0% rule; it does not establish a stable upper bound or a general speedup.
The gallery, kitchen, park and wood-prop controls retain17 exact full-canvas PNGs, full scene payloads, frozen physics and GPU resource counts against accepted-main retained evidence.
Each still makes28HTTP requests; fresh/warm navigation transfers1777/97additional bytes for shared frontend changes.
Fresh complete-scene times are312.1/314.3/370.1/206.8ms against retained324.1/316.9/348.6/205.6ms; warm changes also vary.
The observations are not simultaneous ABBA and do not establish causal timing/memory gains.
Copies naturally add scene geometry and physics work; no new dependency, packaged asset, texture request or render pass is introduced by the feature itself.

## Evidence and remaining scope

Evidence lives outside Git at ../robotforge-evidence/2026-10-24-prop-duplication/.
verification-inputs.json freezes566 runtime/test/config files; summary.json verifies150 functional, package, receipt, resource and control properties.
raw-index.json and the ExperimentStore record referenced by record-summary.json retain editable source/helper snapshots, raw timestamped frames, checkpoint bytes, complete receipts and logs.
Early fixture/assertion/await mistakes and an analyzer's incorrect undo-action name remain retained separately from final passing stages; framing revisions changed only QA camera observation.
Source/tests/config stayed unchanged throughout frozen formal checks and final recordings.

Coverage exercises a small creative course, not every plugin/model/task combination.
Physical phones, Windows, other GPU drivers, arbitrary install failures and general temporal stability remain unverified.
Next work returns to shared material motion stability and indoor/outdoor visual quality, prioritizing same-scene visible benefit against measured loading/rendering cost.
