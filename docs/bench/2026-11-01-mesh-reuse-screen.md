# Decoded mesh reuse screen (2026-11-01)

Do not promote this candidate: both real fixtures fail the warm editing latency gate fixed before measurement.
The experimental implementation eliminates repeated imported mesh transfers and GPU attribute uploads, but this is insufficient to establish the intended local editing improvement.
Only this report, its machine record, STATUS and progress are landed; product code and dependencies remain unchanged.
The broader visual, lightweight and creative-play goal remains active; single-action success tuning stays deferred.

## Fixed comparison

Baseline is natural-rock source f4fc01295ae74fe9e197e22795c567198cdae207, tree-equivalent to main7318b9984372cf0b78b314edb889adc994e0c2f4, with793 frozen inputs.
The isolated candidate has796 inputs and adds optional SHA256/length scene metadata plus one current decoded buffer/geometry record.
A hit requires identical full payload identity, length and imported layout; reusable payloads are limited to64MiB, which is not a total memory cap.
Old-server fallback and the raw no-store endpoint/stale409 remain compatible in scoped tests.
The earlier gzip transport rejection is unchanged; no compression parameter search was performed.

The original saved Malo workshop and Go2 warehouse, real cached imported assets, paused time zero,1280x800 CSS and actual DPR2 were used.
Native UI edits target existing primitive boxes marker and crate_42, respectively: Wood, Steel, Original, Undo, ending at Steel.
No extra object, composite-surface eligibility, floor, robot pose or original source was changed.
Each fixture runs baselineA1, candidateB1, candidateB2, baselineA2 serially, four edits per cohort.
The same frozen external recorder observes actual WebSocket sends, scene reception, normal main render submissions, READY, fetches and native GL calls.
The typeahead helper delay and later snapshot hashes are excluded from action timing.
First submitted render is not GPU completion or presentation; instrumentation adds paired overhead.
Two cohorts per side are local observations, not broad hardware statistics.

## Prespecified result

Warm latency must improve by at least max(50ms,5% of baseline median) in both fixtures.
Cold median may regress by at most max(50ms,5%); whole-cohort server high-water may rise by at most64MiB; maximum measured heartbeat lateness may rise by at most max(50ms,5%).
Warm edits must also avoid all mesh responses/imported attribute uploads and retain actual geometry/attribute identity.
Thresholds were fixed in plan-v1.md before product execution and were not relaxed after observing results.

| Observation | Malo baseline | Malo candidate | Go2 baseline | Go2 candidate |
|---|---:|---:|---:|---:|
| Warm send to first main submission median,ms |396.60|392.25|540.05|512.50|
| Warm sample range,ms |366.20–492.00|359.00–476.00|523.80–561.00|487.20–555.00|
| Fresh-page navigate to observed READY median,ms |421.16|416.01|427.51|434.42|
| Mesh response bytes across eight warm edits |376517376|0|162660096|0|
| Imported GL attribute upload bytes across eight warm edits |376517376|0|162660096|0|
| Server scene-rebuild thread CPU median,ms |25.43|40.13|16.05|22.41|
| Maximum measured heartbeat lateness,ms |85.66|104.81|344.17|348.85|

Malo improves by4.35ms and Go2 by27.55ms, both below the50ms gate.
The cold, whole-cohort memory, heartbeat and transfer/upload guards pass; warm latency fails in both fixtures.
Per-phase raw samples and all eight successful cohort manifests remain retained; no resampling was used to obtain a passing result.
The native scene payload is47064672B for Malo and20332512B for Go2 and stays byte-identical through every edit.
Candidate imported geometry and BufferAttribute identities remain identical through each complete cohort; renderer geometry counts remain50 for Malo and137 for Go2.

Most observed action waiting precedes receipt of the replacement scene: baseline/candidate medians329.8/342.8ms for Malo and461.4/460.05ms for Go2.
Scene-receive to main-submission medians improve62.90→43.55ms and75.70→50.85ms.
These separately computed medians do not sum to the total median; they locate a remaining upstream waiting path, not a separately measured compiler cost.
The server still rebuilds the expanded payload and hashes it on every edit, adding CPU work to the candidate.
Do not infer an uninstrumented speedup, remote-network benefit or compiler bottleneck measurement from this local screen.

Whole-cohort server high-water is about2.04–2.07GB for Malo and1.00GB for Go2.
It includes model compilation, retained full-resolution evidence and capture instrumentation; it is not viewer-only RSS, incremental cache memory or the standard dog cold-start memory metric.
No formal install/viewer budget result is claimed for this rejected candidate.

## Correctness and real evidence

40 canonical native2560x1600 snapshots cover initial, Wood, Steel, Original and Undo in all eight cohorts.
All30 phase-aligned comparisons against the corresponding A1 frame match full RGBA bytes, all481 model ndarray fingerprints, payload/layout and qpos/qvel/ctrl/time/paused/world state.
Material tables intentionally change across phases; invariance is asserted between matching phases, not across edits.
Every saved/live world equals the original except the target surface field appropriate to that phase.

70 distinct scoped pytest cases pass on final candidate sources:3 pure/server tests and67 actual browser/GPU cases.
They cover cache identity/length/layout/cap/ownership, warm reuse, replacement/empty retirement, stale responses, builder/partial failures, retry, context restore, borrowed preview clearing and legacy fallback.
Existing startup, part surfaces, camera/continuous placement, native textures and all-surface GPU cases also pass.
Source and Python reviews found no blockers; Ruff and formatting pass.
No full default-suite, full same_behavior run, new install/wheel measurement, Windows or all-browser claim is made.
The optional type/security tools were not installed or run; no paid CI was dispatched.

All793 baseline inputs,796 candidate inputs,791 retained unpromoted CMF inputs and13 protected original WIP files remain exact in their respective frozen runs.
All eight admitted cohorts complete independent reader/ACK/socket/owned Chrome parent/heartbeat/server/renderer cleanup.
Exact owned profile arguments are observed absent after terminal return; no whole-process-group claim is made.

Four complete native UI movies are retained, fully decoded, and reviewed through24 decoded samples.
Malo baseline/candidate:13.100/12.567s and393/377 decoded frames; Go2:13.133/12.933s and394/388 frames.
They cover the complete timestamped workflow without a duration cap;30fps CFR resampling does not promise one decoded frame per input JPEG.
Sampled still review does not establish continuous dynamic quality or a perceptible speed improvement.
The candidate films are experimental evidence, not footage of a landed optimization.

The first Malo recorder attempt failed before any mutation because its expected-world dictionary explicitly inserted a None surface field omitted by model serialization.
The raw original/runtime worlds were identical and all cleanup passed.
Recorder v2 changes only expected optional-field serialization and the versioned method path; actions, timing, cameras and gates remain unchanged.
That unsuccessful v1 attempt and the initial formatting failure are preserved.

## Retained reproduction record and next seed

[Machine record](2026-11-01-mesh-reuse-screen.json) binds source freezes, plan/method/recorder, all admitted manifests, verification receipts, candidate patch and movie hashes.
Local raw evidence is retained in robotforge-evidence/2026-11-01-mesh-reuse alongside both frozen source worktrees; no large model/media files are committed.
The candidate patch is retained locally with SHA256 e4ad6bd0ea24194ec3d74d81e6eef26de57725a387df4de92fd6e3cfc4dc9daa.
Do not revive this candidate as a latency improvement without a new justified workload and new prospective acceptance.

Next seed: persistent named arrangements of supported rigid props, insertable into an existing world without replacing it.
Existing group copy/move tokens are transient, while saved layouts replace the whole scene.
Reuse existing placement validity, authoring, atomic edit and latest Undo rules; store authored relative transforms and parameters, never stale runtime geometry IDs or prepared tokens.
This targets a concrete creative-play gap using existing assets; it is not yet an implemented or validated capability.
