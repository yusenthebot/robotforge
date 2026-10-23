# Named creative layouts, 2026-10-24

Users can keep a named starting layout, explore another map, restore their creation and drive again.
This round improves repeatable creative play while preserving the accepted visual baseline; it does not tune individual manipulation success rates.
The broader visual quality, lightweight loading and playability goal remains active.

## Scope

Scene & build → Maps adds Save current, Refresh, Restore and confirmed Remove in English and Chinese.
Each world keeps up to 16 immutable records, with normalized unique names of up to 64 characters and at most 512 KiB of authored world YAML per record.
The full authored World descriptor is retained, including robot starts, props, terrain and configuration; this is not a live pose/velocity/time, controller, sensor-state or policy snapshot.
Model/plugin/policy references keep the original world asset base; their external contents are not bundled.
Named layouts are separate from the 50-entry pruned undo history, and ordinary world undo can recover the design preceding a successful restore.
Listings are requested when needed by the panel or scene transitions; no periodic layout-directory scans or new frontend module requests are added.
MCP and script signatures remain unchanged.

Restore validates names, schemas, sizes, checkpoint hashes, current-world hashes and scene version/generation, then compiles before publication.
Staging publishes derived XML before authoritative YAML, with best-effort rollback on installation failure.
Built-in runtime participants preserve old model/data, controllers, sensors, maps, flow and task-container identity on an ordinary failed install.
Active navigation/errands, experiments, demonstration recording/replay and custom reset/controller callbacks must stop first; their external effects are not claimed reversible.
Committed temporary-file cleanup and history-pruning failures are warnings rather than false failed-restore results.
Individual file replacements are atomic; the combined disk/model operation is not crash-atomic.
A rollback I/O failure pauses and invalidates guarded layout editing and asks the user to reopen; generic Play remains available.

## Actual creative workflow

The same rover/barrier/ramp fixture is used on accepted main `9b110b041b66de57b701ecdb7f478776be029090` and candidate runtime `6ba2cec5161cc5afdb05dce2addc0a6a471071a793906da8b59eded0dcdfbaf7`.
Both recordings add a pin through native browser input.
The baseline then switches to Empty floor: its props/terrain are replaced and no native named-layout control exists, although ordinary undo and agent APIs remain available.
The candidate saves the edited creation, switches map, cancels a restore once, confirms restore, reloads the page, cancels removal, drives and confirms removal without changing the current world.
Actual server-state observations, complete WebSocket events, original checkpoint bytes and timestamped screencast frames remain retained.

| Recording | Duration | Fully decoded frames | Actual restored-course drive |
| --- | ---: | ---: | ---: |
| Baseline, desktop English | 4.60 s | 138 | No restored-course segment |
| Candidate, desktop English | 8.67 s | 260 | 0.511 m |
| Candidate, phone-sized Chinese | 8.50 s | 255 | 0.510 m |

The phone-sized run uses native emulated touch for layout/build controls and keyboard input for driving; it does not verify a physical phone or touch-only driving.
All three videos encode the full retained timestamp span without a duration cap and fully decode as H.264.
Six decoded stills per video were visually inspected; this is sampled inspection, not a claim of watching every frame or certifying temporal quality.
A separate two-interpreter run saves a layout, changes to the warehouse, exits, reopens the world in a new process, verifies the identical record ID/hash, restores the pin and drives 0.570 m.
This extends the unit test that creates another Playground in the same process.

| Local WebSocket action reply | Desktop English | Phone-sized Chinese |
| --- | ---: | ---: |
| Save | 11.0 ms | 16.4 ms |
| Restore | 74.3 ms | 42.9 ms |
| Remove | 16.2 ms | 16.4 ms |

These are single-run request/reply latencies from actual CDP network receipts, not first-render or general asset-loading guarantees.

## Compatibility and costs

735 scoped default cases, all 120 browser cases and eight additional slow asset/sensor cases pass: 863 distinct pytest cases, with one scoped default case skipped.
The complete default suite was not rerun.
The unchanged seven-case framework baseline and actual source/fresh offline locked-wheel CLI, viewer, 60-tool MCP and script smoke pass.
The 48 new Python cases cover immutable save boundaries, schemas, capacity, relative models, stale requests, publication/rollback failures, runtime participants and exclusions.
Seven new real Chrome cases cover four language/viewport journeys, confirmed removal and three delayed list/pose/mutation generation races.
Ruff, formatting, 22 JavaScript parses, exact wheel/source equality and independent Python/frontend source reviews pass.
Optional mypy, pyright, bandit and pip-audit are unavailable and were not installed.
No new dependencies, packaged assets, render passes, releases or paid Actions are introduced.

Exactly one unchanged install/viewer budget run passes all 11 measured limits.
No threshold, sample duration or aggregation rule was changed, and the budget was not repeated until green.

| Metric | Observed candidate |
| --- | ---: |
| Wheel | 3,756,943 B; +11,280 B |
| Installed size / package count | 163.2361 MB / 42 |
| Dependency download | 52.5149 MB, unchanged |
| CLI first frame | 0.4978 s |
| Standard / asset scene FPS, DPR 2 | 60.099 / 60.061 |
| Standard / asset draw | 2.5 / 3.7 ms |
| Dog-server peak RSS / idle CPU | 133.0 MB / 5.2% |

The CPU sample passes the existing 6.0% rule; it does not establish a stable upper bound or a general speedup.
The fixed gallery, kitchen, park and wood-prop controls reproduce 17 exact full-canvas PNGs, complete scene payloads, frozen physics and GPU resource counts against the retained accepted-main evidence.
Each control still makes 28 HTTP requests; fresh navigation transfers 12,557 extra bytes and warm navigation 1,278 extra bytes for shared frontend changes.
Fresh complete-scene times are 324.1/316.9/348.6/205.6 ms versus retained baseline 327.4/305.6/372.0/192.7 ms; warm times are also mixed.
Those baseline observations were retained from the preceding round, not a simultaneous ABBA run, so no causal timing or memory improvement is claimed.

## Evidence and limits

The evidence root is `../robotforge-evidence/2026-10-24-saved-layouts/` outside the repository.
`verification-inputs.json` freezes 564 runtime/test/config inputs; `summary.json` checks 110 functional, package, receipt, resource and control properties.
`raw-index.json` and the ExperimentStore record in `record-summary.json` preserve source/helper snapshots, failed attempts, logs, original media and successful final receipts.
Early fixture/API mistakes, unsupported build arguments and helper setup failures remain retained and do not count as passing final verification.
No product runtime changed during the frozen formal checks or recordings.

Real browser coverage exercises a small creative course; complete-World serialization does not establish every task/plugin/model combination.
Missing relative model files fail before replacing the current creation, but layouts are not portable asset archives.
Windows, physical phones, other drivers and crash recovery across arbitrary callbacks remain unverified.
Next priority is reusing configured props for course variants through existing selection/placement controls, alongside continued shared visual-quality work.
Single-action success tuning stays deferred.
