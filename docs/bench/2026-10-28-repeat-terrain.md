# Repeated local terrain — 2026-10-28

The native ramp/stair picker now offers opt-in Keep placing. Each success keeps
the preset, heading and snap spacing, waits for the matching world to render,
then clears the old point. Done/Escape exits. Default single placement and
existing terrain edits retain their original behavior. Every addition still
recompiles, returns all bodies to saved starts and pauses; this is not batching,
automatic slope joining or a controller-success change.

The next request refreshes version, generation and world digest only after
matching state/scene/poses/rendered versions and terrain metadata agree. Missing,
mixed or malformed metadata ends repetition; a current committed addition keeps
Undo, while a stale reply cannot restore it. Existing occupancy, persistence,
rollback and latest-only Undo remain server-owned, with no new wire/API/schema.
The new hint keeps the saved-start/pause behavior visible after every addition.

## Real behavior and unchanged appearance

Four native Chrome/Metal movies use copied original Malo workshop and Go2
warehouse worlds. Both versions build the same three standard pieces, include
one overlapping-footprint rejection/retry, undo only the last piece, save a
layout, switch maps, restore and reload. Final saved worlds agree exactly.
The old UI enters the preset three times; the new UI once. This is an observed
interaction-count reduction, not a human timing or driving-success claim.
Malo uses 15° ramps; Go2 uses 15°/30° stairs and native touch drag/pinch without
another submission. All eight per-cohort request guards match actual world
version/generation/digest. Targets are mathematically projected, physics paused.

| Recorded workflow | Before | Candidate |
|---|---:|---:|
| Malo desktop,1280×800 | 27.367 s | 22.333 s |
| Go2 Chinese touch emulation,320×812 | 18.100 s | 16.967 s |

All four full timestamped MP4s decode; root inspected24decoded sampled frames.
These samples do not certify every video frame. Full raw frames, timestamp
ordering, encoder/decode reports and reload boundaries are retained. Movies
include explicit reloads; they are not locomotion or physical-phone validation.

Three untouched Malo/Go2/Panda worlds retain48static screenshots and24pixel-exact
before/candidate pairs at actual DPR1/2, studio/daylight and ordinary/detail views.
The 485model/state fields per world stay unchanged within each session; compared
compiled arrays agree except separately retained transient path metadata.
Changed absolute mesh paths are independently resolved/hash-compared; identical
relative labels are not claimed as independently resolved source-file checks.
Camera eyes/targets use1e-10tolerance; projection, buffers, materials, draw work,
GPU resources/programs and request-path multiplicity agree. No new shader,
texture, pass, asset or dependency, and no new shader-aesthetic gain claimed.

## Verification and cost

Frozen runtime: `bd83fe3b4eee1e7ec8059194cdd4767ad54c8ee8c194b4147db8748108502b61`,
761source/test/script/dependency inputs. Final scoped selections:15terrain browser,
88shared creation/touch/snap/heading/view/copy/layout browser,579backend and9GPU
cases:691distinct pytest passes, zero skips/failures. Seven unchanged canonical
framework cases and source/fresh offline installed CLI/viewer/MCP smoke pass
with60tools. Ruff/format, three JS parser checks and offline wheel pass.
This is not a complete default, slow or all-browser suite. Optional mypy/pyright/
Bandit/pip-audit are unavailable; failed module attempts and limited added-line
secret audit are disclosed separately.

Only app.js, creation.js and i18n.js change in the413product package files.
Source/wheel/fresh installed files match; all backend/asset/dependency files stay
exact. Wheel3941555B(+261B);163.6382MB installed,42packages. Original13WIP files
remain unchanged. The original11install/viewer budget checks all pass once,
without changed thresholds: standard CLI first frame0.8366s,60.11/60.23FPS,
2.9/3.6ms draw,134.3MBdog-server peak and4.2%idleCPU. These standard benchmark
worlds are distinct from the original Malo/Go2 workspace timing below.

Fresh HTTP encoded resources increase1137B, warm329B, with30requests unchanged
in the initial load cohort. The standard budget pass does not make workspace
loads cost-free. Retained candidate-minus-before sample means:

| Cohort/order | World | Browser-cache-cold | Warm reload |
|---|---|---:|---:|
| Initial ABBA | Malo | +103.55 ms | +54.25 ms |
| Initial ABBA | Go2 | +174.20 ms | +74.10 ms |
| Supplemental BAAB | Malo | +7.15 ms | +4.95 ms |
| Supplemental BAAB | Go2 | −32.05 ms | −8.70 ms |

The one reversed-order supplement investigates the initial slower samples; it
never replaces them or the original budget. Combining all four samples per
cohort/world gives cold+55.35/+71.08ms and warm+29.60/+32.70ms respectively.
Small samples and order variance do not establish causality or universal speed,
steady memory or zero-loading-cost claims. Servers are already compiled and
local model/OS caches warm. Ready is main scene submission plus2RAF/gl.finish,
not all sensor previews compiled, server cold start, remote download or human
input latency. RSS snapshots are not VRAM/peak measurements.

Retained failures: initial new-test await/count precedence and mistaken receipt
removal expectation caused3of6tests to fail; corrected tests assert actual stale
Undo refusal. Two review gaps were strengthened with exact saved bytes/receipt/
version after occupied failure and canvas-hit/zero-send counts during waiting.
A123character assertion was split with identical Python AST for Ruff, then all15
terrain tests reran. A bare system-Python provenance import lacked pydantic;
formal checks use frozen project Python. Initial orchestration was corrected to
the project interpreter before smoke. No product algorithm or budget was tuned
to make a check pass. Independent code and Python-test reviews approve the scope.

Evidence: `../robotforge-evidence/2026-10-28-repeat-terrain/`, including frozen
before/candidate static files, original and final inputs, all failed/final XML
and logs, real recordings/model arrays, comparison/load cohorts, package bridge
and source/installed smoke. User-facing behavior is in [CREATIVE_PLAY.md](../CREATIVE_PLAY.md).

The broad material/lightweight/play goal stays active. The next candidate screen
is negotiated, current-version cached mesh transport compression: actual
Malo/Go2 responses are47,064,672B/20,332,512B. Preserve decoded mesh/interface bytes
and first measure CPU, event-loop responsiveness, memory and cold/edit loads.
No compression is implemented by this change; individual action success stays
deferred.
