# Native image prefilter screen — 2026-10-28

The frozen external BOX → LANCZOS candidate is rejected. Product runtime remains
`88dfa18d0afff74c525b4bf261944f5021bd2273bcfd3cbb9b07bf3519099d90`
on merged base `1f8d5e3`; this PR records a screen rather than a material upgrade.

Only the Pillow resampling enum changed in an external module. Existing 512-side,
512² total-pixel, 64-image limits, deduplication, dimensions, aliases, palette and
frontend were fixed. No sharpening, reducing-gap tuning, new shader or dependency.
The plan fixed both ordinary-distance two-asset benefit and a maximum 10% gzip
increase before capture. Neither gate was established/passed; no filter retuning.

| World | Existing gzip | Candidate gzip | Increase | RGBA bytes |
|---|---:|---:|---:|---:|
| Skydio X2 | 126978 B | 142403 B | 12.15% | 1048576 B |
| ANYmal C | 149100 B | 182255 B | 22.24% | 1045568 B |
| Two X2 + rover + arm | 126978 B | 142403 B | 12.15% | 1048576 B |

The composite still shares one image between its two drones, and does not count
as a second independent material benefit. Ordinary views of X2 and ANYmal retain
similar overall detail; close inspection differences do not establish a clear
improvement for both. Equal raw pixel budgets do not mean equal network cost.
Single cache-miss payload samples were 66.35→66.92 ms,193.50→360.20 ms and
47.43→51.31 ms respectively; these are diagnostic samples, not ABBA or first-frame
latency measurements. Cost already rejects the candidate.

Actual current Chrome/Metal captures retain 36 PNGs: three real cached-model
worlds, three views, actual DPR1/2 and before/candidate. CSS1280×800 and buffers
1280×800/2560×1600 were asserted. Root inspected 18 ordinary/detail images; DPR2
images were displayed resized by the tool. This is not a pixel-resolution or
ringing-free temporal certification. No new movie, formal candidate budget,
full suite or installed smoke was run after static rejection.

The final comparator passes 1810 checks. Frozen physics, compiled arrays except
separately retained transient path metadata, mesh/scene/texture layouts,
per-view camera/projection/light/shadow fit, GPU resource/program counts, draw
work and request-path multiplicity agree. Cached source files were independently
hashed after capture against the retained asset manifest. Original13WIP files
and760source/test/script/dependency inputs remain exact. Offline wheel3941294B
contains413product files equal to the current source and previous merged wheel.

All64 selected product tests pass: native texture backend/browser contracts and
experiment records. Ruff check/format and offline wheel pass. This certifies the
unchanged product, not the rejected candidate. Optional mypy/pyright/bandit/
pip-audit are unavailable; there is no current full-suite/budget claim.

Retained harness limits: v1 final aggregation omitted teardown booleans, but its
three actual baseline cases stopped and passed, separately audited; candidate v2
adds aggregation and frozen-candidate validation. DPR1 page errors were not saved
before the next navigation, so no all-page-error-free claim. Initial comparator
failures came from asynchronous frame counters and resource completion order;
v2 excludes only the counter and compares path multiplicities, retaining exact
render work. The original failed report remains. An initial pytest filename typo
ran no tests; its exit4/log remains alongside the final64-pass run.

A separate static transparency inventory did not justify a shared shadow change:
existing-world matches are hidden collision or independent water; Upkie has one
potential torso visual but has not been imported/rendered. This is not an
exhaustive source-image alpha audit or a real visual positive.

Evidence: `../robotforge-evidence/2026-10-28-native-resampling/`, including frozen
plan/module/fixtures, raw PNGs/payloads/model arrays, indexed asset hashes,
comparison attempts, final decision, logs/XML and package comparison. The broad
quality/lightweight/play goal remains active. Next inspect repeated ramp/stair
building through the existing protected native placement and Undo flow.
Individual manipulation success rates remain deferred.
