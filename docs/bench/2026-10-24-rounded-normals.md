# Shared rounded-box surface normals - 2026-10-24

Round 67 corrects planar and bevel-tangent shading in the shared rounded-box visual generator.
This is a local shading improvement for generated furniture and robot parts, not a broad texture redesign or a fix for wood-floor shimmer.
The broader shared material-quality, lightweight-loading and creative-play goal remains active; individual manipulation-success tuning is deferred.

## Problem and behavior

`design/visuals/mesh.py` used area-weighted loft normals for rounded boxes.
At the bevel boundary, those normals lean across otherwise planar sides; the existing slab/fence code already documents this failure mode.
The selected table/bench cases measured about 3–7.5 degrees of side tilt and 15–23 degrees at cap tangents.
The corrected parameter normals make those flat/tangent directions axial while keeping the rounded edge smooth.
Radius/center epsilon clamps, shared quantized smoothing keys and singular prior-normal fallback are handled explicitly.

Only `src/robotforge/design/visuals/mesh.py` changes in the runtime package.
All original vertices, triangle winding/indices, smoothing keys, caps and dimensions remain; no new topology, texture, shader, render pass or dependency is introduced.
The public call signature, immutable cached meshes, chamfered-box wrapper and transformed/mirrored normals stay compatible.
Plain primitives, imported CAD normals, ground textures, other loft shapes and lighting are not redesigned.
Normal values can change downstream shading, but authored colors/UVs/material settings and physical data remain unchanged in the controlled worlds.

## Actual scene findings

Actual production gallery, kitchen, park and wooden-prop worlds were rebuilt and observed in Chrome at 17 prescribed views.
Kitchen panel/edge and bench/table edge shading becomes more even; the whole-scene difference is modest and mostly local.
The coordinator inspected actual final stills and decoded before/after camera sheets rather than treating a diagnostic material plane as visual approval.
The earlier external prototype is retained separately and is not substituted for final production-source captures.

Every world compares 480 compiled model arrays; only `mesh_normal` differs.
All other model arrays, scalar fields, collision/inertia data, initial/final frozen physics, geom poses, contacts and saved-world digests compare exactly.
`model_identity=id(model)` is excluded only from cross-process observation metadata, not from physical data.
Expanded viewer payload positions/colors/UVs/heightfields remain byte-exact outside the changed normal blocks, with identical layout/byte count and source normals verified through MuJoCo and `build_mesh_payload`.
All four scene descriptions and all 17 view resource counts/cameras match, with only 1e-12 tolerance on reconstructed camera floats.
The actual floor-only detail image remains pixel-exact; this does not establish wood-floor temporal improvement.

## Verification

Baseline: `eb5d1d4389119f243453a4d8021994b2ba30a825`, runtime `04e265220deedbac623587b7610a34b7b102fa760b79cce9e4bd1ab296505779`.
Final runtime: `eab1c5dd1335f11d71c0d3a1e6a5bbb355fb1b730c57e61af553d83974f43b0c`; all 571 frozen source/test/budget files remain unchanged through final verification.

- 1232 scoped default cases pass, one skipped; complete default suite not rerun.
- 151 complete browser cases and eight additional slow asset/sensor GPU cases pass, totaling 1391 distinct passing pytest cases.
- 32 new normal tests cover planar/tangent normals, small corners, coincident rings, coarse bevels, epsilon keys, finite/unit directions, actual viewer payload and rotated/nonuniform/mirrored chamfered boxes.
- All 18 flat-normal regressions fail against unchanged baseline source before implementation.
- Seven unchanged canonical framework cases and 17 actual physics scenarios match 392 sampled state hashes plus the existing terminal outcome.
- Source and fresh offline locked-wheel CLI/viewer/MCP smoke with 60 tools, Ruff lint/format, 23 JS parses and 392 byte-exact package/source files pass.
- Independent Python and code/compatibility reviews approve their stated scopes without outstanding findings.
- mypy,pyright,bandit,pip-audit are unavailable; these tools were not run or installed.
- 139source/stage/model/payload/resource/media assertions pass in the retained summary.

The first physics wrapper falsely rejected the normal `placed=True` suffix as a non-hex hash, despite identical raw baseline/candidate output.
The original exit1, log and raw files remain; offline correction validates the script's exact suffix protocol and all 392 actual hashes without rerunning physics or selecting results.
The original driver and resumed later stages are both retained, including their real commands and exit codes.
The finite-difference unit-test probe also retains a documented initial cancellation issue; larger geometric sampling spans preserve the original 1e-7 orthogonality tolerance.

## Cost and loading

Exactly one unchanged official install/viewer budget run passes 11/11; other budget groups were not run.
No thresholds, sample counts or aggregation rules were changed and no repeat-until-green measurement was used.

| Metric | Final |
| --- | ---: |
| Wheel |3763628B, +573B |
| Installed / runtime packages |163.2582MB /42 |
| Dependency download |52.5149MB |
| CLI first complete frame |0.5008s |
| Standard / asset FPS atDPR2 |60.1269 /60.1669 |
| Standard / asset draw |2.8 /5.0ms |
| Dog server peak / idle CPU |132.6MB /5.2% |

Both fresh and warm HTTP bytes and 29 requests stay exact in all four worlds.
Calls, triangles, textures, geometries, programs and drawing-buffer sizes remain unchanged at all 17 views.
New work is at mesh construction; per-frame shader code and resource layout stay unchanged.
A single sequential final-then-baseline fixed cohort retains mixed load/draw/RSS observations, not an ABBA or general speedup/stableCPU claim.

| World | Fresh baseline→final ms | Warm baseline→final ms | Chrome tree RSS baseline→final MiB |
| --- | ---: | ---: | ---: |
| gallery | 333.7 → 334.2 | 174.3 → 206.9 | 1997.66 → 1994.61 |
| kitchen | 292.1 → 316.4 | 197.2 → 194.5 | 1866.83 → 1872.69 |
| park | 354.1 → 354.3 | 252.7 → 246.2 | 1981.22 → 1968.86 |
| wood_props | 198.8 → 203.3 | 129.9 → 102.5 | 1825.02 → 1835.56 |

Draw medians also stay mixed: gallery near 3.90→4.55ms and park near 1.45→2.90ms increase, while bench detail 3.50→3.25ms and table detail 1.60→1.50ms decrease.
All original samples are retained; process-tree RSS can count overlapping/shared pages more than once and is not GPU VRAM.
The existing formal budget remains the unchanged acceptance guard, without claiming every timing sample improves.

## Real media

| Recording | Duration s | Encoded frames | File under evidence root |
| --- | ---: | ---: | --- |
| Bench before | 4.300000 | 129 | motion-bench-v2-baseline/park_motion.mp4 |
| Bench after | 4.300000 | 129 | motion-bench-v2-final/park_motion.mp4 |
| Table before | 8.300000 | 249 | motion-baseline/wood_props_motion.mp4 |
| Table after | 8.300000 | 249 | motion-final/wood_props_motion.mp4 |
| Native dynamic rover | 7.500000 | 225 | video-dynamic/screen-drive.mp4 |

Bench and table clips use actual paused simulation scenes and identical prescribed camera trajectories, without changing geometry or materials at runtime.
They show local appearance and do not certify dynamic physics or a reduction in temporal aliasing.
The separate native desktop pointer-only rover video drives into the exact wall at five distinct positive-force contact times, reverses/turns/releases, pauses and resets through the real app.
No keyboard/gamepad, direct teleop injection, pose writes or manual stepping occurs during that dynamic recording; the course/camera is prepared before capture.
All nine encoded originals fully decode and six deterministic decoded frames per movie were inspected, not every animation frame.
The generic park clip framed stairs, and the first focused bench clip inherited an initial low-view camera teleport; both remain as preliminary framing attempts.
Final bench clips set the matching first keyframe before recording; only QA camera setup changed.
Physical phones, Windows, other drivers and every plugin/model/task remain unverified.

## Record and next direction

Evidence is outside Git at `../robotforge-evidence/2026-10-24-rounded-normals/`, with baseline/editable-source archives, source/harness/fixture hashes, raw compiled model/payload files, tests/logs, failed attempts, actual inputs and original screencast frames.
`summary.json`, `raw-index.json` and the unified ExperimentStore `record-summary.json` identify this scoped verification after sealing.
External model-cache paths may require relocation; this is not a portable environment image.
The rejected analytical-noise experiment is retained at `../robotforge-evidence/2026-10-24-analytic-grain/` and indexed into the evidence record.
Its filtering-transition error reduction missed the predefined 15 percent screen; diagnostic near-field gains did not establish actual-scene quality gains, and none of its shader changes entered production.
A failed CC0 webpage read added no asset; blanket new bevel geometry was rejected because existing furniture already had geometric bevels.
Next priority remains a visible shared material improvement, especially indoor/outdoor surface detail and wood-floor motion quality, evaluated independently of this normal correction and against the existing lightweight budgets.
