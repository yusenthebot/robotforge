# Natural wood ground, 2026-10-24

Round69 evaluates an unmerged shared ground-material candidate; the broad material-quality and playability goal remains active.
The same ground image pair is used on wood floors, ramps, stairs and heightfields; object/table/robot wood remains unchanged.
Normal-size gallery and kitchen stills show irregular fibres and knots instead of the old parallel comb pattern.
512 colour was selected over 256 because the actual floor-detail view retains useful fine fibres; repeating knots remain visible.
This is an appearance improvement, not proof that temporal shimmer, repetition or all material quality issues are solved.

## Source and compatibility

Baseline `db0f4f1c95144adeed423f45e2af01696fd885a0` / runtime `eab1c5dd1335f11d71c0d3a1e6a5bbb355fb1b730c57e61af553d83974f43b0c`.
Candidate runtime `3b6c8f04601d3dd768c47c759e687a75ea6c342d5efa3e95cad6d268371051b7`; 619 source/test/script/lock/budget files remain unchanged through verification.
Image data comes from [Oak Wood Planks](https://polyhaven.com/a/oak_wood_planks), by Dimitrios Savva / Poly Haven, under [CC0](https://polyhaven.com/license).
Exact original SHA256s, the offline recipe and derived hashes are retained; no asset service, API key, runtime network dependency or new package is introduced.
Linear-light BOX filtering and grayscale mean normalization preserve the authored ground tint; the decoded albedo multiplier mean changes by about 0.00236 percent.
The original two map slots, 1.2m tile scale, filtering and shader remain.
The old synchronous three-argument groundMaterial API keeps its procedural result; the app passes the prepared optional fourth argument.
Actual wood usage triggers one local dynamic module; both images decode atomically before installation behind the existing scene-request gate.
A 1000ms scene wait bounds optional loading; errors/timeout retain a stable procedural fallback and warn once.
Late completion can populate the shared CPU cache for future scenes; it cannot replace an installed fallback or a newer scene.
Thumbnail preparation and WebGL restoration are covered by actual browser regression tests.

## Verification

1450 distinct passing pytest cases; 11 new browser cases are included once.

- focused-default: 1280 passed, 1 skipped, 0 failed.
- browser-full: 162 passed, 0 skipped, 0 failed.
- slow-assets-sensors: 8 passed, 0 skipped, 0 failed.

Seven unchanged framework baselines, source CLI/viewer/MCP smoke and a fresh offline locked-wheel install/smoke pass; the MCP smoke lists 60 tools.
Ruff, formatting and JavaScript parser checks pass.
Complete default suite was not rerun; mypy, pyright, bandit and pip-audit were unavailable and were not installed.
Five actual worlds have identical compiled numeric model arrays, scalars, frozen physical states, expanded mesh bytes and scene payloads.
The terrain capture retains exact positions, UVs and repeats, with independent face-edge/projected-XY checks of the 1.2m pitch.
All 22 view cameras, resource counts and draw geometry controls are compared; actual selected image material is recorded.
226 of 229 scoped checks pass; budget failures and the corrected initial terrain timeout remain explicit; pixel differences are separately retained, including any unchanged-park mismatch. They are not silently treated as exact controls.
External preview first park-far captures differed despite equal post-shot cameras/resources; the cause remains unproven.
The production capture records an explicit readback before each screenshot; this does not establish the cause of the earlier mismatch.

## Cost

Derived image data is 86,881B; the lazy local module is 116,493B.
512 colour / 256 detail adds about 1MiB nominal RGBA+mip storage relative to two 256 maps; CPU and driver allocations are separate.
Rebuilt wheel 3,853,599B versus baseline 3,763,628B, delta +89,971B; all 394 packaged source files match byte for byte.
Exactly one unchanged official install/viewer run passes10/11: standard warehouse GPU-synchronized frame time14.7ms exceeds the8ms maximum. Candidate remains unmerged. No threshold was relaxed; other budget groups were not run.

| Measurement | Candidate |
|---|---:|
| Installed / packages | 163.3734MB / 42 |
| Dependency download | 52.5149MB |
| CLI to first frame | 0.5495s |
| Standard / asset FPS | 60.1009 / 56.4784 |
| Standard / asset draw | 14.7 / 7.7ms |
| Dog server peak / idle CPU | 133.3MB / 4.2% |

Supplemental four-sample ABBA on the same standard warehouse retains all results and confirms zero natural-wood module requests. It diagnoses the failed measurement and does not replace the official budget.
The original ABBA was baseline3.1 / candidate14.9 / candidate14.4 / baseline2.9ms, with equal camera/geometry and zero wood module requests.
A subsequent fixed2x2 baseline/old-app/old-ground/candidate isolation yielded12.8/11.4/11.6/11.3ms; all18 compiled shader sources, cache keys, context and resource counts matched.
One additional instrumented baseline/candidate loop measured12.2/7.6ms. Both had zero texture uploads, mip generation and program creation/linking, equal draw counts and0.9ms median CPU render submission; readback medians11.25/6.7ms differed.
Texture content/sampler state, lights and shadow state matched. Two transparent selection materials had version688/686; captured model/model-view matrix uniforms differed by up to0.000147. This is not complete state equality or a proven cause.
The cohorts disagree on a stable version effect. No source defect or environmental cause is established; original budget failure remains the merge gate.
Fixed-world samples below were collected sequentially baseline then final, not ABBA; every slow sample remains in JSON.
They quantify this local capture and do not prove a general speedup, stable CPU, VRAM usage or reduced memory.

| World | Cold first-frame ms before / after | Warm ms before / after | Chrome tree RSS MiB before / after |
|---|---:|---:|---:|
| gallery | 512.9 / 691.3 | 212.2 / 251.0 | 2031.3 / 2008.1 |
| kitchen | 512.4 / 640.5 | 194.5 / 224.5 | 1915.8 / 1902.9 |
| wood_props | 468.5 / 455.3 | 117.6 / 125.6 | 1868.9 / 1872.6 |
| park | 686.7 / 537.5 | 328.4 / 262.7 | 1997.4 / 2001.6 |
| wood_terrain | 544.1 / 433.7 | 144.2 / 145.9 | 1881.2 / 1876.7 |

Navigation request/byte deltas, draw samples, FPS and the existing flicker results are retained in JSON, including mixed/regressive values.
The old harness counts the two data-image responses as HTTP resources: wood29→32 comprises one new local HTTP module plus two embedded data images; actual HTTP29→30. Encoded HTTP increment is118879B fresh and150B warm in these wood captures.
All five final park views are pixel-exact after the explicit readback; the old preview mismatch remains a separately retained unresolved observation.
Nonwood park navigation does not request the image module; opening an explicit wood thumbnail is allowed to load it.

## Actual media and failures

Seven original MP4s fully decode with strict FFmpeg error handling; raw frame bytes, timestamps, hashes, ffprobe output and six decoded review frames per movie are retained.
Gallery, kitchen and wood-terrain before/after camera movies retain paused physics.
The separate native desktop rover movie uses trusted pointer events for drive, positive-force barrier contact, release, reverse, turn, pause and reset.
The dynamic clip requires the actual natural-oak 512/256 material before and after recording; it verifies driving compatibility, not a new playability feature or action-success improvement.
Six sampled decoded frames per original were visually reviewed; not every-frame visual certification.
Initial lossless-512 preflight exceeded the unchanged 128KiB image cap and is retained; fixed JPEG90 was selected before actual candidate observation.
The first development run failed because its fixture changed a different world from the served qa world; it was interrupted, preserved and corrected with the public set_floor API.
The first terrain-only capture also timed out the inherited time>0 gate on a pausedtime0 world. Its failed log/script/input set remains; the four successful baseline scenes were kept and only the missing terrain resumed using actual renderedVersion/loading plus2RAF.
The corrected 11-case development pass is not counted again in formal totals.
Physical phones, Windows, other drivers, every plugin/model/task and temporal shimmer elimination remain unverified.

## Reproduction

Raw evidence: `../robotforge-evidence/2026-10-24-natural-wood/` relative to the checkout.
The directory retains frozen inputs, original/source/fixture archives, local and transitive helper hashes/copies, all previews/failures, official budget logs, JUnit, model/payload snapshots, original frames and videos.
The ExperimentStore record seals a failed-budget verification; broad goal status stays active.
External robot mesh cache paths remain environment dependencies; the evidence bundle is not a portable operating-system image.
Recipe and license: `src/robotforge/viewer/static/materials/README.md` and `scripts/prepare_ground_wood.py`.

The first seal attempt indexed its live redirected stdout before final output changed it.
Its failed record and original script/index/log are retained; corrected sealing excludes live stdout and is verified separately after completion.
