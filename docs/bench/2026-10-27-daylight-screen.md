# Daylight reflection environment: rejected screen (2026-10-27)

The single candidate is **rejected**. It makes the steel sample's reflection more continuous, but ordinary views of Malo, Go2 and Panda mainly become darker and cooler. Robot joints, wood grain and floor detail do not gain enough common readability to justify another environment texture. No candidate renderer code is shipped; the shared material and creative-play goal remains active.

## Controlled experiment

Base main: `f59987a9544f627d75a03e60dcf96d217b13bd2a` (PR #34). Baseline active-source SHA256: `154677d641d7627062d44c0ed35ca49ccb816f4278ba3926bc45e4725deb43d7`. External candidate: `18facda9412945e5b8ec6cc29f562dbe5e50d5e7cd03878267a2428dd12cdddc`.

Before inspecting images, `screen-plan.json` declared one candidate, two fixed views per world and an ordinary-distance acceptance gate: reject a mainly global color shift or a benefit limited to close metallic props. The external copy changes only `look.js` and `lighting.js`: the existing water sky gradient/sun supplies daylight reflections, with its sun aligned to the existing directional key at `[-7,-9,11]`. PMREM resolution stays 256. Direct, hemisphere and environment intensity, exposure, tone mapping, shadow size/algorithm, author materials, background, viewport and physical state stay fixed. This is a visual screen, not a production-ready cache implementation.

Actual Chrome captures cover the existing Malo workshop, Go2 warehouse and Panda workbench, with identical wood/plastic/steel samples. Each has an ordinary view and a detail view at 1280 × 800. Cameras and projection agree within `1e-10`. All 486 compiled arrays per model agree exactly except Malo's isolated mesh-cache path bytes/addresses; those paths resolve to matching file hashes. World YAML bytes match. Draw calls, triangles, geometry and shader-program counts match; all GL checks return zero and pages report no errors.

## Observed result and cost

| Scene | Ordinary-view observation | Textures, before → candidate | Warm compile + Chrome interactive, seconds |
| --- | --- | --- | --- |
| Malo | White shell turns blue; ground/table/wood lose brightness without clearer joints | 8 → 9 | 3.435 → 3.550 |
| Go2 | Boxes and floor darken; no clear leg/joint or wood readability gain | 6 → 7 | 2.212 → 1.976 |
| Panda | Some curved-shell contrast, but dark table and broad floor shadow dominate | 5 → 6 | 1.757 → 1.757 |

Independent visual review reached the same rejection after ordinary views, then detail/original images. The steel sphere's local reflection improvement is insufficient across the three real scenes.

The additional PMREM target is actually 768 × 1024, RGBA half-float, with a depth buffer. Its nominal color-pixel payload is **6,291,456 bytes (6 MiB)**. This is arithmetic, **not measured VRAM**; depth storage, driver overhead and transient generation allocations are excluded. Steady geometry/program/draw counts stay fixed; PMREM generation's temporary work is not separately measured. No downloaded asset or dependency is added by this prototype.

Each view uses 40 synchronous GPU-finished draws (last 30 measured) and 121 RAF timestamps (last 120 intervals). Observed p50/p95 draw times are about 0.10/0.20 ms for Malo and Panda and 0.20–0.30/0.30–0.355 ms for Go2; the candidate Panda ordinary median rises from 0.10 to 0.15 ms. RAF is about 60 Hz with physics paused. These are single sequential warm observations, not cold loading, active-simulation FPS, speedup or installation-budget acceptance. Whole-canvas affine-color residuals in the raw analysis are post-hoc diagnostics dominated by floor/background, not perceptual quality scores or acceptance thresholds.

## Actual recordings and limits

Six original browser screencasts follow the same six-second scripted camera path: ordinary view, ease to detail, hold, return. Physics remains paused; no physics steps or pose writes occur during recording. These show actual rendering, **not native gameplay input**. Initial/final qpos, qvel, control, simulation time and world bytes remain unchanged. Each recording closes Chrome and its server cleanly.

All six movies strictly decode in full. Their complete timestamped JPEG inputs and original manifests are retained; 30 fps CFR encoding resamples those inputs and does not promise every JPEG appears in the movie. The root inspected all 36 sampled decoded stills. Samples do not certify every dynamic frame or temporal quality.

Raw evidence lives outside Git at `../robotforge-evidence/2026-10-27-daylight-environment/` relative to this checkout. It contains the predeclared plan, two immutable source copies, helpers, source receipts, 12 original PNGs, compiled arrays, `screen-analysis.json`, ordinary/detail comparison sheets, and `movie-{before,after}-{malo,go2,panda}/workflow.mp4` with raw frames, manifests and decode QA. The companion JSON records hashes and concise results; it is a local benchmark report, not a registered experiment-run manifest.

Cache ownership, context restoration, multiple renderers, sensor boundaries and shared water/daylight lifecycle are **not certified for the rejected prototype**. No candidate full suite, installed smoke or original 11-budget run was performed after visual rejection. PR #34's production validation remains scoped to its unchanged runtime; it is not relabeled as candidate validation. This documentation-only change passes an offline wheel build, 407 product-file byte comparisons, Ruff lint/format, 98 experiment-record tests and 3,484 evidence integrity checks (including 2,161 raw JPEGs). The original checkout's 13 pending files retain their hashes. Optional mypy/pyright/bandit/pip-audit are unavailable, not passed or installed for this report.

Keep the frozen sky candidate as negative evidence; do not retune it until it looks acceptable. The next finite visual investigation should address local contact/depth readability and test its real cost, preserving the same ordinary-distance gate, sensor behavior and lightweight loading constraints. Single-action success tuning stays deferred.
