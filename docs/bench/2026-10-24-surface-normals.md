# Surface detail that keeps its strength across resolutions

This increment makes shared procedural grain and ground-height shading retain their physical slope when drawing resolution changes.
It covers textured robot parts, plugin materials, scene props, floors and terrain.
The visible changes in full application frames are subtle; the measured benefit is consistent surface relief across pixel densities, rather than a new art direction.
Geometry, collisions, controls, material colors, roughness settings and public interfaces remain unchanged.
Weave/cloth remains unchanged.

## Implementation

Previously, normalized position derivatives were combined with per-pixel height derivatives, making resolved bump strength shrink with increasing output resolution.
One shared GLSL helper preserves their relative units and rescales them together to avoid tiny determinants.
Degenerate derivatives retain the original normal, and double-sided direction follows the existing Three.js convention.
Procedural grain height is eight percent of feature pitch times its existing bump coefficient and strength.
Existing per-frequency filtering still attenuates unresolved grain slopes; the previous pixel-dependent normal envelope is removed.
Ground uses its existing metre-scale height setting and a symmetric one-source-texel gradient projected through transformed UV derivatives.
This replaces a forward difference measured over a screen pixel, which also varied with drawing resolution.
Ground height sampling uses four texture reads instead of three.
There are no additional dependencies, texture allocations, meshes, network requests or rendering passes.
The two shader cache keys distinguish the changed programs.

Baseline: `e293c2cb93302866f47bf007e4dbe8072f227d76`.
Candidate runtime SHA256: `3519db85aac71b55fdc983870e9ca57044339f07f0976b2b996b6ba672a4f605`.
Raw evidence: `../robotforge-evidence/2026-10-24-surface-normals/`.

## Resolution and image evidence

An actual Chrome/Metal diagnostic renders the served materials on explicitly synthetic planes at 128 and 512 square pixels with the same world footprint, material, camera and illumination.
Seventy-five paired cases cover eight procedural kinds, three local normal axes, two procedural feature scales and nine ground styles.
The three local axes are viewed head-on from their positive sides; this diagnostic does not establish backface, oblique or minification behavior.
Signed post-bump float normals bypass output tone mapping; complete shaded pictures use the application's neutral tone mapper.
Only the center half is compared against the same source's own high-resolution output, using 4-by-4 normal averaging and RGB BOX downsampling.
Old shaded pixels are not treated as candidate correctness ground truth.

The 63 nonflat strength cases comprise 42 procedural and 21 ground configurations.
Across seven changed procedural kinds, the old 128/512 normal-energy ratio ranges from 3.99112 to 4.00004; the candidate ranges from 0.997781 to 1.000048.
For nonflat grounds it changes from 3.86789–4.02010 to 0.998850–1.004359.
Cloth, checker and flat-interior tile samples are excluded from that cohort.
Tile grout edges are not covered by this diagnostic.

Mean complete-shading resolution error improves for all seven changed procedural kinds and wood ground.
Wood changes from 0.15807 to 0.06329, and knurl from 0.63487 to 0.16656, measured in 8-bit RGB levels rather than normalized 0–1 units.
Six other ground styles have slightly higher absolute error: grass changes from 0.08455 to 0.13349 and dirt from 0.02604 to 0.06345.
Ground wood's absolute normal error also increases as its nonzero relief becomes stronger, despite the improved energy ratio.
Resolution consistency is improved; universal absolute-image-error improvement is not claimed.

All 24 cloth PNG/raw-normal control hashes match exactly, as do all six checker complete PNGs.
Checker's 128-pixel raw normals have at most one float32 epsilon of component roundoff; its 512-pixel raw normals match exactly.
The literal combined exact-hash check remains false in the retained summary.
Separate accepted checks verify finite/error-free normals, exact cloth and checker PNGs, and the explicitly documented float32 component tolerance.
Physics comparisons remain exact.

## Actual scenes and interaction

Four serial A1/B1/A2/B2 runs reuse the exact prior frozen gallery, kitchen, park and wood-table worlds and integration states.
All 87 source, fixture, scene, physics, resource-count and GL checks pass.
Draw calls, geometry counts, texture allocations, scene descriptions and paused initial/final physics remain unchanged.
Actual near, far, tabletop and low-angle ground pictures were inspected; they retain plausible relief without adding geometry.
The wood-floor threshold flicker count changes only from 6925 to 6915 flips, or 483 to 482 per megapixel, in both repeats.
That small change does not establish a general temporal improvement.

Actual native drawer clicks after warm navigation open the panel in all 16 cases.
Pointerdown to two animation-frame callbacks measures 30.0–32.5 ms.
This is UI responsiveness after warm navigation, not first interaction on a pristine machine or GPU presentation timing.

## Loading and rendering costs

Every world still requests 28 HTTP resources, with 1610 additional encoded bytes on fresh navigation.
The wheel grows by 693 bytes, with exactly the two intended runtime files changed.
All raw slow samples are retained.
Gallery fresh-profile times in A1/B1/A2/B2 order are 347.6/1123.3/357.1/334.7 ms; kitchen is 297.1/733.4/285.0/312.5 ms.
Initial shader compilation is a possible contributor to the slow first candidate samples, not a demonstrated attribution.
The machine's browser/driver caches were not reset.

Medians of two navigation observations per source, milliseconds:

| World | Fresh before/after | Warm before/after |
| --- | ---: | ---: |
| Gallery | 352.35 / 729.00 | 189.95 / 207.45 |
| Kitchen | 291.05 / 522.95 | 192.80 / 181.00 |
| Park | 342.30 / 414.15 | 207.35 / 229.10 |
| Wood table | 193.85 / 211.80 | 93.60 / 92.70 |

The gallery near-view synchronized draw median increases from 3.425 to 4.425 ms, with pooled p95 from 5.1 to 5.2 ms.
Other views show mixed costs; no overall speed or memory-saving claim is made.
Navigation readiness is a two-animation-frame proxy on an already-running server, while the standard budget measures CLI startup separately.
Synchronous draws include one-pixel readback; paused RAF cadence does not measure active physics throughput.
Process-tree RSS can double-count shared pages and is not GPU memory usage.

## Verification and videos

All 91 browser-marked cases, eight additional slow asset/sensor cases and 300 focused default cases pass: 399 distinct tests, including six new normal-resolution cases and 21 existing JavaScript parser checks.
The seven unchanged canonical framework cases and real CLI/viewer/MCP smoke pass, including 60 MCP tools and an actual script subprocess.
A second smoke run uses the built wheel installed into a fresh temporary environment with the exact locked dependencies; it passes in 6.5 seconds.
Wheel build, Ruff check/format and diff checks pass.
The complete default Python suite was not rerun for this viewer-only change.
Optional mypy, pyright, Bandit and pip-audit are unavailable locally.

All 11 existing install/viewer budgets pass without relaxation: 3645 KB wheel, 163 MB installed, 42 packages, 52.5 MB dependency download, 0.51 s CLI-to-first-frame, 60.1/60.2 FPS standard/assets at DPR 2, 2.20/3.50 ms render, 132 MB server peak and 5.60% idle CPU.
Import, physics-throughput and MCP timing budgets are outside this partial run.
The accompanying machine report links the immutable experiment record, raw evidence index and complete measurements.
Independent source and Python review found no blocking issues; the evidence review corrected units and excluded flat numerical-noise samples from strength results.
The first GPU attempt lacked local proxy exclusions and did not produce valid material measurements; its terminated log is retained.
The old-source negative test then fails with a measured wood strength ratio of 3.99867.
The first candidate exposed the ground forward-difference issue, which was corrected before final captures.
Its failed test log is also retained.

`surface-normals-before.mp4` and `surface-normals-after.mp4` concatenate actual A2/B2 Chrome screencast frames in gallery, kitchen, park and table order.
Both are H264, 1280 by 800, 30 FPS, 14.233 seconds and 427 decoded frames.
All 16 individual clips and both concatenations fully decode; timestamped source JPEGs remain available.
MuJoCo is paused while the camera moves: these videos show material appearance, not a new gameplay or manipulation capability.
The empty-robot callout is hidden equally by externally logged CSS after navigation measurements.

Use `uv run --frozen`, the pinned baseline archive and saved external harness/fixture hashes to reproduce.
Run heavy tests, captures, simulations and performance measurements serially.
Windows, physical mobile devices and other GPU drivers remain unverified.
The broad visual-quality, lightweight-experience and creative-play goal remains active.
