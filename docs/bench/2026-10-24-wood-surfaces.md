# Irregular wood for floors, terrain and props

This increment replaces the conspicuous regular bands in shared wood shading with quieter, longitudinal figure and fibres.
The goal is a less distracting ground surface and readable wood on tables, cabinets and park benches while retaining the existing resource layout.
It changes viewer shading only; geometry, collisions, authored base RGBA colours, physics and public interfaces remain unchanged.
The broad visual quality, lightweight experience and playability goal remains active.
Individual manipulation-success tuning remains deferred.

## Implementation

The generated wood floor keeps its 256 by 256 cached tile, eight-board layout and two GPU texture channels.
Two deterministic rectangular noise fields provide long figure and finer fibres, with independent longitudinal shifts and mild tone variation between boards.
Both fields wrap at the tile boundary.
The old 64-cycle sine pattern and hard-threshold joints are removed.
Narrow two-millimetre joints are averaged over the source texel footprint instead of expanding to full dark texels.
Height, colour and roughness variation are controlled separately, so visible wood does not require deep grooves.
The two additional temporary float fields are generated once per cached wood tile and are not retained in its texture cache.
All other ground styles use the same random sequence and arithmetic as before.

Object wood uses two filtered, elongated 3D noise components rather than an equally spaced wave.
The same filtered mean of 0.575 is retained, with separate mean-preserving colour contrast.
The metric bump gradient and existing per-octave filtering remain.
The longest cached geometry-bounds axis becomes a per-material direction uniform.
This matters because MuJoCo may rotate mesh vertices into a principal-axis frame: the real table's local bounds are approximately 0.025 by 0.8 by 1.2 metres, making local X its thin dimension rather than its length.
Keeping a fixed local-X fibre direction made the first candidate's tabletop look mottled.
The actual table and bench select local Z in the final candidate.
One uniform object is shared by the material and its compiled shader; changing its value does not create a new program variant.
Other material kinds return immediately from the orientation helper.

This remains one direction per mesh, with no per-board directions in merged parts, wood growth model or end-grain reconstruction.
There are no new dependencies, requests, texture allocations, meshes or rendering passes.
Existing shaders for nonwood materials remain unchanged in behaviour.

Baseline: `95479c4bfbb60abc467d04b454a59f9e0a198862`.
Candidate runtime SHA256: `04297ab0c66986dc47119a014a1f7f00834106014d5a7cfdfd046da2ea570d67`.
Raw evidence is outside Git at `../robotforge-evidence/2026-10-24-shared-surface-finish/`.

## Visual acceptance and limits

Actual same-camera gallery and kitchen frames show reduced comb-like floor lines and more restrained joints.
Real table and park-bench closeups retain directional, nonuniform detail.
The first candidate removed the bands but looked too flat and mottled on the table; a geometry-aware direction and then separate colour contrast were evaluated in subsequent retained previews.
These are visual observations rather than a claim of photorealism or a general material-quality score.
Independent source and evidence review accepts this scoped static-appearance and bounded-sampling increment.
The coordinator inspected matched PNGs, supplementary terrain views and frames decoded from the actual videos; this frame inspection does not establish dynamic stability.
Robot materials, grass, stone and cloth are not being redesigned in this increment.

The formal comparison uses the same frozen gallery, kitchen, park and wood-table worlds, exact initial integration states and identical cameras.
Near, wide and grazing views include the actual table surface and bench, rather than relying on the older low park view that is largely occluded by concrete blocks.
Supplementary wood-terrain views check stairs and ramp tops/sides separately from the formal performance cohort.
Material-plane controls and three-axis raw-grain diagnostics supplement real scenes; they cannot certify appearance on their own.
Camera-motion videos retain actual Chrome screencast frames and timestamps with paused physics.
They show material appearance while viewing, rather than a new dynamic-task success result.

All heavy tests, simulations, captures, encoding and performance runs are serialized.
Fresh-profile navigation starts on an already-running server and does not clear machine/driver caches.
Synchronous draw timings include one-pixel GPU readback; RAF cadence with paused physics does not establish active simulation throughput.
Process-tree RSS may double-count shared pages and is not GPU memory.
Physical phones, Windows and other GPU drivers remain unverified.

## Sampling and motion tradeoff

The fixed-screen reversal probe regresses reproducibly in both formal repeats.
Table-detail counts rise from 946 to 1926 and floor-grazing counts from 6915 to 19194.
These whole-frame counts include real texture movement and do not isolate aliasing.
They are retained as failed nonincrease checks rather than being pooled into an improvement.

A separate actual-scene channel-isolation diagnostic localizes most of the increased count to the floor colour map, including the floor background in the table camera.
Removing that channel changes table counts to 45/37 and floor-grazing counts to 4040/4604 for baseline/candidate.
Removing floor bump instead leaves 924/1970 and 7013/19191.
Channel removal changes appearance and approximate means, so this is localization rather than independent quality certification.

A second diagnostic compares each source against its own two-times-resolution actual canvas reference at five matched camera positions, 0 through 20 mm in 5 mm increments.
The same renderer, default framebuffer, camera projection/pose, CSS size, sRGB output, NeutralToneMapping, exposure and shadow settings are asserted between the paired resolutions.
Pixel ratio and diagnostic material changes are restored; the camera intentionally advances through the five positions.
Predefined table and floor regions exclude HTML UI and avoid most silhouette edges; a narrow table back-edge strip of about one or two rows may remain.
BOX reduction is evaluated both in encoded sRGB and decoded linear RGB.
The higher-resolution output is a bounded reference, not ground truth.

| Region | Encoded RGB mean absolute error, before/after, 8-bit | Mean change of error fields, before/after, 8-bit | Linear RGB mean absolute error, before/after, unit RGB |
| --- | --- | --- | --- |
| Table | 0.208286 / 0.146130 | 0.192747 / 0.089221 | 0.00076458 / 0.00055191 |
| Floor | 0.376733 / 0.189799 | 0.324264 / 0.171567 | 0.00143612 / 0.00072364 |

These observations support better same-source sampling consistency in these two regions.
They do not establish motion-compensated alias energy, universal temporal improvement or alias-free motion.
Table spatial luminance variation is about 2.34 to 2.92 8-bit levels; floor variation decreases from about 7.35 to 5.43, consistent with the intentionally quieter floor.
These statistics also include illumination and cannot separately quantify wood-fibre contrast.
The original reversal regression remains an open limitation of the increment.

## Fixed-scene integrity

Four final A1/B1/A2/B2 runs use identical frozen worlds, initial integration states, scene payloads and capture helpers.
All 15 nonwood material-plane controls match byte for byte, as do three actual concrete/grass regions across all four runs.
The whole low park frame differs because a real wood table and bench are visible; it is not a nonwood control.
Its 6134 changed pixels, maximum 11/255 difference and original failed whole-frame equality check are retained.
Four camera-coordinate literal equalities fail at less than 1e-15 metres; the separately defined 1e-12 metre numerical check passes.
Physics hashes, resource counts, program counts and GL/console/runtime checks pass.

Supplementary real wood stairs/ramp captures retain geometry and mapping on top and side faces.
All 60 checked face edges retain approximately 0.8333333 repeats per metre, corresponding to the existing 1.2 metre tile.
This does not reconstruct physically correct end grain on cut terrain faces.

The final actual Chrome videos fully decode at 1280 by 800 and 30 frames per second.
The baseline concatenation contains 547 frames over 18.233333 seconds; the candidate contains 546 over 18.200000 seconds.
Raw timestamp spans, encoding duration differences, complete decode results and individual clips remain available alongside both exports.

## Descriptive performance

Each source has two formal samples, taken serially in alternating order on the same Mac.
These are descriptive measurements without a statistical noninferiority claim.
Fresh profile means a new browser profile, not a cold machine or cleared shader cache.

| Scene | Fresh profile proxy, before/after ms | Warm navigation proxy, before/after ms |
| --- | --- | --- |
| Gallery | 319.45 / 323.95 | 209.15 / 194.35 |
| Kitchen | 291.10 / 306.15 | 195.20 / 184.30 |
| Park | 360.00 / 348.45 | 216.20 / 227.30 |
| Wood props | 186.80 / 220.65 | 118.30 / 101.80 |

Draw costs are mixed, including gallery near median 3.975 to 4.800 ms, kitchen low 1.850 to 2.400 ms and park bench detail 3.250 to 2.450 ms.
Gallery near pooled p95 rises 5.3 to 5.6 ms; kitchen low rises 2.5 to 3.4 ms.
Raw samples, all 17 views, RAF cadence, interaction proxies and process-tree RSS remain in the JSON/evidence rather than being reduced to an overall speed or memory-saving claim.
Fresh HTTP encoded resources increase by 1959 bytes in every formal scene; warm bytes are unchanged and the request count remains 28.

## Application verification

All 104 actual Chrome browser cases, eight additional slow assets/sensors GPU cases and 300 focused default cases pass, giving 412 distinct cases.
The three new tests check x/y/z local wood directions on the actual GPU, including uniform identity, finite values, directional gradients and browser errors.
The complete default suite was not repeated for this viewer-only increment.
The wheel contents match package sources exactly; only app.js, ground-look.js and look.js change relative to the baseline wheel.
Ruff, formatting, JavaScript parser checks, diff hygiene and changed-file secret/debug scans pass.
Optional mypy, pyright, Bandit and pip-audit are unavailable and were not installed.
Independent source, Python and evidence reviews approve the stated scope.

Seven unchanged canonical framework cases pass.
Actual CLI, viewer, script and MCP smoke checks pass with 60 tools, including drive, simulation and PNG observation.
A fresh temporary virtual environment installs cached dependencies exported from the unchanged lockfile and the built wheel, then repeats the real smoke checks from outside the checkout.
All 11 existing install/viewer budgets pass without relaxation; other performance sections were not run.
Measured values are 3647.249 KB wheel, 163.1669 MB installed, 52.5149 MB dependency download set, 42 packages, 0.5073 seconds CLI-to-frame, standard/assets 60.1089/60.071 FPS at DPR 2, 3.0/5.0 ms draw, 132.4 MB dog-server peak and 5.4 percent idle CPU.
The separately retained warehouse server RSS is 557.4 MB and is not the dog-server budget statistic.
The matched wheel grows from 3734009 to 3734783 bytes, a 774-byte increase.
No dependency or performance budget files changed.

The JSON report identifies the completed scoped experiment record, immutable artifacts and verified raw-file index.
That scoped completion does not complete the broad product goal.
Further work should return to shared visual quality and free editing/exploration, retaining the unresolved temporal and cross-platform limits rather than endlessly tuning an individual wood view.
