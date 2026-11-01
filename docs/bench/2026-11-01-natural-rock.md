# Natural rock ground and terrain — 2026-11-01

Rock ground now uses one small local CC0 image pair across floors, terrain and detailed chooser thumbnails.
The existing rock style, authored colour, 0.8 m scale, metric bump strength and physics remain unchanged.
Natural rock uses deterministic translated samples to reduce conspicuous tile repetition; legacy synchronous material calls and failed optional preparation retain the original procedural surface.
Wood, grass and rock prepare independently and concurrently, with a 1,000 ms deadline per optional pair.
Both rock modules load only when rock is used or its detailed thumbnail is requested.
Late completion warms the CPU cache without changing an already installed scene.
No dependency, geometry, extra render target, rendering pass or single-action success tuning was added.

## Content and visual decision

The source is [Poly Haven Rock Face 03](https://polyhaven.com/a/rock_face_03), photographed by Dario Barresi and processed by Rico Cilliers, under [CC0](https://polyhaven.com/license).
The original approximately 2.7 m surface is artistically rescaled to the existing 0.8 m application tile; this is not a calibrated rock-size claim.
See [source hashes and offline reproduction](../../src/robotforge/viewer/static/materials/README.md#rock) for the frozen recipe.
The 256-square JPEG colour and lossless WebP height/roughness images total 78,882 B; their image module is 105,724 B and sampler module 3,248 B.
The portable offline generator reproduced the exact frozen module bytes with locked local dependencies.
There are no runtime requests to the asset provider.

The first direct photo swap was rejected: it improved small stones but made diagonal repetition conspicuous on the Malo floor.
The replacement keeps exactly those image bytes and blends four translated copies with a quintic square partition and explicit texture gradients.
Colour, roughness and all four height neighbours use the same sampling rule.
This costs 24 logical texture fetches instead of six; it does not reproduce the histogram-preserving method from [the referenced stochastic-texturing method](https://jcgt.org/published/0008/04/02/paper.pdf).
Blending softens stone detail and directional patterns remain.

A separate blind three-way DPR1 screen ranked the stochastic candidate ahead of the accepted procedural surface in all four fixed views after identity decoding.
It ranked the direct photo above the stochastic version for one Go2 view and tied them in the other, so the tradeoff is retained rather than described as universal superiority.
DPR2 fixed full-frame checks also retained geological structure without an obvious square-cell seam in the inspected views.
Full-frame previews were resized by the inspection tool; original 2560×1600 PNGs remain available.

## Real application verification

All application captures use actual Chrome/Metal, a 1280×800 CSS viewport and DPR2 drawing buffer 2560×1600.
The robots remain paused: these are camera-motion/material checks, not robot-action success or physical motion certification.
The isolated Malo world changes only its floor style to rock.
The isolated Go2 world retains its concrete floor and existing terrain and adds a fixed, nonoverlapping stones patch.
Original saved worlds are unchanged.

The production viewer loads its own bundled modules with no image, texture-source or shader-hook injection.
Its two fixed native views for each robot match the admitted prototype byte for byte, including exact camera matrices.
Each session preserves all model arrays, qpos, qvel, controls, world description, generation, version and time.
Across prototype/production Malo sessions, two model metadata arrays differ; no claim of complete cross-session model byte equality is made.
The within-session complete invariance gate remains exact.
After the real six-second camera path, both endpoint/reset captures and the final cost capture recover the initial fixed native bytes exactly.
All retained GL and page-error checks pass.

The final production movies each contain 212 timestamped raw JPEGs covering the actual camera action.
They encode to 7.133333 s / 214 frames at 30 FPS and fully decode without errors.
CFR encoding follows the complete raw timestamp sequence; it does not promise every source JPEG appears in the output.
Decoded still inspection is a sampled quality check, not an every-frame temporal-stability proof.

## Cost

Original install/viewer budgets were unchanged: 11/11 passed for this product source.
The benchmark intentionally covers install and viewer sections only; it is not a complete physics/import/MCP benchmark run.

| Metric | Current observation | Existing limit |
|---|---:|---:|
| Wheel | 4,155,676 B / 4,058.2773 KiB | 4,096 KiB |
| Installed core | 164.0203 MB | 180 MB |
| Runtime packages | 42 | 45 |
| Locked wheel downloads | 52.5149 MB | 60 MB |
| Command to first viewer frame | 0.7965 s | 2.0 s |
| Standard warehouse | 60.079 FPS / 3.0 ms | ≥50 FPS / ≤8 ms |
| Asset scene | 60.1169 FPS / 5.0 ms | ≥50 FPS / ≤12 ms |
| Dog cold-start server peak | 134.6 MB | 160 MB |
| Viewer idle CPU | 5.2% | 6% |

The same benchmark separately records a 561.5 MB warehouse server peak during loading; the existing 160 MB cap applies to the dog cold-start cohort, not every loaded robot world.
Additional target-scene warm observations are separate from those formal budget worlds.
The prototype ABBA screen gives Malo baseline 1.5–2.3 ms versus stochastic 5.3–6.3 ms, and Go2 baseline 2.6–3.1 ms versus stochastic 2.8–3.0 ms; all are approximately 60 FPS.
The final production runs give Malo 6.8/5.6 ms and Go2 3.0/3.6 ms, again approximately 60 FPS.
An earlier production capture passed its application assertions but failed socket cleanup and measured Malo 12.9/4.3 ms; both observations and the failed run remain retained.
This is a visible-quality/GPU-cost tradeoff, not a performance improvement or a guarantee for other hardware.
Fresh-profile Page.navigate-to-READY observations are 0.674 s Malo and 0.671 s Go2, distinct from command cold start.
They have no matched baseline navigation cohort and do not establish a loading speedup.

## Checks and retained failures

- Three new CPU asset/source-guard tests pass.
- Sixteen new actual Chrome cases pass with zero skips, including shader rendering, both optional-module failures/deadlines, cache/retry, independent mixed preparation, scene replacement/disposal and actual context loss/restoration.
- Twenty-nine existing wood/grass/ground Chrome cases and four existing surface GPU cases pass with zero skips.
- Source/wheel comparison finds 423 package files byte-identical.
- Whole-repository Ruff lint/format, wheel build and sampler syntax check pass; optional mypy/pyright/bandit/pip-audit are unavailable and are not reported as passed.
- Complete default regression: 4,205 passed, 129 skipped and 421 deselected in 1,341.00 s; zero errors/failures.
- The admitted default/browser/GPU sets contain 4,254 distinct passed cases; three new CPU cases are included in the full default count, alongside two newly discovered module-parse cases.
- All admitted final stage receipts bind the same 793-input source inventory.

The first motion helper failed exact pixels before any candidate import because OrbitControls reconstructed the camera with approximately 1e-13 coordinate differences; 9,106 byte channels differed by at most one.
The replacement pins the preregistered fixed camera synchronously at capture, without relaxing exact pixel equality.
The initial mixed-deadline browser case used a checker-floor fixture while asserting a noise-ground material; the fixture now explicitly starts on concrete and asserts a nonempty initial surface.
The first production capture failed its three-second socket-close wrapper while the native WebSocket close timeout was ten seconds.
The replacement uses a one-second native close timeout, retains the outer guard, and requires actual CLOSED state.
Earlier lint/format and one incorrect offline-generator CLI invocation are retained as failed preparatory checks.

Owned Chrome parents, viewer threads and sockets are closed before postprocessing; root separately observed no live process argv carrying each owned profile.
This is not a claim that all descendant process groups were certified closed.
The 791-input retained CMF worktree and all 13 original WIP files remain unchanged after every execution stage.
No all-browser, all-slow, Windows, physical-device, installed-viewer or universal temporal-quality certification is claimed.
The broad material-quality/playability goal remains active; individual action-success work remains deferred.

## Local evidence

Raw evidence lives outside the package in `../robotforge-evidence/2026-11-01-natural-rock/` relative to the repository checkout.
The source contract is `production-source-freeze-v3.json` (793 inputs); experiment, review, generation, native-frame, movie, cleanup, test and budget receipts are retained there.
The complete production movies are `motion-media-malo-product-v2/workflow.mp4` and `motion-media-go2-product-v2/workflow.mp4`.
The accepted procedural controls and rejected direct-photo screen remain in the same evidence directory.
