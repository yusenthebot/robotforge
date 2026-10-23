# Bounded native images for imported and composed robots

This scoped visual increment supports the shared material-quality and playability goal, which remains active.
The old viewer sampled imported images only at triangle corners, losing interior markings, ports and rotor details and assigning the first image to every use of a shared mesh.
The native path now renders the compiled model's RGB/RGBA images through its authored UVs and binds each geom's material independently.
Template surfaces, terrain, physics, controllers, tasks, MCP tools and script signatures remain unchanged.
Baseline is `e6d0fc2cb9a4c9853bfccb412939a6ee325490bc`; verified runtime is `8f26dfb7fcf4630ef4a2c7460a399d947e37d050d3ab72546b526f40a91c21db`.
Raw evidence, full before/after MP4s and matched full-canvas images are retained outside Git at `../robotforge-evidence/2026-10-24-native-model-textures/`.

## Visible result and bounded resources

Actual cached Skydio X2, ANYmal C and a composition of two X2s with template rover/arm robots use identical saved worlds, initial states and cameras before and after.
The coordinator inspected the matched actual images and decoded video contacts: X2 markings and rotor shading, ANYmal ports, bolts and lidar-glass details become visible while their established dark/red palette remains consistent.
Distant composition views show a smaller benefit; close views expose details that vertex-only sampling cannot retain.
No asset-specific brightness or exposure adjustment is used.
The files are actual viewer captures, with physics paused and a moving camera; they do not demonstrate locomotion or manipulation success.

Images are deduplicated by original pixels before budgeting, so the two X2s share one image and retain the same resolution as one X2.
The whole model payload is limited to 64 unique images, a 512-pixel maximum edge and 262,144 base pixels in total.
This bounds decoded base RGBA data to 1 MiB, rather than granting every image its own 1 MiB allowance.
Mipmaps, texture/sampler objects and browser/driver overhead add memory; this is not a GPU-VRAM bound.
Different repeats share a Three.js pixel source, and identical bindings reuse a texture.
The image bundle is gzip-compressed once per model version; the client loads its small module and bundle only for native images.
No new dependency, packaged image asset, frontend build chain or render pass is added.

| Scene | Extra UV mesh data | Gzip image data | Base RGBA data | Unique images / resolution |
|---|---:|---:|---:|---|
| X2 | 61,296 B | 126,978 B | 1,048,576 B | 1 / 512 x 512 |
| ANYmal C | 818,568 B | 149,100 B | 1,045,568 B | 17 / 124 x 124 each |
| Two X2s + rover + arm | 122,592 B | 126,978 B | 1,048,576 B | 1 / 512 x 512, two aliases |

The old float32 geometry/vertex-color protocol remains valid, with optional UV offsets and an additive image endpoint/description.
Old constructors and describe calls retain their defaults, and legacy clients retain approximate vertex colors.
Untextured scenes never request the new module or image endpoint.
Invalid/missing UVs, unsupported channel formats and image-count overflow retain the approximate legacy fallback; full fidelity is not promised there, including its existing shared-mesh limitations.
RGB/RGBA pictures on meshes with authored UVs are the scope; generated primitive UVs and native normal/roughness maps are outside this change.
Compiled RGBA alpha is retained; MuJoCo's default three-channel image loading does not preserve a source PNG's alpha unless configured for four channels.

## Color policy and compatibility

Native pictures deliberately use the viewer's existing sRGB picture-decoding convention.
Compiled MuJoCo color-space enums are retained as provenance, rather than used to change the established palette.
The initially implemented compiled-linear interpretation visibly washed out the real X2 and ANYmal images and was rejected.
Its exact source and captures remain retained, and a separately labelled external sRGB diagnostic is not counted as product verification.
The final product was then captured and tested without that diagnostic override.
This choice preserves the current viewer convention; it does not establish explicit-linear-image fidelity or pixel parity with MuJoCo's classic renderer.
Repeat scaling follows the pinned MuJoCo 3.15 explicit-UV behavior, including nonpositive repeats and size-relative scaling ([upstream implementation](https://raw.githubusercontent.com/google-deepmind/mujoco/3.15.0/src/render/classic/render_gl3.c)).

## Actual loading and rendering cost

Four predetermined A1/B1/A2/B2 cohorts run serially with the same frozen fixtures and capture helper.
A fresh profile measures first complete scene rendering on an already-running local server with a compiled/restored model and existing cached assets.
The machine, filesystem and driver caches are not cleared; this is not a remote-download or process-cold-start test.
Warm means a second navigation of the same browser profile and server, with the server's per-version image payload retained.
The identical cache-miss timing wrapper on each source never builds the payload before the first navigation.

| Scene | Fresh baseline A1 / A2 | Fresh candidate B1 / B2 | Warm baseline A1 / A2 | Warm candidate B1 / B2 |
|---|---:|---:|---:|---:|
| X2 | 219.4 / 210.7 ms | 188.9 / 203.7 ms | 125.8 / 122.1 ms | 120.0 / 120.5 ms |
| ANYmal C | 211.5 / 234.3 ms | 334.0 / 365.8 ms | 151.3 / 108.9 ms | 136.4 / 104.3 ms |
| Composition | 253.4 / 274.1 ms | 251.8 / 256.1 ms | 157.9 / 161.2 ms | 158.3 / 161.1 ms |

ANYmal's first payload build rises from 21.9/22.3 ms to 155.7/157.2 ms, accounting for most of its observed first-view cost.
X2 builds rise from 2.4/2.6 ms to 10.3/9.9 ms, and composition from 4.5/6.8 ms to 14.3/13.4 ms.
Fresh native HTTP totals increase by 192,091 B, 971,485 B and 253,387 B respectively, with 30 requests instead of 28.
The larger UV payload is included in those measured totals.
There is no general loading-speed improvement claim.

Forty synchronous GPU/readback samples per view and independent one-second RAF traces retain every raw timing.
Native-scene median draw samples range from 0.9 to 2.6 ms versus baseline 1.2 to 2.4 ms; individual views improve, worsen or stay similar.
The RAF samples remain approximately 60 FPS at desktop 1280 x 800, DPR 1, on this Mac's Chrome/Metal stack.
Draw calls, triangle counts and geometry counts remain unchanged; live texture counts rise by one for X2/composition and 17 for ANYmal.
Chrome process-tree RSS is about 4 to 26 MiB higher across the paired native scenes; shared pages can be counted repeatedly, so these totals are not additive and do not measure GPU VRAM.
Native World-panel clicks after warm navigation respond in about 30 to 33 ms; this measures that later click response, not the earliest possible startup interaction.
The complete raw first/warm, frame, RSS and resource observations remain in the companion JSON and external manifests.

## Existing budgets and regression controls

Exactly one unchanged full install/viewer budget run on this runtime passes all 11 measured checks.
No threshold, sample rule or aggregation changed, and no extra official runs were used to search for a passing sample.
It remains a partial benchmark: import, physics and MCP performance sections were not rerun.

| Metric | Measured value |
|---|---:|
| Wheel | 3,745,663 B, +3,906 B from main |
| Runtime install / packages | 163.2012 MB / 42 |
| Locked wheel download | 52.5149 MB |
| CLI to first frame | 0.4960 s |
| Standard / asset FPS, DPR 2 | 60.253 / 60.085 |
| Standard / asset draw | 3.3 / 5.0 ms |
| Dog-server peak RSS / idle CPU | 133.4 MB / 5.8% |

The 5.8% CPU sample passes the existing 6.0% rule but does not establish a stable upper bound.
Earlier CPU variability and failed historical records remain unchanged.
The existing gallery, kitchen, park and wood-prop controls reproduce all 17 exact full-canvas PNGs, complete scene payloads, frozen physics and GPU resource counts against the accepted main runtime.
They retain 28 requests and load neither native image resources nor the new module; the shared app script itself grows by 1,213 raw bytes.
Their mixed timing observations remain retained and are not a general speedup result.

486 focused default cases, all 113 browser cases and eight additional slow GPU/asset/sensor cases pass: 607 distinct cases on the frozen runtime.
The complete default suite was not rerun; the previous full-suite result belongs to its original runtime.
The unchanged seven-case framework baseline, actual source CLI/viewer/MCP/script smoke and a fresh offline locked-dependency wheel installation pass, including all 60 MCP tools.
New cases cover interior image graphics, UV orientation, sRGB GPU decoding, distinct images and alpha on shared geometry, unchanged plain materials, repeat semantics, total budgets, gzip caching, failed-load retry, superseded loads and repeated resource release.
Ruff, formatting, 22 JavaScript module parses, package/source equality, scoped secret/debug checks and independent code/Python source reviews pass.
Optional mypy, pyright, bandit and pip-audit tools are unavailable and were not installed.

The analysis verifies 348 source, fixture, scene, geometry, physics, image, resource and control properties.
Both full three-scene H.264 videos are 24.9 s / 747 frames and completely decode; all six original 8.3 s / 249-frame clips and timestamped raw frames remain retained.
Sampled decoded-contact inspection is not a claim of watching every frame or certifying temporal aliasing.
The evidence also preserves failed fixture/helper attempts, the rejected color implementation and the localhost-proxy-stalled test attempt; none is counted as a passing final run.
Cached source assets and licenses, source snapshots and imported helper bytes are retained for reproducibility without adding them to the install or Git repository.

## Limits and next direction

Downsampling trades tiny lettering for bounded resources, and many distinct images share the same total budget.
Explicit linear images, normal/roughness maps, physical phones, Windows, other GPU drivers and comprehensive temporal-quality evaluation remain open.
The native-scene first-view and RSS costs above are accepted costs of this bounded detail improvement, not evidence that every experience is faster.
The broader visual/playability goal remains active; the next increment should let users preserve a named layout, explore another map and restore their creation, alongside shared visual quality.
Individual manipulation-success tuning, releases and paid Actions remain deferred.
