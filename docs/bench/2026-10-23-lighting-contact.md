# Shared lighting and contact readability

The dominant direct light now shares the studio reflection panel's direction.
One existing 2048 x 2048 shadow map fits visible caster bounds in a fixed light basis, follows streamed movement and includes ground receivers beneath elevated robots.
Adaptive bias replaces the previous 20 mm normal offset, improving small wheel, foot and tabletop contact readability.
Body-local bounds, cached fixed transforms, an idle shortcut, guard bands and texel snapping limit fitting work and unnecessary movement of the projection.
Imported movable bodies, transformed floors, rotor envelopes and deforming meshes are covered.
Reset refits the projection; viewing camera movement does not change lighting.
No extra shadow maps, rendering passes, material formulas, dependencies, physics or public interface changes are introduced.

## Matched evidence

Baseline commit: 714806452a8f3b8b61c6ccdf39793a4dc302d099.
Accepted runtime SHA256: 3d820d1407dab3407a5108d64396bd4b714a7b0df20553729a18cb70ce69920d.
The runtime remains unchanged throughout accepted measurements and verification.
Before, after, before2 and after2 run serially with four identical worlds and frozen cameras at 1280 x 800, DPR 1 in actual Chrome/Metal.
World, streamed scene and initial time/qpos/qvel/ctrl files match byte for byte in all four runs.
The contact world includes a template rover, real cached Go2 and an arm on a table; other worlds cover the robot gallery, kitchen and outdoor park.
Accepted captures use application lighting without runtime light overrides.
Earlier diagnostic bias/direction sweeps and candidate-preview captures remain separate.

Kitchen shelves and tables show stronger directional depth, while wheel, foot and arm-base contact shadows are easier to see in the inspected A/B images.
These observations support contact readability and lighting consistency, rather than a universal material-quality score.
No broad shadow acne is visible in the inspected static contact, kitchen and park views.
All 16 scene captures and the final video have zero JavaScript, runtime and WebGL errors.
Corresponding draw calls, triangles and texture counts remain unchanged.

## Measured cost and limits

The candidate adds 7,841 encoded HTTP bytes and two resources; the built wheel grows by 3,179 bytes.
First-frame timing starts with the server already running and includes scene/time readiness plus two animation frames.
A fresh isolated profile does not flush machine-wide GPU caches.
Two samples per side are observations, not strong statistical estimates.
Render timing uses 40 synchronous render and single-pixel readback samples per camera, distinct from active simulation throughput.

| Scene | Fresh before / after, ms | Warm before / after, ms | Near render median before / after, ms |
| --- | --- | --- | --- |
| Gallery | 428.9, 423.8 / 440.1, 429.0 | 177.3, 200.9 / 249.9, 267.6 | 4.2, 4.2 / 4.2, 4.1 |
| Kitchen | 409.1, 420.1 / 397.3, 421.4 | 277.6, 279.8 / 283.6, 258.6 | 2.3, 2.5 / 2.3, 2.2 |
| Park | 486.8, 460.2 / 473.7, 494.0 | 298.7, 304.3 / 229.9, 244.2 | 3.5, 2.4 / 3.0, 3.0 |
| Contact | 325.6, 345.5 / 292.4, 291.3 | 216.9, 200.2 / 179.1, 180.5 | 3.2, 3.2 / 3.4, 3.2 |

Warm gallery loading is slower in both pairs, by about 69.7 ms between the two-sample averages.
There is no overall startup or performance improvement claim.
The low-angle temporal heuristic is mixed: kitchen wood 594 to 642, grass/heightfield boundary 30 to 43, stairs 227 to 234 and gallery wood 897 to 822 flips per megapixel.
This paused 16 x 5 mm camera translation probe includes geometry, material and shadow edges, so it does not isolate shadow shimmer.
Wood temporal stability remains open; no blanket flicker reduction is claimed.
Map width shrinks from 34.925 m to 26.909 m in the park and from 4.057 m to 2.828 m in the contact world, without increasing map resolution.
Bias stays scale dependent, so an arbitrarily large scene can still lose small contact detail.

## Actual interaction recording

The native Chrome/MuJoCo video is 39.766667 seconds at 1280 x 800, 30 FPS.
It covers six placements including a real tabletop arm at surface [0.48, 0.28, 0.75], joint control, rover drive/stop, shadows off/on, stairs/hills/flat, wood floor, desk map and paused reset.
Native W-down/W-up moves the rover 2.375704 m and sends an explicit zero-speed stop.
Its final position projects beyond the initial shadow frustum; the application follows it with updated coverage.
The two-second active sample is 60.002 FPS.
The full video decodes, and the decoded contact sheet and full contact screenshots were inspected.
Camera hooks are used only for framing; actions use native mouse/keyboard events.
Abrupt camera setup cuts and wide multi-robot views make this a functional validation recording.
Phone EN/ZH creation is checked in the current browser suite but is not re-recorded in this lighting increment.

Video: ../robotforge-evidence/2026-10-23-lighting-contact/video-final/desktop.mp4 relative to the repository root.
SHA256: 6032e9d90e4fd219cbb0bd4a69c7df4f5d03ca74d0fdb0c597644d364848e07c.
Raw manifests, screenshots, physics arrays, resource timings, all render samples, native events, decode log and capture harnesses remain in that external evidence directory.
The companion JSON records hashes and the immutable unified verification record.

## Verification and reproduction

186 focused tests, 16 actual Chrome lighting/creation/gamepad cases and eight GPU/asset/sensor cases pass.
The browser checks include actual shadow raster contrast, transformed floor receivers, fixed-class moving imports, flex and rotor bounds, camera invariance and driving/reset beyond initial coverage.
Independent source review and acceptance-evidence review approve the final change.
Ruff check/format, all 20 JavaScript parser cases, wheel build, seven unchanged canonical framework cases and actual CLI/viewer/MCP smoke pass.
The complete default suite is not rerun for this viewer-only increment.
Optional mypy, pyright, Bandit and pip-audit are unavailable locally; native Windows remains unverified here.

All 11 applicable existing install/viewer budgets pass without relaxation: 3,637 KB wheel, 163 MB installed, 42 packages, 0.51 s CLI start-to-frame, about 60 FPS at DPR 2, 2.50/3.40 ms standard/asset frame measurements, 132 MB server peak and 5.20% idle CPU.
Import, physics-throughput and MCP benchmark sections are outside this partial budget run.

```sh
uv run --frozen pytest -q tests/test_viewer_lighting.py tests/test_viewer_js.py tests/test_viewer_padmap.py tests/test_viewer_textures.py tests/test_scene.py tests/test_assets.py tests/test_viewer_server.py tests/test_viewer_creation.py
uv run --frozen pytest -q -m browser tests/test_viewer_lighting_browser.py tests/test_viewer_gamepad.py tests/test_viewer_creation_browser.py
uv run --frozen pytest -q -m slow tests/test_viewer_assets_browser.py tests/test_viewer_sensors_browser.py
uv run --frozen python scripts/framework_regression.py --baseline docs/bench/framework-regression-v1.json --out /tmp/robotforge-lighting-framework
uv run --frozen python scripts/bench.py --only install viewer --save --out /tmp/robotforge-lighting-budget
uv run --frozen python scripts/smoke.py
```

To recreate the detached comparison checkout after cleanup, use git worktree add --detach ../robotforge-lighting-baseline 714806452a8f3b8b61c6ccdf39793a4dc302d099.
The external run_pairs.py orchestrates matching captures; use new output labels or preserve the existing directories first.
Its check_lighting.py helper imports the retained preceding-round visual_baseline.py from ../robotforge-evidence/2026-10-23-visual-playability/.
All helper hashes and copies are retained in the immutable verification record.
No success-rate tuning, paid Actions, release or deployment is part of this increment.
The broader goal remains active; next priorities are creative scene editing/exploration and targeted temporal stability, measured against this baseline.
