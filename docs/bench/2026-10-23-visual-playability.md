# Shared surface quality and lightweight play validation

Date: 2026-10-23, macOS arm64, Chrome 155, ANGLE Metal reporting Apple M5 Max.
Baseline: `4ccf83874580f521870c4800d24eb018fbddc472`.
Final runtime source SHA256: `d1fef8db1b62d98cd0a05399876babc66d5d93e1819b85852e658f62df401d77`.

This is the first shared visual increment under the overall material-quality and playability goal.
It improves existing surfaces and their discovery, rather than optimizing a particular manipulation task.

## Changes

- Ground planes, heightfields and terrain blocks share deterministic 256 x 256 color and height/roughness tiles.
  Wood has finer seams and grain; tile has satin faces and matte grout; grass, concrete, dirt, sand, rock and snow have distinct surface profiles.
  Independent texture transforms retain physical tile scale and floor/heightfield alignment.
- Shared three.js Sources reuse GPU allocations across meshes; disposal releases color and detail textures.
  The existing low-contrast checker floor is retained.
- Built-in wood, bark, stone and fabric props gain local-coordinate surface detail with distance fading.
  Weave scale and strength now apply, and zero texture strength really gives a flat surface.
- The ground chooser uses the same generated color tiles as previews.
  Tiles are generated when needed, with no image downloads.
  Isolated ground-button and training-marker CSS fixes a pre-existing 9-pixel ground-control collision while preserving training markers at 9 x 9 pixels.

No dependencies, mesh geometry, physics, controller formulas, MCP definitions or persisted world formats changed.
Lighting and shadow settings are unchanged; this report does not attribute contrast differences to improved lighting.

## Fixed-view comparison

Three alternating baseline/candidate pairs use a robot gallery, kitchen and park with a heightfield and stairs.
The gallery includes four arm styles, a rover and full cached Menagerie Panda/Go2 visual meshes, rather than offline collision stand-ins.
Resolution is 1280 x 800 at DPR 1, with matching fixed near/far cameras and paused deterministic physics.
The later two pairs reset selection after body poses settle to remove a pre-existing cached selection-ring-size artifact from the comparison.
That capture normalization does not fix the application's initial ring measurement.
World and scene JSON match exactly across all runs; full initial time/qpos/qvel/ctrl arrays match across the later four runs.

| Scene | Fresh-profile first-frame median, before to after | Warm-navigation median, before to after |
| --- | --- | --- |
| Robot gallery | 431.8 to 458.3 ms | 194.6 to 213.1 ms |
| Kitchen | 401.8 to 441.2 ms | 229.3 to 279.6 ms |
| Park | 480.5 to 491.0 ms | 261.4 to 246.5 ms |

These measure navigation with an already running local server and cached robot assets, not installation or command startup.
Fresh browser profiles do not flush machine-wide GPU caches.
The initial candidate samples were slower: 1848.0, 1440.7 and 865.2 ms respectively.
Both repeats were faster; all samples remain in the JSON instead of discarding the initial cost.
Shader compilation is a plausible contributor, not an isolated cause established by this experiment.

Each view pools 120 synchronous render-plus-one-pixel-readback samples.
After-change medians range from 2.30 to 3.20 ms; park near increases from 2.20 to 2.80 ms.
Draw calls and triangle counts match exactly.
Park GPU texture allocations decrease from 12 to 7 through sharing; gallery/kitchen each increase from 4 to 5 for their detail map.
The measured candidate adds one HTTP resource and about 4.5 KB; the final CSS isolation followed these A/B runs, with no subsequent renderer/material change.
These timings are separate from sustained frame-rate measurement below.

## Low-angle and theme checks

The paused camera slides 16 frames in 5 mm increments, using the existing whole-frame A-B-A luminance-flip heuristic at threshold 5.
The kitchen and park also render correctly in separately opened light-theme documents.

| Low-angle view | Before | Final flips per megapixel |
| --- | --- | --- |
| Wood kitchen | 421 | 594 |
| Grass/heightfield boundary | 30 | 30 |
| Concrete stairs | 229 | 227 |

Wood retains a small extra temporal-detail signal, so this is not a blanket shimmer non-regression claim.
A rejected experiment halved floor bump strength and reduced grain contrast; its result was 599, so it was reverted.
Visibility-isolated measurements were floor-only 186 to 231 and props-only 225 to 354.
Disabling only the final ground-plane bump left the full-frame result at 594.
This localizes much of the signal to the new prop detail in this view; the heuristic includes geometry edges, occlusion and shadow changes and is not a perceptual quality score.
Isolated counts are not additive.
Original, rejected and final results remain available for the next temporal-stability pass.
No JavaScript or WebGL errors occurred, and the diagnostic restored visibility and material settings.

## Existing budget checks

The full install and viewer sections of `scripts/bench.py` pass every existing limit without changing `docs/bench/budget.json`.
This is a partial benchmark covering those two sections, not an imports/physics/MCP benchmark.

| Measurement | Final | Existing limit |
| --- | --- | --- |
| Wheel | 3627.44 KB | 4096 KB |
| Runtime installation | 163.11 MB | 180 MB |
| Packages / download | 42 / 52.51 MB | 45 / 60 MB |
| CLI start to first frame | 0.65 s | 2 s |
| Warehouse / assets frame rate | 60.2 / 60.2 FPS | 50 FPS minimum |
| Warehouse / assets render time | 3.30 / 3.70 ms | 8 / 12 ms |
| Server peak memory / idle CPU | 133 MB / 5.20 percent of one core | 160 MB / 6 percent |

The standard frame-rate scenes use 1280 x 800 at DPR 2, a 2560 x 1600 drawing buffer.
The wheel grows by 1931 bytes against a wheel built from the clean baseline with the same environment.

## Actual interaction recording

The actual Chrome/MuJoCo capture is 34.733 seconds, 1280 x 800 at 30 FPS, with 1042 fully decoded frames.
It shows a gallery camera orbit, native mouse clicks switching kitchen tile to wood, and native W/D keyboard driving in the park followed by reset.
The rover travels 1.52043 m; reset returns it within 0.00001074 m in XY, paused at time zero.
This validates existing direct operation and the improved chooser; it does not establish a new game loop or imported-robot walking success.
No JavaScript or WebGL errors were recorded.

Video: `../robotforge-evidence/2026-10-23-visual-playability/video-final/visual-playability.mp4`.
SHA256: `3e5f9f219f4fb05a4291f0c795e2689840d38779b8dc708c68f081a47c287d48`.
Raw clips, native screencast JPEGs/timestamps, screenshots, ffprobe output, decode log and a contact sheet are retained locally.
The decoded contact sheet was inspected, including the tile/wood states and rover drive/reset sequence.

## Verification and reproduction

Each recorded run checks stable source hashes; the final accepted runtime bytes match the SHA above on every accepted verification and video run.
Validation passes: 242 focused tests, four real Chrome GPU/chooser tests, eight gamepad browser tests and all seven unchanged canonical framework cases.
Wheel packaging, Ruff, actual CLI/MCP smoke and independent source/Python review pass.
The four new/expanded browser cases check shader compilation/nonblank pixels, strength-zero equality, weave scale, shared texture reclamation, visible Chinese controls at laptop/phone widths and preserved 9-pixel training markers.
The companion JSON records measurements, source and harness hashes, exact-equality checks and verification results.
The raw local evidence directory contains frozen baseline/candidate worlds, initial arrays, 36 fixed-view screenshots, browser manifests, every timing sample and reproduction commands.
Its `README.md`, `REPEATED-RESULTS.md` and `video-final/README.md` distinguish navigation timing, GPU timing and real-time interaction.

```sh
uv run --frozen pytest -q -m slow tests/test_viewer_assets_browser.py
uv run --frozen pytest -q -m browser tests/test_viewer_gamepad.py
uv run --frozen python scripts/bench.py --only install viewer --save --out /tmp/robotforge-visual-bench
```

This increment does not rerun the complete default suite; the preceding framework round's full-suite result remains a separate historical result.
No paid Actions, release or new project dependencies were added.
Native Windows and optional type/security tools unavailable in the local environment remain unverified.

Next: improve coherent lighting/contact readability and a complete easy-to-start scene play loop, keeping these fixed-view/performance comparisons as guardrails.
