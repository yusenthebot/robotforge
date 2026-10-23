# Shared material detail at a distance

This increment improves the shared procedural grain used by robot parts, plugins and scene props.
Resolved close detail remains, while smaller features converge toward the material's average tone rather than brightening back to its untinted base color.
Wood, brushed metal, speckle, rubber, bark, stone and knurl use the same footprint-aware policy.
Ground tiles, weave/cloth, geometry, lighting, physics, controls and external interfaces are unchanged.

## Implementation

Each noise octave uses its actual coordinate footprint; the wood wave uses its warped phase and knurl accounts for both product frequencies.
Value noise has a different feature scale from a full sine cycle.
The shader carries the filtered height and the raw height's attenuated first derivatives separately.
It never differentiates an expression that already contains a screen derivative.
The previous base-coordinate bump envelope remains on normals, preventing the mean-tone change from amplifying normal perturbation.
Filtering is a conservative approximation, not an analytic integration of arbitrary warped noise.
The shader cache key separates this implementation from previous grain programs.
There are no new texture allocations, network requests, dependencies or rendering passes.

## Quality evidence

Baseline: `03f6ce4250059763814d512504f664095697b1f5`.
Final candidate runtime SHA256: `fd4d371b736efbdbd1fcead9203a6a60f5e0a054ca668bf31d57c4a73224dc41`.
Raw evidence: `../robotforge-evidence/2026-10-23-material-filtering/`.

The isolated GPU sweep uses the exact served material implementation for eight kinds including unchanged cloth, three face-normal axes, two feature scales, two viewing distances and three small camera translations.
Native 256 x 160 pictures are compared with 1024 x 640 renders reduced with a BOX filter.
There are 96 complete-material configurations and 16 additional diffuse-only configurations, each with three translations.
The diffuse subset removes normal and roughness injection externally to isolate frequency behavior.
Complete-material references retain the existing resolution-dependent bump convention and are approximate diagnostics rather than physical ground truth.
Each source's own reference and the fixed baseline reference are retained, with mean bias and mean-centered error reported separately.
Scaling camera and geometry with feature size tests scale covariance, not two independent physical viewing distances.

Initial final-source transition results against the fixed baseline reference (8-bit RGB MAE):

| Material | Before | After |
|---|---:|---:|
| Wood | 3.278 | 1.067 |
| Speckle | 4.476 | 0.998 |
| Bark | 4.479 | 1.017 |
| Stone | 4.465 | 0.764 |
| Knurl | 5.260 | 3.176 |
| Rubber | 4.602 | 1.248 |
| Brushed | 4.502 | 0.656 |

All seven complete-material mean-centered transition errors also improve.
Near wood native PNGs are byte-identical across every tested normal, scale and translation.
Near rubber retains a small detail change; the diffuse contrast loss is about 1.2%.
Every zero-strength native/4x control and unchanged cloth native picture matches exactly.
Whole-scene world/scene descriptions and initial/final physics are frozen and compared independently.

Temporal results are mixed, not a universal flicker reduction.
Complete-material transition residuals improve for six kinds, while wood changes from 0.1563 to 0.1625 of 255.
The diffuse knurl centered error changes from 1.4447 to 1.5159 and its temporal residual from 0.5997 to 0.6805.
These limitations remain in the accepted report and next-stage priorities.

## Retained unsuccessful attempts

The first diagnostic distance let the old shader remove most of the signal, so it is retained as an unsuitable filtering test.
Revision 1 overfiltered resolved noise and did not preserve the distant mean tone.
Revision 2 restored much of the detail and mean tone but visibly amplified knurl normals.
Revision 3 retains the existing normal envelope and is the candidate described above.
All captures and exact source fingerprints remain available; revision 1's source was reconstructed from the actual patch history and its complete-package digest verified against its original capture.

## Regression and resource verification

The final source passes 283 focused default tests and 23 actual Chrome GPU/creation/object/lighting cases.
The seven unchanged canonical framework cases and real installed CLI/viewer/MCP smoke pass, including 60 MCP tools and a real script process.
Wheel build, Ruff and all 21 JavaScript parser cases pass.
This viewer-only increment does not repeat the complete default Python suite; the previous round's full-suite result is historical.
Optional mypy, pyright, Bandit and pip-audit are unavailable locally.
Windows, other GPU drivers and physical phone hardware remain unverified.

All 11 existing install/viewer budgets pass without relaxation: 3642 KB wheel, 163 MB installed, 42 packages, 52.5 MB dependency download, 0.54 s CLI-to-first-frame, about 60 FPS at DPR 2, 3.00/5.00 ms standard/asset render, 132 MB server peak and 6.00% idle CPU.
Idle CPU is at its existing limit and is retained as measured.
Import, physics-throughput and MCP timing budgets are outside this partial run.
The wheel grows by 479 bytes and fresh HTTP responses by 1161 bytes, with 28 requests before and after.
Texture, geometry, triangle and draw-call counts match in every paired view.

Four serial runs alternate A1/B1/A2/B2 across all four worlds, with the same fixtures, cameras, 1280 x 800 viewport, DPR 1 and browser version.
All 73 source, state, scene, resource-count and GL checks pass.
Repeated fresh-profile runs reuse code previously exercised on this machine; they are not pristine machine cold-start measurements.
Medians of two observations per side (milliseconds):

| World | Fresh profile before/after | Warm navigation before/after |
|---|---:|---:|
| Gallery | 389.1 / 409.1 | 195.8 / 233.5 |
| Kitchen | 391.5 / 388.0 | 261.3 / 267.4 |
| Park | 372.6 / 398.5 | 227.2 / 275.9 |
| Wood props | 242.0 / 241.0 | 114.1 / 100.1 |

The original final-source first-use samples are retained separately against an already-used baseline: gallery 410.9 to 1791.6 ms and kitchen 377.1 to 1231.7 ms.
Shader compilation is a plausible contributor, not a proven attribution or a hidden warm-cache result.
Navigation begins with an already-running server, whereas the standard budget's CLI-to-frame measurement includes server startup.
First-frame readiness is a proxy; this run does not separately measure latency to the first user action.
Synchronous rendering samples include one-pixel readback, not isolated GPU timer-query duration.
Pooled p95 changes include gallery near 5.4 to 5.2 ms, kitchen near 2.8 to 3.2 ms and wood detail 2.5 to 2.7 ms.
Paused animation-frame cadence stays about 60 Hz; it does not measure active physics throughput.
Chrome process-tree RSS is noisy and can double-count shared pages; no memory-saving or overall speed-improvement claim is made.

## Actual video

`material-filtering-before.mp4` and `material-filtering-after.mp4` concatenate actual Chrome screencast frames from the final A2/B2 runs in gallery, kitchen, park and wooden-table order.
The simulation is paused; the videos demonstrate camera motion and material appearance, not new control or manipulation behavior.
Both use H264, 1280 x 800 and 30 FPS encoding; raw timestamped frames are retained.
For these matched diagnostic recordings only, the empty-robot callout is hidden through externally logged CSS on both sources after navigation measurements.
The application UI source and simulation state remain unchanged.
The accompanying JSON links the immutable verification record, raw evidence index, video hashes and complete cost/quality records.

## Reproduce

Use `uv run --frozen`, the committed regression and budget files, and the saved external harness/fixture hashes.
Run heavy browser, simulation and benchmark batches serially.
The baseline checkout can be recreated with `git worktree add --detach ../robotforge-material-baseline 03f6ce4250059763814d512504f664095697b1f5`.
External helpers and all attempts live in the evidence directory rather than the installable package.
The broader visual-quality and creative-play objective remains active.
First-use shader latency, remaining wood/knurl temporal behavior and clearer creative navigation remain follow-up work.
