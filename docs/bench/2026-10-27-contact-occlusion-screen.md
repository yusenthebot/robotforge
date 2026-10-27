# Short-distance contact occlusion: rejected screen (2026-10-27)

Reject this frozen candidate: ordinary-distance views of Malo, Go2 and Panda do not show a useful shared gain in contact or joint readability.
Root and independent visual review agree after ordinary views, detail views and original PNG checks.
This does not rule out all ambient occlusion; it rejects this particular bounded implementation without extending it into parameter sweeps or a post-processing project.
No candidate renderer code is shipped.

## Frozen experiment

Base main is `a8b7fe9451ca3ec3715c9cfe4ecc847575cfe3e1`, after PR #35, with active-source SHA256 `154677d641d7627062d44c0ed35ca49ccb816f4278ba3926bc45e4725deb43d7`.
The external copy changes `app.js` and adds `contact-occlusion.js`.
Before inspecting images, the plan fixes a 6 cm radius, maximum 25% indirect-light attenuation, 12 samples, half-resolution targets capped at 1024 pixels on the long edge, and a 1 cm depth-matching gate for upsampling.
There is no noise texture, blur pass or temporal history.
Depth reconstruction supplies geometric normals; only PBR indirect diffuse/specular, clearcoat and sheen are attenuated.
The existing author materials, direct/hemi lights, exposure, tone mapping, shadows and background remain unchanged.
Original shader hooks and vertex edits are composed; non-main cameras bypass the prototype render wrapper.
These are implementation properties and static review findings, not production sensor/context/lifecycle certification.

Nine groups cover three real-model worlds, each with contact, raised and environment cases.
The same three fixed wood/plastic/steel props have explicit XYZ coordinates; the raised case increases their centers by 15 cm.
Malo and Go2 environment cases add compact ramp/stair fixtures; Panda instead uses its existing table and a view of front legs/tabletop contacts.
Those compact terrain parameters are experimental fixtures, not PR #34's creation defaults.
Physics is reset and paused before capture, with no steps or pose writes during measurement.
Each group has ordinary/detail views at 1280 × 800 CSS pixels and DPR 2, plus a disabled-candidate control.

## Observations and costs

All nine pairs have 485 matching compiled arrays except Malo's isolated cache path bytes/addresses, which resolve to matching mesh hashes.
World YAML bytes match within each pair, physical state stays fixed and the largest camera/projection error is `2.22e-15`, below the analysis tolerance of `1e-10`.
All 18 corrected candidate checkpoints have GL error zero; debug readbacks also return zero errors.
Raised static views show no obvious false contact patch, and environment views show no obvious broad darkening or halo.
These observations do not certify every moving edge or prove a perceptual gain.
The main contact forms remain visually dominated by the existing directional shadows.

| Item | Observed candidate increment |
| --- | --- |
| Auxiliary passes | One scene depth pass and one fullscreen AO pass |
| Depth draws | 40–157 per measured view, plus one AO quad draw |
| Geometry / textures | One quad geometry / three textures |
| Shader programs | Four for Malo/Panda, five for Go2 |
| Targets | Two actual 1024 × 640 RGBA unsigned-byte color targets, one with unsigned-int depth texture |
| Nominal color-pixel payload | 5,242,880 bytes (5 MiB), plus unmeasured depth/driver/transient storage |
| Warm synchronous frame increment | Median 0.00–0.20 ms; p95 −0.10–0.20 ms |

The color-pixel arithmetic is not a VRAM measurement.
Depth generation redraws much of each scene; the unchanged final-pass draw count alone must not be presented as unchanged total rendering cost.
The observed timing increments meet the additional screen gate (median ≤1 ms, p95 ≤2 ms), but insufficient visual benefit still rejects the candidate.
Each view uses 40 GPU-finished draws, measuring the last 30, and 121 RAF timestamps, measuring the last 120 intervals.
Paused RAF remains around 60 Hz; these single sequential warm measurements do not establish active-physics FPS, cold network loading, installation budgets or a speedup.
Warm compile plus Chrome-interactive observations are mixed: for example Malo contact rises from 4.394 to 5.910 seconds while Go2 raised falls from 2.357 to 1.800 seconds.
No extra asset or dependency is required; the prototype adds a JavaScript module request, shader work and render targets.

Disabled screenshots are near-identical but **not byte-identical**: whole-image mean absolute differences range from about 0.00001 to 0.02288 on the 8-bit scale, with localized larger differences.
The diagnostic includes HUD and sensor/lidar timing; the original cause of those differences is not established.
No strict all-pixel equality, sensor-image invariance or numerical quality score is claimed.

## Failed capture and measurement repair

The first candidate run stops at the Malo raised ordinary view after a nonzero GL check; its partial images, arrays, traceback, receipt and original helper are retained.
A separate diagnostic observes GL1282 and an all-zero AO readback without shader-program errors.
An initial diagnostic attempt also fails at the local HTTP proxy boundary; the subsequent command restores the existing localhost proxy exclusions.
The measurement repair changes only the helper: synchronous AO readback temporarily unbinds `PIXEL_PACK_BUFFER` and restores it in `finally`, and the disabled control renders without an unnecessary `pixels()` readback.
Candidate source hashes and all rendering parameters remain frozen.
The complete candidate cohort is then recaptured with clean checkpoints; the complete baseline cohort remains intact.

PBO-bound TypedArray readback can cause `INVALID_OPERATION`, and the vendored asynchronous reader retains a PBO while awaiting its fence.
However, no original failing call records its buffer binding, and the corrected calls observe no bound PBO.
The mechanism is plausible and the boundary repair is valid; **the original trigger is not proven**.
Later clean checks do not erase the first failure or retrospectively certify its cause.

## Evidence and verification scope

Raw evidence is outside Git at `../robotforge-evidence/2026-10-27-contact-occlusion/` relative to this checkout.
It contains the predeclared plan, immutable baseline/candidate source copies, original and corrected helper versions, source receipts, model arrays, 36 original comparison PNGs, 18 disabled controls, 18 AO maps, analysis and six comparison sheets.
Eighteen original browser movies follow the same six-second ordinary-distance lateral camera path, 0.45 m across and back, with identical YAML and initial/final physical state.
They are actual paused rendering with scripted cameras, not native gameplay input or joint-motion verification.
Raw timestamped JPEGs and complete strict-decode QA are retained; 30 fps CFR encoding resamples those inputs and does not promise every JPEG appears in the output.
Decoded samples cannot establish every-frame temporal quality.
All 18 movies fully decode; root inspected all 108 sampled decoded frames.
The companion JSON is a local benchmark report, not a registered experiment-run manifest.

The prototype is rejected before production lifecycle work, candidate full-suite/installed-smoke checks or the original 11-budget run.
Current production runtime and dependency configuration remain unchanged; earlier PR #34/#35 checks remain scoped to that source and are not relabeled as candidate acceptance.
The documentation change receives its own build, package-byte, lint, evidence-integrity and diff/secret checks; optional tools absent from the environment are reported as unavailable.
The offline wheel is 3,919,038 bytes, with all 407 product files byte-identical to PR #35's wheel; Ruff check/format and 7,989 integrity checks pass, including hashes of all 6,399 received raw movie JPEGs.
The original checkout's 13 pending files retain their hashes.
Pytest is not rerun for this documentation-only change; optional mypy/pyright/bandit/pip-audit remain unavailable and are not counted as passes.
The broad material, lightweight-loading and creative-play goal remains active.
Next, improve reuse of existing terrain by moving/rotating an existing ramp or stair block with guarded Undo; single-action success tuning stays deferred.
