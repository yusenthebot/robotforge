# Solid wood growth-field screen (2026-10-28)

The unique frozen external candidate is rejected: ordinary-distance table and bench views do not both improve natural-grain readability.
The table loses continuous fibres over a large area; the bench gains regular bands and a flatter seat, while close-ups show embossed ring/stripe patterns.
Root and an independent reviewer agree; close-up cathedral shapes do not compensate for the ordinary-distance failure.
Production stays on source `b58ddc20ac481b2aafc0629cab54afa0f58ae9c0b5e29e9d815738ab71f2961c` at base `60fc1b93490a1da16c6798161fcbc90ceced3726`.
The broad shared-material/lightweight/playability goal stays active; individual action-success tuning remains deferred.

## Finite candidate

The representation uses an off-center, slowly bending pith, continuous 3D growth rings and fine fibres, inspired by the general solid-texture approach in [Liu et al., 2016](https://research.cs.cornell.edu/wood/).
This is an independently written bounded field, not that paper's model, anisotropic fibre BRDF or implementation.
Only external `look.js` differs, by 833 source bytes; no production shader was edited.
Pith offsets60/90mm, bend wavelengths650/1100mm, amplitudes18/9mm and second phase0.7rad were fixed before observing images.
Radial ring spacing6mm, raw-noise phase warp amplitude0.35cycles, noise scales350/50/50mm, separately filtered0.8/0.2harmonics, ring gain0.28 and fine-fibre gain0.10 were not retuned after screening.
The fully filtered height mean remains0.575; finite images are not guaranteed identical mean brightness.
Existing authored tint, color gain, roughness/bump constants, longest-axis choice and scale convention remain.
The `USE_MAP` branch retains the baseline wood field, and zero-strength bypasses grain; ground oak and unrelated materials remain byte-identical.
Phase warp uses raw noise before ring filtering; already filtered derivatives are not differentiated again.

## Actual comparison

Five worlds with 12 views in total per cohort produce 24 original 2560×1600 PNG captures from CSS1280×800 at DPR2 in actual Chrome.
Worlds cover a table, park bench, kitchen, three-axis wood solids and a nonwood rover control.
Studio lighting, camera/projection, reset-paused state and viewport match; an external helper hides the empty-world note equally to keep furniture unobscured.
All five compiled-model array sets and full scene payloads match exactly.
All 2542 comparison checks pass: world/scene/model arrays, cameras<1e-10, draw calls/triangles/resource/program counts, GL0 and nonblank byte-exact native-image/zero-strength/rubber control readbacks.
These checks establish comparison controls, not quality acceptance.
The prototype adds no asset, sampler, render pass, network request or dependency by construction; actual fixed-view texture/geometry/program/draw counts are unchanged.
Retained sequential draw samples are descriptive, including below-clock-precision values; no ABBA group or original formal budget was run after appearance rejection.
No loading/FPS/memory speedup or candidate lifecycle/device certification is claimed.

## Actual camera movies

Four original paused scripted-camera movies traverse the table and bench from ordinary to grazing views, with no pose writes or manual physics steps during recording.
All strictly decode, retain every timestamped original JPEG, verify paused qpos/qvel/ctrl, GL/page errors and process/thread cleanup, and have six sampled decoded frames inspected each.
30fps encoding is not measured application FPS; full decode and 24 sampled stills do not replace a frame-by-frame motion review or prove shimmer/swimming improvement.

baseline-wood_props: 6.567s, 197decoded frames, 387raw JPEGs; videoSHA256 `75a42c3a162fe99c3ccbdbaab72bd97520553d9d43f4001201fdea88cd68dca1`.

baseline-park: 6.567s, 197decoded frames, 386raw JPEGs; videoSHA256 `7669660e75b2187eeb1f12a5a5525873271349f0e5eee2d5f1565adad4583d90`.

candidate-wood_props: 6.567s, 197decoded frames, 386raw JPEGs; videoSHA256 `66832fce4f615f3f84dac42f81e761ed3e2448775f642510d6aee373da3bc952`.

candidate-park: 6.567s, 197decoded frames, 387raw JPEGs; videoSHA256 `0f9ce3935743718e74ce51ae9f00118aae6dfcb43e48269436f7652f06be9c8d`.

## Product and record verification

Offline wheel build succeeds; all 408 product package files match current source and the pre-screen production tree byte-for-byte.
Wheel is 3925041 B; product code/assets/dependencies are unchanged.
41 existing experiment-record tests pass; Ruff/format/diff and evidence integrity checks are recorded for the documentation-only delivery.
This is not a new full-default-suite or candidate installed-smoke/performance certification.
The first baseline driver timed out because it paused before `qa.Tab.navigate`'s time>0 readiness rule; the failed log, output and helper are retained.
Only the driver sequence was corrected; candidate parameters stayed frozen.
All successful capture processes terminated; review identified conditional cleanup gaps on exceptional evidence writes/readers, and the later recording helper has a statically reviewed outer-finally/kill/reap fix.
Used recording code is preserved as `record-v1.py`; the later robustness fix was not fault-injection tested or relabeled as used in the movies.
1632 retained output artifacts have a separate SHA256/size index;13 original uncommitted files remain hash-identical.
Raw evidence and source snapshots: `../robotforge-evidence/2026-10-28-solid-wood-growth/`.

This frozen ring field is stopped rather than retuned.
Next investigate independent color-figure and micro-relief representation: the observed large-area flattening and repeated relief show that changing the shared height field alone did not meet the ordinary-distance goal.
