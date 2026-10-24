# Object wood color screen - 2026-10-24

Round72 rejects two external color-only wood candidates because ordinary scene views do not show enough recognizable natural-fiber improvement.
Production remains at runtime `afabb4e3d7e86a2340472de4177ef8bebcdcb4d95d4c83cb0aa0947c7d06a28e`, based on `86f22f58744634f5edc050da108d8cdd95d57656`.
The shared visual-quality, lightweight-loading and creative-play goal remains active.

## Finite experiment

Current object wood uses procedural longitudinal noise, while the existing ground already uses a local natural oak texture.
The experiment reuses the retained CC0 Poly Haven oak photograph, selecting one board at source rectangle `(128,324,1024,392)` to avoid floor joints on furniture.
The original SHA256 is `f715ed3e4346e94f971ef8dd61221ff6cc774886c20f65381e01e1dc0bb0a634`.
There was no new download, dependency or generated image.

One source/crop and two maximum lengths,256 and512pixels, were fixed before observing candidate scenes.
Preparation converts sRGB to linear luminance, reduces with BOX, normalizes the mean to the existing object-grain mean0.575, and quantizes to8bit grayscale.
There is no contrast boost, seam feathering or post-screen parameter retuning.
The texture is sampled without sRGB decoding and repeats over0.896by0.068meters, scaled by the existing surface scale.

External copied static files replace only the wood color factor in look.js and add one eager texture module.
Mesh-local coordinates follow the existing longest-bounding-box axis; two smooth side projections blend around bevels, and end faces retain the old procedural approximation.
Roughness, bump, clearcoat, authored color, geometry and physics remain unchanged in the prototype.
Native image precedence and zero-strength behavior were reviewed in source; dedicated runtime boundary fixtures were deferred after the appearance gate failed.
The eager module and global texture ownership are explicitly not a production loading/disposal implementation.

## Observed result

Three actual Chrome cohorts capture four frozen worlds, gallery/kitchen/wood_props/park, at17prescribed1280by800DPR1views each.
The51original fixed images and exact source-copy hashes are retained outside Git.
Root and an independent reviewer inspected ordinary table/bench/kitchen views and close-ups.
Both candidates introduce weak curves at close range;512is slightly finer, but normal scene distance does not show enough readable natural fibers.
The kitchen mainly loses the old broad bands without gaining a convincing natural surface.
This is insufficient benefit to justify another sampler, image and loading lifecycle.

The crop is not periodic: mean source-edge differences are about18.3to21.7byte, depending on axis and size.
That is a separate source-wrap risk, not proof of a visible screen seam or the cause of motion shimmer.
No shimmer fix, moving-wood stability, new action success or shared-material upgrade is claimed.

| Candidate | Decoded dimensions | PNG bytes | Eager module bytes | Mean material factor ratio | Nominal RGBA8 mip payload upper estimate |
| --- | ---: | ---: | ---: | ---: | ---: |
|256 |256x19 |3247 |4974 |1.0000046032 |26005B |
|512 |512x39 |12872 |17806 |0.9999891607 |106560B |

The ratios pass the one-percent material-mean guard but do not measure final screen brightness.
Mip payload estimates exclude driver allocation/alignment and are not measured VRAM.
The candidate adds one renderer texture in these wood-containing worlds; primitive/call counts are unchanged in the recorded loop.

## Controls and retained analysis failures

The final178offline control/integrity checks retain171passes and seven literal camera-eye equality failures.
All scene hashes, initial/final paused physics states, recorded GL-error checkpoints and primitive/call counts match their intended controls.
Projection and target values match; the seven eye-coordinate differences are floating-point residuals, at most7.10543e-15.
They remain failed exact comparisons, without rounding or recapture, so this is not an all-controls-green or pixel-exact certification.

The first analysis retained45failed checks.
It incorrectly compared the cumulative renderer.frame counter in34draw-count checks, and expected four complete floor-labelled pictures to remain identical even though their visible wooden table changed.
The original script/result are preserved as analyze-v1.py and analysis-v1.json.
The corrected draw comparison uses calls/triangles/points/lines, and ground-source retention is checked by module hashes.
Supplementary post-hoc floor-only rectangles `(0,450,1280,700)` are pixel-exact in all four pairs; they do not replace full-view equality or define a new visual acceptance threshold.
The initial source-root guard also failed before starting Chrome; its log and empty output directory remain.

## Actual recordings and cost limits

Four original paused-viewer camera recordings show table and bench before and rejected512candidate, following the same prescribed five-second keyframe paths.
Raw arrival-order JPEGs/timestamps, complete MP4s and six decoded samples per clip remain available.
All four originals pass strict full decoding with `-xerror -err_detect explode`; the24sampled decoded frames were inspected.
These are camera footage with physics paused, not moving-robot/contact or per-frame temporal certification.
Thirty-fps encoding resamples the complete timestamp sequence and does not promise one output frame per raw JPEG.

| Fresh-profile first complete frame, ms | Baseline |256 |512 |
| --- | ---: | ---: | ---: |
| Gallery |1126.0 |1495.1 |500.6 |
| Kitchen |463.5 |1004.9 |352.5 |
| Wooden props |459.5 |542.6 |369.4 |
| Park |620.3 |548.1 |477.9 |

These are single sequential preview observations, with raw40draw/readback samples per view, not alternating cost cohorts or official budget acceptance.
Faster512navigation samples do not establish a speedup; eager decode, shader/cache state and machine load are not isolated.
No package/install/viewer budget, full pytest suite or source/MCP smoke is rerun for the rejected external shader.
Physical phones, gamepads, Windows, other GPU drivers, runtime fallback/disposal and an actual moving wooden robot remain unverified in this screen.

The documentation-only submission passes an offline wheel build, repository Ruff checks/format, diff whitespace checks and41experiment-record tests.
All621previously frozen runtime/test/script/config inputs and all433wheel ZIP entries remain byte-identical to the preceding production verification; the wheel is still3854686B.
Optional mypy/pyright/bandit/pip-audit are unavailable and were not installed.

## Record and next direction

Machine-readable results are in the adjacent JSON; full evidence is `../robotforge-evidence/2026-10-24-object-wood-screen/`.
The failed experimental record retains exact prototype/static/helper/source snapshots, fixtures, raw frames, failed assertions, reviews and hashes.
Only this report and progress documents enter Git; no candidate texture or runtime code is shipped.

Do not continue tuning this frozen crop/color-only prototype.
Next shared-material work should first establish a clearly readable normal-distance reference and a suitable surface representation, including detail/roughness where justified, then measure its bounded cost.
Creative exploration and composition remain the other main development line; individual weightlifting/drawer success stays deferred.
