# Wood appearance and source sampling screen - 2026-10-24

Round68 rejects the frozen curved-fibre candidate and does not promote source-integration-only changes.
Production runtime remains exactly `eab1c5dd1335f11d71c0d3a1e6a5bbb355fb1b730c57e61af553d83974f43b0c`, based on `107f005e6bc72e678ba9c6db25387fb01b830041`.
The broader visual-quality, lightweight-loading and playability goal remains active.

## Question and finite candidates

Current floor close-ups have broad, soft bands rather than clearly recognizable natural wood.
The old256-square source samples its highest noise field at about2.67texels per cell and then quantizes it to8bit.
This suggested a source-sampling risk, not an established cause of screen shimmer.
Prior output-resolution references shared that same source tile and could not isolate this question.

`plan.json` freezes the prototype and preview source hashes before candidate observation.
The baseline serves an exact copy of production static files.
The integrated variant averages the existing continuous colour/height field over4x4 source-pixel subpositions.
The figured variant uses one frozen32-cycle longitudinal-growth field with low-frequency domain bending and the same4x4 source integration.
Point/2x2/4x4 variants and16x16 numerical references diagnose source convergence rather than offering a series of aesthetic retunes.
All retain256-square colour/detail maps, board proportions, existing joint coverage and the original point-sampled roughness bytes.
External copied static directories override only ground-look.js; product files and serving security remain unchanged.

## Actual result and decision

Fresh actual Chrome paused gallery, kitchen, park and wooden-prop fixtures were captured at17 prescribed views per variant,51initial stills in total.
Normal-size gallery and kitchen images still show repeated parallel stripes in the figured candidate, with no convincing natural-fibre improvement.
The coordinator and independent reviewer both reject that appearance against the predefined two-indoor-scene gate.
No frequency/contrast retuning or relaxed visual threshold was used to get a pass.

The candidate's grayscale albedo tile has mean linear multiplier0.798541 versus baseline0.806906, a ratio0.9896337505 or1.036625percent lower.
This also misses the predefined1percent source-tone guard.
It is a texture-factor statistic, not measured final screen/object brightness.

Source integration alone changes the actual gallery-near and kitchen-near screenshots by at most1byte/channel, with whole-image mean absolute differences0.02367and0.00987byte.
The coordinator inspected those actual images and found no sufficient normal-size appearance gain.
A better source metric alone does not justify shipping this variant.

## Source diagnostics and retained failures

| Source field | Colour RMSE to its16x16 source reference, byte units | Max error, byte units |
| --- | ---: | ---: |
| Existing point field |0.235220 |1 |
| Existing4x4 integration |0.050781 |1 |
| Figured point |0.372469 |2 |
| Figured2x2 |0.177208 |1 |
| Figured4x4 |0.082031 |1 |

These compare the raw256-square source arrays, not real rendered temporal error or physical ground truth.
The16x16 reference retains center-selected boards and the existing joint coverage; it does not integrate those discontinuities, and no32x32reference was run.
The119retention assertions pass: seven modes times eight nonwood styles with two channels, plus wood roughness, including baseline self-comparisons.
They are not119picture-quality or physics tests.

The final241control/integrity assertions retain239passes and two failed park-far exact-image controls.
Other four park views are pixel-exact; the far-view screenshots differ broadly despite equal scene descriptions and frozen physical state.
Preview mode did not record final camera transforms, so settling or another cause remains unproven.
The original differences remain failures and are not attributed to wood shading, filtered away or rerun until green.
All-view pixel/camera equality is not claimed.

The first analysis also compared original ffmpeg PNG-encoding hashes against Pillow-reencoded files, falsely rejecting24decoded samples.
Its original source and failed summary remain.
Offline correction re-extracts the same video frames, validates their original ffmpeg byte hashes and compares decoded pixels with the saved PNGs; no video/browser capture was repeated.
The first external baseline attempt timed out because StaticFiles rejected resource symlinks outside its directory.
That browser log/profile/world and failure note remain; ordinary copied external files fixed the harness without weakening product serving security.

## Real recordings and cost scope

Four actual paused-scene camera clips fully decode: gallery/kitchen before and rejected candidate, each3.3seconds and99encoded frames.
They use identical prescribed3second lateral trajectories and matching first keyframes; six deterministic decoded frames per clip were inspected.
These are actual viewer footage, not moving-robot/contact verification or quantitative temporal-stability approval.
Raw screencast JPEGs and timestamps, complete MP4s, source RGBA arrays, original full PNGs and decoder checks are retained.
The first decoder checked exit status only; after reviewer feedback, a separate offline strict decode of the same four originals uses `-xerror -err_detect explode` and confirms zero exit status and empty error-level stderr for all four, without re-recording.

| Fresh-profile first complete frame, ms | Baseline | Integration only | Rejected figured |
| --- | ---: | ---: | ---: |
| Gallery |844.3 |787.6 |670.3 |
| Wooden props |744.3 |466.5 |461.3 |
| Kitchen |489.9 |724.4 |824.7 |
| Park |654.9 |742.6 |610.6 |

These are single serial preview samples, baseline then figured then integration, not alternating performance cohorts or installed-candidate acceptance.
The original raw navigation/network and process-memory observations remain in the manifests.
The separate Node ImageData-stub source diagnostic took18.98ms for baseline wood generation,86.64ms for integration and242.46ms for figured4x4 in its single cold pass.
Those are diagnostic generation timings, not actual browser loading or per-frame GPU costs.

Because the appearance and tone gates fail, no official install/viewer budget, package build/install, pytest suite, CLI/MCP smoke, output-resolution temporal residual or source-channel attribution run was performed this round.
Existing round67production validation remains historical evidence; it is not relabeled as current candidate validation.
The external candidates retain map dimensions and render code but do not receive a performance approval.
Physical phones, Windows, other drivers and wooden ramp/stair scale remain unverified in this screen.

## Record and next action

Evidence is outside Git at `../robotforge-evidence/2026-10-24-wood-figure/`.
Its plan, exact candidate modules/static copies, original helpers/fixtures, raw captures, failures, summary/index and unified experiment record distinguish a rejected trial from a shipped change.
The trial is recorded as failed against its quality/control gates, with broad goal active.
No candidate shader, image, dependency or runtime change enters production.

Next work should use a natural-material visual reference and a bounded texture or stronger surface representation, explicitly measuring its added resource cost.
Keep source-pixel sampling as a numerical diagnostic; do not continue tuning this frozen sine/warped-value-noise candidate or call it a shimmer fix.
Actual normal-size multi-scene improvement must precede full installation/loading/frame/memory validation.
