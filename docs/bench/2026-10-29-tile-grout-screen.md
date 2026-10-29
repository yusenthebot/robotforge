# Shared tile grout baseline — 2026-10-29

No qualifying ordinary-distance defect was established on both the shared tiled
floor and existing tiled platform paths. No material candidate was made or
promoted. The hard source threshold alone does not justify changing filtered
rendering. This is a recorded baseline screen, not a material-quality gain.

## Real scenes and gate

The preregistered external plan requires repeatable broken joints, width jumps or
stripe interference in ordinary views of two actual paths before one source-pixel
coverage candidate may be tried. Near views can attribute a problem but cannot
qualify it alone. Geometry edges, shadows and compressed-video artifacts do not
qualify as grout defects. Kitchen's default floor is wood, so it was not reskinned
to create a tile example.

One copy of the existing Panda workbench uses normal
`load_scene("desk_lab", seed=0)` with its library Panda retained. The second world
is the built-in default pool with its ROV starter and existing tiled platforms.
The platform studied is `deck_south`; these are two material application paths,
not two independent tile implementations or gains on robot surfaces. Original
world bytes remain unchanged.

Each path has front, oblique and near views at actual browser DPR1 and DPR2 under
studio and daylight: 24 PNGs total. The CSS viewport is1280×800, drawing buffers
are1280×800 and2560×1600. Light media is explicit. Resolved `ground_tile` geometry,
procedural256² color/detail canvases, source hashes, bump scale and visible ray
samples are recorded. Actual camera eye, target and lighting are checked.

Four studio recordings retain the complete five-second scripted camera motion
inside recorded settled endpoint holds. These are real paused viewer renders,
not physics/action demonstrations or human-input timing studies. Raw arrival
timestamps are strictly increasing; an independently declared0.1s maximum frame
gap and motion-epoch endpoint coverage are checked. All raw JPEGs decode at
1280×800. All four H264 movies fully decode; each has189frames/6.3s after CFR
encoding of approximately6.183s raw capture. CFR does not preserve every input
JPEG as a separate output frame.

The root inspected eight ordinary-view PNGs across both DPRs/lights and24 decoded
samples from the four movies. A second reviewer inspected the DPR1 studio
front/oblique pairs and both DPR1 decoded contact sheets. This did not establish
the required shared two-path defect. It does not certify flicker-free playback,
every-frame visual quality, all camera distances or all hardware.

## Integrity and evidence

The captured runtime is
`b399c6c405fedc612c92f0e8323529f5f238b42bd01a03bd2df2fdb768d490ae`,
main commit `b46beb3436999a9c44b6b8d185b06796403b2436`.
All768 source/test/script/dependency inputs remain exact. Each case guards all
compiled ndarray fields plus names/paths, model/data identity, complete MuJoCo
integration state, qpos/qvel/ctrl and paused time; this is not a snapshot of every
Python scalar/option object. The original13 WIP files are preserved. Owned test
servers and Chrome processes stop after capture.

External evidence lives in
`../robotforge-evidence/2026-10-29-tile-grout-screen/`. `plan.json` precedes
capture; final stills and movies are in `v2/desk/` and `v2/pool/`.
`validation-v2.json` independently checks the exact matrix, all preservation
fields, raw continuity, and manifest/video/decoded-sample hashes bound to four
complete decode receipts. Machine-readable summary accompanies this report.

The original `capture.py` and first capture tree are retained as preliminary:
their acceptance checks lacked material/endpoint enforcement. The second helper
adds those guards and integration-state checks. Its general cleanup failure
paths are not unit-tested; the final validator checks the actual successful
cleanup and preservation outcomes rather than relying on nested `passed` alone.
The first validator was also incomplete about the matrix/media binding and is
retained separately. No failed/incomplete receipt is relabeled as final.
The initial package-inventory assertion counted35 ZIP directories as files;
the corrected file-only comparison verifies all416 source/package files exactly.

## Validation and scope

Offline wheel, repository Ruff/format and106 scene/water/terrain/i18n tests pass.
Twenty existing ground-image/filtering/normal browser cases also pass,126 scoped
pytest passes in total; these existing normal diagnostics do not cover tile
grout edges. The actual tile baseline above is separate from those diagnostics.
Optional mypy/pyright/Bandit/pip-audit are unavailable and are not passes.
The wheel is3,949,875B, the same size as the integrated PR52 build; this round
does not perform a fresh installation or recertify installed behavior.
No source, dependency, texture, shader, pass, API or physics change is included.
No new full-suite, canonical-baseline, installed-wheel smoke, budget, loading,
frame-rate, memory or material-improvement result is claimed for this screen.
Existing budgets remain unchanged.

The shared native Mac Chrome zoom check was already completed in the
[surface-inspection milestone](2026-10-28-surface-inspection.md):
80→90→100→110→100, buffer/GL/camera/state guards and restored original zoom.
Earlier locked setup/no-op attempts remain unpassed; this is not monitor-switch
or all-browser certification. This screen does not rerun or relabel that evidence.

The broad visual-quality, lightweight-loading and creative-play goal remains
active. Single-action success tuning stays deferred. This grout candidate stops
at the evidence gate; it should not be revived by a parameter sweep without a new
ordinary-use reproduction.
