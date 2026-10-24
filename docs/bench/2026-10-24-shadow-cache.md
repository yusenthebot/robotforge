# Shared shadow reuse and context recovery, 2026-10-24

Round70 supports the shared material-quality, lightweight-viewer and playability goal.
It follows the natural-wood candidate in PR25; the original round69 budget failure remains in its report.
The implementation reuses rigid-world shadow depth while the casters and framing stay unchanged.
A moving caster invalidates depth even when the fitted shadow bounds stay unchanged; scene rebinding and context restoration also invalidate it.
Worlds with rotors or flexes retain automatic depth rendering because they animate below body transforms.
The existing renderer-wide sensor gate stays unchanged, so temporary depth/segmentation/stereo materials cannot populate the main shadow map.
Shadow resolution, shader, lights, texture slots, physics and public interfaces stay compatible; there is no new dependency or external request.

## Recovery fix

The new real-context RGB test first failed against both the previous source and the cache candidate with the same hashes.
A separate diagnostic records identical camera/projection state: normal RGB after restoration equals rendering with environment lighting disabled.
Forcing a fresh shadow does not repair it; regenerating the studio environment restores the original RGB hash.
The GPU-generated studio reflection map must therefore be regenerated after context restoration for this fixture.
The app now rebuilds it after Three.js restores renderer state, disposes the previous texture and marks shadow depth dirty.
The regression repeats actual context loss/restoration twice and compares the rendered pixels each time.
This evidence does not claim to cover every context-loss scenario or identify a driver cause.

## Diagnostic limits

The finite GPU timer query experiment shows that elapsed GPU work also varies; it does not establish one cause for the earlier frame-time variation.
Its full-size RGBA8 diagnostic target adds 15.625MiB, all four paths retain their actual timings, and the blit path omits final state restoration from its reported inner-loop time.
The diagnostic path labelled A-original uses a frozen pose and a center-pixel readback, whereas the unchanged official budget reads pixel0,0.
These diagnostic paths are not product changes or replacements for the original performance budget.
A new controlled park pair records unequal pre-settle projection offsets and identical images after eight stable actual animation frames.
That establishes a capture confound in the new pair; older samples without projection records cannot be retroactively explained.
Production fixed captures settle and record the actual projection before measured renders.
Original failed diagnostics and the initial development-context failure remain in the evidence directory.

## Verification and cost

Runtime SHA256 `2e30c1a13ef61233bbba8314839ece10ba342d8d2446dcf4eb9ee1f4122b6a7e`; 620 frozen source/test/script/lock/budget inputs remain unchanged.
The cache baseline is commit `e30e4eab11526a4e1edde75c840deb205a83592d`, runtime `3b6c8f04601d3dd768c47c759e687a75ea6c342d5efa3e95cad6d268371051b7`.
1459 distinct pytest cases pass: 1280 focused-default (one skipped), 171 complete browser cases and eight additional slow resource/sensor cases.
Nine new browser cases are counted once; seven unchanged canonical framework baselines, source and fresh offline locked-wheel CLI/viewer/MCP smoke pass (60 MCP tools).
The full default suite was not rerun; mypy, pyright, bandit and pip-audit were unavailable and not installed.
Ruff/format, JavaScript parsing, credential-pattern review and all 394 packaged source files pass.

All five compiled model array sets, scalar values, expanded mesh/payload bytes and frozen physical states match the before-cache source.
All 22 actual fixed views are pixel-exact with matching projection, eye/target and resource counts.
Three actual caster-group moves preserve the fitted bounds but dirty and refresh depth; moved/restored images match the uncached source.
The terrain-only world has no dynamic body and explicitly skips that probe.
Across 22 views with 40 synchronous render/read samples each, GL draw calls fall from 130880 to 60120 (54.1 percent).
Per-view times remain mixed, including slower kitchen-low, park-far and terrain-low samples; a sequential pair does not prove universal frame-time, memory, loading or temporal-quality improvement.

Exactly one unchanged original install/viewer budget run for this new source passes all 11 applicable gates.
Other budget groups were not run; the original round69 failed run remains failed and unmodified.

| Measurement | Current source |
|---|---:|
| Wheel | 3853930 B |
| Wheel versus before-cache / main | +331 / +90302 B |
| Installed / packages | 163.3743 MB / 42 |
| Dependency download | 52.5149 MB |
| CLI first frame | 0.5402 s |
| Standard / asset FPS | 57.3836 / 57.4064 |
| Standard / asset synchronized frame | 2.1 / 11.0 ms |
| Dog-server peak / idle CPU | 132.9 MB / 3.8 percent |

Cache/recovery adds 879 encoded bytes on each fresh navigation and zero on warm navigation; actual HTTP request counts stay 30 for wood worlds and 29 for park.
The historical helper count of 32 in wood worlds includes two embedded image-data responses; it is not 32 local HTTP requests.
Cold/warm fixed-scene first-frame samples are mixed, including slower kitchen loading, and are fully retained in the data and raw manifests.
The existing natural-oak pair remains 86881 B, with the same nominal extra1 MiB RGBA+mip storage relative to main; caching introduces no new shadow resolution, map or image resource.
Repeating knots remain visible and object/robot wood is unchanged.

Seven original recordings pass strict full decode: six 3.3-second actual gallery/kitchen/wood-terrain camera clips plus an 8.2-second actual native desktop rover recording.
The latter records five distinct positive-force rover/wall contact times, reverse, turn, release and pause/reset, with actual natural-oak512/256 maps asserted before/after.
Root inspected six decoded sample frames from each movie (42 total), not every-frame visual certification.
Physical phones, Windows, every model and every task were not verified in this increment.
The initial context failure, evidence-helper path failure and script-construction error remain retained; fixing external harness paths did not change any frozen product inputs or rerun passed tests.
Final numeric report: [2026-10-24-shadow-cache.json](2026-10-24-shadow-cache.json).
GitHub disposition is tracked by PR25 and the external landing receipt; the broad goal remains active.
Raw evidence: `../robotforge-evidence/2026-10-24-gpu-readback/production/`.
