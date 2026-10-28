# Surface inspection — 2026-10-28

Views → Look here lets a qualified surface tap choose the orbit/zoom centre on
robot parts, fixed scenery, local terrain and flex. Top/Front/Side use that point;
Back restores the existing original camera record. This supports exploring and
comparing existing surfaces. Default materials and physics remain unchanged.

## Behaviour and compatibility

The camera retains its viewing direction and eye-to-hit Euclidean radius. It does
not automatically magnify a part or promise identical off-axis projected size.
The point is a world-coordinate snapshot, not live body tracking. Hidden ancestors,
zero-opacity/invisible hit-face materials and camera-axially clipped hits are
ignored. Animated water uses its bounded still level, not wave crests or a large
triangle traversal. No new request, asset, image texture, shader/pass or dependency.

Only qualified tap release selects a surface. Drag, multitouch, stale generation,
version/session/gesture changes, disconnection and animation cannot submit it.
Inspection suppresses grab/navigation/drive inputs, restores prior editing bars,
and clears old placement points. It does not pause physics or submit an edit.
Existing MCP (60 tools), script, world and layout interfaces remain unchanged.

## Real verification

Frozen runtime `0c9e9243b7463a3dbc75a8b7279606d1b3baabe5a2ef939e09ae250c489e52da`;
768 source/test/script/dependency inputs stay exact across the preintegration
final receipts. The later upstream integration is recorded below with a new hash.
663 viewer-default/static-reader tests pass (six network opt-in manipulation
cases skipped); all317 non-slow browser tests pass (one optional pxr collection
skip); nine selected slow GPU tests pass. These are989 distinct pytest passes,
including the seven inspection cases. This is not the whole default/all-slow suite.
Seven unchanged canonical framework cases pass with the actual Python3.12.13
binary required by the existing baseline. Everyday tests run3.12.16; no version
or baseline gate was relaxed. Source and fresh offline installed CLI/viewer/MCP
smoke, Ruff/format, JS parsers and offline wheel pass. Optional mypy/pyright,
Bandit and pip-audit are unavailable, not passes. Added-line secret audit is limited.

Before implementation, trusted Chrome right-button pans on three paused original
worlds visually centred a target but retained an off-surface 3D orbit pivot.
The new actual surface taps keep the point through Side and wheel input and return
the initial eye/target within1e-8. Each candidate session preserves480 model arrays,
qpos/qvel/ctrl, version, paused time0 and original user-world bytes.

| Original category | Before pans | Before pivot error m | After maximum error m | Handler p95 ms |
|---|---:|---:|---:|---:|
| malo-head | 2 | 0.228595 | 5.32e-08 | 18.61 |
| go2-rack | 2 | 0.742993 | 8.79e-08 | 16.81 |
| go2-dock | 3 | 2.281950 | 5.91e-07 | 16.96 |

These are scripted, mathematically informed initial cameras/targets, not a human
navigation-efficiency study. Handler measurements are10 serial warm native taps
per world, pointerup receipt through the source zero-timeout handler and
microtask-observed bar hide. They exclude physical input, CDP delivery and next
GPU presentation; all pass the prospective50ms gate without retuning.

Six complete timestamped H264/30fps workflow movies fully decode. Root inspected
36 decoded sample frames; raw JPEGs, timestamps and full movies remain local.
No duration cap; CFR resampling does not preserve every source JPEG as a frame.
The cameras are broad and the clips demonstrate orbit/zoom/return, not automatic
part magnification, material aesthetic improvement or action-success rates.

| Workflow | Decoded seconds | Frames |
|---|---:|---:|
| before / malo-head | 9.300000 | 279 |
| candidate / malo-head | 7.600000 | 228 |
| before / go2-rack | 9.733333 | 292 |
| candidate / go2-rack | 7.733333 | 232 |
| before / go2-dock | 13.700000 | 411 |
| candidate / go2-dock | 7.166667 | 215 |

Separate before/candidate package roots retain48 fixed PNGs/24 byte-exact main
3D-canvas pairs: untouched Malo/Go2/Panda, explicit light media, actual browser-
emulated DPR1/2, studio/daylight, normal/detail.1770 checks pass. Projection,
default materials, draw work/resources/programs match; eye/target tolerance1e-10.
All cross-process model/state fields agree except retained transient paths and
mesh_pathadr. One whole-window Go2 status-text region differs879 pixels/max167,
bbox[1831,1497,1913,1520]; no full-window identity claim. Root viewed three
representative candidate controls; main-canvas equality is checked separately.

Actual native macOS Chrome shortcuts80→90→100→110→100 pass in an owned
foreground tab in the existing profile, with no device-metrics or zoom emulation.
DPR is approximately1.6/1.8/2/2.2/2; renderer ratio follows min(DPR,2), buffers use
actual floating-point floor sizing, GL/errors remain0, eye/target and all480 model
arrays/state/names/paths stay exact. Returning projection is exact; the original
100% zoom is restored and the test tab/server closed. Root viewed90/110% stills.
This closes the current-code native-zoom pending item, not the earlier failed or
locked captures. It is not physical monitor switching or a fresh fractional-DPR
pixel reference. Developer CDP observations/listeners were confined to the closed
test tab; the served application remained unchanged.

## Weight and measured cost

Wheel3,949,773B, +1,320B (gate≤5,120B); all416 source/wheel/fresh installed
product files agree. Dependencies and the original13 WIP files stay exact.
Observed cold HTTP resources remain31; encoded resource bytes+5,265B in both
original workspaces (gate≤8,192B), excluding WebSocket traffic.

Original11/11 install/viewer budgets pass once, without threshold changes:
163.665MB/42 packages, CLI first frame0.8924s, standard/assets60.087/60.111FPS
atDPR2,3.2/3.5ms draws, standard dog-server peak135.2MB and4.4%idle CPU.
The warehouse diagnostic peak566.6MB is separate from that dog metric. Other
budget groups were not run. Host load was around18 on18cores, not a quiet-machine
or universal performance guarantee.

The prospectively ordered ABBA+BAAB cohort uses fresh package processes and
Chrome profiles, four samples per source/world, each browser-cache-cold navigation
followed by actual warm reload; light media is explicit before navigation.
Candidate-minus-before mean ready-to-GPU-finish differences:

| Original world | Browser-cache-cold ms | Warm reload ms |
|---|---:|---:|
| malo | -5.90 | -4.77 |
| go2 | +4.05 | -13.50 |

All samples, medians/ranges, resource counts/bytes, sampled Chrome RSS and original
order remain in JSON/evidence. Servers/models/OS caches are warm and paused;
ready means matching main scene/no loading/busy plus2RAF/gl.finish. This is not
first-ever remote import, CLI startup, complete sensor compilation, human latency,
or a causal/universal speedup. RSS is a process-tree sample, not peak/steady/VRAM.
Fixed default draw samples show before/candidate medians0.1–0.5ms; raw30 draws
(after10 warmups) and120 RAF intervals per view remain external, not independent
trials or proof of a causal GPU cost change.

## Retained attempts

The JSON lists every selected-out, fixture-target, event-point, ownership, framing,
preinspection-wire-window and Chrome-startup attempt. The earlier v11 seven-case
receipt predates additional test-file assertions; all product inputs match, and
the final317 reruns all seven current cases. The first report generator correctly
stopped on that mismatch before writing documentation; it is retained and corrected
as a historical-run bridge. Keyboard reentry bar restoration and axial clip rules
were corrected before freeze; the two extended object-bar cases also pass within
that final selection. Native v1 failed camera settling before zoom actions;
its cause is unestablished. Final foreground v2 passes without product changes.
A missing temporary WIP list interrupted the precommit audit while dependent shell
commands continued the code push. The persistent13-hash list was immediately
rechecked and exact, and subsequent package/final audits pass; the interrupted
audit is not represented as a precommit pass. Old evidence remains intact.

Evidence: `../robotforge-evidence/2026-10-28-surface-inspection/`, indexed with
hashes in evidence-index.json (`139ccd5862749a136d0ad46675be1192ad1c4ed35517b028e337521bc5379966`).
The JSON report records final receipt hashes, source bridge, native checkpoints,
load samples, media hashes, costs, attempts and scope. Media is local, not a
public GitHub attachment. No paid Actions, release/deployment or per-action success
tuning. The broad visual-quality and creative-play goal remains active.

## Latest-main integration before merge

During verification, main advanced to `ab05dd1` with an independent USD inertia
fix. It was merged normally into this feature branch (`dc1907a`), preserving it.
Integrated runtime is `b399c6c405fedc612c92f0e8323529f5f238b42bd01a03bd2df2fdb768d490ae`.
Only `zenobench/usd_mjcf.py` and its existing test differ among the768 frozen
inputs; the other766, including all viewer files and seven inspection cases,
are byte-exact. The full browser/native/video/control/load/budget evidence above
retains its actual0c9 runtime hash; it is not mislabelled as a new full-suite run.

On the integrated runtime,28 optional USD tests pass in the existing extras
environment, explicitly importing this worktree's module (Python3.12.16,
MuJoCo3.15.0, NumPy2.5.4). Together with the earlier989, these are1017 distinct
phased pytest passes. Full Ruff/format, offline wheel/fresh installed package,
source and installed60-tool smoke, and the unchanged seven canonical cases pass
again. Python review approves the event tests and inertia fallback. No new
dependencies were installed and no passing viewer budget was repeated.

All416 integrated source/wheel/fresh installed product files agree. Final
wheel3,949,875B is+1,422B versus the preceding3b916 closure, comprising the
measured preintegration feature delta1,320B plus102B upstream change. Original
13 WIP files remain exact. The integration-index.json chains to the prior evidence
index and records the new receipts/package/source bridge.
