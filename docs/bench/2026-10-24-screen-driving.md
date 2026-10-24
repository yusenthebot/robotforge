# On-screen driving in creative play - 2026-10-24

Round 66 adds native hold-to-drive controls so touch users can explore worlds they build without a keyboard or gamepad.
The broad shared visual-quality, lightweight-loading and playability goal remains active.
Individual manipulation-success tuning remains deferred.

## Behavior and compatibility

- Coarse-pointer devices show controls by default; desktop users enable them under Robots → Display.
- Buttons follow the selected robot's shared capabilities: ground/boat forward and turn, ROV vertical movement, aircraft vertical and lateral movement.
- Native pointer capture supports independent fingers, opposite directions, same-button multiple pointers and Space/Enter hold.
- Keyboard and screen input combine per axis; existing nonzero gamepad input retains priority.
- Pause, reset, editing, selection, loading, hidden document, capture loss and disconnect release held input.
- The panel does not start the simulation automatically; lessons and arm sliders retain their existing behavior.
- Held commands renew every 250 ms only while held; the browser requests a 750 ms wall-clock hold.
- The server validates finite axes, bounded hold duration, current scene/version and runtime generation before effects.
- Latest-owner expiry/disconnect releases only that robot and controller generation; another client's newer command is preserved.
- Existing commands without hold/context retain their prior protocol and lifetime.
- Successful quick move/heave/stop and ownership notification are atomic under the existing session lock; long planning is not wrapped in that lock.
- A pre-existing nested multi-waypoint flight event is encoded through the already supported JSON path form; single-point and string forms remain compatible.

## Verification

Frozen runtime SHA256: `04e265220deedbac623587b7610a34b7b102fa760b79cce9e4bd1ab296505779`.
All 570 frozen runtime/test/budget inputs remain unchanged throughout final verification.
Baseline is exact main commit `fc4f6074282208f2dc90b77f10dae1a74f169627`, runtime `b6a2e8891de433c55e899b0ce9cf0886bdbc9a1c9848322724b8af3c03d347f6`.

- 1000 scoped default cases passed, one skipped; this is not a complete default-suite run.
- Complete browser selection: 151 passed; additional slow asset/sensor GPU selection: eight passed, totaling 1159 distinct passing pytest cases.
- Seven unchanged canonical framework cases, real source CLI/viewer/MCP smoke with 60 tools, and a fresh offline locked-wheel installation smoke passed.
- Ruff lint/format, 23 JavaScript parser checks and exact equality of all 392 wheel package files with source passed.
- Independent frontend, Python and security reviews found no outstanding actionable issue in their stated source-review scopes.
- mypy, pyright, bandit and pip-audit were unavailable; these checks were not run and no tools or dependencies were installed.
- 119 offline evidence/source/media/control checks passed; the JSON report preserves the individual assertions.

## Real app and media

Native Chrome UI input drives the actual server and simulation; no keyboard/gamepad, direct teleop injection, pose writes or manual stepping is used during candidate recordings.
The fixed course and camera are prepared before recording.
Each candidate drives into the exact rover/wall owner pair at five distinct positive-force simulation times, reverses, turns, releases, pauses and resets.
The Chinese touch-emulated phone additionally holds forward and turn together, then releases forward independently.
All recorded native input events are trusted; phone controls meet 44-pixel touch targets and layout checks find no overlaps, clipping or offscreen controls.
Baseline video shows the missing screen-control entry and idle Play/pause; it is not an equivalent driving-task timing comparison.

| Recording | Duration | Encoded frames | File under evidence root |
| --- | ---: | ---: | --- |
| Baseline | 3.200 s | 96 | video-baseline/screen-drive.mp4 |
| Touch-emulated phone | 7.567 s | 227 | video-phone/screen-drive.mp4 |
| Desktop | 7.533 s | 226 | video-desktop/screen-drive.mp4 |

Every encoded frame fully decodes; original uncapped screencast JPEGs/timestamps and six deterministic decoded stills per video are retained.
The coordinator inspected the decoded contact sheets and control screenshots; this is not every-frame visual certification.
An independent actual WebSocket/wall-clock probe observes a nominal 250 ms nonrenewed hold become commanded zero after 275.54 ms, and latest-owner disconnect after 1.64 ms in this local sample.
Expiry also works with simulation time paused; old-owner disconnect preserves the newer owner, and legacy unbounded commands preserve their original disconnect semantics.
These are commanded-zero observations, not guarantees of instantaneous physical rest or timing on every machine.
Physical phones, Windows, other GPU drivers and every plugin/model/task remain unverified.

## Lightweight and fixed-scene controls

Exactly one unchanged official install/viewer budget run passed all 11 covered checks; other budget groups were not run.
No thresholds, sample counts or aggregation rules changed, and the runtime was not repeatedly measured until green.

| Metric | Candidate |
| --- | ---: |
| Wheel | 3763055 B, +5335 B |
| Installed / runtime packages | 163.2564 MB / 42 |
| Dependency download | 52.5149 MB |
| CLI first complete frame | 0.4909 s |
| Standard / asset FPS, DPR 2 | 60.097 / 60.1269 |
| Standard / asset draw | 2.9 / 4.0 ms |
| Dog server peak / idle CPU | 132.5 MB / 5.4% |

Actual gallery, kitchen, park and wooden-prop scenes retain all 17 byte-identical full-canvas PNGs, exact scene payloads and initial/final frozen physics, and unchanged calls, triangles, textures, geometries and programs.
Reconstructed camera/target floats use only a 1e-12 comparison tolerance; pixels, scene payload and physics remain exact.
One sequential baseline/candidate sample retains fresh/warm navigation, 40-draw GPU/readback samples, RAF and process-tree RSS; this is not ABBA or a general speedup claim.
Fresh HTTP grows by 12057 B and warm HTTP by 2212 B, with 29 versus 28 requests.
Fresh first-frame samples are gallery334.5→342.7, kitchen297.3→304.3, park342.6→375.8 and wooden-props180.4→191.3 ms; warm differences are +28.2/−0.3/+19.0/+4.3 ms respectively.
Draw medians are mixed, including park near1.40→3.10 ms; RAF is about59–60 FPS at DPR1.
Chrome process-tree RSS changes +10.23/+11.84/+4.00/−3.50 MiB respectively; overlapping/shared pages may be counted multiple times and this is not VRAM.
No new dependency, packaged image, shader, texture or rendering pass was added.

## Evidence and next direction

Raw evidence is outside the repository at `../robotforge-evidence/2026-10-24-screen-driving/`.
It retains exact baseline/source archives, frozen inputs, original failed-attempt notes, review scopes, logs/JUnit, fixed fixtures/views, real socket probe, native inputs/wire receipts, raw/encoded media and full decode checks.
`summary.json`, `raw-index.json` and the unified ExperimentStore `record-summary.json` identify the scoped completed verification; they do not complete the broad goal.
Intermediate failures are honestly summarized; exact intermediate source snapshots and complete early stdout are not claimed.
The preceding round65 visual investigation rejected normal-only box edges, roughness-variance adjustment and contact AO after actual scene checks; those prototypes did not enter production.
The next priority remains visible shared indoor/outdoor material improvements and smoother creative exploration, judged against unchanged loading/frame budgets and actual motion evidence.
