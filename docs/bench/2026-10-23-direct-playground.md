# Direct robot creation and placement

This increment makes the existing viewer usable as a local creative playground without requiring a chat client for every robot addition.
The broader goal remains overall visual quality and playability within the existing loading and rendering budgets.
Shared materials from the preceding increment are retained; this change does not claim another material-quality improvement.

## Visible behavior

An empty world offers Add robot, and the existing Robots drawer offers the same four illustrated local starters: rover, robot dog, arm and humanoid.
Choose one, then click or tap an open ground or tabletop surface.
Enter places at the marker, Escape cancels, and normal button focus and activation remain available.
Mouse wheel zoom and right-drag camera movement remain available during placement.
The new robot is selected and framed only after the matching scene, first poses and successful reply have arrived.
Maps, terrain, modes, joints and reset remain in their existing locations.
The empty prompt no longer covers the Maps panel on a phone.

Adding recompiles the world and preserves unchanged robot and object poses using the existing carry behavior, with velocities reset.
It does not promise uninterrupted motion.
Entering placement stops the previous keyboard/gamepad drive and suppresses gamepad arm jogging during placement.
Ground placements continue to follow later terrain changes; tabletop placements retain their explicit support height.

## Boundaries and compatibility

The new action uses the existing local viewer WebSocket world-request path, with strict validation and four server-owned template IDs.
It accepts finite bounded coordinates and an expected scene version, with no file paths, imports, downloads or arbitrary builder parameters.
Plain models, active lessons, owned executions, stale versions and occupied saved/current positions are refused.
The final collision check reads current poses under the commit lock, so movement during preflight cannot silently relocate another robot.
Unique names and serialized edits prevent simultaneous clients from overwriting an existing robot.
World format, public add_robot semantics, MCP definitions and physics/controller implementations remain unchanged.
The catalog gains one additive robot_templates field.
Feature terrain now reads XY from either existing XY or XYZ robot positions.

## Evidence and limits

The frozen baseline is ec14ff98d073ffd31c13920a22f2ab8343e5417a.
Both baseline and candidate use the same empty checker-floor world, isolated fresh browser profiles and the same read-only capture harness at laptop and phone sizes.
Navigation measurements start with an already running server; a fresh profile does not flush machine-wide GPU caches.
Single navigation samples are observations, not evidence of a general speed improvement.
Paused rAF and synchronous render/readback measurements are distinct from active-simulation throughput.
Raw worlds, streamed scenes, screenshots, timings, resource responses, manifests and source hashes remain in the sibling robotforge-evidence/2026-10-23-playground-entry directory.

## Actual creation loop

The accepted source SHA256 is 55a07c1465a1bfca77322195ef4072cae93c4bf364ea325c42cefcf972d910e8.
It remains unchanged across final browser captures and verification.
The real Chrome/MuJoCo desktop video is 33.20 seconds at 1280 x 800, 30 FPS; the phone touch video is 11.333 seconds at 390 x 844, 30 FPS.
All eight placements succeeded, including four different starters, repeated arms and an arm on a real desk at a 0.75 m surface height.
Native W-down/W-up moves the rover 1.1525 m and sends an explicit stop.
The recording also covers arm joint control, stairs/hills/flat, wood floor, the desk map, reset and another placement.
The active rover sample is 60.005 FPS over two seconds.
Both videos fully decode, and their contact sheets and full desktop/phone screenshots were inspected.
There are zero JavaScript/runtime/WebGL errors.
Camera setup uses the viewer's documented diagnostic hooks; the actual actions use native mouse, keyboard and touch.
Camera changes are abrupt test setup, and wide views intentionally show multiple robots; these are functional recordings.
Phone verification covers touch creation and panels, rather than a touch joystick.

Videos remain under journey-final/desktop.mp4 and journey-final/phone.mp4 in the evidence directory.
Desktop SHA256: cdb4a3ae11f5346b4fea05e2a0bf1aeed98fabdcfedb1005be478ba31e687d18.
Phone SHA256: 16c7e4436db82d043edef7f14f68a4bc2532f589b342aabe24270727759b80e5.

## Empty-world comparison

World definitions match exactly before and after.
Streamed scene definitions also match after removing only the new additive robot_templates catalog field.
The frozen-harness observations use DPR 1.

| Measurement | Baseline | Final |
| --- | --- | --- |
| Laptop first frame | 387.0 ms | 210.4 ms |
| Phone first frame | 204.0 ms | 190.2 ms |
| HTTP resources / encoded payload | 23 / 1,231,275 B | 24 / 1,243,949 B |
| Laptop render/readback median / p95 | 0.4 / 0.8 ms | 0.5 / 2.2 ms |
| Phone render/readback median / p95 | 1.4 / 3.9 ms | 0.6 / 1.0 ms |
| Paused rAF, both viewports | about 60 FPS | about 60 FPS |

The extra resource and code add 12,674 encoded bytes; idle draw calls, triangles, textures and geometries remain 2/4/4/2.
Laptop render p95 is higher in this sample, so the table is not a universal rendering-speed improvement claim.
A separate journey observes fresh/warm navigation of 184.4/77.4 ms on desktop and 202.6/78.1 ms on phone, with controlled localStorage state.
These are not pooled with the frozen fresh-profile comparison.
End-of-journey JavaScript heap samples are 24.3 MB desktop and 12.2 MB phone; the worlds differ, and this is not a leak test.
The candidate wheel grows by 6,112 bytes against the clean baseline built in the same environment.

## Verification

184 focused tests pass, including 40 new placement regressions and existing world, carry, placement, mode, viewer-server, execution, capability and design tests.
22 real Chrome cases pass: the new loop in EN/ZH at laptop/phone sizes, held-key/gamepad suppression, all eight existing gamepad journeys, and relevant phone framing, fit-view and focus-layout checks.
Independent source/Python/security review approves the final implementation.
The complete default suite is not rerun for this increment; the earlier full-suite result remains historical.

## Reproduction

```sh
uv run --frozen pytest -q tests/test_viewer_creation.py
uv run --frozen pytest -q -m browser tests/test_viewer_creation_browser.py tests/test_viewer_gamepad.py
uv run --frozen python scripts/smoke.py
uv run --frozen python scripts/framework_regression.py --baseline docs/bench/framework-regression-v1.json --out /tmp/robotforge-entry-regression
uv run --frozen python scripts/bench.py --only install viewer --save --out /tmp/robotforge-entry-bench
```

No new dependencies, budget relaxation, paid Actions, release or deployment is part of this increment.
Native Windows and optional type/security tools unavailable locally are not claimed as verified.

## Existing performance budgets

The install/viewer benchmark passes all 11 applicable existing budgets without changing budget.json.
Imports, physics-throughput and MCP benchmark sections are not part of this partial run.
The standard viewer scenes use 1280 x 800 at DPR 2, with a 2560 x 1600 drawing buffer.

| Measurement | Final | Existing limit |
| --- | --- | --- |
| Wheel | 3633.41 KB | 4096 KB |
| Installed runtime | 163.13 MB | 180 MB |
| Packages / download | 42 / 52.51 MB | 45 / 60 MB |
| CLI start to first frame | 0.56 s | 2 s |
| Warehouse / asset scene FPS | 60.1 / 60.2 | 50 minimum |
| Warehouse / asset render time | 2.80 / 4.30 ms | 8 / 12 ms |
| Dog server peak / idle CPU | 133.1 MB / 3.8 percent of one core | 160 MB / 6 percent |

All seven canonical recorded framework cases also pass with the golden baseline unchanged.
The actual CLI/viewer/MCP smoke passes in 3.4 seconds, lists 60 tools, drives a rover, runs a script and renders a PNG.
Wheel packaging, Ruff, ES-module syntax and manual secret/input-boundary review pass.
The companion JSON and immutable verification experiment record retain the raw evidence hashes and exact verification scope.
