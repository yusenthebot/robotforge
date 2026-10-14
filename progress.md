# progress

## Current state

Round 1 done, 2026-10-14.
`src/robotforge/sim/session.py` owns one MjModel/MjData and advances it in scaled real time (debt-based, capped at 0.1 s per tick so stalls become slow motion, not bursts).
`src/robotforge/viewer/` serves a vendored three.js r178 client (no build step) and a WebSocket: scene JSON per model version, mesh blob over HTTP, binary pose frames (float32 time + xpos + xquat) at 60 Hz, state JSON at 4 Hz.
Client sends ctrl, cmd (play/pause/step/reset), speed, drag_start/move/end.
`robotforge open <mjcf|urdf>` and `robotforge demo` work; demo is a 3-DOF arm with cubes and a ball.

## What worked

- Python-authoritative physics with a thin viewer: realtime factor 1.00 and 120 fps on the demo and on a 263-geom Go2 room scene.
- MuJoCo's own mesh normals (mesh_normal + mesh_facenormal) give correct hard-edge shading without client-side normal computation.
- Spring drag plus subtree gravity compensation: the held body tracks the cursor (test asserts within 1 cm).
- Cold start to a served scene: 0.19 s.

## What did NOT work (and the fix)

- `import robotforge` failed: iCloud Desktop sets the macOS `hidden` flag on `.venv/*.pth`, and Python 3.12 skips hidden .pth files. Fixed with `.venv -> .venv.nosync` symlink.
- `_GEOM_TYPE_NAMES` keyed by `mjtGeom` enums never matched `model.geom_type` ints, so the scene had 0 geoms. Keys are now ints.
- After a restart the server silently moved from 8737 to 8738: the free-port probe lacked SO_REUSEADDR so TIME_WAIT looked busy. Probe now matches uvicorn, and an explicit `--port` is strict.
- Scripted pointer tests in the browser pane failed because the pane is hidden: rAF is throttled (fps 0), so camera matrices were stale, and screenshot px != CSS px (viewport 1024x768 vs 800x600 screenshot). Picking now refreshes matrices itself; tests use `window.__robotforge.screenOf`.
- Instant synthetic drags (down/move/up in one tick) do nothing; that is expected, real drags work.

## Measurements

- Unpacked site-packages incl. dev tools: 153 MB (mujoco 72 MB, numpy 24 MB). Runtime-only footprint still to measure in a clean venv; A7 needs a clear definition (download vs unpacked).
- Go2 room mesh blob 15 MB as non-indexed float32 triangles. Indexed transfer would be ~2x smaller; not needed on localhost yet.

## Next-round seed

Round 2: MCP server with the official Python SDK (mcp 2.3.0 is a new major version; read its API before writing).
Tools: world_open, robot_add (MJCF/URDF path for now), sim (play/pause/reset/step), observe (state + rendered image), view (start viewer, return URL).
One process hosts both the MCP stdio server and the viewer server sharing a SimSession.
Verify with a real headless `claude -p` session that loads the demo, moves an actuator, and reports an observation; then the same with `codex exec`.
Also: headless offscreen rendering for `observe` (mujoco.Renderer) on macOS.

## Frontier

- Current ceiling: interactive viewer for existing MJCF/URDF; no agent, no design, no behaviors.
- Next frontiers: MCP agent loop (round 2), world.yaml + robot templates (round 3), behaviors + scenes (round 4), import doctor + robot library (round 5), MCP App embed.
- Radical ideas weighed: physics in browser via MuJoCo WASM (rejected as primary; kept for sharing).
- Escalation ladder: open-loop gaits -> CPU RL -> GPU backends (MuJoCo Warp on NVIDIA) -> sim2real with real hardware.
