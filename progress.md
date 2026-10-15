# progress

## Current state

Rounds 1-3 done, 2026-10-14.

- `sim/session.py`: one MjModel/MjData, real-time advance (only while a viewer is connected), drag with gravity compensation, `load_model` for compiled worlds.
- `sim/render.py`: offscreen PNG snapshots (CGL on macOS, works off the main thread), view presets + model cameras.
- `viewer/`: Starlette + WebSocket thin client (three.js r178 vendored). Grouped per-robot sliders, collapsible panel, Frame button, no-cache headers.
- `design/`: `components.py` (servo/motor catalog with approximate datasheet values, materials, colors), `templates/` (arm 3-6 DOF + gripper, quadruped 12 DOF with IMU/touch/camera, rover diff-drive), `world.py` (pydantic World -> MjSpec, `<robot>/` prefixes, objects incl. table and bin, `home` keyframe), `store.py` (world dir under $ROBOTFORGE_HOME/worlds, atomic edits, undo history, build/world.xml).
- `playground.py` + `mcp_server.py`: 18 MCP tools (design + control + observe + open_viewer with MCP Apps ui:// resource). `robotforge mcp` runs it on stdio; `robotforge open <world dir|mjcf|urdf>` opens the viewer.

## What worked

- MjSpec `attach(child, prefix, frame)` makes multi-robot worlds and imports clean. Each attach needs a fresh child spec.
- Real-parts sizing: quadruped (sts3215) stands unaided, arm holds a ready pose, rover drives straight at 10 rad/s.
- MCP SDK 2.x (`MCPServer`, `mcp.server.apps.Apps`) covers MCP Apps natively.
- Raising `ToolError` for anticipated failures gives agents actionable text ("Did you mean shoulder?").

## What did NOT work (and the fix)

- mcp 2.x hides messages of unexpected exceptions from clients; tools must raise `ToolError` (wrapped via `_tool_errors`).
- `Apps` snapshots its tools at `MCPServer(...)` construction: register `@apps.tool` before creating the server.
- Viewer showed a stale UI after code changes: browsers heuristically cached index/app.js. Now `Cache-Control: no-cache` (ETag revalidation).
- 604 px-wide pane (same as an inline chat app) made the old fixed panel cover half the view; now collapsible and grouped.
- `claude -p` real-agent run: blocked by expired CLI OAuth (needs user). `codex exec`: first run reached the model but MCP calls needed `mcp_servers.<name>.default_tools_approval_mode="approve"`; later runs fail with "workspace routing discovery failed/timed out" though curl to OpenAI works via the proxy. Needs the user. Also: `codex exec` waits on stdin unless `< /dev/null`.

## Round 4-5 additions

- Behaviors: `behaviors/quadruped.py` (trot, per-leg DLS IK; mj_jac needs mj_comPos, not just mj_kinematics), `drive.py`, `arm.py` (arc sweeps within joint limits, safe travel height, reach pre-check), `manager.py`.
- Viewer: WASD driving, robot picker, follow camera; `robotforge open <world>` uses the Playground so driving works without an agent.
- Scripts: `robotforge/script.py` API + `script_runner.py` lockstep runner over a token-guarded localhost socket.
- What did NOT work: straight-line arm carries through the base (IK singular, cube dropped) -> cylindrical arcs; shortest arc crossed the +-180 deg yaw limit -> limit-aware sweep; reach check from home pose stalled for targets behind -> seeded yaw; rover wheels torque-starved (kv too low) -> stiff speed loop; script budget spin when < 1 step left.
- RL decision (user): SB3 as optional extra + UniLab as external backend with ONNX policy playback. UniLab pins torch 2.7 + mujoco-uni 3.8, so it cannot share our venv.

## A7 measurements (2026-10-14, clean runtime-only install from the built wheel, macOS arm64)

- Download: 49 MB in 40 wheels (mujoco 27 MB, numpy 5.2, pillow 4.6, cryptography 3.7 via mcp, pyopengl 3.0 via mujoco).
- Installed: 157 MB unpacked (mujoco 74 MB, numpy 22, pyopengl 15, pillow 13, cryptography 11).
- So the "<= 100 MB core install" target holds for download size, not for unpacked size; mujoco alone is 74 MB unpacked and cannot be avoided.
- Cold start to a served scene: 0.19 s. World recompile with 5 library robots + a template: < 0.5 s. Viewer: 120 fps / realtime 1.00 on the Mac; 37-57 fps streamed on CI runners.

## Round 6 (2026-10-15): one trunk, a viewer beginners understand

- All feature branches merged into main and deleted; work is trunk-based from here (CLAUDE.md).
- Viewer rebuilt: light/dark tokens, amber = what you control (play button, selection ring with heading notch, drag tether, slider fill, targets).
  Bottom dock (play, reset, speed, fit view), Robots drawer (pick robot, joints), help popover, first-run hint, reconnect banner, English + Chinese (`static/i18n.js`).
- World panel in the app: Maps (6 scene presets + empty floor), Terrain (9 ground styles, 8 terrain types; features go in front of the selected robot), Modes (per robot type, `src/robotforge/modes.py`).
- Server: `hub.slow_handlers` run viewer requests in a worker thread and answer with `{"type": "reply", "id", "ok"}`; state carries each robot's mode; scene carries world, floor, terrain, modes, catalog.
- What did NOT work (and the fix):
  - `done?.(await request(msg))` never sent the request when `done` was unset: an optional call skips evaluating its arguments. Await first, then call.
  - Selection ring vanished under Go2: its foot meshes dip 2 cm below the floor, so "lowest point of the robot" was under the floor. Now a downward ray finds the scenery under the robot.
  - Ring lagged behind a walking robot in a throttled tab: per-frame smoothing. Now time-based.
  - The stand/walk tasks accept a rover (any free-moving robot); the Modes list filters by robot kind (`TASK_KINDS`).
  - Reach / drive-to "random" targets repeat: tasks seed their RNG with 0. Modes draw a fresh point and pass it explicitly.
  - Fake `Hub` objects in tests broke when the hub grew; tests use a real `ViewerHub`.
  - In the hidden browser pane, CSS transitions and rAF stall, so screenshots can show stale colors; not a user-facing bug.

## Next-round seed

Fix the open review findings on main, one at a time: builder HIGH 1-4 + MED 6-9, then UniLab 1-4 (numpy runtime for the shipped Go2 walker first, so the Learned walk mode works without onnxruntime) and naming A-E.
Then: bound the MPPI walk mode (walk for a while, then stand), update the agent skill + lessons for builder/terrain/unilab and the viewer's World panel, re-render the showcase film from main.

## Frontier

- Current ceiling: any robot (templates, parts, library, URDF) in 6 scenes and 8 terrains, driven by gait, MPC/MPPI, LQR, IK or learned policies, all pickable from the viewer.
- Next frontiers: behaviors (round 4), import doctor + robot library (round 5), real-agent evals once CLIs work, MCP App embed in Claude Desktop (needs user to add the server to Claude Desktop).
- Radical ideas weighed: physics in browser via MuJoCo WASM (kept for sharing only).
- Escalation ladder: open-loop gaits -> CPU RL for custom morphologies -> GPU backends (MuJoCo Warp on NVIDIA) -> sim2real.
