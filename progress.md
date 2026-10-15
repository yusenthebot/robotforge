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

## Round 7 (2026-10-15): teaching mode, PID and IK lessons, studio lighting

- User ask: the app only offers WASD; it needs a teaching mode where you learn control, IK, sensors while playing; assets look cheap.
- `src/robotforge/learn/`: Lesson (controls incl. choice, plots, steps with physics checks or quizzes, rig facts for overlays, camera) and LessonRunner (swaps in the lesson rig, restores the world on exit, re-attaches on Reset, ends if the world is swapped, 20 Hz live updates via `hub.lesson_stream`).
- Viewer: `lesson.js` (course picker, step panel, controls, canvas plots, overlays per rig kind, draggable target), `look.js` (studio env map + rf_* PBR presets), i18n strings in `i18n.js`.
- Asset upgrade delegated to an agent on `feat/assets` (visual-only meshes, rf_* materials, physics unchanged).
- What did NOT work (and the fix):
  - PD rig crept at constant speed under full torque: the hub geom overlapped the motor housing, a permanent contact. Lesson arms touch nothing (contype 0).
  - Every gain gave the same 27% overshoot until the grid search showed Kd needs ~3: the Kd slider now goes to 6; 3 stars need the +-1 degree band, which needs feedforward or I (the lesson's point).
  - MjSpec reads joint `range` in degrees unless `spec.compiler.degree = False`: IK joints were pinned near 0.
  - The IK forearm blew up (NaN qacc) under servo damping with explicit Euler; implicit integration + armature fixed it; a stress test guards it.
  - Iterative IK had nothing to show when the analytic solution had already reached the target; switching to it now moves the target.
  - Dragging the target lost the last position to the 30 Hz throttle; the release always sends it.
  - A fast synthetic drag (left_click_drag) does not move a physics body; scripted checks hold the pointer for a second.

## Round 8 (2026-10-15): sensors lesson, learned walk without extras, bounded MPPI walk

- `learn/sensors.py`: rover in a walled arena with encoders, IMU, 24 rangefinders, a bumper touch sensor and a camera; lessons can declare `drive` and the viewer routes W A S D to them; overlays draw lidar rays by distance, the bumper on contact, the filtered acceleration and the camera cone; a picture-in-picture renders the robot's camera view (render target + readPixels at ~12 Hz).
- `bridge/onnx_numpy.py`: minimal ONNX protobuf decoder + numpy evaluator (Gemm, MatMul, Elu, Relu, Tanh, ...); the shipped Go2 walk runs without onnxruntime, matching it to 1e-4 at ~23 us per call.
- `modes.py`: `Then` hands over from one controller to another; MPPI walking walks 8 s, then stands with MPC.
- Agent skill and MCP instructions point at scenes, terrain, the parts builder, learned walkers and the viewer lessons.
- What did NOT work (and the fix):
  - MuJoCo 3.15 rangefinders refuse to compile with `intprm[0] = 0` (the ray data bitmask); set it to the distance bit. The parts builder has the same bug (review HIGH 2), fixed after the asset merge.
  - The raw accelerometer buzzes +-5 m/s^2 from contact chatter at constant speed; the plot shows a 0.12 s low-pass, as real robots filter it.
  - A test of a lesson rig must hold actions long enough for physics; scripted pilots steer from the true pose.

- Later in round 8: merged `feat/assets` (agent): `design/visuals/` mesh library + collector, looks for 6 templates, all props and parts-built robots; physics unchanged bit for bit on 18 models; budgets 30k tris per robot, 200k per scene. The viewer matches `rf_` presets after the last `/` (robots arrive as `dog/rf_shell`).
- Builder review HIGH 1-4 fixed with failing-first tests: rangefinder bitmask (HIGH 2), mirrored slides keep their direction (HIGH 3), IK seed wraps into the base limits (HIGH 4, the -268 degree seed), balancer detection asks physics when a caster sits a few mm high (HIGH 1).
- Open, found while verifying HIGH 4: after the rover drives a turn and stops at an unplanned heading, pick-and-place grasps and carries fine but at release the rover twists 10 degrees and the cube is flung (0.39 m up). Likely gripper or cube against the bin wall; trace in scratchpad/arm_turn_trace.py. Not the same failure as HIGH 4.

## Next-round seed

Round 9: playability phase 3 from docs/PLAY_PLAN.md (challenges with stars on the existing scenes: obstacle course time trial, maze, push-recovery, a controller race) or more lessons (walking gaits, RL) depending on the user's answers; investigate the release fling; builder MED 6-9.

## Frontier

- Current ceiling: any robot in 6 scenes and 8 terrains, driven by gait, MPC/MPPI, LQR, IK or learned policies from the viewer, plus two interactive lessons that teach PID and IK by doing.
- Next frontiers: behaviors (round 4), import doctor + robot library (round 5), real-agent evals once CLIs work, MCP App embed in Claude Desktop (needs user to add the server to Claude Desktop).
- Radical ideas weighed: physics in browser via MuJoCo WASM (kept for sharing only).
- Escalation ladder: open-loop gaits -> CPU RL for custom morphologies -> GPU backends (MuJoCo Warp on NVIDIA) -> sim2real.
