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

## Round 9 (2026-10-15): challenges

- `learn/challenges.py`: Challenge base on the lesson engine (kind "challenge", real scenes via `scene_model`, real behaviors via `lesson.playground`, timer from the first key, stars, Reset retries). Maze escape (seeded, par from the shortest path at 0.35 m/s), shove survival (hold / step in place / MPC stand; 0.2 m/s more per level), obstacle course (stars at the ends of the ramp, stairs, rough ground).
- Viewer: the picker groups lessons and challenges, generic `live.readout` and `results.summary` + stars, best stars kept per browser, the camera follows `drive.follow`.
- CI: the walkers note test failed on CI only; my venv still had onnxruntime from rendering the film. `uv sync` (default install) before trusting a local run.
- What did NOT work (and the fix):
  - Nothing finishes the obstacle course yet: template dog gait falls at the 4 cm curbs (~11-12 m), Go2 gait on the stairs (5.9 m), the Go2 learned walk stalls on the ramp (trained on flat ground). Scored by distance instead; a terrain-trained policy is a frontier item.
  - Shove robustness differs less than hoped between controllers (hold 4, gait 5, MPC 5-7 levels); thresholds and text match the measurements.

## Round 10 (2026-10-15): reinforcement learning, trained live

- `learn/rl.py`: cart-pole; release (it falls), reward quiz, random weights (they fail), ARS training in a background thread paced at 0.7 s per round (reaches an average of 9.7/10 in ~20 rounds, ~17 s), push test, quiz on scale (Go2 walk: 24.6 M steps). The live cart runs the policy being trained; attempts restart when the pole falls.
- `control/ars.py`: `train_ars(policy=..., stop=...)`; `Lesson.close()` stops background work on exit or swap.
- What did NOT work (and the fix):
  - The push step was flaky (4 of 6 runs): a push late in an 8 s attempt was cut off by the attempt timeout before its 3 s answer. Attempts now wait 4 s after a push; 6 of 6 pass. A real user would have hit the same "I pushed and nothing happened".
  - In the browser pane, an emulated viewport can render into a corner after the pane resizes; reset with preset desktop before trusting a screenshot.

## Round 11 (2026-10-15): real robots, face-offs, flicker (plan v2, phase A)

- User ask: compare real robots (G1, PiPER) with our templates; lessons other than the cart-pole need a big lift; tuning and RL training must show a visible difference; terrain still flickers.
- Library: AgileX PiPER (Menagerie `agilex_piper`, MIT, manifest regenerated from a sparse checkout at the pinned commit). Library arms declare `tool` (body, grasp distance) and `gripper`, so move_arm / pick_and_place / reach work on PiPER and Panda.
- `learn/compare.py`: FaceOff base (robots in a row, spec table from the compiled models, results table in the viewer's right column, name tags that follow the robots).
  - Arm face-off: ours / PiPER / Panda; workspace clouds; reach-out test with the gripper pointing down (0.2-0.7 m): ours 0/6 at 20 mm error (P-only servos sag), PiPER 3/6 (wrist limit), Panda 5/6; gravity compensation switch: ours 0.3 mm and 2/6, Panda 6/6; pick race from the home pose: all three place their cube.
  - Humanoid face-off: ours (ZMP preview, 0.45 m, 0.107 m/s) / G1 (RL, 1.33 m, 0.5 m/s): G1 4.7x faster in m/s but only 1.6x in body heights per second; energy per metre 0.8 vs 5.2; both last ~3 shove levels.
- G1 walking: UniLab g1_walk_flat, PPO, 2200 iterations, 108 M steps, 62 min on MPS; exported card checked against a rollout recorded in UniLab (all terms match up to UniLab's training noise); shipped in `walkers/`; runs on numpy ONNX.
- Flicker: `__robotforge.flicker()` counts pixels that flip there and back while the camera slides 5 mm per frame. Causes: one texture repeat per terrain box (squeezed 12x on risers) and per-texel white-noise grain plus non-wrapping blotches. Box UVs in meters (stairs 144,071 -> 327), periodic value noise (grass 12,087 -> 0, no tile seams), heightfields drawn 4 mm higher.
- What did NOT work (and the fix):
  - Single-start DLS said Panda and PiPER could not reach points their workspace covers; 24 seeded restarts, keeping the collision-free answer nearest to now. Self-collision rejection needs "already touching at home" pairs ignored (coarse meshes).
  - Joint-space moves into another IK branch are smooth but not obstacle-aware: from far poses they knocked the cube or tilted it out of the grip. The face-off tests now start every arm from home (fair benchmark protocol); real motion planning is a frontier item.
  - The tool snapped from horizontal to pointing down on the first tick (162 mm tracking spike on PiPER); the down direction now blends in along the move (35 mm).
  - The release fling took three hypotheses to find (bin wall contact, IK rejecting the rim, DLS divergence: all falsified; the third trace showed the planned point circling). Root cause: `_sweep` judged the arc's turn from tool angles clipped to the base joint limits. Regression test in test_builder.
  - The G1 veered at 0.5-0.8 m/s. Not the observation contract (matched) and not the 2 ms vs 6.7 ms physics step (same drift at both); the 1-hour policy itself drifts ~6 deg/s. Heading hold (Isaac Lab style) fixed it for every learned walker.
  - A flicker metric with 2 cm camera steps aliased the test itself; realistic per-frame steps (5-10 mm) and an in-page A/B against rebuilt old textures gave trustworthy numbers. Cloning a three.js texture shares its image source (the first A/B compared old with old).
  - My `_real_humanoid` first only checked "from the library", so Go2 reported its RL top speed; it now also checks the catalog kind.

## Round 12 (2026-10-15): the experiment bench (plan v2, phase B part 1)

- Engine: `Bench` (title, curve axes, params with short names, `Metric`s with lower_is_better / ranked, a reference line, `Sweep`s); `LessonContext.record_run` keeps the last 5 runs; actions `clear_runs` and `sweep` (slow handler `lesson_bench`, on copies of the rig); the record goes to the viewer only when it changed, and with the scene on (re)connect.
- Viewer: an "experiment record" section under the live plots: overlay chart (one colour per run, newest thicker, dashed target), table (run, settings, metrics, best underlined), sweep buttons (locked until the knob unlocks) and one small chart per swept metric (hollow marks: never got there).
- PID: `grade()` scores a curve for live tests and sweeps alike; Kp sweep 2-80 shows overshoot 0 -> 36% (without feedforward it only settles at high Kp), Kd sweep shows overshoot damped to 0 from Kd 1.5; each sweep 40-50 ms.
- RL: knobs α (step size), ν (noise), directions; fast-forward; every training starts from zero weights and is recorded; new "tune" step needs two differently tuned runs. Measured (30 rounds): α 0.1 learns at round 18-19, α 0.02 never, α 0.5 jumps around and learns at 27-28; ν 1.0 never; 2 directions noisy, 16 learn at 16 for twice the simulation.
- What did NOT work (and the fix):
  - "Fewest simulation steps" was marked best for the run that never learned: metrics can be unranked.
  - A test helper returned the last of two ticks, and the record only rides on the first one after a change; tests read single ticks.
  - In the browser pane a synthetic drag works when pointer events are dispatched on the canvas with moves spread over a second (the PID lesson's first step needs a drag).

## Round 13 (2026-10-15): twins on a tuning bench (plan v2, phase B part 2)

- `learn/twins.py`: TwinsLesson (kind "bench", picker group 调参台): `Twin`s (name, label, colour, starting values), per-twin `Knob`s become controls `<knob>_<twin>` in a group the viewer heads in the twin's colour, `shared` controls (the test, the target) come first; `TwinValues` gives each controller its own view (own knobs, shared, fixed); a `Watch` controller measures every physics step and samples plots at 100 Hz.
- PID triplets (A Kp 10 Kd 0.3, B Kp 40 Kd 0.5, C Kp 40 Kd 3, weights in their colours): step to 120 deg: A 24.8% and never settles, B 36.2% and 2.95 s, C 0% and 0.41 s; steps: compare, tune A to two stars, feedforward for all (A and C 0.36 s).
- LQR twins (cart-poles in their colours, Q in {1, 10, 50, 200, 1000}, R in {0.02, 0.2, 2, 20}): push 0.6 rad/s: R 20 uses 44% vs 100% effort for 3.40 vs 3.29 deg and 1.18 vs 1.04 s; Q 1000 vs 1: 1.92 vs 3.86 deg, cart 17.1 vs 11.9 cm; the table also shows the angle gain LQR worked out.
- Engine: `Control.group`, `Lesson.groups`; `pd.rig_spec` / `pd.add_rig` build named rigs; JointPD takes a prefix and a values view.
- What did NOT work (and the fix):
  - Peak effort read backwards (A 10%, B 21%) when measured at the 20 Hz lesson tick: LQR's answer to a push is a spike of a few ms. Measured every physics step now (100% vs 44%).
  - The results key "test" collided with the PID lesson's result banner ("秒到位，超调 %"); the twins use "triplet".
  - Lock order: the reset hook took the runner lock inside the session lock while ticks and input take them the other way round (a latent deadlock). Fixed in the hook.
  - The selection ring and "our template" subtitles made no sense on twins: hidden in lessons without driving; twin tags are one line.

## Round 14 (2026-10-15): MPC and gait twins (phase B complete)

- `DogTwins`: twin template dogs in coloured lanes, put back on their start line with fresh controllers for every test (`restart`), name tags on the trunks.
- MPC twins: stand MPC per dog (horizon {0.1, 0.2, 0.35, 0.6} s, futures {4, 16, 32, 64}, noise {0.02, 0.05, 0.1, 0.3}); escalating sideways shoves (0.2 m/s more per level, every 2.5 s); the table counts physics steps per plan. Two MPC dogs run 2.5-4.6x faster than real time on the Mac.
- Measured over six seeds (noise in levels survived): noise 0.05 -> 5-6, 0.3 -> 6-10; horizon 0.1 / 0.35 / 0.6 s -> 6 / 8.5 / 6.8 on average. Calm standing is no shakier with big noise (only the best try is executed). Each shoving gets fresh seeds.
- Gait twins: `quadruped.for_template(period=, lift=)`; walks at one shared speed on the flat (5 s) and over a 3 cm curb (12 s). Period 0.25 s: top 0.52 m/s, energy 13.1 per metre; 0.5 s: 0.26 m/s, 8.4; 0.7 s cannot reach 0.2 m/s. Lift x0.5 costs 4.6 vs 15.8 at x1.6 on the flat and is stuck at the curb (x1 and x1.6 cross).
- Finding worth a decision later: the stand task's default MPC noise (0.05) is the weakest setting for shoves; 0.3 survives more with no calm jitter. Changing the default would also change the shove challenge's balance.
- What did NOT work (and the fix):
  - One run per setting is not evidence: a single seed said noise 0.05 vs 0.3 was 5 vs 10 levels, the lesson's own run said 6 vs 7. Measured six seeds and the text quotes ranges and averages; the test asserts mechanics, not one chaotic run's order.
  - Re-running a shove gave the identical outcome (seed 0 every time), contradicting "shove a few times": seeds now change per run.
  - A camera across the shove direction hid the shoves; MPC twins look from the front. The gait camera now covers the curb and beyond.
  - Result tables lost track of the settings that produced them once a knob moved; each result section shows "ran with".

## Round 15 (2026-10-15): inverse kinematics II on six-joint arms (phase C part 1)

- CI: the RL lesson test failed on Windows only: its waits shared one 60 s deadline and the slow runner checked the run record mid-training. Each training now waits up to 10 minutes and the test fails loudly if the thread is still alive.
- `learn/ik_arm.py` (IK II): our template arm (dof 6, orange) and the PiPER side by side; `IKFollow` is resolved-rate IK (one DLS step per 20 ms tick, lambda from a choice, optional gripper-down axis) for the arm picked in "arm following the target"; live: hand, target, the SVD of the hand's position Jacobian; plots: error, manipulability (home = 100), largest joint step per tick.
- Measured (pulling the target outward): our arm reaches ~0.45 m; with lambda 0.05 it stretches calmly and w falls to 0; lambda 0.0001 jumps 55-106 deg per tick beyond reach (flailing), 0.2 lags 22 mm behind; the PiPER reaches 0.7 m.
- Steps: drag two targets, flatten the ellipsoid (w < 30), small vs large lambda at the edge, gripper down, the PiPER beyond our reach, a quiz on singularities.
- Viewer: rig kind `arm3d` (target ball dragged on a camera-facing plane, mapped back to tx/ty/tz of the active arm; dashed error line; ellipsoid from the SVD axes and radii; name tags). The planar target drag sends a dict of controls now.
- What did NOT work (and the fix):
  - With the default lambda 0.01 the arm already flailed at the edge, so it never stretched out and the ellipsoid never flattened; the default is 0.05 and the learner tries 0.0001 on purpose.
  - The "reached a new target" marker did not reset when the arm, lambda or gripper setting changed, so the gripper-down step could not count the same spot.

## Round 16 (2026-10-15): sensing II, robot vacuums on a lidar rule (phase C part 2)

- `learn/autopilot.py` ("lidar_autopilot", chapter 感知): two rovers in walled rooms, one behind the other, run the same reactive rule at 50 Hz (drive on unless the lidar sees something closer than the stop distance ahead, then turn on the spot toward the more open side); each twin has its own stop distance, speed, lidar noise and median filter (5 readings).
- A 30 s run paints the floor cells each rover covered in its colour (viewer rig `cleaner`, InstancedMesh tiles); the bumper counts bumps and the seconds spent pressed against something; the table shows "ran with" settings per run.
- `sensors.Rover` takes a name prefix and `sensors_spec(floor=)` builds the room, so two rovers share one world; twin knobs gained a `toggle` kind.
- Measured (30 s, noise 0): stop 0.15 m covers 6.9% and spends 22 s pressed against walls (the lidar sits 12 cm behind the nose); 0.3 m covers 25.7% with no bumps; 0.9 m covers 20.1% (corners and edges stay dirty); 0.45 m/s covers 31.2% with 9 bumps; with noise 0.3 m the filtered rover covers 22.2% vs 18.8% unfiltered.
- Steps: stop distance, too careful, a little faster, sensors lie (noise + filter), a quiz on why vacuums still carry bumpers (sensor fusion).
- What did NOT work (and the fix):
  - Odometry counted the jump back to the start line as driving: the restart clears the rover's last encoder and pose readings. A test now checks a noise-free rerun repeats exactly.
  - Rooms side by side put rover A under the lesson panel: the free area between the panels is portrait-shaped, so the rooms stack front to back under a near top-down camera.
  - The browser pane again rendered an emulated viewport into a corner after the server restarted; reset with preset desktop, then set the size again.

## Next-round seed

Round 17 = phase C part 3: the PID lesson rebuilt to the cart-pole standard around triplets and sweeps (what each knob does, seen on three joints at once, then the learner tunes one to stars, then the bench sweep), and plan the gait lesson (template dog, then Go2) and a humanoid balance lesson (our template, then G1). Odometry drift corrected by lidar stays optional.

## Frontier

- Current ceiling: any robot in 6 scenes and 8 terrains, driven by gait, MPC/MPPI, LQR, IK or learned policies (Go2, G1) from the viewer; 6 lessons, 3 challenges, 2 real-robot face-offs and 4 tuning benches with measured comparisons.
- Next frontiers (2026-10-15): the experiment bench (phase B); obstacle-aware arm motion (joint-space RRT-Connect with the collision check we now have, taught as a lesson); G1 trained with shoves and on rough ground (domain randomization made visible); custom-morphology RL.
- Next frontiers: behaviors (round 4), import doctor + robot library (round 5), real-agent evals once CLIs work, MCP App embed in Claude Desktop (needs user to add the server to Claude Desktop).
- Radical ideas weighed: physics in browser via MuJoCo WASM (kept for sharing only).
- Escalation ladder: open-loop gaits -> CPU RL for custom morphologies -> GPU backends (MuJoCo Warp on NVIDIA) -> sim2real.
