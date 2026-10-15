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

## Round 17 (2026-10-15): PID I rebuilt, PID II on the PiPER shoulder (phase C part 3)

- Engine: `Quiz(predict=True)` is a guess before an experiment: any answer is taken once, the step's check waits for it, and the verdict (with the right answer marked) only shows after the experiment passes. `LessonContext.step` names the current step. `Bench.zoom` offers a closer look at the run chart around the target.
- `pd.JointLesson`: one-joint PID lessons in general (a graded, recorded step test, sweeps on copies, a live formula with a row per term). `JointPD` takes any joint, a sign and offset (lifting is always a bigger angle), a torque limit and the controller's own gravity model; the integral has anti-windup. `grade()` takes start and target and reports `end_err` only once the arm has come to rest.
- Viewer: "控制器此刻在算什么" panel (P, D, I, feedforward, motor with a saturation note, gravity, zeta), predict quiz states (guess in amber, then right/wrong plus the answer), result banner says how far off a run came to rest, a "not counted" note, run chart zoom to +-6 degrees, the joint dial takes a body-frame arm direction and a range.
- PID I (6 steps, each P/D/steady/FF step starts with a guess and uses the step test): P only, Kp 10 and 40 give nearly the same swing (9 to 170 degrees) because Kp * e is far past the 4 N*m limit; D sweep at Kp 40 shows overshoot falling to 0 by Kd 3 and settling fastest there (0.41 s, 0.95 s at Kd 6); Kp 80 / Kd 5 halves the steady error (2.7 -> 1.33 degrees); feedforward ends at 0 in 0.4 s while Ki 10 is still 1.1 degrees off after 3 s (Ki 40 catches up, Ki 60 starts to overshoot).
- PID II "pid_piper" (Menagerie PiPER, shoulder joint2 as a torque motor, gravcomp off, other servos 5x stiffer to hold the pose, a box at the tool point whose 0.5 kg the controller's model does not know): Kp 40 / Kd 3 sag 8.7 degrees (gravity 6.05 N*m vs 1.8); with feedforward they overshoot 25% (zeta 0.39: I = 0.37 vs 0.076 on the small arm); Kp 80 / Kd 10 (zeta 0.92) settles in 0.26 s with no overshoot; box full with feedforward ends 2 degrees short, Ki 20 fixes it; challenge: three stars without the model (needs Ki about 60; Ki 20 earns one).
- Menagerie's own PiPER servo (Kp 80, Kd 5) has zeta about 0.45 and overshoots 15% in the same test with gravcomp on.
- What did NOT work (and the fix):
  - In the old step test Kp made no visible difference: a 120 degree step saturates the 4 N*m motor for any Kp above 10. Kept and taught (the formula shows the clipped sum) instead of raising the limit; D and steady error carry the Kp story.
  - The PiPER showed no sag at all: every Menagerie PiPER link has `gravcomp="1"`. Switched off for the lesson and named in the text.
  - With gravcomp off the held elbow and wrist sagged 4 and 8 degrees under their file servos; stiffened 5x at the same damping ratio.
  - Without anti-windup Ki 5 looked as good as feedforward by luck (the integral charged during the climb); conditional integration makes the integral honest.
  - Swinging runs reported a meaningless "ended off by 42 degrees" and won "best"; end_err is now None unless the arm came to rest.
  - The PiPER challenge passed instantly with the previous step's settings; it now has to be done without the model, and a run with feedforward says it does not count.

## Round 18 (2026-10-15): quadruped gaits lesson (phase C part 4)

- `learn/gait.py` ("gait", chapter 走路): our template dog and the Menagerie Go2 in two lanes on the same gait code; every "走 5 秒" puts both back on the start line with fresh gaits and walks the chosen one; the record keeps distance over time, roll and pitch (RMS after the first second); a summary line reports each walk.
- `behaviors/quadruped.py`: `PATTERNS` (trot, pace, bound as phase offsets), `QuadrupedGait.set_pattern`, `set_period` (top speed follows; a library robot's known top speed still caps it), `leg_length`.
- Engine and viewer: `Plot(kind="footfall")` (one row per foot, filled while it touches the floor, from real contacts); rig kind `gait` (name tags, the support strip between two feet or the patch under three or four, foot discs, the centre of mass dropped to the floor, drawn over the robots); `live.follow` makes the camera walk along with a body (retried until the scene's bodies exist; orbiting away stays the learner's); summaries without stars show no empty stars; record tables show choice labels.
- Steps (each starts with a guess): trot (diagonal feet together), pace (rocks sideways), bound (nods, hardly moves), step period at 0.6 m/s, the Go2 with the same settings, a challenge (Go2: 2.7 m in 5 s, roll and pitch under 2 degrees), a quiz on why robots trot.
- Measured at 0.2 m/s, T 0.36: our dog trot 1.03 m, roll 0.63, pitch 0.43; pace 0.86 m, roll 4.2; bound -0.14 m, pitch 3.9 (Go2 bound pitch 13.3, 0.26 m). At 0.6 m/s: T 0.25 2.6 m, T 0.36 1.76 m, T 0.5 1.31 m (our dog); Go2 T 0.36 2.66 m, T 0.25 2.83 m.
- What did NOT work (and the fix):
  - A statically stable crawl (one leg at a time, duty 0.75) does not work on our gait: the centre of mass is inside the support triangle only 12 to 38% of the time, it stalls at 0.1 m/s, and a quick body-sway attempt (feet shifted toward the centroid of the feet that stay down) made it fall. Left out of the lesson; a real crawl with planned body shifts is a frontier item.
  - The Go2's free joint has no name, so finding the trunk by joint name failed; it goes by the body under the free joint.
  - A fixed camera over the whole 3 m walk made the dogs and the support overlay tiny, and a closer one lost the end of the walk behind the panels: the camera now follows the dog being tested.
  - The support line drawn as a 1 px line under the dog was invisible: it is a 3 cm strip with foot discs, drawn without depth test after the robots.
  - The stride formula used the asked-for speed even when the period capped it; it uses the speed the gait really steps for and notes the cap.

## Round 19 (2026-10-15): humanoid balance lesson, and a viewer deadlock (phase C part 5)

- `learn/balance.py` ("balance", chapter 走路): our template humanoid and the G1 stand side by side along the push direction, seen side on; each push puts both back on their marks with fresh controllers, lets them settle (ours crouches first if asked) and kicks the chosen one at a set speed, forward or to the left; the floor shows the support patch, the centre of mass and the capture point x + v sqrt(z / g); a live formula compares it with the foot edge; plots show that margin and the body tilt; the record keeps tilt after the push.
- Steps (each starts with a guess): stand (0.1 m/s), push until it falls, sideways, crouch, the G1 steps, a quiz on stepping.
- Measured (ours, 0.45 m, centre of mass 22.8 cm): forward 0.3 m/s stands (capture point 4.6 cm, toes 5.1 cm), 0.4 falls (6.1 cm); sideways 0.5 stands (7.6 against 7.3: the ankles save a little), 0.6 falls; crouched 0.3 (centre of mass 17.6 cm) 0.4 stands (5.4 against 5.2), 0.5 falls. G1 (centre of mass 66.6 cm): forward 0.6, 0.8 and 1.0 m/s all stand with capture points 16 to 27 cm against a 10 cm foot edge, because it steps (feet 40 cm forward at 0.6, 117 cm at 1.0); sideways 0.6 fell in the lesson run (stood in one probe).
- `HumanoidController.crouch(drop)` sinks the hips and holds them (custom poses next to POSES).
- Viewer: the gait rig draws the capture point (ring joined to the centre of mass) when a lesson sends one; `live.view` frames each robot its own way; switching the followed robot glides the camera over.
- What did NOT work (and the fix):
  - Ankle feedback gains, the obvious knob, make no visible difference to pushes (off: 4 forward levels, default 3, four times 3): the lesson is built on the capture point instead, which predicted every outcome to the centimetre.
  - Crouching via `_pose_level` did nothing: the controller's opening stand pose overwrote it. A real `crouch()` pose instead.
  - The viewer hung opening the lesson: a lock-order inversion between the lesson start (lesson lock, then session lock) and the hub's scene build (session lock, then the lesson's description). The G1's long mesh build widened the window. Extras are now built after the session lock is released; a regression test checks another thread can take the session lock meanwhile. Found by reproducing through the websocket with `faulthandler.dump_traceback_later`.
  - Robots side by side across the view hid the small one in front of the G1; they now stand along the push direction, and each robot has its own framing (the G1's head was cut off with the small robot's).
  - The G1's policy treads in place even at zero command (it drifted 19 cm sideways in 3 s), so its support patch changes from moment to moment.

## Round 20 (2026-10-15): Review round (phase C milestone)

- Every lesson, challenge, face-off and bench started headless with its buttons pressed (18 of 18 run), and each opened in the browser and looked at.
- Fixed: the picker now reads as one course (PID I and II, IK I and II both under 运动学, Sensing I and II) and no longer says "next is gaits"; the humanoid face-off showed the G1 standing on top of our small humanoid (lanes 1.8 m apart, camera higher); the gait twins started half behind the panel (camera moved, then follows them); a new scene silently dropped a lesson's follow camera (the lesson asks again).
- Pruned definitions nothing used: three mesh helpers, inverse action scaling, `TorqueMap.max_torque`, `reward_rate`, `_pose_distance`, `joint_of`, `look_direction`, `SimSession.load_xml`, five stale constants. Dynamic dispatch (`script_runner._op_*`) and SB3 callbacks kept.
- README describes the in-viewer courses (it only listed the written chat lessons).
- What did NOT work: nothing new broke; one screenshot showed the picker over a started lesson, but repeating the steps by hand showed the normal flow is fine (a timing artefact of the scripted batch).

### Ambition critic (what an expert would find missing)

- The promise is "any robot, make it do anything", yet nothing in the viewer teaches a robot the user built to move by learning; RL stops at the cart-pole.
- Lessons cannot be checked as a beginner would play them except by hand: there is no playtest agent.
- The agent (Claude or GPT) cannot see lesson progress to tutor (needs new MCP tools: a gate).
- Floor A5 (MCP App embed in Claude Desktop) and A6 (Claude Code session, blocked by the expired CLI login) are still not verified.
- No obstacle-aware arm motion, no crawl gait, no terrain policy.

## Round 21 (2026-10-15): any body learns to walk (phase D part 1, Frontier round)

- `control/cpg.py`: `Body.find` (a robot's free trunk and its position-servo hinge joints, any morphology), `Rhythm` (target = home + offset + amplitude sin(2 pi f t + phase), 3 numbers per joint), `Trainer` (rollouts on a private model of the robot alone, reward = distance - 0.5 drift - w energy / (m g) - 1 per fall, ARS with 8 directions both ways, top 4), `RhythmController` for the live world.
- `learn/walk_rl.py` ("learn_to_walk", RL II): template dog, a three-legged starfish built from parts (no gait code knows it), the template humanoid, and "your robot" when the world the learner came from has one that moves freely with servo joints (file imports skipped). Training runs on a thread; the viewer replays the newest rhythm every 4 s from the start line with the camera following; each training is a learning curve in the record; per-training seeds.
- Measured (30 rounds, 4 seeds): dog 1.1 to 1.8 m in 4 s (about 11 s of training; 60 rounds pass 2 m), energy 76 to 148 J; energy cost 0.1: 1.0 to 1.4 m on about 50 J; 0.3: 3 of 4 stand still; starfish 0.5 to 0.85 m; humanoid falls every time (open loop).
- Why a rhythm and not ARS on a linear feedback policy: the ARS paper needed tens of thousands of episodes for MuJoCo walkers; a rhythm with 3 numbers per joint learns in tens of rounds on a laptop and works for any number of legs.
- Engine guard: a test checks every lesson's controls unlock in some step (this lesson first shipped with Stop, Rounds and Frequency locked forever).
- What did NOT work (and the fix): the camera framed the dog at the panel edge and the other robots out of view (side on to the active lane now, following); one seed for every training made each setting give the same number every time (per-training seeds, ranges in the text); name tags floated at one height for all bodies.

## Round 22 (2026-10-15): reflexes for the learned rhythm (phase D part 2)

- `control/cpg.py`: `Rhythm(reflexes=True)` adds 4 gains per joint on the trunk's roll, pitch and their rates (`trunk_senses`, the free joint's body-frame angular velocity); `Reward(survive=True)` makes a fall also cost the seconds not stayed up; `train(pushes=...)` shoves every try once at random (the same shove for both sides of a direction); `push_test` counts survivals under 12 random shoves; the live `RhythmController` reads the trunk too.
- RL II: step 4 now tells the truth about the humanoid (with a distance-only reward it learns to lunge half a metre and fall, 5 of 6 trainings); new step 5 "给节律加上反射" with two switches (reflexes; shove it while training, points for staying up), a push test after every training (new record column), and the replay shoved at 2 s when shoving is on.
- Measured (humanoid, 80 rounds, 6 seeds, test = 12 random 0.4 m/s shoves): distance only: 5 of 6 lunge and fall, 0 of 12 survived; survive + shoves, no reflexes: stands nearly still (-0.05 to 0.16 m), 2 to 10 of 12; survive + shoves + reflexes: 4 of 6 walk 0.26 to 0.55 m without falling and survive 6 to 12, 2 of 6 collapse (0 of 12); standing untrained: 8 of 12. In the browser: no reflexes 0.03 m / 7 of 12, reflexes 0.36 m (fell calm) / 8 of 12, a headless run 0.55 m / 12 of 12.
- What did NOT work (and the fix):
  - Round 21's claim "the humanoid falls whatever it learns" was a reward artefact: with a survival reward even the open-loop rhythm stops falling (it learns to stand still). The text now says what was really measured.
  - Reflexes are not a clean win: 140 numbers instead of 60 make random search unreliable (2 of 6 trainings collapse). Kept, and taught as the reason deep RL uses gradients.
  - A reflex test with large gains drifted as much calm as shoved; tests check the reflex arithmetic directly and the shove on a still rhythm.
  - CI failed on d320f40 (Ubuntu and Windows): the RL II test judged the replay by the last tick's 50 ms of samples, and slower runners ended training at another phase of the replay (0 m right after a restart). It now collects a whole attempt; locally the old check read 0.07 to 1.07 m by phase, the new one 0.94 to 1.14.

## Round 23 (2026-10-15): obstacle-aware arm motion (phase D part 3)

- `behaviors/motion_plan.py`: `ArmSpace` (joint space and collision check in a private model copy; fixed geoms kept `CLEARANCE` 4.5 cm away by `mj_geomDistance`, the floor `FLOOR_CLEARANCE` 5 mm, other overlaps by contact, the robot's own home overlaps ignored but never an obstacle's), `rrt_connect` (Kuffner and LaValle), `shortcut`, `resample`, `joint_length`.
- `learn/arm_plan.py` ("arm_planning", chapter 运动学): our six-joint arm and the PiPER, each with a fence and a target on each side; Straight (every joint at once) hits; Plan and go (RRT-Connect, optional shortcuts, step size 0.1 to 0.8 rad) goes round; the PiPER step; a quiz on the curse of dimensionality. Arms follow paths with motor gravity compensation (`body_gravcomp` + `jnt_actgravcomp`, switchable at run time because the PiPER compiled with it) and the path waits whenever a joint lags more than 0.1 rad; a straight run ends a second after it hits.
- Viewer: rig kind `plan` (straight sweep as dots and a dashed line, the clash point drawn through the fence, both trees growing in edge by edge with node dots, the path as an amber tube, the goal); drawings come once per change and are latched in the viewer; the lesson resends on (re)describe.
- Measured (8 seeds): ours, straight hits at 27% of the sweep; step 0.1 rad: 69 to 145 nodes, 13 to 29 ms; 0.2: 25 to 73, 8 to 23 ms; 0.8: 8 to 13 nodes; shortcuts save 18 to 48% (0.2 rad). PiPER: straight hits at 30%; 0.2 rad: 49 to 93 nodes, 15 to 28 ms. Followed in physics: 24 of 24 plans never touched a fence (ours 1.2 to 2.5 s per move, PiPER 1.9 to 6.1 s).
- Fixed on the way: `ArmMotion` worked out home-pose overlaps before it knew which bodies move (AttributeError on any overlap, then discarded); failing-first test with an arm whose shapes overlap at home.
- What did NOT work (and the fix):
  - The first planner treated an obstacle overlapping the home pose as part of the robot (it ignored the fence's pairs); only the robot's own overlaps are ignored now.
  - Plans that cleared the fence scraped it when followed: the template's P-only servos lagged up to 47 degrees on a fast base turn and sagged about 5 under gravity. Fix in three parts, each measured: 4.5 cm clearance, gravity compensation through the motors, the path waiting for a lagging arm (0.05 rad stalled the PiPER's normal velocity lag, 0.1 works).
  - Clearance by contact margins set at run time missed some pairs in the lesson world (a PiPER finger box 9 mm from its fence made no contact with a 4.5 cm or even 10 cm margin; a two-box model at run time did); measured distances instead. Cause not pinned down.
  - Drawings sent with one update were missed when two updates arrived between frames (hidden pane, throttled frames): latched in the viewer.
  - One-pixel lines for the trees and the sweep vanished at lesson distance: dots and a tube.

## Round 24 (2026-10-15): Review round (phase D milestone)

- Real agent check: `codex exec` works again (the old "workspace routing discovery" failure is gone); with the plugin installed from GitHub main into a scratch `CODEX_HOME` (docs/PLUGINS.md), Codex built a grass world, added the template dog, opened the viewer and walked it 10 s: 3.05 m, upright (A1 re-verified on GPT). It used 430k input tokens (385k cached) for that small task. `claude -p` still fails: OAuth session expired (needs `claude auth login`).
- Picker: RL I renamed to match RL II; the footnote no longer promises learned walking (shipped in round 21) and says what the record does; README lists motion planning and RL II. No dead code found by the scan.

### Ambition critic (round 24)

- The core promise is "any robot, make it do anything" from chat, but what the viewer now teaches does not reach the agent: a rhythm learned for "your robot" in RL II cannot be kept, and `move` cannot use it. Closing that loop (save the learned walk to the robot, `move` and WASD use it when no gait exists) needs no new tool.
- Agent cost: 430k input tokens for A1 suggests the tool list and instructions are heavy; worth measuring what the agent is sent.
- Learned reflexes are unreliable (2 of 6 collapse) and only random search is offered for walking; no gradient-based option in-viewer for user robots.
- Tutor tools (the agent reading lesson progress) and a lesson MCP surface are gated; a playtest agent is now feasible with `codex exec`.

## Next-round seed

Round 25 = phase D part 4: keep what a robot learned. In RL II, a "use this walk for my robot" button saves the learned rhythm (and reflexes) into the learner's world folder; `move` and WASD then drive that robot with it (forward by the rhythm, stop by holding home), so a robot built in chat that no gait code knows can walk on command. Verify with a real Codex session: build a three-legged robot by chat, learn its walk in the viewer, then ask Codex to move it.

## Frontier

- Current ceiling: any robot in 6 scenes and 8 terrains, driven by gait, MPC/MPPI, LQR, IK or learned policies (Go2, G1) from the viewer; 11 lessons (PID on the PiPER, gaits on the Go2, balance on the G1, any body learning to walk, motion planning on the PiPER), 3 challenges, 2 real-robot face-offs and 4 tuning benches with measured comparisons.
- Next frontiers (2026-10-15, re-ranked after the round 24 critic: keep learned walks for the user's robot first): learned walking for robots the user builds (round 21; reflexes and shoves round 22); a playtest agent that plays every lesson; tutor MCP tools (gate); a statically stable crawl with planned body shifts (taught next to the trot); obstacle-aware arm motion (joint-space RRT-Connect with the collision check we now have, taught as a lesson); G1 trained with shoves and on rough ground (domain randomization made visible); custom-morphology RL.
- Next frontiers: behaviors (round 4), import doctor + robot library (round 5), real-agent evals once CLIs work, MCP App embed in Claude Desktop (needs user to add the server to Claude Desktop).
- Radical ideas weighed: physics in browser via MuJoCo WASM (kept for sharing only).
- Escalation ladder: open-loop gaits -> CPU RL for custom morphologies -> GPU backends (MuJoCo Warp on NVIDIA) -> sim2real.
