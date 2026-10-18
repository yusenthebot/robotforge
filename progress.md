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
  - In round 9 nothing finished the obstacle course: template dog gait falls at the 4 cm curbs (~11-12 m), Go2 gait on the stairs (5.9 m), the Go2 learned walk stalls on the ramp (trained on flat ground). Scored by distance instead; a terrain-trained policy is a frontier item.
    - Phase E: with W at 0.3 m/s a key-driven run can now finish, with luck: 7 of 20 runs over a sweep of A/D tap timing finished, 7 fell on the stairs and 6 on the curbs and stones.
      The closing text says it can be finished, just not every time; finishing every run is still the frontier item (a terrain-trained policy).
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

## Round 25 (2026-10-15): keep what a robot learned (phase D part 4)

- Feature: RL II has "Keep this walk for my robot" (把这个走法存给我的机器人). It writes the learned rhythm to `walks/<robot>.json` in the learner's world (`behaviors/rhythm_walk.py`: `SavedWalk`, validated with pydantic, joint names without the robot prefix; `RhythmWalker`).
- `move`, the keyboard and the Modes tab drive a gait-less robot with its kept walk, forward only at its own pace; robots with a gait keep it and get the walk as walker or mode `rhythm`. `robots` lists `learned_walk` (or a `walk_hint` that says how to learn one); world loads say it in their notes. The learner's own robot is selectable from RL II's first step, and a gait-less robot is the one offered.
- Verified in the browser: Codex's tripod trained in RL II (30 rounds in 7.9 s, 1.65 m in 4 s, 12 of 12 shoves), kept, driven with W (the user drove it into a kitchen scene).
- Verified with real Codex sessions (plugin from GitHub main, scratch `CODEX_HOME`):
  - Run 1, before the round's fixes: Codex never called `move`, wrote its own gait, got a bare "Error executing tool" from `observe` and `run_script`, then stepped the simulation by hand: about 180 calls, 646k input tokens, 4.4 cm in 3 s.
  - Run 2, crash messages fixed: no errors, 23 calls, but a hand-written gait again (9.2 cm forward, 9.3 cm sideways) and the `walk_hint` ignored.
  - Run 3, kept walk: `open_world`, `robots`, `move`: 1.657 m in 4.02 s, upright, 18.4 degrees off heading; 6 tool calls, 321k input tokens (291k cached).
  - Run 4, fresh tripod after the guidance change: Codex told the user it can learn a walk in RL II, then drove the viewer's own WebSocket from a Node script to train (1.12 m in 4 s), keep and walk it: 0.831 m in 3 s. It worked, but cost 1.29M input tokens spent reading our source.
- Viewer bugs found in the browser and fixed:
  - Black screen: a page that loads with no size (hidden pane, collapsed panel, minimized window) made the camera aspect 0/0 = NaN and the camera position NaN forever. Zero sizes are ignored, framing waits for a size, a NaN camera reframes itself.
  - The lesson panel slid sideways (the four-choice row could not wrap); drawers never scroll sideways now.
  - Reloading mid-lesson: the whole-scene framing replaced the lesson's camera; the scene message cached the extras (robots, lesson, record) per model version, so a reloaded page got stale ones; the follow camera kept any offset it started with, and a glide lost the replay's jump back to the start line. All fixed, measured frame by frame (the tripod stays at x = 634 px through whole replays).
  - Plot ticks read -0.0 (one decimal for any spacing) or showed a single line; name tags overlapped and floated at guessed heights (now from the model, decluttered); RL II aimed at the dog's lane for every robot; `/favicon.ico` 404 in every console.
- Security: the viewer accepted WebSockets from any web page (cross-site WebSocket hijacking): a site visited while it ran could load scenes into the user's world. Browser sockets must now come from a page this server served (Origin = Host), every request must name this machine or an IP (no DNS rebinding); scripts, agents and `--host` on a LAN still work.
- What did NOT work:
  - The bare Codex errors from `observe` and `run_script` were never reproduced: the same 13 calls replayed over stdio, with Codex's minimal environment, and with a browser attached all worked. Crashes now reach the agent with their type and message, so the next occurrence will say what it was.
  - My first lesson-camera fix silently did not land: a `grep ... &&` chain skipped the edit. Check the file after scripted edits.
  - Measuring the follow camera with timers while the pane was hidden showed a broken follow; frames do not render there. Measure on `requestAnimationFrame`.

## Round 26 (2026-10-15): lessons in a small window (phase D part 5)

- The user's split pane is 672 x 553. Under 700 px the lesson panel and plots covered the 3D view: no robot was visible in RL II, and IK II left a 67 px strip.
- Layout by window size (style.css, measured in the browser at 375 x 812, 672 x 553 and 1280 x 800):
  - Over 1100 px: unchanged (panel left, plots right).
  - 601 to 1100 px: the panel keeps its left column, the plots sit compact at the top of the rest.
  - Phones: the panel is a sheet along the bottom, the plots a strip on top.
  - The panel folds to its title anywhere (chevron beside the exit button; on a phone it points down).
- Plots in a small window (under 1100 px wide or 640 px tall) show one thing at a time behind tabs: live plots first, then the face-off table and the record, tall ones scrolling inside. At 672 x 553 all 20 lessons, benches and challenges leave at least 366 x 208 px of view (most 366 x 266).
- The view centres on the part of the window the panels leave free, in both directions (`freeRect`, exposed on `window.__robotforge`); lesson cameras step back where that part is short (up to 1.6 x), so the robot fits on a phone.
- Bug found on the way: the old sideways view shift never ran, because it tested `offsetParent`, which is null for every fixed panel.
- Bug found in the sweep: the idle G1 drifted about 6.5 cm/s (4.15 m in 30 s headless) and walked into our humanoid's lane in the face-off. An idle policy walker now holds the spot where it stopped: it walks back past 0.2 m (faster than the policy's 0.2 m/s stand threshold) until within 5 cm. At most 0.2 m in 30 s; the G1 walk, push and balance tests still pass.
- What did NOT work: capping the plots box height alone (IK II's three plots and the arm face-off's table still left 67 to 140 px); defaulting the tabs to the face-off table (tall, it hid the G1's head).

## Round 27 (2026-10-15): Review round (phase D milestone)

- CI: every run since 60c46c5 had been cancelled by the next push. Ubuntu passed on 0156aa7 with all of rounds 25 and 26 (the security change included); pushes now wait for a run to finish.
- Real agent check: a fresh Codex session (plugin from main) did a beginner's four-part task in 18 tool calls. It built a world with a Go2 from the library, a template arm, a cube and a bin; the Go2 walked 42 cm in 2 s upright; pick-and-place succeeded; it checked the result in a top view. Its oversized screenshot got a clear error and it retried. Cost: 1.12M input tokens (1.07M cached), about 45k per model turn; ours of that is about 11k of tool definitions, 1.2k of instructions and up to 7.5k of skill files.
- Found by using the viewer as a learner would, and fixed:
  - Opening the G1 face-off from a world could freeze the viewer: the lesson imports the policy module on a worker thread, the stream thread met it half imported, raised, and the stream task died for good (old world on screen, no poses). A half-imported module now means no policy walker yet; a failing tick is logged and the stream carries on; a scene is announced only once built.
  - Dragging a robot held the whole robot's weight, floating it off its feet: the G1 fell at a 10 cm/s tug. A standing robot is now pushed, not carried (objects and arm links still are), and a dragged learned walker holds the spot where it is let go.
- Dead code: `lessonKind` (exported, never used). The other 29 single-use names are validators, `_op_*` handlers dispatched by name and callbacks.
- Docs: the plan's phase D and its decisions now match what was done.
- What did NOT work:
  - Four scripted races of a lesson start against the stream loop did not reproduce the freeze; the traceback only showed when the old server shut down. The stream now logs failures as they happen.
  - Measuring drags on robots with hand-written gaits: the template humanoid tips at a 15 cm drag and the Go2 on its gait flips at a 40 cm one; a softer drag spring stops the flip but then barely moves it. Left open.

### Ambition critic (round 27)

- Agent cost is the weakest part of the chat experience: 45k input tokens per turn. Tool results are verbose (observe lists every joint of every robot), screenshots frame a fixed patch of floor so the robots are small, and 50 tool definitions are sent every turn.
- Robots with hand-written controllers do not take a push like a real robot (the humanoid tips, the Go2 gait flips); the learned policies do. A push-recovery step for the template humanoid and dog would match what the balance lesson teaches.
- Learned walks cannot turn, and agents cannot learn one without driving the viewer's internals (`learn_walk` is waiting on the user).
- No automated playtest: every bug this round and last was found by hand in the browser.

## Round 28 (2026-10-15): a cheaper, sharper agent (frontier)

- Measured first: replaying round 27's 18 tool calls returned only about 26k characters of text and 4 images in total. Most of the 1.12M input tokens was context re-sent on each of about 25 model turns (95% cached); the largest part we send is the tool list, read every turn.
- `observe` views now step back just far enough for what they show to fill a set share of the image (a projected fit of the bounding box, per view direction and aspect), instead of a fixed multiple of the scene's diagonal. The round 27 top view showed the Go2, arm and cube about half as large.
- Tools return text only (an output schema made the SDK send each result twice, as text and as structured content), descriptions lose the docstring indentation, and argument schemas lose a title per parameter that repeated its name: 45.1k to 32.5k characters. Arguments, their checking and result text are unchanged.
- Same Codex task on the leaner plugin (installed from the local checkout): success, 24 tool calls (18 before) for 918k input tokens (1.12M before), about 38k per call against 62k. The extra calls were its own detours: a bin placed out of the arm's downward reach (the error said so; it moved the bin) and a second walk because editing the bin reloaded the world.
- `claude -p` is still logged out (needs the user).
- What did NOT work: comparing single runs is noisy (the agent takes different paths); per-call tokens are the steadier measure.

## Round 29 (2026-10-15): learned walks that turn (phase D)

- Keeping a walk in RL II now goes on, in the background, to learn a turn each way (`learn_turns`): the same random search with a reward for the angle turned less 3 x how far the body strays at its farthest. Two starts per side, the better one kept. Saved as optional fields of walks/<robot>.json (walks kept before still load). `move(turn=...)` and A/D turn the robot on the spot; `robots` reports `turns` and `turn_deg_s`.
- Verified in the browser with the keyboard (Codex's tripod): A for 2 s turned it 88 degrees left while it moved 11 cm, D turned it back 110 degrees, W walked 0.95 m.
- Verified with a real Codex session: it read the turn rate from `robots`, asked for a 4 s left turn (182 degrees, 19 cm drift), then walked 1.16 m in 3 s, upright; 6 tool calls, 505k input tokens (463k cached).
- That session also found a bug, fixed: `move` reported `turned_deg` as the difference of the two headings, so a half turn left read as -177.6 (a right turn). The heading is now followed while the robot moves.
- What did NOT work:
  - Net displacement after the turn as the wander measure: a tripod circled round a point and came back (0.01 m net, 0.5 m away mid-turn in the browser). Measured at its farthest now.
  - A wander weight of 6: the tripod nearly stopped turning (7 deg/s); 3 keeps the spots tight (6 to 9 cm) at 45 to 87 deg/s.
  - One start per side: one run learned 16 deg/s, the next 87.

## Round 30 (2026-10-15): edits keep the world where it is (phase D)

- Every edit compiled the world again and loaded a fresh simulation, so robots jumped back to their start (a Codex session re-walked its Go2 after moving a bin). `sim/carry.py` reads the live state by name before an edit (free body poses, joint angles, motor targets) and gives it back afterwards to every robot and object the edit left unchanged, standing still. New terrain or a scene starts fresh; anything that would land deep inside something new starts fresh too. Leaving a lesson brings the world back as it was.
- Verified in the browser: the tripod walked 1.2 m with W, then stayed put through a floor change and a lesson round trip.
- A real Codex session then broke it: a crate placed where the Go2 had started sent the Go2 back to its start, inside the crate. Mid-stride the Go2's feet sink 26 mm into the ground (the template dog's 6 mm), which read as buried; ground contacts no longer count. A regression test reproduces the session.
- The MCP instructions now say that edits keep what they do not change in place, so agents do not redo moves.
- What did NOT work: a 1 cm overlap threshold against everything, the ground included.

## Round 31 (2026-10-15/16): phase E, first overnight multi-agent round

- Four Workflow agents, each in its own worktree and branch, then an adversarial reviewer per branch; I reviewed the diffs, cherry-picked onto main and ran everything serially.
- E1 playtester: `learn/playthrough.py` plays every lesson to the end the way a learner would (quiz answers, knob settings, button presses, WASD at the viewer's share), `scripts/playtest.py` prints the table.
  It found 5 lesson problems, all fixed: RL I's push step could not pass after the tuning step left unlearned weights (Push now brings back the best ones and says so), the starfish step did not say to reset the energy cost, the PID twins feedforward step, the IK step count, and the obstacle course's W speed and closing text (keys alone finish it 7 runs in 20).
  18 of 20 lessons finished before, 20 of 20 after.
  Slow playthroughs are marked `slow`; CI runs them on Ubuntu.
- E2 push recovery: a quadruped on its gait that is shoved or dragged steps along with it (swing feet aimed as if the trunk were level, sideways strides follow the trunk).
  8 flips in 24 bench cases before, 0 after (`scripts/push_bench.py`).
  Review found a high-severity regression: a dog blocked by a stair riser wobbled like a pushed one and recovered without end, so the obstacle course fell from 12 m to 4.6 m.
  Walking recovery now stops after 3 s and rests 3 s (a dog pinned on a wall: 7.1 s of recovery before, 2.9 s now), and the course passes again.
- E3 imported robots learn to walk: RL II offers a robot imported from a URDF as "Your robot"; joints a URDF leaves unlimited can learn, wheels never.
- E4 colors: hex codes like "#9c6b3f" work everywhere a color does.
- Real E2E with Codex installing the plugin from the public GitHub repo: session 1 created a world and imported the Unitree A1 URDF (the doctor fixed 5 problems), walked it 0.39 m on the gait and told the user how to make it learn its own walk.
  In the browser pane I trained it in RL II (30 rounds, 13.8 s: 1.46 m in 4 s, 12 of 12 shoves survived), kept the walk and its turns (88 deg/s).
  Session 2 opened the world, saw `learned_walk`, walked 1.54 m with `walker="rhythm"` and turned left, upright; 189k input tokens.
- Found and fixed on the way:
  - Ubuntu CI: the IK arm lesson stuck at the gripper-down step.
    Flailing with lambda 1e-4 can throw the arm onto its yaw, shoulder and elbow limits, where every clipped step points out of range.
    A 1e-9 nudge of the aimed joints jammed 7 runs in 20 on macOS too; a jammed arm now starts over from home (0 of 40).
  - A data reset without `SimSession.reset()` left every controller idle until the old clock time came round; controllers now run when the clock goes back.
  - The first-visit hint floated over the open Learn panel; the Learn panel ran under the raised drive chip on panes narrower than 820 px.
  - Agent instructions said only robots without a gait can learn a walk; Codex read the plugin's source to find out (826k input tokens).
    Any legged robot can.
  - RL I's note about swapping in the best weights stayed up after new weights; a hex color in a YAML comment line stopped a world from opening; a hexapod refused `walker="gait"`.
- What did NOT work:
  - The round 31 workflow stalled after the E2 review (no fix agent started); I did the fix myself from the reviewer's tested recipe.
  - Worktrees for workflow agents start from an older commit than local main; agents now check out the exact base first.
  - `codex exec` without `< /dev/null` waits on stdin forever (lost 10 minutes again).
  - The A1's learned forward walk arcs (42 degrees in 4 s) and a 1 s turn right after walking gets only 16 degrees of its 88 deg/s; round 32 works on both.

## Round 32 (2026-10-16): phase E, second multi-agent round, plus E5 docs

- Three Workflow agents with up to three adversarial review rounds each; two branches merged, one held back.
- Agent error audit (36 commits): `scripts/agent_errors.py` calls every tool with plausible agent mistakes and grades the answers; forgiving forms where one reading exists (names in other capitals, "the dog", numbers as text, null as not given), and every refusal names the argument, the value and what would work. Tool descriptions and instructions were trimmed.
  Review follow-ups I fixed: worlds saved under the old folder names ("Café" in caf) open by their name again; a pick-and-place point within reach is no longer told to "move it nearer".
- Quadrupeds stand on their gait from load (14 commits): a never-moved dog, Go2, built dog or hexapod takes a viewer drag or a shove: fresh-robot bench 12 falls in 32 before, 0 after. Legs set by hand keep their angle through edits and lessons; robots on ramps stand as before.
- Push recovery follow-up from round 31's E2 fix agent (it left patches when its worktree was removed under it): a hand that keeps dragging is still a push, so the 3 s cut-off only applies to a blocked robot (long drags: 2 of 8 flips before, 0 after); the terrain bench tells crossing from going around.
- Straight walks and quick turns: held back. The heading term and MJCF imports work, but switching out of the walk tipped the template dog over on a right turn; round 33 continues the branch. Its independent fix (the autopilot arenas take the sensors arena's physics options) is merged.
- E5: `scripts/screenshots.py` takes the README's pictures of the real viewer (headless Chrome over the DevTools protocol, lessons played through page clicks); README now shows the park, the kitchen, the imported A1, RL II with the A1 and with the lesson's robots, the PID twins and the obstacle course.
- Found while taking the pictures, fixed: the obstacle course chip said 驾驶小车 for a dog; "-0.0 m"; the course plot magnified millimetre settling; at 1280 x 800 the third PID twin reached under the plots and on a phone off the screen (lesson cameras can now ask to fit every robot); run labels broke inside words.
- Verified: 877 passed / 20 skipped, 3 slow, smoke, playtest 20 of 20, push bench 0 falls in 24; CI green on 35e2765.
- What did NOT work:
  - Three review rounds were not enough to bring any branch to "ship"; the reviewers kept finding real regressions, so I merge what is better than main and keep a list.
  - A second WebSocket client starting a lesson leaves the page that shows it without plot history (screenshots now play lessons through the page; real users start lessons from their own page).
  - Every quadruped in a world now runs its gait: headless simulation is about 20 percent slower with several dogs; a forward drag by the back still flips a never-moved template dog.

## Rounds 33 and 34 (2026-10-17): assets and phase F, eight agents in parallel

- The user asked for better and more unique robot assets (heterogeneous, spliced, complex), then for a bigger goal worked overnight; phase F in docs/PLAY_PLAN.md: robots that do whole jobs in real scenes.
- Merged into main and pushed:
  - Gallery: ten designs built from parts, added by name (`add_robot(name, design=...)`): wheel biped, lift picker, rocker-bogie mars rover, wheel dog, centaur, scorpion, Go2 with a PiPER, a real Franka Panda on a wheeled base, twin-arm torso, tentacle. Each stands, drives or walks, turns and picks in real runs with the real models.
  - Visual dressing for robots built from parts (shells with parting lines, joint covers, screws, LEDs, materials that look different); physics bit-identical.
  - F2 navigation: `move(robot, to=[x, y] or a name)` plans around what blocks the robot (a 2D map from the compiled scene, A*, a follower on the robot's own walker), draws the path in the viewer, double-click to send; the Go2 reaches the warehouse dock landing 24 of 24 after review.
  - F4 imitation lesson: the learner demonstrates picks, k-nearest neighbours and a small numpy MLP copy them, DAgger corrects; F5 seeing lesson: color masks, centroids, back-projection, and `observe(..., find="red")` for agents; F6 `scripts/agent_eval.py` runs fixed tasks through `codex exec` with deterministic checks; F7 the viewer records the 3D view as a video.
- Held back: the real-robot library and splicing (assets/library: ANYmal C, H1, Z1, UR5e, Kinova Gen3, xArm 7, Booster T1, Barkour; arms on a Go2's or Spot's back with payload limits). Its arm planner and the gallery's grew apart in behaviors/arm.py; round 35 integrates them. Straight walks: still a high finding, continued in round 35.
- Found and fixed while merging: lesson.js lost the closing lines two new rigs shared, so the viewer drew nothing (a module-parse test now guards every viewer script); the dressed camera-in-part rule left stale segmentation ids; a screenshot helper lost its return; two tests written for one branch's wording; Windows CI file paths and a course test timing.
- Verified: 1227 passed / 27 skipped, 5 slow, smoke, 22 of 22 lessons played; screenshots of the gallery sent to the user.
- What did NOT work:
  - Resolving conflicts by keeping both sides broke code twice where a hunk ended inside a function; read every conflict with its shared context.
  - Two branches implementing the same feature differently (pick-and-place reach messages, carry planning) cannot be merged mechanically.
  - A workflow agent's API connection dropped (ECONNRESET) on its last fix; the branch was left as it was.

## Rounds 35 to 37 (2026-10-18/19): G1 to G3 and G7 landed, live landing, new priorities

- Round 35 was cut off by API connection drops; round 36 (G1 finish) and the G2/G3/G7 workflow both stopped on a session limit with ten lanes reviewed but none merged.
- The user said (2026-10-19) they do not want branches piling up for a manual merge, parked drones, and put first legged robots, mobile manipulation, mounting anything on anything, and generating robots, environments and tasks (docs/ROADMAP.md, "2026-10-19 调整", G8 to G11).
- New: `scripts/land.py` lands a lane on main after every verified step: a lock in the git dir (one lane at a time), merge main in, ruff, the whole suite and smoke (optionally playtest), fast-forward main, push. All ten lanes landed through it within about four hours, with conflicts resolved by each lane's agent in context.
- Landed:
  - Real-robot library merged with the gallery: ANYmal C, H1, Z1, UR5e, Kinova Gen3, xArm 7, T1, Barkour, ALOHA, OP3, Robotiq; arms on dogs (Go2 + PiPER + SO-100, Spot + Z1, ANYmal + UR5e + Robotiq, Barkour + SO-100). ANYmal + UR5e floor picks on 60 fixed samples: 42 placed, 18 refused, 0 missed, 0 touches (library branch alone: 35 placed, 11 missed, 14 touches).
  - Mobile picking (F3): `pick_and_place` sends a robot with an arm over when the object or place is out of reach (behaviors/fetch.py on the merged planner). 29 of 29 feasible runs (kitchen apple with mobile_panda and lift_picker, warehouse parcel to the dock, desk_lab), 3 correct refusals (go2_arm cannot reach over a 0.93 m island).
  - Sensor suite (G2): cameras (rgb, depth, segmentation), 2D/3D lidar, IMU, encoders, force-torque, touch, rangefinder, GPS, odometry on any robot, with rate, noise, latency and dropout; `add_sensor`, `read_sensors`, `open_viewer(show=...)`; the viewer draws rays, points, frustums, pictures and plots. Go2 3D lidar in the warehouse: 99.4% of 2789 rays within 1 mm of an analytic truth; depth camera nearest within 0.1 mm.
  - Water (G3): harbor, lake and pool scenes, boats and ROVs with pilots; forces from the current step (thin plates no longer fly off).
  - Mechanisms and end effectors (G3): drawers, doors, valves, buttons, conveyors, cloth, suction and magnets; `operate`. Gravity was compensated twice for arms that already had it (fixed); the real Panda opens a drawer 0.2 m at x 0.55 and 0.7; the PiPER reaches 0.12 to 0.14 m by its geometry and says so.
  - Drones (parked): quad, hex, octo, Skydio X2, Crazyflie; landings refuse anything that is not ground.
  - Straight walks: a kept walk steers back to its line; steering is dropped per walk when it makes turns fall more (out-of-sample bench seed 91: kept 72 falls, base 111, in 7200 sequences).
  - Agent evals round 2 (codex exec), robot placement on free spots, videos only in the world's folder; a pixel pass at phone, laptop and wide sizes; `scripts/bench.py` with 39 budgets (lazy Pillow and trimesh keep startup light).
- Verified: every landing passed ruff, the suite (1446 passed / 29 skipped at the water landing) and smoke; playtest for lesson lanes; CI green on Windows and Ubuntu at b3e4be2 before the landings.
- What did NOT work:
  - Two workflows ended on a session limit, losing fix and integration agents; the cure was landing per step, so a cut-off lane loses minutes, not a night.
  - A semantic clash git could not see: main's lightweight test failed on a lane that imported Pillow at module level (land.py's suite caught it).
  - A built sub with strong up thrusters was taken for a drone after the drones merge (fixed in the water lane).
  - iCloud left "name 2.py" conflict copies in every worktree and "main 2" refs in .git; now ignored, and the broken refs moved out.
  - The library line-up still shows a grey arm lying down (the Z1 and xArm 7 library homes are Menagerie keyframes folded back); go2_arm cannot fetch from counter height.

## Next-round seed

Round 37 phase 2 is running (wf_ccf0a9ac-491): five lanes from the new main that land after every verified step: G8 legged (every library legged robot walks, stairs and gravel with terrain sensing, payloads), G9 mobile manipulation (base and arm plan together, drawers, doors, buttons, conveyors, two rooms), G10 mount anything on anything (checks, quick-change end effectors with dexterous hands, saved combinations), G11a environments and tasks from a sentence (success checks, rewards, task suites), G11b robots from a sentence. Then: library home poses for the Z1 and xArm 7, README pictures, a review round.

## Frontier

- Current ceiling: any robot in 6 scenes and 8 terrains, driven by gait, MPC/MPPI, LQR, IK or learned policies (Go2, G1) from the viewer; 11 lessons (PID on the PiPER, gaits on the Go2, balance on the G1, any body learning to walk, motion planning on the PiPER), 3 challenges, 2 real-robot face-offs and 4 tuning benches with measured comparisons.
- Done in round 25: learned walks kept with the user's robot and driven by `move` and the keyboard.
- Done in round 26: lessons in narrow windows and on phones; idle learned walkers hold their spot.
- Done in round 29: learned walks turn on the spot (A/D, move turn).
- Done in round 30: edits and lesson round trips keep robots and objects where they are.
- Done in rounds 35 to 37: real-robot splicing on one merged planner, mobile pick and place, the sensor suite, water, mechanisms and end effectors, drones, straighter learned walks, a 39-budget bench, and lanes that land live.
- Done in rounds 33 and 34: a gallery of ten unique robots, dressed parts, navigation, imitation and seeing lessons, agent evals, video recording.
- Done in round 32: agent errors that say how to fix them, quadrupeds that stand on their gait from load, README pictures from the real viewer.
- Done in round 31: a playtester that plays every lesson (20 of 20 finish), quadrupeds that take a push on their gait, URDF imports that learn to walk, the public repo installed and driven by Codex end to end.
- Next frontiers (2026-10-15, after round 26): a `learn_walk` tool (gate); learned walks that turn (a steering term or a second rhythm); measuring and cutting agent token cost (321k for a 6-call task); learned walking for robots the user builds (round 21; reflexes and shoves round 22); a playtest agent that plays every lesson; tutor MCP tools (gate); a statically stable crawl with planned body shifts (taught next to the trot); obstacle-aware arm motion (joint-space RRT-Connect with the collision check we now have, taught as a lesson); G1 trained with shoves and on rough ground (domain randomization made visible); custom-morphology RL.
- Next frontiers: behaviors (round 4), import doctor + robot library (round 5), real-agent evals once CLIs work, MCP App embed in Claude Desktop (needs user to add the server to Claude Desktop).
- Radical ideas weighed: physics in browser via MuJoCo WASM (kept for sharing only).
- Escalation ladder: open-loop gaits -> CPU RL for custom morphologies -> GPU backends (MuJoCo Warp on NVIDIA) -> sim2real.
