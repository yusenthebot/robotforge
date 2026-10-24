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

## Rounds 37 and 38 (2026-10-19/20): G8 to G12, lanes that land live, Sonnet to save tokens

- Every lane landed through scripts/land.py after each verified step; nothing waited on a branch. The user asked (2026-10-19) to save tokens, so the second half ran on Sonnet with one review round per lane.
- The repo went private on 2026-10-19 at the user's request; CI now runs only by hand (`gh workflow run CI --ref main`) at the end of a milestone.
- Landed:
  - G8 legged: one interface for library quadrupeds and humanoids; terrain sensing; Go2, A1, ANYmal C, Barkour and Go1 cross 8 x 30 cm stairs at 60 percent speed (14 of 15); Spot stands before stairs it cannot do; humanoids walk flat only (G1, T1, OP3). scripts/legged_bench.py, docs/LEGGED.md.
  - G9 mobile manipulation: base and arm plan together; drawers, doors, buttons, conveyors (the base drives beside the belt), a two-room home; held-out seeds 8 to 27: doors and buttons near 20/20, Go2 fetch 19/20, wheeled drawer take 7/20, ANYmal fetch 7/20. docs/MOBILE.md.
  - G10 mounting: `mount` puts any robot, end effector or sensor on any body with an adapter plate and checks; LEAP and Allegro hands, suction, magnet; saved designs mount again; Z1 and xArm 7 homes stand up.
  - G11 generation: load_scene(description=...), add_task and task suites (19 of 20 generated tasks done for real); add_robot(describe=...) builds robots from a sentence (snake sweep 213 of 216).
  - G12 breadth: add_robot(library=...) and list_library over 200+ open robots from robot_descriptions with a trial at add time (55 legged tried: 6 walk, 21 collapse); new sensors (stereo, fisheye, event camera, sonar, wheel odometry, tactile skin, heat and gas); occupancy mapping, particle-filter localization, navigation on the robot's own map; user plugins (parts, sensors, controllers, behaviors), MJCF and URDF export, one robot interface in scripts (docs/EXTENDING.md).
  - README as a product page with a 68 s promo video from real footage (docs/media/promo.mp4).
- What did NOT work:
  - Network drops (ECONNRESET) killed whole workflows three times; landing per step kept the loss to minutes, and resuming in the same worktrees worked.
  - Auto-triggered CI was cancelled by every next landing, so Windows went untested for a day; five Linux and Windows failures surfaced once runs completed (now fixed).
  - iCloud keeps copying ref files ("main 3") and sources ("manager 2.py"); land.py sets the refs aside, .gitignore hides the copies; moving the repo out of iCloud is the real cure (the user's call).
  - Humanoids beyond flat walking, the wheeled drawer take and ANYmal fetching through narrow gaps stay unreliable.

## Rounds 39 and 40 (2026-10-20): G5, G6, humanoids, G13, and version 0.0.1

- Landed: G6 control depth (joint and Cartesian impedance, operational space, press and wipe at a force, peg in hole, grasp planning from hulls); G5 flexible interfaces (behavior trees and state machines with a tree panel, gamepad, record, replay and export as JSON and NPZ); humanoids standing by whole-body balance, arm handover, rovers taking turns; G13 asset engine (parametric generators, four styles, physical materials), 24 generated objects with grasp hints and harder tasks (mug on shelf, stack, rack, sort, pour, hand a tool), any-algorithm training with a live tuning view (ARS, CEM, PPO, SAC, plugins; held-out scoring).
- scripts/land.py: first come first served, takes over a lock whose holder died, runs the suite in parallel shards balanced by recorded times, reruns files that fail only beside others, and skips tests over 90 s the branch does not touch.
- iCloud deleted .git/refs/heads/main mid-landing and resurrected deleted refs and worktrees; the repo moved to ~/dev/robotforge (fresh clone; the old folder is kept as a backup).
- Version 0.0.1 (user, 2026-10-20: wrap up as a closed loop): a review round tried 39 abilities for real (26 PASS, 12 PARTIAL, 1 FAIL) and ran 8 codex sessions; a fixer landed all 17 high and medium findings (ANYmal + UR5e + Robotiq mount, imported arms recognised, wheeled URDFs, stretched-servo check, the README bedroom-desk example, kitchen apple, lidar box, G1 dead zone and others). docs/RELEASE_0.0.1.md lists what works, how it was verified and the known limits.
- What did NOT work:
  - GitHub Actions refused to start the 0.0.1 CI run (account billing), so Ubuntu and Windows are unverified for 0.0.1.
  - Network drops killed whole workflows twice more; resuming in the same worktrees recovered everything.
  - The machine ran at load 20 to 70 with six lanes, so the suite took 18 to 38 minutes per landing until heavy tests were skipped when untouched.
  - Still unreliable: humanoids off flat ground, some catalogue robots, URDF round trip (motors lost), wheeled drawer take and ANYmal fetch (about a third), handover mid-way collisions.

## Round 41 (2026-10-21): runtime boundaries and measured mobile manipulation

- Local changes, not committed or pushed: MJCF metadata detachment, complete script joint reads, exclusive script clock ownership, bounded cancellation/output cleanup, POSIX process groups and Windows Job Objects; default ARS routing, coordinator plugin initialization and shared queued/active/evaluation deadlines.
- Mobile planning now rehearses complete grasp, fold and loaded departure paths at perturbed poses, with independent IK warm starts. A real Go2 regression caught during development led to keeping walker current-stance behavior unchanged; newly searched poses still rehearse complete paths.
- Frozen regression improved from 2/7 to 6/7: Panda drawer seeds 8–11 from 0/4 to 4/4, Panda fetch and Go2 drawer still pass, ANYmal fetch seed 8 still falls before grasping.
- Untuned holdout: 25/26, including Panda drawers 19/20, Panda cross-room fetch 3/3 and ANYmal + UR5e cross-room fetch 3/3. Seed 50 misses the drawer grasp. Three ANYmal successes do not establish broad stability.
- Saved every attempt's generated world, stage trace, physics state and source/model/dependency provenance. Report: docs/bench/2026-10-21-mobile-boundaries.md; raw local evidence: ../robotforge-evidence/2026-10-21/.
- Verification: wheel and Ruff passed; complete default selection ran 3020 cases (2891 passed, 127 skipped, two test expectation/isolation failures). Both test issues corrected, affected files 188 passed/3 skipped, deliberate reordering 87 passed. No full rerun after test-only corrections. Production and final test diffs reviewed. Real installed CLI/MCP smoke passed, omitted algorithm trained through ARS without RL extras, and Chrome final-state screenshots were inspected with zero JS errors.
- Remaining boundaries: native Windows untested here; deliberately detached POSIX descendants and blocking threads/plugins need separate treatment. ARS checks step budgets between whole iterations (64 requested, 815 used in the smoke), while wall deadlines are checked inside rollouts. Static type and Bandit/pip-audit tools were unavailable.
- Corrected stale release context: v0.0.1 tag exists at e418898; the latest pre-edit CI passed Ubuntu but Windows had a conveyor-test failure. No paid CI run started this round.

## Round 42 (2026-10-21): shared execution kernel, first two adapters

- User direction: framework optimization first; pause drawer and other individual success-rate tuning. Approved sequence: execution kernel, capability descriptions, task/algorithm contracts, common experiment records.
- Added session-scoped ExecutionKernel/ExecutionRun with synchronous ownership admission, shared watched/headless stepping, cumulative simulation/wall/optional-step budgets, reset generation invalidation and measured run outcomes. move and ScriptRunner use it; robot algorithms are unchanged.
- Preserved MCP definitions and existing response fields, continuous moves, recording, heading tracking, settling and interrupted navigation reports. Legacy controller commands use admission tokens; plugin behaviors still use scripts. Viewer teleop/manual steps cannot steal an owned clock.
- Cancellation tracks actual commanded controllers: an arm-only or refused command does not claim an existing base. Replacement/reset controllers are preserved. Script cleanup reaps the child first, cancels owned replays and retains the lease until cooperative planning threads finish; queued threads recheck validity before mutation.
- Real installed CLI/MCP before/after turning move: 0.559 m both times, exact advertised move/run_script tool definition match. Chrome observed live move and script, saved a WebP, disconnected mid-run without preventing completion, and recorded no JS errors. A real waiting script was reaped 0.016 s after reset; another move then succeeded. Standard smoke passed in 2.9 s.
- Verification: wheel and Ruff passed; scoped adversarial review approved after fixing legacy admission, controller ownership and per-step deadline cases. Full default selection: 2920 passed, 127 skipped, zero failures, both shards green in 590.2 s; source fingerprint unchanged. Expanded selection 273 passed/1 skipped and final kernel/process selection 45 passed.
- Evidence: docs/EXECUTION.md and docs/bench/2026-10-21-execution-kernel.md; raw local logs, before/after requests, live screenshots and source snapshots in ../robotforge-evidence/2026-10-21-execution-kernel/.
- Limits: this migrates two adapters, not every legacy lifecycle. Blocking threads/plugins/native calls remain cooperative, native Windows is unverified here, and type/security package checks were unavailable. No dependencies added, no commit or push.

## Round 43 (2026-10-21): one robot capability description

- Templates, imported models, library robots and compositions now share a compiled-model capability description across MCP, script Robot objects and the viewer. Existing tool signatures and legacy fields are preserved; missing TCPs, unpowered tools, standalone grippers and untuned humanoids no longer gain capabilities from their kind alone.
- Extracted shared read-only bindings and speed formulas from the runtime controllers, without constructing controllers, solvers or ONNX models on discovery. Active limits and current marine drag are refreshed; structural discovery is cached per model. Saved walks must match the current joints before establishing support.
- Script snapshots can refresh, each arm exposes its gripper, and multiple endpoints require an explicit choice. The viewer shares drive/profile/mode decisions and broadcasts changed descriptions at 1 Hz, with model-version buffering during mesh loads and no update consumed by a joining client.
- Real installed CLI/MCP/HTTP/script values agree for six source/body cases, with a cached Panda added in Chrome. Imported rover displacement was 0.176 m; a JGA25-equipped mounted-arm rover moved 0.137 m and reached its arm target within 10.7 mm per coordinate. Delayed-mesh and two-viewer refresh checks passed, with no JS errors.
- A default small rover carrying an arm did not meet the early short-run displacement threshold; that attempt is retained and no production controller or success-rate tuning followed. Capability support describes an interface and binding, not arbitrary configuration performance.
- Compatibility: move/run_script definitions and legacy result fields matched; turning displacement remained 0.559 m. All 320 physics markers in 14 affected controller scenarios matched 96763e0. Final focused tests 60 passed; standard smoke passed in 3.4 s. Independent review approved after fixing explicit learned-mode compatibility and two viewer update races.
- Full default selection: 2965 passed, 127 skipped, zero failures (3092 cases, 666.1 s). Both shards green, with source and tests unchanged throughout. Existing real-Chrome gamepad regression: 8 passed in 33.84 s, using injected Gamepad API input and actual browser/physics behavior.
- Evidence and boundaries: docs/CAPABILITIES.md and docs/bench/2026-10-21-capabilities.md; raw source, tool, trajectory, script, model, screenshot and test evidence in ../robotforge-evidence/2026-10-21-capabilities/. No dependencies, commits, pushes or release. Native Windows and unavailable static/security package checks remain unverified.

## Round 44 (2026-10-22): shared task and algorithm contracts

- Built-in, saved-world and plugin tasks share discovery, schema validation and model/robot/actuator binding across MCP, scripts, viewer and training. Legacy TASKS, decorators, public calls and all 60 MCP definitions stay compatible.
- Registered task identity stays separate from Task.name dispatch. Stored policy parameters and new/legacy metadata rebuild consistently, including Reach wrappers and default-named RewardTask implementations.
- Control methods share aliases, signatures/schemas, finite settings and structural adapters. Training keeps the existing Algorithm/Hyper registry, rejects unselected/duplicate settings, checks dependencies before allocation and closes environments on setup failure.
- Script controller attachment shares execution budgets and cancellation ownership; viewer metadata refresh covers task and algorithm reload/removal, without executing factories to list them. Saved-world wrong-robot playback is refused.
- Real verification: isolated installed CLI/MCP, script process, separate training coordinator/workers and rendered Chrome; plugin algorithm plus ARS runs saved and replayed policies, nondefault goal/tolerance survived all entry points, actual Training Play resumed a paused world, viewer task_run policy reply matched, no JS errors.
- Behavior: 14 built-in scenarios and all 280 physics hash markers match b0522a0; no control/reward/manipulation success-rate tuning.
- Validation: 3072 passed, 127 skipped, zero failures in the complete 3199-case default selection (718.1 s); 107 new contract tests; 8 real Chrome gamepad cases (33.03 s); wheel, Ruff, JS syntax, smoke and independent final review passed. Source/tests frozen throughout.
- Evidence: docs/TASK_CONTRACTS.md, docs/bench/2026-10-22-task-contracts.md/.json and ../robotforge-evidence/2026-10-22-task-contract/. Foundation checkpoint b0522a0 and contract checkpoint 7508d34 were pushed after verification at the user's request. No dependency or release added; native Windows and unavailable type/security tooling remain unverified.

## Next-round seed

Continue incremental arm-planning and multi-phase execution migration against the unchanged seven-case common-record framework baseline.
MCP pose and gripper now share the recorded execution lifecycle with run_controller; move and ScriptRunner retain their shared kernel.
Preserve carrier settling, queued work, phase budgets and wind/fast-only simulation semantics.
Manipulation success-rate work stays paused.
Native Windows and the prior conveyor failure remain pending; publishing/licensing are separate decisions.

## Frontier

- Current ceiling: any robot in 6 scenes and 8 terrains, driven by gait, MPC/MPPI, LQR, IK or learned policies (Go2, G1) from the viewer; 11 lessons (PID on the PiPER, gaits on the Go2, balance on the G1, any body learning to walk, motion planning on the PiPER), 3 challenges, 2 real-robot face-offs and 4 tuning benches with measured comparisons.
- Done in round 25: learned walks kept with the user's robot and driven by `move` and the keyboard.
- Done in round 26: lessons in narrow windows and on phones; idle learned walkers hold their spot.
- Done in round 29: learned walks turn on the spot (A/D, move turn).
- Done in round 30: edits and lesson round trips keep robots and objects where they are.
- Done in rounds 37 and 38: legged on terrain, mobile manipulation with mechanisms, mount anything on anything, worlds, tasks and robots from a sentence, 200+ open robots by name, perception and mapping, user plugins and export.
- Done in rounds 35 to 37: real-robot splicing on one merged planner, mobile pick and place, the sensor suite, water, mechanisms and end effectors, drones, straighter learned walks, a 39-budget bench, and lanes that land live.
- Done in rounds 33 and 34: a gallery of ten unique robots, dressed parts, navigation, imitation and seeing lessons, agent evals, video recording.
- Done in round 32: agent errors that say how to fix them, quadrupeds that stand on their gait from load, README pictures from the real viewer.
- Done in round 31: a playtester that plays every lesson (20 of 20 finish), quadrupeds that take a push on their gait, URDF imports that learn to walk, the public repo installed and driven by Codex end to end.
- Next frontiers (2026-10-15, after round 26): a `learn_walk` tool (gate); learned walks that turn (a steering term or a second rhythm); measuring and cutting agent token cost (321k for a 6-call task); learned walking for robots the user builds (round 21; reflexes and shoves round 22); a playtest agent that plays every lesson; tutor MCP tools (gate); a statically stable crawl with planned body shifts (taught next to the trot); obstacle-aware arm motion (joint-space RRT-Connect with the collision check we now have, taught as a lesson); G1 trained with shoves and on rough ground (domain randomization made visible); custom-morphology RL.
- Next frontiers: behaviors (round 4), import doctor + robot library (round 5), real-agent evals once CLIs work, MCP App embed in Claude Desktop (needs user to add the server to Claude Desktop).
- Radical ideas weighed: physics in browser via MuJoCo WASM (kept for sharing only).
- Escalation ladder: open-loop gaits -> CPU RL for custom morphologies -> GPU backends (MuJoCo Warp on NVIDIA) -> sim2real.


## Round 45 (2026-10-22): shared experiment records

- Common schema/store across control comparison, training run/segment and live control/policy playback.
  UUID execution identities, semantic fingerprints, environment/task/algorithm/seeds/budgets/version/metrics and relative SHA256 artifact manifests are retained together.
  MJB + complete integration state preserve compiled assets, controls, external forces, mocap and warm starts.
- Existing 60 MCP definitions remain byte-for-byte equal to the previous tool snapshot; result references are additive.
  Low-level Python comparisons remain opt-in; MCP and script/viewer adapters record automatically.
  Scripts can query bounded history and inspect a canonical UUID in the current world.
- Training artifacts survive named-policy replacement and halving rungs.
  Actual weight hashes select winning/trial lineage, including after a later segment fails; unknown legacy sources remain unknown.
  Checkpoint input SHA participates in the specification identity.
  Worker, submission and cancellation failures close the appropriate records without concurrent parent writers.
- Stop/replacement/reset/model load/shutdown finalize live observations before discarding state.
  Controllers and scheduling remain unchanged; unavailable contact/plugin observations are null rather than fabricated scores.
  Corrupt/path-traversing artifacts, unsafe policy names, foreign parent writes, hidden physics dependencies and unknown callbacks have regression coverage.
- Real installed MCP comparison reproduced in a separate process.
  A detached ARS worker trained Reach from seed 0: 64 requested steps, 800 actually consumed, preserving existing whole-iteration semantics.
  Repeating its recorded RunSpec in a fresh process yielded identical policy SHA256.
  Real Chrome Training Play and a token-authenticated script replay retained both training run and segment origins.
  The 28-second 1920x1080/30fps MP4 is an actual viewer capture with verification-only labels populated from results, not generated animation.
- Real live IK replay did not match bit-for-bit: maximum qpos error 3.875e-10, qvel 6.116e-10, ctrl 0.
  The record exposes strict equality separately from per-field absolute errors and a 1e-8 numerical check.
  Prepared controller state, recomputed derived fields, interactive changes and external runtime state limit generic live reproduction.
- Current checks: wheel, Ruff, independent review, 128 targeted tests and standard smoke pass.
  Full default regression passed all 3297 selected cases: 3170 passed, 127 skipped, zero failures/errors in 599.6 seconds.
  All 8 real Chrome gamepad cases passed in 33.71 seconds, and 280 physics markers in 14 scenarios match bcf2cf7.
  Source, scripts and tests stayed unchanged across final verification.
- Next seed: common-record framework acceptance comparisons and incremental execution adapter migration.
  Keep manipulation success-rate tuning paused; current Windows and optional runtime validation remain separate.


## Round 46 (2026-10-22): framework regression baseline and controller execution

- Added a fixed seven-case acceptance suite using common records and verified artifacts.
  Captured against old core aae9283 with only the new harness untracked, repeated independently, then compared the migrated candidate without replacing the baseline.
  Runtime mismatch is unsupported; missing cases, unavailable/nonfinite metrics, changed inputs or physics and corrupt artifacts fail closed.
- Migrated MCP run_controller to shared ownership, cumulative preparation/stepping deadlines and reset generations.
  Snapshot workers leave the pose lock free; repeated cancellation drains real work before lease release, and existing stop behavior clears only owned controllers.
  Execution parent records link original controller observations and preserve actual reasons without masking cancellation on recording errors.
- All 60 MCP definitions stay identical and existing response fields remain, with an additive execution reference.
  Real Chrome headless/playing/paused/disconnect checks have identical final arrays, 400 steps and push at step 100.
  Actual stop/reset, later requests, old script controller flow and rover move pass with no browser errors.
  The real 19.23-second 1080p/30fps test MP4 was decoded, inspected and shown to the user.
- Checks: 3221 passed, 127 skipped, zero failures/errors in the complete 3348-case selection (1020.7 s); 79 final focused tests including 51 new cases; 8 real Chrome gamepad tests (47.60 s).
  Wheel, Ruff, installed smoke (3.4 s) and independent Python reviews pass; 14 scenarios and all 280 physics markers match aae9283.
  Source/scripts/tests/dependencies stayed frozen throughout final verification.
- Evidence: docs/FRAMEWORK_REGRESSION.md and docs/bench/2026-10-22-framework-regression.md/.json; raw logs, records, hashes and MP4 in ../robotforge-evidence/2026-10-22-framework-regression/.
  Implementation checkpoint 62557d2 was pushed before full regression; no control/reward/success-rate tuning, new dependencies, release or paid Actions.
  Native Windows, optional RL backends and unavailable type/security packages remain unverified.
- Next: migrate remaining pose/gripper and arm families separately against this baseline; explicitly preserve multi-phase and fast-only/wind lifecycle behavior.


## Round 47 (2026-10-22): pose and gripper execution

- Extracted RecordedExecution from the controller adapter and migrated MCP pose/gripper to its lease, cumulative preamble deadlines, generation checks and common terminal records.
  Tool definitions and existing fields remain compatible, with an additive execution reference; the controller adapter remains a task-specific subclass.
- Preserved reused ArmMotion queues, five-second gripper cap, controller attachment and position-servo targets.
  Cancellation only cleans the exact endpoint; a moving carrier remains owned by its previous caller.
  Humanoid stop retains zero-speed balance and any pending/current pose; later stepping may continue that pose.
  Script bool/None wrappers and cumulative lease remain; read-only polling cannot adopt replacement controllers for cleanup.
- The seven-case committed framework golden remains byte-for-byte unchanged and passes.
  Ten additional actual MCP cases and all 60 complete tool definitions match clean old core fac7d71 exactly; 14 control scenarios retain all 280 physics markers.
  Actual installed MCP/Chrome verifies identical initial and final arrays across four viewing states for pose (2260 steps) and gripper (260 steps), stop/reset/recovery, real scripts and a mounted-arm moving rover.
  Browser errors are empty; the actual decoded/inspected 44.6-second 1080p/30fps MP4 was shown to the user.
- Verification: 3248 passed, 127 skipped, zero failures/errors in the complete 3375-case default selection (720.5 s); 166 focused tests with 27 new cases, 8 real Chrome gamepad cases (34.46 s), wheel, Ruff and installed smoke (3.7 s).
  Independent Python review approved after fixing preamble-timeout mutation and final metric persistence failure; both tools have regressions.
  Source/tests/scripts/dependencies remained unchanged during final verification.
- Implementation checkpoint 86c939b was pushed after focused, baseline and real Chrome verification.
  Evidence: docs/bench/2026-10-22-pose-gripper.md/.json and ../robotforge-evidence/2026-10-22-pose-gripper/.
  No controller/reward/success-rate tuning, dependencies, release or paid Actions; native Windows, optional RL and unavailable type/security tools remain unverified.
- Next: arm planning and multi-phase adapters in separate increments, preserving carrier settling and phase budgets.

## Round 48: arm planning and multi-phase execution (2026-10-23)

- Migrated MCP move_arm, pick_and_place and operate to RecordedExecution/ArmExecution; tracked script arm planners and kept one script family ledger.
  The existing factories and controller math still own queues, catches, relays, fetch errands and mechanism visits.
  Carrier waits, original phase caps, legacy settle rounding and script cumulative limits remain; whole-call metadata/preparation and all phases consume one anchored budget.
- Added controller actor provenance and same-step invalidation checks.
  Foreign replacements survive cleanup; delayed callbacks are disarmed; owned navigation ends; repeated cancellation retains native workers and reaps the script child.
  Owned Errand contact/grip changes restore before reset/model replacement discards old IDs.
- Frozen prior-core 6ac862e captures: 13 completed cases exact in steps, public response and qpos/qvel/ctrl.
  The moving-carrier unreachable request still refuses after the same 670 settlement steps and final state; only its newly attached empty arm is now removed.
  This is not successful mounted-arm coverage and is not counted as such.
- Actual installed MCP and Chrome: movement, handover and mobile operation agree in initial state/results/steps/final arrays across headless, playing, paused and disconnected modes.
  Actual stop/reset, script wrapper flow and empty browser error list pass; the decoded/inspected 141.80 s MP4 is retained for the user.
- Verification: 3285 passed, 127 skipped, zero failures/errors over all 3412 default-selected cases (830.2 s); 117 focused tests, 37 new cases, 8 real Chrome gamepad tests, wheel, Ruff, CLI smoke and seven unchanged framework baseline cases pass.
  Independent Python review approved after mounted alias, ownership, deadline, old-settle rounding and repeated script cancellation regressions.
  Source/tests/scripts/dependencies remained unchanged during final verification.
- Evidence: docs/bench/2026-10-23-arm-execution.md/.json and ../robotforge-evidence/2026-10-23-arm-execution/.
  No success-rate/reward/controller tuning, new dependencies, release or paid Actions; native Windows, optional RL and unavailable type/security tools remain unverified.
- Next: migrate the remaining direct simulation/control and admission-only adapters incrementally against the unchanged framework baseline.

## Round 49: overall visual quality and playability goal (2026-10-23)

- User set the broader goal: improve simulation materials and overall visual quality while keeping the project lightweight and quick to load, and improve playability.
  Single-action success-rate work such as weight lifting or drawer operation stays deferred.
- Updated GOAL, ROADMAP and STATUS so the next run starts from this direction.
  Existing execution, capability, task and experiment infrastructure serves the experience work; remaining adapter migrations no longer define the default next milestone.
- Next: representative robot/indoor/outdoor visual and interaction baselines, measured cold/warm loading and rendering costs, then shared visual improvements and a playable scene loop selected by observed benefit.
  Existing performance budgets and relevant physics/interface regressions remain guardrails.
- This is a documentation and goal-setting change only.
  No material, renderer, gameplay or success-rate code changed; no new visual or performance result is claimed.


## Round 50: shared surface quality and direct exploration (2026-10-23)

- Established three-scene visual/performance comparisons with fixed cameras, real cached Panda/Go2 meshes, paused matching physics and cold/warm browser navigations.
  Three alternating baseline/candidate pairs retain every sample, including initial candidate first-frame delays up to 1.848 seconds.
  World/scene JSON and captured initial physics arrays match; later captures normalize an existing startup selection-ring-size artifact without changing app behavior.
- Added shared 256-pixel ground color/detail tiles and distinct roughness/bump profiles across planes, heightfields and terrain blocks.
  GPU Source sharing and disposal avoid per-object image allocations; park textures fall from 12 to 7.
  Added wood/bark/stone/fabric detail and corrected texture strength/scale semantics.
- Ground previews use the same tiles; separate ground-button/training-marker CSS fixes tiny ground hit areas while keeping training markers 9 x 9 pixels.
  Native browser interaction switches tile/wood, drives the rover 1.52043 m and resets within 0.00001074 m in XY.
  The actual 34.733-second video is decoded and visually inspected.
- Validation: 242 focused, four actual GPU/chooser, eight gamepad browser and seven unchanged canonical framework cases pass; wheel, Ruff, actual CLI/MCP smoke and independent review pass.
  Full install/viewer budget sections pass without changing limits: 163 MB installation, 0.65-second CLI first frame and about 60 FPS at DPR 2.
  No complete default-suite rerun for this viewer-only change; prior full framework validation stays separate.
- Residual: wood whole-frame low-angle flips increase from 421 to 594 per megapixel; grass boundary stays 30 and stairs 229 to 227.
  Weaker floor grain/bump did not improve it and was reverted; isolation puts much of this view's extra signal in prop detail, with all evidence retained.
  No lighting or shadow settings changed, and no blanket visual/performance improvement or new game loop is claimed.
- Evidence: docs/bench/2026-10-23-visual-playability.md/.json and ../robotforge-evidence/2026-10-23-visual-playability/.
  No new dependencies, physics/controller/interface changes, individual manipulation tuning, release or paid Actions.
  Next: coherent lighting/contact readability, temporal stability and a complete easy-to-start scene play loop against this baseline.


## Round 51: directly create, place and play (2026-10-23)

- Empty worlds and the Robots drawer now offer four illustrated local starters, with click/touch placement on ground or a tabletop, automatic selection after reply/scene/poses, existing control/map/terrain/reset reuse and EN/ZH copy.
- The viewer action validates an allowlist, finite bounded surface coordinates and version; blocks lessons/plain models/owned execution; serializes unique naming; checks saved and current occupancy atomically before committing through the existing world edit path.
- What failed during verification: raw surface Z embedded mobile roots; stale preflight poses could allow a moving robot into the target; entering placement suppressed the stop message and let arm pad jogging resume; XYZ positions broke feature terrain and stopped following new ground; far placement targets left the tight initial view.
  Fixed support offsets, final live-state collision checks and store/generation guards, selected-robot stopping and pad suppression, ground XY/table XYZ classification, XY feature extraction and retained wheel/right-drag camera input.
  The first keyboard test also omitted CDP Enter text, so it generated key events without native button activation; corrected the input harness and retained native focus/activation checks.
- Final verification: 184 focused tests including 40 new boundary/terrain/tabletop cases; 22 actual Chrome EN/ZH creation, keyboard/gamepad and framing/layout cases; seven unchanged canonical framework cases; wheel, Ruff, JS syntax and actual CLI/viewer/MCP smoke pass.
  Independent Python/source/security review approved.
  The complete default suite was not rerun; prior full-suite results remain historical.
- All 11 existing install/viewer budgets pass without relaxation: 3633 KB wheel, 163 MB runtime, 42 packages, 0.56 s CLI start-to-frame and about 60 FPS at DPR 2.
  Wheel grows by 6,112 B; fresh empty HTTP by 12,674 B and one request.
  Single fresh-profile timing observations and the laptop p95 increase remain in the report, without claiming general speed improvement.
- Actual native videos: desktop 33.20 s and phone 11.333 s, fully decoded and contact-sheet/full-frame inspected; eight placements including a real tabletop arm, rover drive/stop over 1.15 m, joint control, stairs/hills/flat, desk map, floor and reset; no JS/runtime/GL errors.
  Frozen baseline/current empty world and streamed scenes match except the additive template catalog.
  Source stayed 55a07c1465a1bfca77322195ef4072cae93c4bf364ea325c42cefcf972d910e8 throughout accepted final runs.
- Evidence: docs/bench/2026-10-23-direct-playground.md/.json and ../robotforge-evidence/2026-10-23-playground-entry/; unified completed verification record 5eb68fd7-4bf4-4642-a4e1-60dba2ed12ff verifies all 34 immutable artifacts.
- Next: continue shared visual/contact readability and creative editing/exploration within the same lightweight budgets.
  The broad visual/playability goal remains active; single-action success-rate tuning remains deferred.


## Round 52: shared lighting and moving contact shadows (2026-10-23)

- Aligned direct sunlight with the studio's existing brightest reflection panel and replaced the 20 mm normal bias with bounded texel-scale bias.
  Kept one 2048-square map, existing light intensities and unchanged materials/physics/interfaces.
  Body-local cached bounds, guard bands, texel snapping and an idle shortcut fit moving casters and ground receivers beneath tabletop/airborne bodies.
- Review caught empty-world sun orientation, a floor plane treated as world-local and movable imported mocap bodies cached as fixed.
  Fixed key initialization, transformed floor planes after streamed poses and fixed-body matrix invalidation.
  Real GPU regressions also cover rotor envelopes, flex deformation, camera invariance, shadow raster contrast and drive/reset beyond initial coverage.
  Test harness fixes update the probe camera matrix and pause before reset so the brief reset-time state is observable.
- Final runtime stayed 3d820d1407dab3407a5108d64396bd4b714a7b0df20553729a18cb70ce69920d throughout accepted A/B, video and verification.
  Four serial alternating old/current runs retain exact scene/world/initial arrays, all timing samples and zero browser/runtime/GL errors.
  Observed stronger shelf/table shadows and clearer contact, with unchanged corresponding draw/triangle/texture counts.
- Validation: 186 focused, 16 real Chrome lighting/creation/gamepad, eight GPU/asset/sensor and seven unchanged canonical framework cases pass; wheel, Ruff, JS syntax and actual CLI/viewer/MCP smoke pass.
  Independent source and acceptance-evidence review approved.
  The complete default suite was not rerun; unavailable optional type/security tools and native Windows remain unverified.
- All 11 existing install/viewer budgets pass: 3637 KB wheel, 163 MB install, 42 packages, 0.51 s CLI first frame, about 60 FPS at DPR 2, 2.50/3.40 ms standard/asset frame time, 132 MB server peak and 5.20% idle CPU.
  HTTP grows by 7841 B/two resources and wheel by 3179 B.
  Warm gallery loading is 69.7 ms slower between two-sample averages; temporal flips/Mpix change kitchen wood 594 to 642, grass boundary 30 to 43, stairs 227 to 234 and gallery wood 897 to 822.
  No overall speed or flicker improvement claim; wood temporal stability remains open.
- The actual 39.766667 s 1280 x 800/30 FPS video is fully decoded and inspected: six placements, tabletop arm, native 2.375704 m drive/stop beyond initial shadow bounds, joints, shadows off/on, terrain/map/floor and reset.
  Evidence and immutable verification record: ../robotforge-evidence/2026-10-23-lighting-contact/; report: docs/bench/2026-10-23-lighting-contact.md/.json.
- Next: creative scene editing/exploration and targeted temporal stability within the same budgets.
  No individual success-rate tuning, new dependencies, paid Actions, release or deployment; the broader goal remains active.


## Round 53: build, interact and undo local props (2026-10-23)

- Added World-menu Build palette with ball/block/pin/barrier/table, three server-owned scales, quarter-turn controls and one transient wire size envelope.
  Native desktop/touch placement supports ground, tabletop and loose-block surfaces, excludes robot bodies, and retains the four robot starters and existing controls.
  Geometry disposes on replacement/cancel/success; there is no new light, shadow caster, render pass, texture asset or dependency.
- Reused edit/admission ownership and private saved/live preflight, including fixed-fixed pairs and libccd fallback for coincident native-CCD zeros.
  Undo add uses a bounded opaque receipt tied to exact entry/store/version/generation/file digest, removing only that entity rather than popping world history.
  Exact carry preserves free-body/joint/control fields even in existing deep contacts, with velocities restarted; compile-time disk-write detection remains optimistic.
- Review fixes: coincident pins bypassed native distance queries; generic undo carry teleported unrelated deep-contact bodies; dynamic block tops were initially excluded; a fixed robot base bypassed a dynamic-only support rule.
  Added regressions and an actual imported fixed-body browser case.
  The initial native size picker harness did not behave as assumed on this Mac; replaced it with three visible keyboard-accessible buttons.
  Browser setup also needed localhost proxy bypass and an open collision-probe point rather than clicking a robot surface that correctly refused dispatch.
- Final frozen runtime ed392c58959f2d1d459ce0ed898835672bc0733d97948265d99c5bdecce67016 passes the complete default selection: 3400 passed, 127 skipped, 82 deselected, zero failures/errors across 3527 selected cases, 1145.45 seconds.
  It includes all 61 new object core cases and 21 JS parsers.
  Eight new object browser cases plus 16 existing creation/gamepad/lighting cases pass; seven unchanged canonical framework cases, wheel, Ruff and real installed CLI/viewer/MCP smoke (60 tools) pass.
  Independent Python, frontend/source and evidence review approved; optional unavailable type/security tooling and native Windows remain unverified.
- All 11 existing install/viewer budgets pass unchanged: 3642 KB wheel, 163 MB installed, 42 packages, 0.70 s CLI first frame, about 60 FPS at DPR 2, 2.50/7.50 ms standard/complex-asset rendering, 132 MB server peak and 5.20% idle CPU.
  HTTP grows by 9080 B/one request; wheel by 5511 B.
  Four alternating frozen-world desktop/phone pairs retain all 16 captures: warm desktop navigation medians are about 20% slower and props-phone render median/p95 increase from 1.25/1.80 to 1.60/2.35 ms.
  Workload stays 34 calls/14566 triangles in the prop world; no overall speed or memory improvement claim.
- Accepted native videos: desktop 39.70 s and Chrome touch-emulated phone 40.90 s, both fully decoded and inspected, with all props/preview/stacking/tabletop arm/joints/refusal/retry/undo, actual named rover-ball/pin contacts, terrain/floor/map/reset.
  Desktop ball/pin displacements are 0.889806/0.545388 m; phone 0.522893/0.571355 m.
  An earlier final phone gesture began at x=-12.79054 outside the viewport and failed acquisition; its 31.77 s partial recording remains retained.
  One diagnostic-camera-only retry checks in-viewport coordinates and actual acquired chassis bodies; force-drag tips rovers, so no phone joystick or hardware-performance claim.
  Pre-static-support-fix captures stay explicitly historical; no physics/controller/force-strength tuning was used.
- Report: docs/bench/2026-10-23-prop-playground.md/.json; guide: docs/CREATIVE_PLAY.md.
  Raw evidence: ../robotforge-evidence/2026-10-23-prop-playground/; all 6579 indexed raw artifacts verify, and completed unified record c7fd332d-c8d3-4fbb-bc71-8b028523b52a verifies 213 immutable artifacts.
  Candidate-source.patch applies cleanly to baseline 86305fb; raw frames stay externally indexed and videos/source/tests/results are copied immutably.
- Next: improve navigation and free editing clarity, including the four-tab panel that currently wraps Modes onto a second row, plus targeted wood temporal stability.
  The broad visual/playability goal remains active; individual action success rates, releases and paid Actions remain deferred.

## Round 54 — shared material filtering (2026-10-23)

- 本轮按整体视觉与轻量体验推进，没有调整举铁、开抽屉或其他动作成功率。
- 共享 look.js 按每层噪声和条纹相位的实际像素覆盖范围过滤木纹、拉丝、微粒、橡胶、树皮、石材和滚花；远处保留平均材质色调，法线沿用旧的衰减范围。
- 初版过度柔化近处噪声；第二版虽改善平均色调，却放大滚花凹凸；第三版修复法线强度并保留全部较差样本和经哈希验证的源文件。
- 283 项针对性测试、23 项真实 Chrome 材质/搭建/道具/光照回归、七项不变框架基线、真实 CLI/viewer/MCP 60 工具冒烟、wheel/Ruff/21 项 JS 语法检查通过；本轮没有重跑完整默认测试。
- 全部 11 项既有安装/查看器预算通过：3642 KB 包、163 MB 安装、42 个包、0.54 秒 CLI 首帧、约 60 FPS、3/5 ms 标准/资产渲染、132 MB 服务端峰值、6.00% 空闲 CPU（达到既有限制）。没有放宽预算。
- 安装包增加 479 B；首次 HTTP 增加 1161 B，仍为 28 次请求。四轮 A1/B1/A2/B2 对照的世界、场景、初始/最终物理和资源数量一致，73 项证据检查通过。
- 七种变化材质的完整着色器过渡区固定参考和去均值误差均下降，近处木纹所有法向/尺度/平移样本逐字节相同。橡胶近处有轻微细节变化；木纹完整时间残差和滚花 diffuse 时间/去均值结果仍有小幅退化，没有宣称全面消除闪动。
- 首次使用新代码相对于本机已用过的基线，展示场景 411→1792 ms、厨房 377→1232 ms；重复加载有升有降，原始慢样本独立保留，未用暖缓存结果替代。着色器编译是可能原因，尚未单独归因。
- 真正的前后 Chrome 相机实录覆盖展示、厨房、公园和木桌，仿真暂停；对照期间仅在两侧统一隐藏空机器人提示以观察桌面，并记录该外部 CSS 处理。
- 证据、失败样本与复现脚本保存在 ../robotforge-evidence/2026-10-23-material-filtering/；详见 docs/bench/2026-10-23-material-filtering.md/.json。
- 整体目标保持进行中；后续优先检查首次着色器加载、剩余木纹/滚花时间稳定性和自由搭建导航。
- 本轮统一验证记录 5d4a6188-7f2c-47d1-99fc-5fc8a3de00a9 已完成，80 个不可变产物校验通过；14,383 个原始文件由主进程再次逐项校验。原始帧在 Git 外保存。

## Round 55 — draw the posed scene once (2026-10-23)

- 按整体视觉和快速加载目标推进，移除姿态到达前的 provisional world 绘制，避免随后雾、光照和相机拟合触发重复程序准备。材质、几何、物理和现有接口不变。
- 以场景版本确认姿态，接受时间为零的暂停；新网格加载期间继续确认 WebSocket 帧，丢弃过期请求，并统一保护拾取、双击导航、选择圈和辅助渲染。主视图正常提交后才隐藏加载提示，显卡上下文丢失使标记失效。
- 网格加载失败显示中英文原生重试按钮，重连获取最新场景，没有自动网格重试循环。
- 全部 71 项 browser 测试、另八项 slow GPU 资产/传感器测试、283 项针对性默认测试通过，合计 362 项不同用例。新增五项边界测试包含在 browser 套件中。旧版负对照记录 21 次过早绘制，预期失败。
- 七项不变框架基线、实际 CLI/viewer/MCP 60 工具冒烟、wheel/Ruff/21 项 JS 解析检查通过。完整默认测试未重跑，可选类型/安全扫描工具本机不可用。
- 四轮 A1/B1/A2/B2 的 114 项来源、状态、资源、图像和 GL 检查通过，六个固定视角逐字节一致。所有新版本首次绘制均有姿态；旧版 11/12 次首次绘制未有姿态，其中六次新 profile 均未有姿态。
- 首次完整场景两帧代理指标中位数：展示 398.6→327.6 ms，厨房 385.9→311.9 ms，公园 422.3→343.0 ms；缓存导航改善 31–76 ms。不是 GPU 或系统呈现时间，机器和驱动缓存不保证冷。旧 1.79 秒慢样本未复现，也未宣称解决。
- 全部 11 项既有安装/查看器预算通过：3643 KB 包、163 MB 安装、42 个包、0.50 秒 CLI 首帧、约 60 FPS、2.50/4.90 ms 渲染、133 MB 服务端峰值、5.80% 空闲 CPU。包增加 853 B、HTTP 编码资源增加 2779 B，仍为 28 次请求。
- 首次绘制仍有约 88–103 ms 同步阻塞；稳定帧时有升有降，公园近景中位数 2.05→2.75 ms。旧 gate-1 暖公园帧间隔退化也保留。没有全面帧率或内存改善的宣称。
- 两段各 20.033 秒的实际桌面/手机触摸模拟视频均完整解码并查看，包含故障注入、原生重试、实际创建球体、播放暂停、零时刻重置和镜头移动。原始帧、失败 harness 和被中止的重叠预算尝试均在 Git 外保留。
- 报告 docs/bench/2026-10-23-viewer-startup.md/.json；证据 ../robotforge-evidence/2026-10-23-viewer-startup/。整体目标保持进行中，后续继续材质时间稳定性和更清楚的自由搭建导航，单项动作成功率继续暂缓。
- 本轮统一验证记录 2ad3f0e7-4359-4ae9-b837-5c47c63f93c5 已完成，96 项不可变产物通过校验；主进程独立重验 3264 个索引原始文件（451,577,826 B）。

## Round 56 — continuous creative building (2026-10-23–24)

- 按整体可玩性目标增加默认关闭的连续放置：一次选择道具，保留尺寸和朝向，每次独立提交、检查占用并等待新场景姿态和正常绘制，然后清空旧落点。完成/Escape 退出，撤销仍只移除最后一次添加。
- 顶部“场景 · 搭建”包含四个同排页签，支持左右/Home/End、游走焦点、废弃持久化值回退和 Escape 焦点返回。搭建页直接打开现有机器人选择器，模型文件可以跳到地图选择。
- 请求增加可选 strict generation，服务器入口检查，显式请求才返回提交代次；旧请求/回包形状不变。状态携带已有代次，防止零时刻重置、提交后回复前重置、延迟状态和旧异步回调重新激活连续放置。
- 完整 browser 85 项、另八项 slow GPU 资产/传感器、300 项针对性默认用例均通过，共 393 项不同用例，包含 14 项新浏览器和六项 Python 回归。完整默认 Python 套件未重跑。
- 七项不变框架基线、CLI/viewer/MCP 60 工具真实冒烟及实际脚本进程、wheel/Ruff/21 项 JS 解析和独立前端/Python 审查通过。旧源码页签负对照如预期失败，页签顶部相差 30 像素。
- 11 项既有安装/查看器预算全部通过：3645 KB 包、163 MB 安装、42 个包、0.50 秒 CLI 首帧、标准/资产 60.2/60.1 FPS、2.90/5.00 ms 渲染、133 MB 峰值、6.00% 空闲 CPU（达到原限制）。包增加 1422 B，没有新依赖、纹理、几何或渲染通道。
- 迭代中保留了测试 harness 的 async-generator、未暂停零时刻前提、课程已分离 store 和绕过 open_model 的模型加载错误；最终用真实 MJCF 打开和原生选择地图完成验证。
- 报告 docs/bench/2026-10-23-creative-build-flow.md/.json，Git 外证据 ../robotforge-evidence/2026-10-23-creative-build-flow/。目录按 23 日起始命名，验收跨入上海时区 24 日，原始 UTC 时间保留。
- 本轮不改变材质或动作成功率；整体目标保持进行中。手机在放置中触摸会添加物体，需要先完成再拖动相机，这一交互仍是后续方向。
- 四轮加载/状态/资源对照 115 项检查、独立 UI 隐藏画面对照 42 项和媒体 31 项检查通过；六组 3D PNG 文件哈希完全一致，完整图片保留仅头部变化。相机数值最大差 8.53e-14，观察容差 1e-9；物理和图像仍严格一致。
- HTTP 编码资源增加 5211 B，仍为 28 次请求。展示新 profile 代理指标 327.9→335.9 ms，厨房 314.1→297.2 ms，公园缓存 206.8→221.35 ms；帧时有升有降，没有整体提速或节省内存的宣称。
- 两端前后四个球的位置、尺寸和朝向完全相同；道具选择/打开道具面板 4→1，增加了显式连续放置和完成两次操作，没有把自动化时长当作人类可用性研究。
- 五段正式 H264 实录完整解码并抽帧查看：前后桌面/手机各约 24 秒，实际驾驶推球 13.9 秒。检测到 219 条接触观察，小车移动 0.69833 m，球移动 0.45289 m；控制和物理参数不变。
- 保留了相机浮点逐字节比较过严、漏掉无名称碰撞几何的接触观察器和首帧 794/后续 800 高度导致的视频格式问题；最终视频从原始帧等比补边导出，没有裁剪或重演物理。
- 本轮统一验证记录 5d63c2be-a90b-49f0-879d-40cff03c9ad5 已完成，134 项不可变产物通过校验；主进程独立重验 8197 个原始文件（1,236,680,437 B），原始帧和失败尝试保留在 Git 外。


## Round 57 — resolution-stable surface relief (2026-10-24)

- Shared metric bump gradients retain height/position units for seven procedural material kinds and textured ground/terrain; source-texel central differences replace ground's screen-pixel forward difference. Colors, roughness settings, weave/cloth, geometry, physics, controls and interfaces remain unchanged.
- 63 nonflat real GPU plane cases bring 128/512 normal-strength ratios from about four to about one. All seven changed procedural kinds improve complete-shading resolution error; six ground styles have small absolute-image-error regressions. Actual full-scene changes are subtle; wood-floor flicker barely changes and is not claimed solved.
- All 91 browser cases, eight slow assets/sensors cases and 300 focused default cases pass (399 distinct, six new normal cases). Seven unchanged framework cases, actual CLI/MCP/script smoke and a fresh locked-wheel installation smoke pass. Wheel/Ruff/21 JS parser checks and independent source/Python/evidence reviews pass; full default suite not rerun.
- All 11 existing install/viewer budgets pass without relaxation: 3645 KB wheel, 163 MB install, 42 packages, 0.51 s CLI frame, 60.1/60.2 FPS, 2.20/3.50 ms render, 132 MB server peak and 5.60% idle CPU. Wheel +693 B; fresh HTTP +1610 B; still 28 requests and no new textures, passes or dependencies.
- Four serial frozen-world comparisons pass 87 source/scene/physics/resource/GL checks. Candidate first gallery/kitchen samples are slower (1123/733 ms); gallery near draw median rises 3.425 to 4.425 ms. These costs remain visible alongside the consistency benefit.
- Actual before/after camera-motion videos each decode 427 frames at 1280x800/30 FPS for 14.233 s, with paused physics. Literal checker raw-float equality fails by at most one float32 epsilon; complete PNG and physics controls stay exact, with separate accepted numerical controls recorded.
- Report: docs/bench/2026-10-24-surface-normals.md/.json; evidence: ../robotforge-evidence/2026-10-24-surface-normals/. Completed scoped record f82e3f76-a885-473b-a3f7-0f75e8695e40 verifies 51 artifacts and 4653 raw files (1,125,306,260 B). Runtime SHA256 3519db85aac71b55fdc983870e9ca57044339f07f0976b2b996b6ba672a4f605.
- Broad goal stays active. First-use shader cost, temporal behavior, broader visual art direction and creative/mobile editing remain follow-ups; individual action success tuning remains deferred.


## Round 58 — camera gestures while building (2026-10-24)

- Placement now commits a qualified primary tap on release while OrbitControls keeps the complete pointer stream. Drag, out-and-back motion, pinch with one finger remaining, mouse chords and cancellation cannot become accidental additions. Queued placement rechecks gesture epoch, busy state, creator session, scene version and runtime generation.
- Repeat size/yaw and latest-only undo remain; camera input still works during the 500 ms repeat guard. Held pointers cannot use Enter/Place here to submit old points. English/Chinese hints describe the camera gestures.
- Actual candidate recording exposed a genuine small-world visibility defect: the retained camera exceeded scene-sized fog, hiding a sole prop completely; Done also showed a robot-only empty card. Above-water fog now follows camera-target distance and contracts again, leaving underwater attenuation and clipping planes unchanged. Visible props, terrain, water and cloth suppress the empty card.
- All 101 browser, eight additional slow GPU assets/sensors and 300 focused default cases pass (409 distinct, ten new); seven unchanged framework cases, 11 existing budgets, actual CLI/MCP/script smoke and fresh locked-wheel smoke pass. Optional type/security tools are unavailable; complete default suite not rerun. Independent source/Python/evidence reviews pass.
- Four final serial frozen-world comparisons retain identical scene/state/resource counts and all nine PNGs. Fresh-profile timings are mixed: gallery 325.00 to 314.15 ms, kitchen 294.65 to 318.15 ms, park 358.45 to 354.00 ms; park warm rises 219.10 to 229.00 ms and gallery pooled draw 3.80 to 4.00 ms. No general speed claim.
- Wheel +1259 B, fresh HTTP +4030 B, warm +0 B, still 28 requests and no new dependencies/textures/passes. Existing budgets pass at 3646.49 KB wheel, 163.165 MB install, 42 packages, 0.5009 s CLI frame, 60.13/60.08 FPS and 2.30/5.00 ms standard/assets draw.
- Native touch-emulation videos fully decode at 390x844/30 FPS, before 7.800 s and after 8.033 s. Same inputs cause two unwanted old-source additions and zero candidate additions; two deliberate taps add two props, Done retains them and Undo keeps the first visibly rendered without reframing. Physics paused at 0.1 s; no new dynamic-task success result is claimed.
- Scoped record 89a93948-e255-44df-8010-40354db3b58f verifies 68 immutable artifacts and 4035 raw files (252907235 B), retaining superseded/failed attempts and harness/dependency snapshots. Report: docs/bench/2026-10-24-touch-build-camera.md/.json; evidence: ../robotforge-evidence/2026-10-24-touch-build-camera/. Runtime SHA256 a1d2c2ef95b4fe09e1c1b3abb0963fc1ebe841ae553d9d9be530f1abf1c4443e.
- Broad quality, lightweight experience and playability goal stays active. Physical phones, Windows, other drivers and extreme clipping-range zoom remain unverified; source guards for blur/unexpected capture loss/outside release have no separate new browser cases. Individual action success remains deferred.


## Round 59: irregular shared wood surfaces (2026-10-24)

- Shared floor/terrain wood retains its 256-square tile, eight-board layout and two texture channels, replacing regular comb bands with deterministic longitudinal figure/fibres and two-millimetre source-texel-covered joints.
- Object wood uses filtered elongated noise and the longest geometry-bounds axis; real MuJoCo tabletop principal-axis geometry selects Z rather than its thin X dimension. A shared uniform preserves shader-program reuse. Base RGBA, geometry, physics and interfaces remain.
- First previews looked flat/mottled; retained iterations add geometry-aware direction and separate mean-preserving colour contrast. V1 has only a runtime digest and preview evidence, while v2/v3 have exact source snapshots. Current runtime SHA256 04297ab0c66986dc47119a014a1f7f00834106014d5a7cfdfd046da2ea570d67.
- 104 browser, eight slow GPU assets/sensors and 300 focused cases pass (412 distinct). Three new GPU tests cover x/y/z wood directions. Seven unchanged framework cases, real CLI/viewer/MCP/script and fresh locked-wheel smoke, Ruff and JS parsing pass; full default suite not rerun, optional type/security tools unavailable.
- Four final serial comparisons pass 339 integrity/media/UV/diagnostic assertions; nonwood controls and three real concrete/grass regions are byte-exact. Physics and resource counts remain equal. Supplementary real stairs/ramp retain all 60 UV edge pitches. Literal camera comparisons differ below 1e-15 m; the low park whole image legitimately changes at visible wood, so its failed equality remains with exact nonwood crops.
- Raw fixed-screen reversal counts regress reproducibly, table view 946 to 1926 and floor-grazing 6915 to 19194. Isolation localizes most additional reversals to moving floor colour. Five-position actual-canvas 1x/2x comparisons improve bounded table/floor sampling errors, but neither this nor decoded contact sheets establishes alias-free motion or general temporal improvement.
- All 11 existing install/viewer budgets pass without relaxation: 3647.249 KB wheel, 163.1669 MB install, 42 packages, 0.5073 s CLI frame, about 60.1 FPS, 3.0/5.0 ms standard/assets draw and 5.4% idle CPU. Matched wheel +774 B, fresh HTTP +1959 B, warm bytes unchanged, still 28 requests. No new textures, passes or dependencies.
- Timings remain mixed: gallery near median 3.975 to 4.800 ms, kitchen low 1.850 to 2.400 ms, bench detail 3.250 to 2.450 ms; wood-world fresh proxy 186.80 to 220.65 ms. No overall speed or memory reduction claim.
- Actual Chrome before/after camera-motion exports fully decode at 1280x800/30 FPS, 18.233333 s/547 frames and 18.200000 s/546 frames. Physics is paused; actual dynamic-task success is not remeasured. Coordinator inspected matched PNGs, terrain views and decoded video frames; no dynamic stability proof is claimed.
- Scoped record 5eb51556-6fca-4114-8655-8ff05e99e1e5 verifies 132 immutable artifacts and 5019 raw files (728839704 B). Report docs/bench/2026-10-24-wood-surfaces.md/.json; raw evidence ../robotforge-evidence/2026-10-24-shared-surface-finish/.
- Broad visual quality, lightweight experience and playability goal remains active. Next work should return to free editing/exploration and shared visual improvements; original temporal regressions and physical-phone/Windows/other-driver gaps remain explicit. Individual manipulation success remains deferred.


## Round 60 draft: reversible existing-prop editing (2026-10-24)

- Existing saved sphere/box/cylinder/table/bin entries can be selected by exact name or explicit native scene tap, moved/90degree rotated, removed and restored by one-use Undo edit. Ordinary physics dragging remains; camera/input gestures reuse the placement flow. Identity/size/material remain; target-only historical pose/order is restored, unrelated poses/controls carried, velocities/time restart.
- Strict version/generation/file-byte and compiled-descriptor guards, saved/live occupancy, task-reference and support checks reject stale/unsafe edits. Fixed bin walls and anonymous visual meshes map to actual compiler ownership. Race/tilted-support/Escape fixes are covered; interfaces remain additive.
- Complete default3447pass/127skip, browser107pass and slow GPU8pass (3562 distinct), unchanged framework7, real source and offline fresh-wheel60-tool smoke, Ruff/format/JS parser/package equality and scoped independent reviews pass. Initial sensor test read world first/version4 old points while awaiting second/version7; test now waits actual world/version without weakening shown assertion. Initial direct source smoke omitted localhost bypass, timed out; explicit NO_PROXY run passes3.3s, freshwheel5.6s. All retained.
- Four serial A1/B1/A2/B2 fixed cohorts keep all17 fullcanvas views byte-exact and all rendering scene fields except new object_edit. Native desktop/touch workflows exercise cancel/move/rotate/remove/Undo/refusal/retry; native desktop W reaches39 contact samples and reset keeps new wall. Independent receipt supplements preserve7 raw request/reply pairs per device;1203 scoped checks and full uncapped MP4 decode pass. Phone is touch emulation, not touch driving or physical-device verification.
- Wheel+6873B, freshHTTP+8448B/warm+979B,28 requests; no new deps/textures/passes. Draw/navigation timing mixed; gallery near/low rise2.375/2.150 to3.300/3.350ms. Final standard/assets budgets show60.1409/60.0591FPS,3.0/5.0ms; install163.1909MB/42packages.
- HOLD MERGE: both full budgets fail idleCPU6.2% against unchanged6.0%, with other10checks passing. Original five-second A/B repeats5.8/6.4 baseline vs7.4/6.0 candidate are mixed. Fixed instrumented ABBA30second results6.9/6.9 vs7.133/6.833 do not identify continuous scene work or prove equivalent/passing cost. GC values are wall durations; counters are full child totals. No threshold, physics or GC-policy change; failed first diagnostic writer retained. Source/receipt acceptance is not performance approval.
- Failed scoped record89af1f39-7f9f-4d1d-b087-8d9233f1d6a1 verifies431 artifacts and11066 raw files(881429837B). Runtime0475f89353459a332be50f558f1b203ca2ccc329a1e9554a5eca3dc9e232d091. Report docs/bench/2026-10-24-object-editing.md/.json; raw ../robotforge-evidence/2026-10-24-object-editing/. Publish as draft and keep main unchanged; broad goal active, individual action success deferred.


## Round 61: remove unchanged viewer catalog copies (2026-10-24)

- A ten-second actual-server profile identifies repeated deep copies of the cached task directory before unchanged broadcasts are discarded. Compare the private catalog, detach only a changed outgoing message, and assign the prior-description snapshot only after successful copying. Public APIs, live registry/world-task/learned-capability refresh, welcomes and polling remain compatible; no simulation/controller/rendering change.
- 319 focused default +107 browser +8 additional slowGPU cases pass (434 distinct), unchanged framework7, actual source and fresh offline locked-wheel60-tool smoke, Ruff/format/package equality and scoped review pass. Three new regressions all fail on archived905 and pass here. Complete default suite not rerun; prior3447-pass run is historical. Optional type/security tools unavailable.
- Exactly one changed-runtime official install/viewer budget run passes11/11 at4.6% idleCPU against unchanged6.0%. Supplementary fixed ABBA baseline5.4/6.0 vs candidate6.2/5.8 stays mixed; candidate6.2 still exceeds6.0. No stableCPU reduction/upper-bound/all-performance-samples-green claim, no threshold/sampling/aggregation change, no repeats until pass. Prior905 failed record and both6.2 official failures stay unchanged.
- Wheel+101B to3741757B, install163.1911MB/42packages, firstframe0.5068s,60.1069/60.1989FPS atDPR2,3.3/3.1msdraw,132.2MBdogserverpeak. Only playground.py runtime bytes change; no frontend/dependency/texture/pass or physical changes. Profiles remove53487 recursive copies across9catalog calls, but instrumentation is not an uninstrumented speed measurement.
- Four frozen scenes retain17byte-identical PNGs, exact payload/physics/resources. Replays retain7same-connection raw edit request/reply pairs each;197receipt checks and195additional summary properties pass. Desktop39actual barrier-contact samples plus saved-layout reset; uncapped desktop24.0s/720frames and touch-emulation14.966667s/449frames fully decode. Coordinator inspects decoded contact sheets, not everyframe. Four observed camera arrays differ<=8.882e-16; first exact-float summarizer failure retained, reconstructed camera-only1e-12 tolerance documented while prescribed inputs, images, payload and physics stay exact.
- Completed scoped recordce300e01-8a69-4c16-894a-22d9bd38b803 verifies425immutable artifacts and4500indexed rawfiles(401195174B). Runtimeee36b9a8de315fb9debe8fbfb8b070a59bc0720fec145796cfb0d3558ffc4799. Report docs/bench/2026-10-24-viewer-idle-cost.md/.json, raw ../robotforge-evidence/2026-10-24-idle-cost/. This scoped completion does not complete the broad goal. Shared material motion quality and scene creation/exploration remain next; individual action-success tuning deferred.

## Round 62: bounded native model pictures (2026-10-24)

- Shared visual ceiling: vertex-only texture sampling erased interior graphics and assigned one image per mesh. Optional authored UVs/native RGB/RGBA image bundles now recover markings, ports, bolts and rotor detail across imported files, library robots and compositions; each geom gets its own material. Old payload/defaults/MCP/script interfaces stay compatible, and physics/templates/terrain/controllers remain unchanged.
- Deduplicate original images before a shared1MiB baseRGBA/512edge/64image budget; twoX2s share one512square image. ANYmal17unique images share124square resolution (1045568decodedB). Gzip is cached once/version and frontend resources load only on native scenes. Mipmaps and browser/driver costs are extra, not capped VRAM. Global legacy sRGB preserves the established palette; exact rejected-linear source/frames and separately labelled diagnostic remain retained. Explicit-linear parity is not claimed.
- Frozen runtime8f26dfb7fcf4630ef4a2c7460a399d947e37d050d3ab72546b526f40a91c21db passes486 focused default+113 browser+8additional slowGPU cases=607distinct,7unchanged framework cases, actual source and fresh offline locked-wheel60-tool smoke, Ruff/format/22JS parser/package-source equality and independent code/Python/evidence/product reviews. Complete default suite not rerun; optional type/security tools unavailable. All559 frozen inputs remain unchanged.
- Exactly one unchanged install/viewer budget run passes11/11: wheel3745663B (+3906B), install163.2012MB/42packages, download52.5149MB, CLI firstframe0.4960s,60.253/60.085FPS atDPR2,3.3/5.0msdraw, dogserver133.4MBpeak and5.8% idleCPU. No budget or sample-rule change, no repeats until pass, no stableCPU upper-bound claim; prior failed records stay failed.
- Fixed native A1/B1/A2/B2 retains the tradeoff: ANYmal fresh211.5/234.3→334.0/365.8ms, payload21.9/22.3→155.7/157.2ms. X2/composition differences are smaller; frame medians mixed and RAF about60FPS atDPR1. Paired Chrome process-treeRSS+4–26MiB is not VRAM. Native freshHTTP+192091/+971485/+253387B,30vs28requests; UV data is included. No general speedup claim.
- 348 source/fixture/scene/geometry/physics/resource checks pass.17 untextured full-canvas PNGs plus complete scene/physics/GPU resources match accepted main exactly, still28requests and no new image/module loads; shared app itself grows1213rawB. Actual fixed-world before/after camera videos24.9s/747frames each and all six8.3s/249frame originals fully decode. Sampled image/decoded-contact review sees clear nearby gains, smaller distant gains; paused-camera videos do not validate locomotion/manipulation or temporal stability.
- Completed scoped record2251df29-650a-4790-9db0-048bad7bec30 verifies301immutable artifacts and4387indexed rawfiles(566662216B), including cached licensed assets, source/harness snapshots, failed fixture/helper/proxy attempts, rejected color implementation, raw frames and final media. Report docs/bench/2026-10-24-native-model-textures.md/.json; evidence ../robotforge-evidence/2026-10-24-native-model-textures/. Broad product goal remains active.
- Next seed: durable named design-layout checkpoints and transactional restore for the real journey build→save→explore another map→restore→drive. Keep asset reference bases, validate before replacing the current layout, guard stale requests, and distinguish authored design from live velocities/time and pruned Undo history. Single-action success tuning, releases and paidActions remain deferred.


## Round 63: durable named creative layouts (2026-10-24)

- Maps now offers bilingual Save current/Refresh/Restore/confirmedRemove. Each original world keeps16 immutable authored checkpoints, separate from50-entry transientUndo; no live physics state or external asset contents are archived.
- Fixed original asset base and bounded records, strict schema/name/hash/version/generation checks, compile-before-publication and explicit built-in idle runtime participants protect normal failed restoration. Unknown/custom callbacks and active irreversible work are refused; postcommit cleanup warns, rollbackIO failures pause/invalidate guarded editing. Multi-file crash atomicity is not claimed.
- Fixed first-pose/list-generation/mutation-reply races, all covered in realChrome. 735 scoped default pass/1skip,120 browser,8 slow=863distinct;7 unchanged framework cases, real source/fresh offline locked-wheel60tool smoke, Ruff/format/22JS parses, exact package/source equality and independent reviews pass. Complete default suite not rerun; optional type/security tools unavailable.
- Exactly one unchanged full install/viewer budget passes11/11: wheel3756943B (+11280B), install163.2361MB/42packages, unchanged52.5149MB dependencies, CLI firstframe0.4978s,60.099/60.061FPS atDPR2,2.5/3.7ms draw,133.0MBdogserverpeak,5.2% idleCPU. No threshold/sample/aggregation changes or stableCPU/general speedup claim.
-17 fixed PNGs/fullscene/physics/GPUresource controls exactly match accepted main, still28HTTP requests; fresh/warm frontend bytes+12557/+1278. Retained previous-round timing observations are mixed and not simultaneousABBA.
- Actual native pin building→save→map→restore→reload→keyboarddrive→remove succeeds in desktopEnglish and touch-emulated Chinese phone viewport, driving0.511/0.510m. Full baseline4.60s/138frame, desktop8.67s/260frame and phone8.50s/255frame videos fully decode; six decoded stills pervideo inspected. Separate interpreters preserve identical recordID/hash across exit/reopen, restore and drive0.570m.
-110 checks and564 frozen inputs; runtime6ba2cec5161cc5afdb05dce2addc0a6a471071a793906da8b59eded0dcdfbaf7. Report docs/bench/2026-10-24-saved-layouts.md/.json; evidence/failedattempts/source+helper snapshots/index/ExperimentStore record under ../robotforge-evidence/2026-10-24-saved-layouts/. No new dependencies/assets/renderpasses, releases or paidActions. Broad goal remains active.
- Next seed: reuse configured rigid props through selection/placement and saved course variations, alongside shared visual quality. No physicalphone/Windows/everyplugin-model-task or temporal-quality certification; individual manipulation-success tuning stays deferred.


## Round64: reuse configured props in native creative play (2026-10-24)

- Build selection adds bilingual Duplicate, copying full server-owned authored properties into bounded unique names. Existing placement supports90degree rotation, source stacking, guarded repeat/fresh-point behavior, inverse, saved variants and strict version/generation/world digests. Source task/support references stay untouched; existing move/public interfaces remain compatible.
-40 new Python+8Chrome cases;775 scoped default pass/1skip,128fullbrowser+8slow=911distinct.7 unchanged framework cases, real source/fresh offline locked-wheel60tool smoke, Ruff/format/22JS parses, package/source equality and independent source/evidence review pass. Complete default suite not rerun; optional type/security tools unavailable.566 frozen files remain unchanged at runtimeb6a2e8891de433c55e899b0ce9cf0886bdbc9a1c9848322724b8af3c03d347f6.
- Exactly one unchanged11/11install/viewer budget pass: wheel3757720B (+777B), install163.2393MB/42packages, download52.5149MB, CLI firstframe0.4921s,60.105/60.207FPS DPR2,3.0/4.9msdraw,132.8MBdogserverpeak and5.4%idleCPU. No thresholds/sample/aggregation changes, repeats until green or general speedup/stableCPU claim.
-17 fixed PNGs/fullscene/physics/GPUresources equal retained main;28HTTP requests remain, fresh/warm bytes+1777/+97. Retained paired timings mixed, not simultaneousABBA. Copies add geometry/physics work; no new dependency, packaged asset, texture request or renderpass.
- Actual desktopEnglish/Chinese touch-emulated phone journeys copy/repeat/rotate/undo/save/map/restore/reload, drive and contact the exact copied wall at5distinct positive-force times, then native pause/reset keeps the layout and resets the rover. Videos baseline10.97s/329frames, desktop17.80s/534frames, phone17.93s/538frames fully decode; six stills each inspected. Restored drive0.515/0.514m; fresh interpreter reopens actual recorded saved bytes, restores and drives0.570m. Earlier close/narrow framing recordings and analyzer-action-name mistakes retained; only QA camera changed for clearer contact views.
-150 evidence checks pass. Report docs/bench/2026-10-24-prop-duplication.md/.json; editable source/helper snapshots, raw frames/receipts/index and ExperimentStore record under ../robotforge-evidence/2026-10-24-prop-duplication/.
- Existing edit publication boundary remains, not a new crash-atomic transaction. Copies preserve authored settings, not live tilt/velocity/task bindings; successful recompile resets velocities/time. No physicalphone/Windows/allplugin-model-task or temporal-quality certification. Broad goal stays active; next priority shared material motion stability and indoor/outdoor visual quality. Individual action-success tuning, releases and paidActions remain deferred.


## Round65: bounded shared-surface investigation (2026-10-24)

- Real fixed scenes and diagnostic controls rejected roughness-variance adjustment, normal-only box edges and contactAO: no convincing bench/table gain and dirty indoor halos/additional GPU cost. No prototype entered production.
- Nineteen bounded checks and six exact white-factor controls support rejection only, not production-quality approval. Raw evidence ../robotforge-evidence/2026-10-24-surface-motion/investigation-conclusion.json; broad goal stays active.

## Round66: touch exploration through on-screen driving (2026-10-24)

- Shared-capability native direction buttons enable phone driving without keyboard/gamepad; desktop opt-in, independent multi-pointer holds and existing input priority. Server uses bounded wall-clock latest-owner holds with scene/generation protection; old clients remain compatible. No individual manipulation-success tuning.
-1000 scoped default pass/1skip+151fullbrowser+8slowGPU=1159distinct,7unchanged framework, actual source/fresh offline locked-wheel60tool smoke, Ruff/format/23JS parses/392package-source files and independent reviews pass.570 frozen inputs unchanged. Full default suite not rerun; optional type/security tools unavailable.
- Exactly one unchanged11/11install/viewer budget pass:3763055Bwheel(+5335B),163.2564MB/42packages,0.4909sCLI firstframe,60.097/60.1269FPS DPR2,2.9/4.0msdraw,132.5MBdogserverpeak/5.4%idleCPU. No thresholds/sampling changes or general speedup/stableCPU claim.
-17 actual fixed-scene PNGs/scene/physics/GPUresources match main. Single sequential before/after timings mixed; fresh HTTP+12057B/warm+2212B,29vs28requests. Process-treeRSS is not VRAM. No new dependencies/images/shaders/textures/passes.
- Actual native pointer desktop and Chinese touch-emulated phone recordings drive into the exact wall with5distinct positive-force contacts, reverse/turn/release/pause/reset. Baseline3.20s/96frames, desktop7.53s/226frames, phone7.57s/227frames fully decode; six stills each inspected. Baseline shows absent screen controls, not equivalent task-time comparison; physicalphone/Windows unverified.
- Actual WebSocket/normal wall-clock probe verifies nonrenewal, latest-owner disconnect, paused expiry, old-owner preservation and legacy behavior.119 checks pass; runtime04e265220deedbac623587b7610a34b7b102fa760b79cce9e4bd1ab296505779. Report docs/bench/2026-10-24-screen-driving.md/.json, raw/archives/ExperimentStore under ../robotforge-evidence/2026-10-24-screen-driving/. Broad goal remains active; next shared material quality and creative exploration, no release/paidActions.


## Round67: correct shared rounded-box surface normals (2026-10-24)

- Analytical noise-gradient shader screen missed its predefined transition-error threshold and was rejected; no shader changed. Blanket new bevel geometry rejected because existing furniture already rounded. Failed CC0 webpage lookup added no asset.
- Shared rounded_box now uses parameter tangents instead of area-weighted bevel normals, correcting3–23degree selected flat/tangent deviations. Radius/center clamps, quantized smoothing keys, singular fallback and cached immutability handled; vertices/faces/keys/caps and physics unchanged. Actual kitchen/bench/table gains are local and whole-scene change modest; no floor-color/shimmer fix claim.
- 1232focused default pass/1skip+151complete browser+8slowGPU=1391distinct,32new tests;7framework,17actual physics cases/392identical sampled hashes, source/fresh offline locked-wheel60tool smoke, Ruff/format/23JS parses/392package-source files and independent reviews pass.571frozen inputs; complete default suite not rerun, optional type/security tools unavailable.
- 4worlds×480compiled arrays change only mesh_normal; other arrays/scalars, physical state and expanded payload positions/indices/UV/colors/hfields exact.17scene/camera/resource controls unchanged and actual floor-detail PNG exact. Final production captures separate from external prototype.
- Exactly one unchanged11/11official install/viewer budget pass:3763628Bwheel(+573B),163.2582MB/42packages,0.5008sfirstframe,60.1269/60.1669FPS DPR2,2.8/5.0msdraw,132.6MBdogserverpeak/5.2%idleCPU.29requests/HTTPbytes unchanged. Final-then-baseline timing/RSS mixed, not ABBA or general performance/VRAM claim; other groups not run.
- Real paused bench/table camera before/after and independent native rover contact/reverse/turn/release/pause/reset videos retained.9originals fully decode, six stills each inspected; generic stairs/initial focused-camera teleport clips kept, final bench matching keyframe fixed only QA setup. No physicalphone/Windows/everyframe visual certification.
- Original physics wrapper exit1 incorrectly rejected legitimate placed suffix; raw files byte-identical. Corrected known-protocol validation reuses all392hashes without physics rerun or sample filtering; original logs remain. Finite-difference cancellation and helper versions documented honestly.
- 139assertions; runtimeeab1c5dd1335f11d71c0d3a1e6a5bbb355fb1b730c57e61af553d83974f43b0c. Reports docs/bench/2026-10-24-rounded-normals.md/.json, source/helper/fixture/model/payload/media/archive/ExperimentStore evidence ../robotforge-evidence/2026-10-24-rounded-normals/. Broad goal active; next shared material detail and wood-floor motion quality, no action-success tuning, new dependency/texture/pass, release or paidActions.
