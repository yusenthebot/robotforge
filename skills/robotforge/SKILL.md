---
name: robotforge
description: Use for robots and embodied AI with the robotforge MCP tools (a MuJoCo playground for beginners) to build, load or import (URDF) robots, make them walk, drive, fly or grasp, compare controllers, write robot scripts and teach lessons. 机器人仿真，具身智能入门。
---

# robotforge

robotforge is a robot playground on the MuJoCo physics engine, made for beginners.
The user describes a robot and a goal in plain words; you build it with the robotforge MCP tools, make it move, check the result and explain the idea behind each step.
The tools come from the `robotforge` MCP server: Claude Code names them `mcp__plugin_robotforge_robotforge__<tool>`, Codex `mcp__robotforge__<tool>`; search your tool list for `robotforge`.
Clients that load MCP tools on demand (recent Codex) show them only through their tool search: search it for `robotforge` before deciding they are missing.
The very first start installs the server's Python packages and can take a minute, so if the tools are not there yet, wait and look again before anything else.
Do not work around missing tools by importing robotforge's Python modules yourself: nothing would show in the user's viewer. Follow [references/troubleshooting.md](references/troubleshooting.md) instead.

## Workflow

1. **Understand the wish.** If the request is clear, act with sensible defaults. Ask one short question only when a choice really matters (which robot, what it should do). "Teach me" or "lesson N" means the learning path: call `lessons`.
2. **World.** `new_world(name, floor)` creates a project; `open_world` reopens one.
   - Ready-made scenes (`list_scenes`, `load_scene(preset)`): desk_lab, kitchen, factory, warehouse, park, maze, obstacle_course, and with water harbor, lake, pool. Robots already in the world move to the scene's places (boats and ROVs onto the water).
   - Ground and terrain: `set_floor(style)` (checker, grass, wood, concrete, dirt, sand, rock, snow, tile); `set_terrain(type)` for the whole ground or `add_terrain(type, pos)` for one feature (hills, rough, stairs, ramp, stones, curb, platform).
   - Water: `set_floor(water=1.2)` fills the world 1.2 m deep (or `water={"level": 1.2, "waves": 0.03}`); higher terrain is dry land. Light things float, heavy ones sink; `add_object(shape="buoy")` is a moored buoy, `shape="dock"` a wooden dock.
3. **Robot.** Pick the source that fits:
   - template (`list_templates`): arm, quadruped, humanoid, rover, cartpole, boat, rov (an underwater robot), drone (4, 6 or 8 rotors), tuned with `params` (sizes, servo, color);
   - library (`list_library`): real robots such as `go2`, `panda` or the `skydio_x2` and `crazyflie` drones, downloaded on first use (4-55 MB); `mounts` puts one on another, an arm on a dog's back (`add_robot(name="go2", library="go2", mounts=[{"library": "piper"}])`, then the arm is `go2/arm`, and a walking dog stops before it works) or a gripper on an arm's flange (`ur5e` with `robotiq_2f85`, also on a dog: `anymal_c` with both); each walker carries only what `list_library` says (`carries`);
   - file: `add_robot(file=...)` imports URDF, xacro, MJCF or a mesh and returns the import doctor's report;
   - parts (`builder_guide`, then `add_part` / `update_part` / `remove_part`): any robot from bodies, legs, wheels, rotors, arms, grippers and sensors (hexapods, snakes, self-balancers, an arm on a rover, a hexacopter);
   - design (`list_templates` lists them): a ready-made unusual robot from the gallery, `add_robot(name, design="centaur")` (wheel-legged biped, Mars rover, centaur, scorpion, a Go2 carrying a PiPER arm, tentacle...), saved as parts so `update_part` changes it;
   - nothing fits: write a small URDF and import it (see recipes).
   - combining: `mount(what, on)` puts anything on anything in one call, on an adapter plate: an arm on a base (`mount("ur5e", on="cart")`), an end effector on that arm's flange (`mount("robotiq_2f85", on="cart/arm")`, also `panda_hand` or the dexterous `leap_hand` and `allegro_hand`, whose fingers close together as the gripper; another one swaps it, a quick change), a sensor on its wrist (`mount("lidar", on="cart/arm")`); `at` names a body, `shift` and `rotate` place it. Read `checks`: `mass_kg`, `stands.margin_m` (how far the centre of mass is inside what it stands on), each arm's `reach_m`, `reaches_floor`, `self_collision`, `holds`, `tips_robot`. A robot that would tip or fall is refused with what to change. `save_as="name"` keeps the combination; `add_robot(design="name")` or `mount("name", on=...)` uses it again; `remove("cart/arm")` takes a mount off.
   Add props with `add_object` (box, sphere, cylinder, table, bin; `material="steel"` for what a magnet holds).
   Things that move on a joint of their own are objects too: drawer, cabinet (a door), button, switch, lever, valve, conveyor (`speed` in m/s), and soft ones: rope, cloth.
   - End effectors: any arm takes `end_effector="suction"` (a suction cup: flat things fingers cannot get round, a tile, a plate) or `"magnet"` (steel only: `material="steel"` things, and the steel bars of drawers and doors) in `add_robot`, or later in `update_robot` (a tool change; `"gripper"` goes back to its fingers). `gripper(robot, "close")` turns a cup or magnet on, `"open"` off; `pick_and_place` uses them as it uses fingers.
   Sensors go on any robot's body with `add_sensor` (camera rgb, depth or segmentation, 2D or 3D lidar, IMU, encoders, force_torque at a wrist, touch arrays on feet, rangefinder, GPS, odometry; rate, noise, latency in `options`); `read_sensors` reads them, `remove("robot/sensor")` takes one off.
   The viewer draws what added sensors sense: lidar rays and points, a camera's picture beside the view, IMU, encoder and force plots, touch pads and force arrows; `open_viewer(show=["go2/front_cam"])` also draws one the robot came with and brings a camera's picture up.
4. **Viewer early.** Call `open_viewer()` as soon as a robot exists and tell the user what they can do: drag the robot, drive with W A S D, and use the panels.
   The top left "Maps, terrain, modes" panel switches scenes, ground and terrain, and how each robot is controlled (keyboard gait, learned walk, MPC, LQR, IK...); the Robots panel has the joint sliders; Learn opens hands-on lessons.
   While the viewer is open, actions play out in real time.
5. **Make it move**, with the highest-level tool that fits:
   - `move` / `stop` / `pose`: walk a quadruped or humanoid, drive a rover or a boat, humanoid poses; a boat's `move(to="dock")` plans round buoys and the shore; an ROV's `move(to=[x, y, z])` dives to that point and holds depth and heading (z is height above the floor like every z, depth = water level - z: 0.5 m under a surface at 1.2 m is z = 0.7; the result's `depth_m` is the depth reached; a point on land or under the bottom gets `why` with the z range that works; its `depth` sensor reads meters under the surface);
   - aircraft (kind `aircraft` in `robots`): `move(robot, to=[x, y, z])` takes off and flies there, then hovers; a list of points flies a path, e.g. a 3 m square at 2 m: `to=[[3,0,2],[3,3,2],[0,3,2],[0,0,2]]`; a last point with z 0 lands there (one call can take off, fly the path and land); z is a height above the world's zero, not above the ground under it, so over a hill or a table check `clearance_m` in the result; a point under a tree's crown or a table hovers there (z at the table's height lands on it); a landing where something stands (a ball, a crate, a robot), on top of what moves or a crown, or at an edge is refused with a spot nearby on the same surface that has room; legs go over or round what is in the way; `stop` hovers in place; `simulate(seconds, wind=[x, y])` blows a gust to show how it holds its place; the viewer flies it with W and S (forward, back), A and D (sideways), the arrow keys (turn), E (up) and Q (down); dragged with the mouse it holds where it is let go; `compare_controllers(drone, task="hover", push_at=3)` compares its own flight controller (`cascade`: position, attitude, body rates) with `lqr` and `mpc`, and `run_controller` flies it with one of them until `stop`, which hands it back to the flight controller;
   - `move_arm` / `gripper` / `pick_and_place`: arms; with an arm on a robot that drives or walks (`mobile_panda`, `lift_picker`, `go2_arm`), `pick_and_place` fetches: the robot goes over, picks the thing up, carries it and puts it down ("bring the apple to the table"), and says what it did or why it cannot before it moves; `pick_and_place(robot, object, "drawer")` puts things in an open drawer;
   - `operate(robot, target, action, amount)`: an arm opens or closes a drawer (amount in m) or a cabinet door (degrees), presses a button, flips a switch or a lever, turns a valve (open, close or turn, degrees); it holds the handle and follows the joint's own line or arc. Read `success` and `notes`: a note says when the arm could follow only part of the way (move the thing nearer) or did not take hold (move it a little and try again);
   - a conveyor runs by itself: `set_controls({"<conveyor>/speed": 0})` stops it, any other value (m/s) runs it;
   - `run_controller` / `compare_controllers`: a named algorithm on a task (reach, balance, stand, walk, drive_to), see `list_algorithms`;
   - `set_controls` + `simulate`: raw actuator targets;
   - `write_script` + `run_script`: custom sequences, loops and logic the user can keep;
   - learned walkers: `list_walkers`, then `move(robot, speed, walker="rl")` walks with a policy trained by reinforcement learning (a Go2 walk ships with robotforge; `walker="gait"` goes back);
   - a robot that no gait code knows, built from parts or imported from a URDF or MJCF file (`move` says so), can learn a walk of its own in the viewer's RL II lesson ("Keep this walk for my robot") if position servos move its legs; `robots` then lists its `learned_walk` and `move` uses it: forward at its own pace (holding its heading and the line it started on once that is learned, `holds_heading`), and once its turns are learned (they follow keeping the walk) turning on the spot: `turn_deg_s` is the slower side's rate from standing, `turn_deg_s_from_walk` each side's average in a turn's first second when the turn follows 1.5 s or more of walking with no pause (viewer closed, or walking with `seconds=null` and then the turn) that did not start right out of a turn; single turns vary (most within a quarter of it, a few far off), so read `turned_deg` from `move` and correct;
   - an MJCF file whose legs have torque motors (most Menagerie quadrupeds) cannot learn one (`move` says why); the same robot from the library (`add_robot(library=...)`) walks with its gait;
   - when the user asks such a robot (it has a `walk_hint`) to walk, say so before anything else and offer that lesson: about ten seconds of learning, and the walk is kept; a hand-written gait script is the fallback if they decline;
   - `train_policy` when the tool list has it: reinforcement learning (optional `rl` extra); `unilab_check` / `unilab_train` for the UniLab backend.
   - a video to keep or share: `move(..., record=true)` or `run_script(name, record="show.gif")` saves the motion, playing in real time (record=true: a GIF; a file name picks `.gif` or `.webp`, and `.mp4` is saved as a GIF) in the world's `videos` folder; the result's `video.path` says where, to copy it from. No browser is needed.
6. **Check, then explain.** Verify with numbers and an image (next section), explain what happened and why, offer a next step.

## Conventions

- Names are lowercase with underscores: worlds, robots, objects, scripts.
- A robot's parts carry its name as a prefix: actuators, joints, sensors and cameras are `<robot>/<name>` (`dog/front_left_knee`, `rover/front_cam`). Read them from `describe_model` or the add_robot result; never guess.
- Units: meters, kilograms, seconds; joint angles in radians; yaw and heading in degrees, counter-clockwise from +x; speed in m/s; turn in rad/s, positive is left.
- `pos` is [x, y] on the floor (height is automatic) or [x, y, z] for objects. z is up; a robot at yaw 0 faces +x.
- A world is a folder, `~/robotforge/worlds/<name>/` (or under `$ROBOTFORGE_HOME`), with `world.yaml` and `behaviors/*.py`. `show_world` prints it; `undo` reverts the last edit.

## Check your own work

Never say something worked because the call returned. Read the numbers, and look at an image for anything visual.

| After | Check |
|---|---|
| `add_robot`, `add_object` | it is listed; `observe()` shows it where expected; for imports read `doctor` (fixed, unfixed, notes) |
| `move` | `distance_m`, `turned_deg`, `to.upright`, any `note` (why it fell short); with `seconds` it has stopped already (`stopped: true`, no `stop` needed) |
| `move` with `to`, an aircraft | `arrived`, `to.position` against the target, `max_tracking_error_m`; landing: `landed` and `touchdown_m_s` (below about 0.4 is gentle); `robots` shows its `flight.state` and `altitude_m` |
| `pose` | `upright` |
| `move_arm` | `tool_position` against the target (within about 3 cm) |
| `pick_and_place` | `success`, `object_position` |
| `operate` | `success`, `moved` (m or degrees), `state`, any `notes` (how far the arm could follow); `observe()` lists every mechanism under `mechanisms` |
| `run_controller` | `success`, `failed_at`, `reward`, `stopped_early`; `keeps_running` means call `stop(robot)` later |
| `compare_controllers` | the `table` and `best`; explain with its numbers |
| `run_script` | `ok`, `output`, traceback |
| `add_sensor` | `sensor` name, `position_m`, the first `reading`; then `read_sensors` (lidar `ahead` and `nearest` with `bearing_deg`, depth `nearest` distance; `center_m` is depth along the axis) |
| `set_controls` + `simulate` | `observe()` joints against controls; any `warning` |

`observe(view=...)` takes default, front, back, left, right, top, a robot's name (`view="dog"`: the default view framed on that robot), or a robot camera such as `dog/front_cam`.
`observe(view, find="red")` (a color name, a hex code, or "all") lists what that view sees of the color: each find's world `position`, the `object` name `pick_and_place` takes, and `about_cm`, how far off the position may be.
A flag on a find (cut_off, hidden, side_on, raised, held_by) means it may be much further off; the result's `how` says what each one it carries means.
A lidar (a sensor on a robot built from parts) comes whole in `observe()`'s `lidars`: `nearest_m`, `nearest_bearing_deg` from the robot's front (positive is left), `nearest_side`, the nearest hit on each side, and every ray in `ranges_m` (null: nothing within range).
A lidar or rangefinder sees past its own robot's parts (a column, the wheels), so its nearest hit is something else.
Without OpenGL it returns the state with a note instead of an image: rely on the numbers and ask the user to look at the viewer.
When something fails, say so plainly, read the error (it says what to do), fix it and check again.
Report real numbers, including disappointing ones.

## Teaching

- Answer in the user's language. For Chinese users, give the English term once in parentheses (逆运动学 inverse kinematics).
- Before each step, one or two sentences: what you will do and the idea behind it. One new concept at a time, with a plain analogy.
- Invite the user to try things by hand (sliders, dragging, W A S D) and to change one number and see what happens.
- Write code the user can read, with comments. Keep messages short and end with a concrete next step.
- More in [references/teaching.md](references/teaching.md).

## Lessons

`lessons()` lists nine beginner lessons; `lessons(number)` returns one lesson's plan: ideas to teach, tool steps, checks and a closing challenge.
Follow the plan and adapt it to the learner. MCP clients with prompt support also list them as prompts `lesson_1` to `lesson_9`, plus `start`.

The viewer has its own hands-on lessons behind the Learn button (学习): PID control, inverse kinematics, motion planning, sensors, finding an object with a camera, gaits, balance and reinforcement learning (a cart-pole, then any body learning to walk, the user's own robot included), real-robot face-offs (our arm next to the AgileX PiPER and Franka Panda, our humanoid next to the Unitree G1), plus challenges with stars (maze escape, shove survival, obstacle course).
Each is a guided experiment with live plots, quizzes and checks that run on the physics, and it works without you.
When a beginner wants to understand how robots work, open the viewer and suggest one; afterwards, offer to build on it in their own world.

## References

- [references/recipes.md](references/recipes.md): walk a dog, pick and place, compare LQR and MPC, write a script, import a URDF, build a new robot, train a policy.
- [references/scripts.md](references/scripts.md): the script API and a complete closed-loop example.
- [references/troubleshooting.md](references/troubleshooting.md): common mistakes and how to recover.
- [references/teaching.md](references/teaching.md): tone, short explanations of core ideas, how a teaching turn flows.
