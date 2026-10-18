---
name: robotforge
description: Use for robots and embodied AI with the robotforge MCP tools (a MuJoCo playground for beginners) to build, load or import (URDF) robots, make them walk, drive or grasp, compare controllers, write robot scripts and teach lessons. 机器人仿真，具身智能入门。
---

# robotforge

robotforge is a robot playground on the MuJoCo physics engine, made for beginners.
The user describes a robot and a goal in plain words; you build it with the robotforge MCP tools, make it move, check the result and explain the idea behind each step.
The tools come from the `robotforge` MCP server: Claude Code names them `mcp__plugin_robotforge_robotforge__<tool>`, Codex `mcp__robotforge__<tool>`; search your tool list for `robotforge`.
The very first start installs the server's Python packages and can take a minute, so if the tools are not there yet, wait and look again before anything else.
Do not work around missing tools by importing robotforge's Python modules yourself: nothing would show in the user's viewer. Follow [references/troubleshooting.md](references/troubleshooting.md) instead.

## Workflow

1. **Understand the wish.** If the request is clear, act with sensible defaults. Ask one short question only when a choice really matters (which robot, what it should do). "Teach me" or "lesson N" means the learning path: call `lessons`.
2. **World.** `new_world(name, floor)` creates a project; `open_world` reopens one.
   - Ready-made scenes (`list_scenes`, `load_scene(preset)`): desk_lab, kitchen, warehouse, park, maze, obstacle_course, and with water harbor, lake, pool. Robots already in the world move to the scene's places (boats and ROVs onto the water).
   - Ground and terrain: `set_floor(style)` (checker, grass, wood, concrete, dirt, sand, rock, snow, tile); `set_terrain(type)` for the whole ground or `add_terrain(type, pos)` for one feature (hills, rough, stairs, ramp, stones, curb, platform).
   - Water: `set_floor(water=1.2)` fills the world 1.2 m deep (or `water={"level": 1.2, "waves": 0.03}`); higher terrain is dry land. Light things float, heavy ones sink; `add_object(shape="buoy")` is a moored buoy, `shape="dock"` a wooden dock.
3. **Robot.** Pick the source that fits:
   - template (`list_templates`): arm, quadruped, humanoid, rover, cartpole, boat, rov (an underwater robot), tuned with `params` (sizes, servo, color);
   - library (`list_library`): real robots such as `go2` or `panda`, downloaded on first use (4-55 MB);
   - file: `add_robot(file=...)` imports URDF, xacro, MJCF or a mesh and returns the import doctor's report;
   - parts (`builder_guide`, then `add_part` / `update_part` / `remove_part`): any robot from bodies, legs, wheels, arms, grippers and sensors (hexapods, snakes, self-balancers, an arm on a rover);
   - design (`list_templates` lists them): a ready-made unusual robot from the gallery, `add_robot(name, design="centaur")` (wheel-legged biped, Mars rover, centaur, scorpion, a Go2 carrying a PiPER arm, tentacle...), saved as parts so `update_part` changes it;
   - nothing fits: write a small URDF and import it (see recipes).
   Add props with `add_object` (box, sphere, cylinder, table, bin).
4. **Viewer early.** Call `open_viewer()` as soon as a robot exists and tell the user what they can do: drag the robot, drive with W A S D, and use the panels.
   The top left "Maps, terrain, modes" panel switches scenes, ground and terrain, and how each robot is controlled (keyboard gait, learned walk, MPC, LQR, IK...); the Robots panel has the joint sliders; Learn opens hands-on lessons.
   While the viewer is open, actions play out in real time.
5. **Make it move**, with the highest-level tool that fits:
   - `move` / `stop` / `pose`: walk a quadruped or humanoid, drive a rover or a boat, humanoid poses; a boat's `move(to="dock")` plans round buoys and the shore; an ROV's `move(to=[x, y, z])` dives to that point and holds depth and heading (z is height above the floor: 0.5 m under a surface at 1.2 m is z = 0.7; the result's `depth_m` is the depth reached);
   - `move_arm` / `gripper` / `pick_and_place`: arms;
   - `run_controller` / `compare_controllers`: a named algorithm on a task (reach, balance, stand, walk, drive_to), see `list_algorithms`;
   - `set_controls` + `simulate`: raw actuator targets;
   - `write_script` + `run_script`: custom sequences, loops and logic the user can keep;
   - learned walkers: `list_walkers`, then `move(robot, speed, walker="rl")` walks with a policy trained by reinforcement learning (a Go2 walk ships with robotforge; `walker="gait"` goes back);
   - a robot that no gait code knows, built from parts or imported from a URDF (`move` says so), can learn a walk of its own in the viewer's RL II lesson ("Keep this walk for my robot"); `robots` then lists its `learned_walk` and `move` uses it: forward at its own pace, and once its turns are learned (they follow keeping the walk) turning on the spot;
   - when the user asks such a robot (it has a `walk_hint`) to walk, say so before anything else and offer that lesson: about ten seconds of learning, and the walk is kept; a hand-written gait script is the fallback if they decline;
   - `train_policy` when the tool list has it: reinforcement learning (optional `rl` extra); `unilab_check` / `unilab_train` for the UniLab backend.
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
| `pose` | `upright` |
| `move_arm` | `tool_position` against the target (within about 3 cm) |
| `pick_and_place` | `success`, `object_position` |
| `run_controller` | `success`, `failed_at`, `reward`, `stopped_early`; `keeps_running` means call `stop(robot)` later |
| `compare_controllers` | the `table` and `best`; explain with its numbers |
| `run_script` | `ok`, `output`, traceback |
| `set_controls` + `simulate` | `observe()` joints against controls; any `warning` |

`observe(view=...)` takes default, front, back, left, right, top, or a robot camera such as `dog/front_cam`.
`observe(view, find="red")` (a color name, a hex code, or "all") lists what that view sees of the color: each find's world `position`, the `object` name `pick_and_place` takes, and `about_cm`, how far off the position may be.
A flag on a find (cut_off, hidden, side_on, raised, held_by) means it may be much further off; the result's `how` says what each one it carries means.
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
