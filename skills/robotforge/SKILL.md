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
2. **World.** `new_world(name, floor)` creates a project (floors: checker, grass, wood, concrete); `open_world` reopens one.
3. **Robot.** Pick the source that fits:
   - template (`list_templates`): arm, quadruped, humanoid, rover, cartpole, tuned with `params` (sizes, servo, color);
   - library (`list_library`): real robots such as `go2` or `panda`, downloaded on first use (4-55 MB);
   - file: `add_robot(file=...)` imports URDF, xacro, MJCF or a mesh and returns the import doctor's report;
   - no fitting template: write a small URDF and import it (see recipes).
   Add props with `add_object` (box, sphere, cylinder, table, bin).
4. **Viewer early.** Call `open_viewer()` as soon as a robot exists and tell the user what they can do: drag the robot, move sliders, drive with W A S D. While it is open, actions play out in real time.
5. **Make it move**, with the highest-level tool that fits:
   - `move` / `stop` / `pose`: walk a quadruped or humanoid, drive a rover, humanoid poses;
   - `move_arm` / `gripper` / `pick_and_place`: arms;
   - `run_controller` / `compare_controllers`: a named algorithm on a task (reach, balance, stand, walk, drive_to), see `list_algorithms`;
   - `set_controls` + `simulate`: raw actuator targets;
   - `write_script` + `run_script`: custom sequences, loops and logic the user can keep;
   - `train_policy` when the tool list has it: reinforcement learning (optional `rl` extra).
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
| `move` | `distance_m`, `turned_deg`, `to.upright` |
| `pose` | `upright` |
| `move_arm` | `tool_position` against the target (within about 3 cm) |
| `pick_and_place` | `success`, `object_position` |
| `run_controller` | `success`, `failed_at`, `reward`, `stopped_early`; `keeps_running` means call `stop(robot)` later |
| `compare_controllers` | the `table` and `best`; explain with its numbers |
| `run_script` | `ok`, `output`, traceback |
| `set_controls` + `simulate` | `observe()` joints against controls; any `warning` |

`observe(view=...)` takes default, front, back, left, right, top, or a robot camera such as `dog/front_cam`.
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

## References

- [references/recipes.md](references/recipes.md): walk a dog, pick and place, compare LQR and MPC, write a script, import a URDF, build a new robot, train a policy.
- [references/scripts.md](references/scripts.md): the script API and a complete closed-loop example.
- [references/troubleshooting.md](references/troubleshooting.md): common mistakes and how to recover.
- [references/teaching.md](references/teaching.md): tone, short explanations of core ideas, how a teaching turn flows.
