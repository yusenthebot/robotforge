# Recipes

Each recipe is a tested path from a beginner's request to a checked result.
Adapt names and numbers; keep the checks.

## Contents

1. Walk a robot dog
2. Pick and place with an arm
3. Compare controllers (LQR, MPC, ...)
4. Write a script
5. Import a URDF (or xacro, MJCF, mesh)
6. Build a robot that no template covers
7. Train a policy
8. Drawers, doors and a suction cup

## 1. Walk a robot dog

"给我一只四足狗，让它在草地上走."

1. `new_world(name="dog_walk", floor="grass")`
2. `add_robot(name="dog", template="quadruped")`, or a real one: `add_robot(name="go2", library="go2")` (first use downloads about 31 MB; if the call times out, call it again, the download continues).
3. `open_viewer()`; tell the user W A S D drives it.
4. `robots()` gives `max_speed` (template dog about 0.36 m/s; humanoids about 0.1 m/s and slow to start and stop).
5. `move(robot="dog", speed=0.25, seconds=10)`.
6. Check: `distance_m` above 1 m and `to.upright` true; `observe()` image. Foot touch sensors (`dog/front_left_touch` ...) show stance and swing.

Explain: a trot moves diagonal legs together; stance pushes back, swing lifts forward; leg inverse kinematics turns foot positions into joint angles.
Turning: `move(robot="dog", speed=0.2, turn=0.5, seconds=5)`; `seconds=null` keeps walking until `stop(robot)`.

## 2. Pick and place with an arm

"做一个机械臂，把红方块放进盒子."

1. `new_world(name="pick")`, `add_robot(name="arm", template="arm")` (default reach 0.45 m).
2. `add_object(name="cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.25, -0.12], color="red")`
3. `add_object(name="box", shape="bin", size=[0.14, 0.14, 0.05], pos=[0.05, 0.28], color="blue")`
4. `open_viewer()`, then `pick_and_place(robot="arm", object="cube", target="box")`.
5. Check: `success` true and `object_position` inside the bin; `observe(view="top")`.

Keep small objects 0.15 to 0.35 m from the base of a default arm, and at most about 5 cm wide (gripper opening).
"cannot reach": move the object or target closer, or raise the arm's `reach` param with `update_robot`.
If the user wants code they can keep (or asks for "a script"), write `behaviors/pick.py` with `sim.pick_and_place(...)`, or step by step with `sim.move_arm` and `sim.gripper` (see scripts.md), run it and check its printed result.
`move_arm(robot, [x, y, z], gripper="open"|"close")` moves the gripper, pointing down, to a world point: use it to teach inverse kinematics.

## 3. Compare controllers

"用 LQR 和 MPC 比一比."

- `list_algorithms()` lists tasks, fitting robots and a one-line explanation of each method.
- Tasks: `reach` (arm), `balance` (cartpole), `stand` and `walk` (quadruped), `drive_to` (rover).
- Show one live: `run_controller(robot="pole", task="balance", method="lqr", push_at=2)`; the controller stays attached (`keeps_running`), release it with `stop(robot="pole")`.
- Compare off-screen from the current state: `compare_controllers(robot="pole", task="balance", methods=["none", "pd", "lqr", "mpc"], push_at=3)`.
- Targets: `task_params={"target": "cube"}` or `{"target": [0.3, 0.1, 0.2]}` for reach, `{"goal": [1.0, 0.5]}` for drive_to.
- Tuning: `method_params={"samples": 64, "horizon": 0.6}` (MPC), `{"q_weights": [5, 50, 0.5, 1], "r_weight": 0.5}` (LQR), `{"omega": 20, "zeta": 1.0}` (arm PD).

Read the table columns to the user: reward (higher is better), success, the task's own number (distance, angle, height), effort in joules, wall/sim (computer seconds per simulated second; below 1 is faster than real time).
The usual story: classic control (PD, LQR) is cheap and precise near its design point; MPC needs no hand-designed law but costs hundreds of physics rollouts per second; hold and gait fall when shoved, MPC on top of the gait recovers.

## 4. Write a script

Use a script when the job is a sequence, a loop, or logic the user wants to keep and rerun.

1. `write_script(name="square", code=...)` saves `behaviors/square.py` in the open world.
2. `run_script(name="square")` runs it in lockstep with the simulation and returns `ok`, `output` and any traceback.
3. `read_script(name="square")` shows it again.

`run_script` runs real Python on the user's computer (not a sandbox): say so the first time, and only run code you wrote for this task.
Write closed-loop code (measure and correct), print the result the user cares about, comment each step.
API and a full example: [scripts.md](scripts.md).

## 5. Import a URDF

"打开这个 URDF" / "导入我的机器人."

1. A world: `new_world(name="my_robot")`.
2. `add_robot(name="myrobot", file="/path/to/robot.urdf")`. Also accepts xacro, MJCF (.xml) and plain meshes (.stl, .obj, .dae, .glb). A first import of a large robot can take up to a minute (mesh conversion and a hold test).
3. Read the result's `doctor` report to the user in plain words:
   - `summary`, then each finding: `fixed` (found and repaired, `action` says how), `unfixed` (could not repair, `action` says what happens instead), `info` (worth knowing).
   - Typical fixes: `package://` mesh paths resolved by searching the folder, DAE meshes converted, missing mass and inertia computed from shapes, motors added sized to the load, overlapping collision pairs excluded, the robot lifted out of the floor, a hold test run.
4. `describe_model()` lists the new actuators (`myrobot/<joint>`); `set_controls`, `simulate`, `observe` drive it; the viewer sliders work too.
5. `robots()` shows its `kind`: an imported quadruped or a wheeled robot can use `move`; an arm with no `tcp` site and kind `other` is driven with `set_controls` or a script.

Import options go in `params` (or `options` for `open_model`):

- `base`: `auto` (default; the doctor guesses), `fixed` (bolted down, like an arm on a table) or `floating` (free, like a dog). Re-import with the other one when the guess is wrong: `update_robot(name, params={"base": "floating"})`.
- `respect_effort_limits`: by default, torque limits too weak to hold the robot up are relaxed and reported; `true` keeps the file's limits (the robot may sag or fall: that is the honest result).
- `decompose`: `true` splits concave collision meshes into convex pieces (slower first import, cached).
- `scale`, `mass`: for plain meshes (scale defaults to a meters versus millimeters guess).
- `xacro_args`: values for `<xacro:arg>` names.

For a URDF with meshes, give the path inside its package folder so `package://` paths resolve.
`open_model(path, options)` instead replaces the whole scene with that one file (no world.yaml, no undo): use it only to inspect a model quickly.

## 6. Build a robot that no template covers

First check the tool list: if it has a parts or builder tool, follow its description.
Otherwise:

1. Start from the closest template and tune its params (`list_templates` shows sizes, servos, materials, limits).
2. For a new body plan (a snake, a six-legged spider, a gantry), write a small URDF with links (box, cylinder, sphere shapes) and revolute joints with limits, save it in the world folder and import it (recipe 5). The doctor adds mass, inertia and motors; start simple, then add parts.
3. Drive it with `set_controls`, `simulate` and scripts; check it stands with `observe` before adding motion.

Say clearly which parts are approximations.

## 7. Train a policy

"让它自己学会."

- Built in, no extra install: `run_controller(robot="pole", task="balance", method="ars", push_at=2)` trains a linear policy for a few seconds, then runs it live; compare with `compare_controllers(robot="pole", task="balance", methods=["ars", "lqr"], push_at=3)`.
- Deep RL, when the tool list has `train_policy` (optional `rl` extra with PyTorch):
  1. `train_policy(task="balance", robot="pole")` or `train_policy(task="walk", robot="dog", minutes=5)`; it returns at once with a name and a time estimate.
  2. `training_status(name=..., wait_seconds=60)` to follow progress (reward curve).
  3. `run_controller(robot=..., task=..., method="policy:<name>")` to watch it, or put `"policy:<name>"` in `compare_controllers`.
- If `train_policy` reports that the rl extra is missing, tell the user what it adds (PyTorch and Stable-Baselines3, a large download) and point to docs/ALGORITHMS.md; do not install anything without their yes.

Explain: reward, trial and error, a policy maps observations to actions; test it with a push it never saw.

## 8. Drawers, doors and a suction cup

"打开厨房的抽屉，再用吸盘把桌上的薄板放进抽屉."

1. `load_scene("kitchen")`: the island's arm has a chest with a drawer (`drawer`) beside it, and the island a plate.
2. `operate(robot="arm", target="drawer", action="open", amount=0.2)`. Check `success`, `moved` (m) and `state.open_m`.
3. The plate is 1.5 cm thick and 14 cm wide: fingers cannot get round it, a suction cup can. `update_robot(name="arm", end_effector="suction")` (a tool change; the drawer stays open).
4. `pick_and_place(robot="arm", object="plate", target="drawer")`. Check `success`; `operate(..., action="close")` slides it in with the plate.

A door: `operate(robot, "cabinet", "open", 60)` (degrees). A button: `operate(robot, "button", "press")`, then `observe()` shows `mechanisms.button.presses`. A valve: `operate(robot, "valve", "open", 180)` turns it half a turn, a quarter at a time.
"cannot take hold of ...'s handle": the handle is out of the arm's reach pointing down; move the arm or the thing nearer each other (a template arm reaches pointing down from about 0.1 to 0.3 m of its base at table height).
Notes saying it could follow the handle only part of the way mean the same, partway: move them nearer.

The factory (`load_scene("factory")`): a conveyor brings colored boxes to a stop next to an arm with a suction cup and a red, a green and a blue bin. `observe(view="top", find="red")` names the red boxes; `set_controls({"conveyor/speed": 0})` stops the belt while picking; `pick_and_place("arm", "box_3", "red_bin")` sorts one.
Explain: a suction cup holds by pressing a flat face against it (air pressure), a magnet holds only steel, fingers hold by squeezing; the right tool depends on the thing.
