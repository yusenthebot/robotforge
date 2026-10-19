# Extending robotforge

This is the short version of docs/EXTENDING.md in the repository, which has every example run by the tests.

## Plugins

A plugin is a Python file that registers things with `robotforge.extend`.
Where it lives:

- `~/.robotforge/plugins/` (or the folders in `ROBOTFORGE_PLUGINS`): the user's own, loaded when robotforge starts.
- `<world>/plugins/`: travels with the world. `write_script(name, code, plugin=true)` writes one there and loads it. Trust covers every file under the folder as one set (helpers and subfolders too). Files that were not written on this computer do not load until the user approves them in a terminal with `robotforge plugins trust WORLD`; you cannot. Read them with `plugins(action="show", world=NAME)` (it works before the world is open) and tell the user what they do.

```python
import math
from robotforge.extend import Box, Cylinder, behavior, controller, look, part, sensor


@part("signpost", size="[height]", default_size=[1.2], fixed=True, color="brown")
def signpost(size, rgba):
    h = size[0]
    return [Cylinder(0.05, h, at=[0, 0, h / 2]), Box([0.4, 0.03, 0.2], at=[0, 0, h - 0.1], color=1.3)]


@sensor(
    "geiger",
    channels=("level",),
    params={"source": "hot", "radius": 2.0},
    at="top",
    appearance=look("cylinder", [0.07, 0.035], "yellow"),
)
def geiger(ctx):
    d = ctx.distance(ctx.params["source"])
    return {"level": max(0.0, 1.0 - d / ctx.params["radius"]), "distance_m": d}


@controller("sweep", params={"joint": "shoulder", "amplitude": 0.4, "hz": 0.5}, rate_hz=50)
def sweep(ctx):
    p = ctx.params
    ctx.set(p["joint"], p["amplitude"] * math.sin(2 * math.pi * p["hz"] * ctx.time))


@behavior("square", params={"side_s": 1.5, "speed": 0.2})
def square(robot, side_s, speed):
    for _ in range(4):
        robot.move(speed=speed, seconds=side_s)
        robot.move(speed=0.0, turn=1.2, seconds=1.0)
    return {"position": robot.position}
```

- Part: `add_object(name, shape="signpost", pos=[1, 0])`. A part returns `Box(size, at, color, rotate, collide)`, `Cylinder(diameter, height, ...)` and `Sphere(diameter, ...)`; `at` is the middle of the shape in the part's frame (z up from the ground).
- Sensor: `add_sensor(robot, type="geiger", options={"source": "cube", "radius": 3, "noise": 0.02, "rate_hz": 10})`. `ctx` has `pos`, `rot`, `params`, `time`, `ctx.position(name)`, `ctx.distance(name)`, `ctx.ray(direction, max_range)`, `rng`, `model`, `data`. An option the plugin did not list is refused with the ones it has. A class with `def __init__(self, ctx)` and `def read(self, ctx)` keeps state.
- Controller: `plugins(action="run", name="sweep", robot="arm", options={"amplitude": 0.6}, seconds=5)`; it keeps running until `stop(robot)`. `ctx` has `ctx.joint(name)`, `ctx.velocity(name)`, `ctx.joints()`, `ctx.set(actuator, value)`, `position`, `heading_deg`, `ctx.position_of(body)`, `ctx.sensor(name)`, `time`, `dt`, `params`; names are the robot's own, without the `robot/` prefix. A controller that raises stops itself and says why in its status.
- Behavior: a function `def run(robot, **options)` written with the script robot interface, run as a script (`plugins(action="run", name="square", robot="dog")`); what it returns is the result.

- Training algorithm: `@algorithm("hill_climb", hyper={"sigma": Hyper(0.2, low=0.05, high=0.5, log=True)})` on a function `def run(ctx)` that loops `while ctx.has_budget()`, tries policies with `ctx.evaluate(theta)` or `ctx.rollout(act)`, reports with `ctx.log(name=value)` and returns the flat parameter vector (shape set by `ctx.new_shape(hidden)`); `ctx.state` keeps arrays between a search's stretches. Then `train_policy(algo="hill_climb")` or `algo=["ars", "hill_climb"]` to compare, and a `search` draws from its `Hyper` ranges. `list_algorithms` shows it. The training process loads it from the plugin folders, so write it with `plugin=true` (docs/TRAINING.md has the contract).
- Training task: `@task("stay_upright")` on `def build(model, robot, params)` returning the result of `reward_task` (`from robotforge.training.custom import reward_task`; pass it `model, robot, params` and `reward=lambda ctx: ...`, `failed=`, `success=`, `reset=`, `seconds=`); `ctx.joint(name)`, `ctx.body(name)`, `ctx.base_velocity()`, `ctx.height()`, `ctx.up()`, `ctx.effort()`. Then `train_policy(task="stay_upright", robot=...)` on any robot.

When a plugin file fails, `plugins()` shows `not_loaded` with the file and line.
After editing the file, `plugins(action="reload")`.

## Export

`export_model()` is the open world as MJCF in the world's `exports/` folder; `robot="go2"` just that robot; `format="urdf"` (needs `robot`) as URDF; `to=` a new or empty folder (in the world, the home folder or the temp folder; `overwrite=true` to replace an earlier export).
Meshes are OBJ files in `meshes/`.
`check` says the file was loaded again in MuJoCo and compared (`same` is false if bodies, joints, masses, joint limits or any mesh vertex differ for MJCF; bodies, joints and limits for URDF).
URDF cannot say a floating base, closed loops, capsules or ball joints: `notes` lists what changed.
`python -m mujoco.viewer --mjcf=<file>` opens an MJCF.

## Robot interface in scripts

```python
from robotforge.script import connect

sim = connect()
robot = sim.robot("dog")  # or "go2": the same script
robot.on_step(lambda r: print(r.time, r.sensors["imu"]["orientation_deg"]), every=0.5)
robot.move(speed=0.3, seconds=2)  # walks or drives, then stops
arm = sim.robot("panda")  # or a template arm
arm.arm.move_to([0.4, 0.0, 0.3], gripper="open")
arm.gripper.close()
```

`robot.kind`, `robot.sensors[name]` (also `.names()`, `.get(name, default)`, `.imu`), `robot.move(speed, turn, seconds=None)`, `robot.stop()`, `robot.pose(name)`, `robot.arm` / `robot.arms` (`move_to`, `pick_and_place`), `robot.gripper` (`open`, `close`), `robot.position`, `robot.heading`, `robot.joints`, `robot.set_controls`, `robot.start(controller, **options)`, `robot.on_step(fn, every)`, `robot.run(seconds)`.
A robot that cannot do something raises an error that says what it can do instead (a dog has no arm).
Blocking calls (`arm.move_to`, `pick_and_place`) do not call the per-step callback.
