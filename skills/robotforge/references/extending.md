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

## Recording and replaying demonstrations

```python
from robotforge.script import connect

sim = connect()
dog = sim.robot("dog")
with dog.record("square_walk") as demo:  # keeps joints, actuator targets, base pose, sensors and commands
    dog.move(speed=0.25, seconds=2.0)
    dog.move(speed=0.0, turn=1.2, seconds=1.0)
print(demo.result["command_counts"])  # {"move": 2, "stop": 2}
sim.robot("go2").replay("square_walk")  # the same commands on another quadruped
sim.export_demo("square_walk", "npz")  # demonstrations/square_walk.npz
```

The same from an agent: `demonstration(action="start", robot="dog", name="square_walk")`, act, `demonstration(action="stop")`, `demonstration(action="replay", name="square_walk", robot="go2")`, `demonstration(action="export", name="square_walk", format="npz")`.
`mode="commands"` (default) gives the recorded commands again at their times; arm targets are kept relative to the arm's base, so an arm standing elsewhere reaches the same place relative to itself.
`mode="joints"` puts the robot in the first pose and plays the actuator targets back as recorded: exact, open loop, for arms and hands on the same actuators.
The NPZ has `time`, `joint_pos`, `joint_names`, `ctrl`, `ctrl_names`, `base_pos`, `base_quat`, `sensors`, `sensor_names`, `command_t`, `command_op` and `commands` (JSON text).

## Behavior trees and state machines

```python
from robotforge.flow import Action, Condition, Fallback, Retry, Sequence, Tree, sensor_is
from robotforge.script import connect

sim = connect()
robot = sim.robot("go2")
tree = Tree(
    Sequence(
        sensor_is("lidar", "ahead.distance_m", ">", 0.5, name="path clear"),
        Retry(Action(lambda ctx: ctx.robot.arm.pick_and_place("cube", "tray"), name="fetch the cube"), times=3),
        name="errand",
    ),
    name="cube errand",
)
print(tree.run(sim, robot))  # SUCCESS or FAILURE; the viewer's tree panel follows it
print(tree.summary())
```

Nodes: `Action(fn)` (fn(ctx) returns True, None, False or a Status), `Condition(fn)`, `sensor_is` (a condition on a sensor, as in the example), `Wait(seconds)`, `WaitFor(test, timeout)`, `Sequence`, `Fallback`, `Parallel(success="all"|"any"|n)`, `Retry(node, times, wait)`, `Repeat`, `Inverter`, `Timeout(node, seconds)`, `StateMachine(name, start)` with `State(...)` and `fsm.transition(a, b, when)`.
`ctx` has `sim`, `robot`, `time`, `bb` (shared dictionary) and `ctx.log(text)`.
A call the playground refuses inside an `Action` is a failed node with its message as the note, so `Retry` and `Fallback` can handle it.
See docs/EXTENDING.md in the robotforge repository for every node with an example.

