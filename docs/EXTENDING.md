# Extending robotforge

robotforge is meant to be bent to what you need.
This page covers five ways to do that without touching its source:

- [Plugins](#plugins): your own parts, sensors, controllers and behaviors, written as small Python files that robotforge finds and uses like its own.
- [Export](#export): a world or one robot written out as MJCF or URDF with meshes, so other simulators open it.
- [One robot interface in scripts](#one-robot-interface-in-scripts): `robot.sensors`, `robot.move`, `robot.arm`, `robot.gripper` and a per-step callback, so one script runs on a template dog and on a Go2, or on a template arm and on a Panda.
- [Demonstrations](#recording-and-replaying-demonstrations): what a robot did, recorded, replayed on it or another robot of its kind, and exported as JSON or NPZ for training.
- [Behavior trees and state machines](#behavior-trees-and-state-machines): task logic in a script, with the viewer showing which node runs.

Every example below is run by the test suite (`tests/test_extending_docs.py`), so what you read here works.

## Plugins

A plugin is a Python file that registers things with `robotforge.extend`.
It is your own code running on your own computer with your own rights, like a script.
Nothing is sandboxed.

### Where plugins live

| Place | Loaded | Trusted |
|---|---|---|
| `~/.robotforge/plugins/` (or the folders in `ROBOTFORGE_PLUGINS`, joined with `:` on macOS and Linux, `;` on Windows) | when robotforge starts, and on `plugins(action="reload")` | always: you put the files there |
| `plugins/` inside a world's folder | when the world is opened | only when every file under it was written on this computer, or approved by you in a terminal with `robotforge plugins trust WORLD` |

A world can come from somebody else, and a plugin in it is code that would run the moment you open it.
So a world's plugin files that were not written here do not load.
Trust covers everything under the folder as one set: helper files (`_helper.py`), subfolders and packages a plugin imports are as much code as the plugin.
One file nobody approved, or one added or changed later, and none of the folder loads.
A world's plugin never takes the place of a standard module of the same name (`random.py` beside it is not `import random`).
`open_world` says which files did not load and why; a world that uses a part from them does not open until they are approved, and says how.
The agent can read them (`plugins(action="show", world=NAME)` prints every file) and tell you what they do, but it cannot trust them: that decision is yours.
You approve in a terminal: `robotforge plugins show WORLD` prints the files, `robotforge plugins trust WORLD` prints them again and asks `yes` (`--yes` skips the question once you have read them).
A world copied to another place is not trusted again.
`write_script(name, code, plugin=true)` writes a plugin into the open world, trusts it (you asked for it, as for any script it runs) and loads it, unless the folder holds other files that nobody approved: then it is saved and waits with them.

If a file fails to load, nothing it registered is kept, and the error names the file and the line:

```text
plugins() ->
  not_loaded: {"lab/plugins/geiger.py": "line 12: NameError: name 'distnce' is not defined"}
```

The other plugin files still load.
`plugins()` lists what is loaded and what failed, `plugins(action="reload")` reads the folders again after an edit, and `robotforge plugins` on the command line does the first without an agent.
In Python, the same registry is `robotforge.extend.REGISTRY`.

### Parts: new shapes for `add_object`

A part is a function from a size and a color to a list of boxes, cylinders and spheres.
Placing it on the ground or a table, yaw, mass, saving it in `world.yaml` and drawing it in the viewer are what any shape gets.

```python title="plugins/signpost.py"
from robotforge.extend import Box, Cylinder, part


@part("signpost", size="[height]", default_size=[1.2], fixed=True, color="brown", summary="a post with a board")
def signpost(size, rgba):
    height = size[0]
    return [
        Cylinder(0.05, height, at=[0, 0, height / 2]),
        Box([0.4, 0.03, 0.2], at=[0, 0, height - 0.1], color=1.3),  # a number: lighter than the part's color
    ]
```

Then `add_object(name="sign", shape="signpost", pos=[1, 0])`, or `shape: signpost` in `world.yaml`.

- `size` says what the numbers mean, `default_size` is used when none is given, `sizes=(1, 2)` allows more than one length.
- `fixed=True` makes it never move; otherwise it is loose, with `mass` kg (default 5).
- Shapes: `Box(size, at, color, rotate, collide)`, `Cylinder(diameter, height, ...)`, `Sphere(diameter, ...)`.
  `at` is the middle of the shape in the part's own frame (x forward, y left, z up from the ground), `rotate` is degrees about x, y, z, `collide=False` makes it a picture only.
- A part is checked when it is registered, at its default size, so a mistake shows up with the plugin's file named.

### Generators: modelled objects with grasp hints

A part is a handful of boxes.
A generator is for a real thing: a mesh that looks like it, convex pieces that collide like it, a mass and friction from what it is made of, and grasp hints (where a gripper takes hold of it) that the arm planner uses.
It is a function that gets a builder, `g`, and describes the object once in its own frame (origin at the middle of its base, z up, resting the way it is meant to stand or lie).

```python title="plugins/coaster.py"
from robotforge.design.visuals import mesh
from robotforge.objects import Range, generator, kit


@generator(
    "coaster",
    size="[diameter]",
    default_size=(0.09,),
    color="brown",
    material="wood",
    summary="a cork coaster",
    keywords=("coaster", "mat"),
    params={"edge": Range(0.001, 0.004, 0.002, "how round its edge is, m")},
)
def coaster(g):
    d, t = g.size[0], 0.006
    disc = mesh.cylinder(d / 2, t, 32, bevel=g.p("edge"), z0=0.0)  # how it looks
    g.part(disc, "wood", [kit.cylinder(d / 2, t / 2, (0, 0, t / 2))])  # what it is made of, how it collides
    g.grasp("edge", (d / 2 - 0.004, 0, t / 2), (1, 0), 0.012, 1.0, "pinch its rim")  # where to take hold of it
    g.anchor("top", (0, 0, t))
```

Then `add_object(name="c", shape="coaster")`, `add_object(name="c", describe="a big cork coaster", seed=3)` or `shape: coaster` in `world.yaml`.

- A seed varies what `params` lists (`Range(low, high, default)`, `Choice(...)`, `Flag(...)`): the same shape, size, seed and parameters always build the same object, and a caller can fix any of them with `params`.
- `g.part(mesh, material, pieces)` adds a solid.
  Its mass is the mesh's volume times the material's density (`robotforge.objects.materials`), its friction the material's.
  `pieces` are convex shapes from `robotforge.objects.kit`: `box`, `cylinder`, `capsule`, `sphere`, `ellipsoid`, `hull(mesh)`, and for hollow things `vessel(...)`, a floor and a ring of boxes round a wall.
  `g.decor(mesh, rgba)` adds a look with no mass (a printed label: `robotforge.objects.labels`).
- `g.grasp(kind, pos, jaw, opening, score, note)` is a place a two-finger gripper takes hold of, `pos` where the tool point goes, `jaw` the horizontal direction the jaws close along (`(0, 0)`: any), `opening` the width they must open to straddle that part.
- Meshes come from `robotforge.design.visuals.mesh` (lathe, loft, tube, rounded_box, ...) and `robotforge.objects.profiles` and `shapes`.
  A mesh that is not convex is cut into convex pieces by `robotforge.objects.collision.decompose` (CoACD: slow, so the result is kept).
- A generator is checked when it is registered, at its default size, and appears in `plugins()` as a part.

### Sensors: new types for `add_sensor`

A sensor is a function that looks at the simulation and returns what it measures.
It sits on any body of any robot, and has a rate, noise and latency like the built-in ones.

```python title="plugins/geiger.py"
from robotforge.extend import look, sensor


@sensor(
    "geiger",
    summary="how close the hot source is: 1 on top of it, 0 at the radius",
    channels=("level",),
    params={"source": "hot", "radius": 2.0},
    at="top",
    appearance=look("cylinder", [0.07, 0.035], "yellow"),
)
def geiger(ctx):
    distance = ctx.distance(ctx.params["source"])
    return {"level": max(0.0, 1.0 - distance / ctx.params["radius"]), "distance_m": distance}
```

Then `add_sensor(robot="dog", type="geiger", options={"source": "cube", "radius": 3, "noise": 0.02, "rate_hz": 10})`.
The reading comes back from `read_sensors`, `observe` and `sim.sensor(robot, "geiger")` in scripts; the viewer's sensor list shows it, and the yellow puck is drawn where it sits.

- `channels` are the names in the reading that `noise` applies to.
- `params` are the options `add_sensor` takes for it, with defaults; an option not listed is refused with the ones that are.
- `ctx` is a `SensorContext`: `ctx.pos` and `ctx.rot` (where the sensor is and which way it faces), `ctx.params`, `ctx.time`, `ctx.position(name)`, `ctx.distance(name)`, `ctx.ray(direction, max_range)` (the first thing a ray from the sensor meets), `ctx.rng`, and `ctx.model`, `ctx.data` for anything else.
- A class works too: `__init__(self, ctx)` once, `read(self, ctx)` at every reading, so a sensor can keep a count or a filter.
  An edit of the world starts it again.
- A sensor that raises does not stop the simulation: its reading is `{"error": "..."}` until it is fixed.

### Controllers: feedback inside the simulation

A controller is called every `1 / rate_hz` seconds of simulated time and sets actuators.
It keeps running until `stop(robot)`, a `move`, or another controller takes the robot.

```python title="plugins/sweep.py"
import math

from robotforge.extend import controller


@controller("sweep", summary="swings one joint to and fro", params={"joint": "shoulder", "amplitude": 0.4, "hz": 0.5})
def sweep(ctx):
    p = ctx.params
    ctx.set(p["joint"], p["amplitude"] * math.sin(2 * math.pi * p["hz"] * ctx.time))
```

Start it with `plugins(action="run", name="sweep", robot="arm", options={"amplitude": 0.6}, seconds=5)`, or from a script with `robot.start("sweep", amplitude=0.6)`.

- `ctx` is a `StepContext`: `ctx.joint(name)`, `ctx.velocity(name)`, `ctx.joints()`, `ctx.set(actuator, value)`, `ctx.position`, `ctx.heading_deg`, `ctx.position_of(body)`, `ctx.sensor(name)`, `ctx.time`, `ctx.dt`, `ctx.params`.
  Names are the robot's own, without the `robot/` prefix.
- A class with `__init__(self, ctx)` and `update(self, ctx)` keeps state.
- A controller that raises stops, and its status says why; the world goes on.

### Behaviors: routines for any robot

A behavior is a function written against the [robot interface](#one-robot-interface-in-scripts), so it runs on any robot that can do what it asks.
It runs like a script: its own process, in lockstep with the simulation, with a time limit.

```python title="plugins/square.py"
from robotforge.extend import behavior


@behavior("square", summary="walks or drives a square", params={"side_s": 1.5, "speed": 0.2})
def square(robot, side_s, speed):
    start = robot.position
    for _ in range(4):
        robot.move(speed=speed, seconds=side_s)
        robot.move(speed=0.0, turn=1.2, seconds=1.0)
    end = robot.position
    return {"kind": robot.kind, "gap_m": round(((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5, 2)}
```

Run it with `plugins(action="run", name="square", robot="dog", options={"side_s": 2})`; the result is what it returns.
Your own script can call it too, on any robot; the script's process reads the plugin folders itself:

```python title="behaviors/patrol.py" robots="dog,go2"
from robotforge.extend import call
from robotforge.script import connect

sim = connect()
robot = sim.robot(sim.robots()[0])
print(robot.kind, call("square", robot, side_s=1.0))
```

### Algorithms and tasks: training with your own

A training algorithm is a function that spends a step budget and returns the policy it found.
It sits beside ARS, CEM, PPO and SAC: `train_policy(algo="hill_climb")`, a list to compare it with them, and a search can draw its settings from the ranges you give.
[TRAINING.md](TRAINING.md) has the whole contract; this is a complete one.

```python title="plugins/hill_climb.py"
from robotforge.extend import Hyper, algorithm


@algorithm(
    "hill_climb",
    summary="keep a random change to the policy when it scores better",
    hyper={"sigma": Hyper(0.2, low=0.05, high=0.5, log=True, help="size of each random change")},
)
def hill_climb(ctx):
    theta = ctx.state.get("theta", ctx.new_shape().zeros()).copy()  # ctx.state is kept between a search's stretches
    best = ctx.evaluate(theta)
    while ctx.has_budget():
        candidate = theta + ctx.params["sigma"] * ctx.rng.standard_normal(theta.shape)
        score = ctx.evaluate(candidate)
        if score > best:
            theta, best = candidate, score
        ctx.log(best_return=best)  # a line on the Training panel
    ctx.state["theta"] = theta
    return theta  # the policy's flat parameters; ctx.policy(theta) is the policy they make
```

- `ctx.env` is the environment (Gym API, actions in [-1, 1]), `ctx.params` the settings, `ctx.rng` a generator seeded per trial, `ctx.obs_dim` and `ctx.act_dim` its sizes.
- `ctx.rollout(act)` plays one episode with `act(obs)`; `ctx.evaluate(theta, episodes)` plays the policy with parameters `theta`.
  Both count against the budget and draw on the episode returns of the Training panel.
- `ctx.has_budget()` is false when the steps are spent, a stop was asked for, or the time is up.
- `ctx.new_shape(hidden=(16,))` sets the policy's layers (none: a linear policy); `ctx.update_norm()` scales the inputs by what they have been.
- Return the flat parameters, or any object with `__call__(obs)` and `save(path)`.
- `needs=("torch",)` names modules the algorithm imports, so a missing one is explained before training starts.

A task is a function that builds a `Task` for a robot.
`reward_task` makes one from a reward function, so any robot can be trained on anything that can be measured.

```python title="plugins/stay_upright.py"
from robotforge.extend import task
from robotforge.training.custom import reward_task


@task("stay_upright", summary="hold a pole straight up", params={"limit": 0.8})
def build(model, robot, params):
    return reward_task(
        model,
        robot,
        params,
        reward=lambda ctx: 1.0 - abs(ctx.joint("hinge")) - 0.01 * ctx.effort(),
        failed=lambda ctx: abs(ctx.joint("hinge")) > ctx.params["limit"],
        reset=lambda ctx, rng: ctx.set_joint("hinge", rng.uniform(-0.25, 0.25)),
        seconds=3,
        task_name="stay_upright",
    )
```

Then `train_policy(task="stay_upright", robot="pole", algo=["ars", "hill_climb"])`.
The reward function sees a context with the robot's state by its own names (`ctx.joint`, `ctx.body`, `ctx.base_velocity()`, `ctx.height()`, `ctx.up()`, `ctx.effort()`).
A plugin algorithm or task runs inside the training process too: it loads from the plugin folders there, so keep it in `~/.robotforge/plugins` or a trusted world.

### Using plugins from Claude or GPT

The agent writes the file for you (`write_script(name, code, plugin=true)`), then uses it with the ordinary tools: `add_object(shape="signpost")`, `add_sensor(type="geiger")`, `plugins(action="run", ...)`.
Ask for what you want ("add a sensor that tells me how close the red cube is") and it writes the plugin.

## Export

`export_model` writes the open world, or one robot, for other simulators.

- `export_model()` writes the whole world as MJCF.
- `export_model(robot="go2")` writes just that robot (on the floor) as MJCF.
- `export_model(robot="go2", format="urdf")` writes the robot as URDF.
- `to="/some/folder"` chooses the folder; the default is `exports/` in the world's folder.
  A `to` must be a new or empty folder inside the world's folder, your home folder or the temporary folder; `overwrite=true` replaces an earlier export in a folder that is not empty (it removes only the `.xml`, `.urdf` and `meshes/` an export writes).

On the command line: `robotforge export <world> [--robot NAME] [--format mjcf|urdf] [--to FOLDER] [--overwrite]` does the same and exits 1 if the check found a difference.

Beside the `.xml` or `.urdf` there is a `meshes/` folder of OBJ files, one per mesh.
The result says where, how many files, and a `check`: the export is loaded again in MuJoCo and compared with the world.
For MJCF the check compares bodies, joints, actuators and geoms by name, the total mass (to 0.1 percent), joint limits, the position of every body and every mesh vertex in the world (agreeing to 10 micrometers), and height fields; `same` is false if any of them differs.
For URDF it compares bodies, joints (by name, type and limits) and mass.

Open an MJCF with `python -m mujoco.viewer --mjcf=<file>`; any tool that reads URDF opens the other.
The sensors robotforge added are kept in the MJCF as custom text, which other tools ignore.

URDF is a tree of rigid links, so it cannot say everything MuJoCo can; `notes` lists what changed:

- a floating base (a dog, a humanoid) is the root link with no joint;
- a body with several joints gets massless links between them, a ball joint becomes three revolute joints;
- capsules become cylinders and ellipsoids become spheres;
- closed loops and couplings (a Panda's two fingers, an arm's gripper) are not written.

```python title="check_export.py"
import mujoco

model = mujoco.MjModel.from_xml_path("exports/go2_urdf/go2.urdf")
print("bodies", model.nbody, "joints", model.njnt)
```

## One robot interface in scripts

`sim.robot(name)` gives a script one interface to any robot, so the same script runs on a template dog and on a Go2, or on a template arm and on a Panda.

```python title="behaviors/tour.py" robots="dog,go2"
from robotforge.script import connect

sim = connect()
robot = sim.robot(sim.robots()[0])
seen = []
robot.on_step(lambda r: seen.append(round(r.time, 1)), every=0.5)

start = robot.position
robot.move(speed=0.25)  # walks, or drives, until robot.stop()
robot.run(2.0)  # two simulated seconds pass; the callback fires as they do
robot.stop()
print(robot.kind, "moved", round(abs(robot.position[0] - start[0]), 2), "m; callback ran", len(seen), "times")
```

```python title="behaviors/reach.py" robots="arm,panda"
from robotforge.script import connect

sim = connect()
robot = sim.robot(sim.robots()[0])
x, y, z = robot.position
tool = robot.arm.move_to([x + 0.2, y, 0.15], gripper="open")
robot.gripper.close()
print(robot.kind, "tool at", [round(v, 2) for v in tool])
```

What a `Robot` has:

| | |
|---|---|
| `robot.kind` | `arm`, `quadruped`, `humanoid`, `rover`, ... as `robots` says |
| `robot.sensors` | `robot.sensors["lidar"]`, `robot.sensors.imu`, `robot.sensors.names()`, `robot.sensors.get("x", default)`, `"imu" in robot.sensors` |
| `robot.move(speed, turn, seconds=None)`, `robot.stop()`, `robot.pose(name)` | walk, drive or swim; with `seconds` it runs that long and stops |
| `robot.arm`, `robot.arms`, `robot.gripper` | `arm.move_to([x, y, z], gripper=None)`, `arm.pick_and_place(obj, target)`, `gripper.open()`, `gripper.close()`; an arm on a dog is `go2/arm` |
| `robot.position`, `robot.heading`, `robot.joints`, `robot.time` | where it is |
| `robot.set_controls({"shoulder": 0.3})` | raw actuator targets by the robot's own names |
| `robot.start(controller, **options)` | hand it to a plugin controller |
| `robot.on_step(fn, every=0.02)`, `robot.run(seconds)` | `fn(robot)` is called every `every` simulated seconds while time passes through `run()` or `sim.step()` |

A robot cannot do what its body cannot, and says so with what it can do instead:

```text
dog is a quadruped and cannot work an arm; it can walk or drive: use robot.move
```

Blocking calls (`arm.move_to`, `pick_and_place`) run to their end without calling the per-step callback.
A Go2 carrying a PiPER arm has both: `robot.move` walks it and `robot.arm` works the arm.

## Recording and replaying demonstrations

A demonstration is what a robot did, kept as data: its joint positions, the actuator targets, its base pose and sensor readings at a fixed rate, and the commands that steered it (a key or a gamepad stick in the viewer, `move`, `move_arm`, a script).
It can be played again on the same robot or on another robot of the same kind, and written out as JSON or NPZ for training.

```python title="behaviors/record_walk.py" robots="dog,go2"
from robotforge.script import connect

sim = connect()
robot = sim.robot(sim.robots()[0])
with robot.record("walk_demo") as demo:  # keeps frames and commands until the block ends
    robot.move(speed=0.25, seconds=1.5)
    robot.move(speed=0.0, turn=0.8, seconds=1.0)
saved = demo.result
print(
    saved["kind"],
    saved["frames"],
    "frames,",
    saved["command_counts"],
    "->",
    saved["path"].split("demonstrations")[-1][1:],
)

played = robot.replay("walk_demo")  # the same commands again (another robot of its kind would do as well)
print("replayed", played["commands_issued"], "commands in", played["seconds"], "s")
print("wrote", sim.export_demo("walk_demo", "npz")["size_kb"], "kB")
```

An agent does the same with the `demonstration` tool: `start` (robot, name), act, `stop`, then `replay` (name, robot), `export` (name, format) and `list`.

| | |
|---|---|
| What is kept | `time`, `joint_pos` (rad or m), `ctrl` (the actuator targets), `base_pos` and `base_quat` (world frame), the numbers of the sensors the robot carries (not cameras and lidars unless named with `sensors=[...]`), and the commands, each with the time it was given |
| `mode="commands"` (default) | gives the recorded commands again at their times; the robot's own controllers do the rest, so it works on any robot of the same kind. A slower robot gets the speeds scaled down (`speed_scale`) and a command it cannot do (out of reach) is listed under `refused`. Arm targets, and the places of `go` and `pick_and_place`, are kept relative to the robot's own frame, so another robot goes the same way from where it stands (a saved file without `"frame": "robot"` on its `go` is read as world coordinates). A second replay on a robot takes over from the first |
| `mode="joints"` | puts the robot in the first pose and plays the actuator targets back as recorded: exact for the same robot or one with the same actuators, open loop, so for arms and hands: a walking, driving or flying robot is refused |
| Files | `stop` saves `demonstrations/<name>.json` in the world; `export` writes `json` or `npz` there (`to=` a file name; its `.json` or `.npz` suffix sets the format). An export never replaces a file that exists or another demonstration unless `overwrite=true`. A reading missing in some frames is `null` in the JSON (strict parsers read it) and NaN in the NPZ. The NPZ holds `time`, `joint_pos`, `joint_names`, `ctrl`, `ctrl_names`, `base_pos`, `base_quat`, `sensors`, `sensor_names`, `command_t`, `command_op` and `commands` (JSON text); LeRobot's parquet needs a library robotforge does not install |

A recording ends on its own when the world is edited or reset, and says so. Stopping one that holds nothing (no time passed) is an error that also ends it, so the robot can be recorded again. Demonstrations belong to the world: opening another world lists that world's.
What an errand does inside the simulation (a fetch driving the robot over and stopping it) is part of the command that started it and is not recorded twice.

```python title="behaviors/record_reach.py" robots="arm,panda"
from robotforge.script import connect

sim = connect()
robot = sim.robot(sim.robots()[0])
x, y, z = robot.position
with robot.record("reach_demo") as demo:
    robot.arm.move_to([x + 0.2, y, 0.15], gripper="open")
    robot.gripper.close()
print(demo.result["command_counts"])
print(robot.replay("reach_demo")["commands_issued"], "commands replayed")
```

## Behavior trees and state machines

`robotforge.flow` is a small library for the logic of a task: what to do first, what to do when it fails, what to check on the way.
It needs nothing but a script's `sim`, and the viewer draws the tree while it runs: one row per node, the running one lit, a note such as `attempt 2 of 3` on a retry.

```python title="behaviors/walk_tree.py" robots="dog,go2"
from robotforge.flow import Action, Fallback, Retry, Sequence, Tree, WaitFor
from robotforge.script import connect

sim = connect()
robot = sim.robot(sim.robots()[0])
start = robot.position


def walked(metres):
    def test(ctx):
        x, y, _ = ctx.robot.position
        return ((x - start[0]) ** 2 + (y - start[1]) ** 2) ** 0.5 >= metres

    return test


tree = Tree(
    Sequence(
        Action(lambda ctx: ctx.robot.move(speed=0.25), name="start walking"),
        Fallback(
            WaitFor(walked(0.4), timeout=6.0, name="walked 0.4 m"),
            Retry(WaitFor(walked(0.2), timeout=3.0, name="walked 0.2 m"), times=2),
            name="far enough",
        ),
        Action(lambda ctx: ctx.robot.stop(), name="stop"),
        name="short walk",
    ),
    name="walk tree",
)
print(tree.run(sim, robot), "after", round(sim.time, 1), "s")
print(tree.summary())
```

Every node answers success, failure or running, and the tree is ticked every 0.1 simulated seconds (`tick=`) until the root is done:

| Node | What it does |
|---|---|
| `Action(fn)` | `fn(ctx)`: `True` or `None` is success, `False` failure, or return `SUCCESS`, `FAILURE`, `RUNNING`. A call that blocks (`arm.move_to`, `pick_and_place`) blocks inside the tick. A call the playground refuses (out of reach) is a failure, with its message as the node's note (in a `Condition` too, and a sensor the robot lacks) |
| `Condition(fn)`, `sensor_is("lidar", "ahead.distance_m", ">", 0.5)` | success when it holds; a sensor the robot lacks is a failed condition, not an error |
| `Wait(seconds)`, `WaitFor(test, timeout)` | running until the time has passed, or the test holds (failure at the timeout) |
| `Sequence(*nodes)`, `Fallback(*nodes)` | all in order, or the first that works; they carry on from the child that was running, `reactive=True` re-checks the earlier ones every tick and stops (halts) the later ones when an earlier one decides |
| `Parallel(*nodes, success="all")` | all in the same tick; `"any"` or a number of successes needed. Children that block (arm calls) run one after the other, not together |
| `Retry(node, times=3, wait=0)`, `Repeat(node, times=None)`, `Inverter(node)`, `Timeout(node, seconds)` | decorators. `Timeout` stops a child that keeps running too long; a child that blocks cannot be stopped in the middle, so a child that ends after more than `seconds` inside one tick fails with a note (`took 10.0 s, over 2 s`) |
| `StateMachine(name, start)` with `State(name, do, enter, exit, result)` | states and `transition(source, target, when)`; a state with `result=SUCCESS` or `FAILURE` ends it; `do` may be a node and `when` may be `SUCCESS` or `FAILURE` of it |

A call that blocks (`arm.move_to`, `arm.pick_and_place`, `robot.pose`) holds the whole tree until it is done, because the clock only runs between ticks.
So two blocking actions in a `Parallel` do not overlap, a `Timeout` around one cannot cut it short, and `max_seconds` of `Tree.run` is checked between ticks.
`Parallel` is for conditions, waits, state machines and calls that only start something (`move`, `stop`, `jog`).

The context `ctx` has `ctx.sim`, `ctx.robot`, `ctx.time`, `ctx.bb` (a dictionary the nodes share) and `ctx.log(text)`, which goes into `tree.summary()` with when each node ended and how.
A tree is a node too, so a state machine goes inside a tree and the other way round:

```python title="behaviors/turn_machine.py" robots="dog,go2"
from robotforge.flow import SUCCESS, Action, State, StateMachine, Tree
from robotforge.script import connect

sim = connect()
robot = sim.robot(sim.robots()[0])

fsm = StateMachine("lap", start="straight")
fsm.add(State("straight", do=Action(lambda ctx: ctx.robot.move(speed=0.25, seconds=1.0), name="walk 1 s")))
fsm.add(State("turn", do=Action(lambda ctx: ctx.robot.move(speed=0.0, turn=1.2, seconds=1.0), name="turn 1 s")))
fsm.add(State("done", result=SUCCESS))
fsm.transition("straight", "turn", when=SUCCESS)  # the walk finished well
fsm.transition("turn", "done", when=SUCCESS)

tree = Tree(fsm, name="one corner")
print(tree.run(sim, robot), "in", round(sim.time, 1), "s")
```

`Tree.run(sim, robot, tick=0.1, max_seconds=120, blackboard=None, show=True)` returns the status of the root; a tree that is still running after `max_seconds` simulated seconds stops with a failure.
A node may appear once in a tree: make a second one for a second place.

