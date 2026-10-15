# Robot scripts

Scripts are plain Python files in the world's `behaviors/` folder.
`run_script` starts them in their own process and runs them in lockstep with the simulation: time passes only when the script calls `sim.step()` or a blocking action, so the script can read, think and act without the world running away.
Limits: 60 simulated seconds by default (`max_sim_seconds` up to 300) and 180 seconds of computer time.
Shown live when the viewer is open.

## API

```python
from robotforge.script import connect

sim = connect()  # only works when started by run_script

sim.time  # simulated seconds since the world started
sim.step(seconds=0.02)  # let time pass; returns the new time

sim.position(name)  # [x, y, z] in meters: an object, a robot (its base) or a body like "dog/trunk"
sim.heading(robot)  # degrees, counter-clockwise from +x
sim.joints(robot)  # {joint: angle in radians or slide in meters}
sim.sensors(robot)  # {sensor: value}, e.g. IMU, foot touch, tool position

sim.set_controls({"arm/shoulder": 0.5})  # raw actuator targets (names from describe_model)
sim.move(robot, speed=0.2, turn=0.0)  # start walking or driving; keeps going until stop()
sim.stop(robot)
sim.pose(robot, "wave")  # humanoid pose; waits; True if still upright
sim.move_arm(robot, [x, y, z], gripper=None)  # "open" / "close" / None; waits; returns the tool position
sim.gripper(robot, "open")  # waits
sim.pick_and_place(robot, obj, target)  # target: object name or [x, y]; waits; True if it landed there

print(...)  # everything printed comes back in run_script's output
```

A refused request raises `ScriptError` with the reason (unknown name, out of reach, ...).
Newer installs may add more calls (for example `sim.run_policy(robot, name)` with the rl extra): check the `write_script` tool description.

## Writing good scripts

- Start with a docstring that says what the script does, then constants for the numbers a learner may want to change.
- Closed loop: measure (`position`, `heading`, `sensors`) inside the loop and stop on what you measured, not on elapsed time.
- Step in small slices (0.02 to 0.05 s) inside loops, and always `stop()` a robot you started.
- Print the result the user cares about (distances, corners, success) so `run_script` can be checked.
- Comment the idea, not the syntax; comments in the user's language are fine.

## Example: a closed-loop square (tested)

```python
"""Drive the rover around a 0.5 m square, measuring as it goes (closed loop)."""

import math

from robotforge.script import connect

sim = connect()
ROBOT = "rover"
SIDE = 0.5  # meters per side
SPEED = 0.2  # m/s on the straight parts
MAX_TURN = 1.0  # rad/s, fastest turn in place


def heading_error(target_deg):
    """How far the rover still has to turn, in degrees, between -180 and 180 (positive = turn left)."""
    return (target_deg - sim.heading(ROBOT) + 180) % 360 - 180


def drive_straight(distance):
    start = sim.position(ROBOT)
    sim.move(ROBOT, speed=SPEED)
    while math.dist(start[:2], sim.position(ROBOT)[:2]) < distance:
        sim.step(0.02)  # let 20 ms of simulated time pass, then measure again
    sim.stop(ROBOT)


def turn_to(target_deg):
    # Proportional control: turn fast when far from the target heading, slowly when close.
    while abs(error := heading_error(target_deg)) > 1.0:
        rate = max(0.15, min(MAX_TURN, abs(math.radians(error)) * 2.0))
        sim.move(ROBOT, speed=0.0, turn=math.copysign(rate, error))
        sim.step(0.02)
    sim.stop(ROBOT)


start = sim.position(ROBOT)
target = sim.heading(ROBOT)
for corner in range(1, 5):
    drive_straight(SIDE)
    target += 90  # left turn at every corner
    turn_to(target)
    x, y, _ = sim.position(ROBOT)
    print(f"corner {corner}: x={x:.2f} m  y={y:.2f} m  heading={sim.heading(ROBOT):.0f} deg")

end = sim.position(ROBOT)
print(f"ended {math.dist(start[:2], end[:2]):.2f} m from the start")
```

Measured output: corners within 0.07 m of the ideal square, ending 0.01 m from the start.

## Example: pick and place step by step (tested)

```python
"""Put the red cube in the tray, one motion at a time, and report where it landed."""

from robotforge.script import connect

sim = connect()
ARM, CUBE, TRAY = "arm", "cube", "tray"
LIFT = 0.12  # m above the table while carrying

cube = sim.position(CUBE)
tray = sim.position(TRAY)

sim.move_arm(ARM, [cube[0], cube[1], cube[2] + LIFT], gripper="open")  # above the cube
sim.move_arm(ARM, [cube[0], cube[1], cube[2]])  # down around it
sim.gripper(ARM, "close")
sim.move_arm(ARM, [cube[0], cube[1], cube[2] + LIFT])  # lift
sim.move_arm(ARM, [tray[0], tray[1], tray[2] + LIFT])  # carry
sim.move_arm(ARM, [tray[0], tray[1], tray[2] + 0.05])  # lower
sim.gripper(ARM, "open")
sim.step(0.5)  # let it settle

x, y, z = sim.position(CUBE)
print(f"cube at x={x:.3f} y={y:.3f} z={z:.3f}")
```

Measured output with the cube at [0.25, -0.12] and the tray at [0.05, 0.28]: `cube at x=0.054 y=0.270 z=0.025`, inside the tray.

When the step-by-step version drops the cube, compare with `sim.pick_and_place(ARM, CUBE, TRAY)`, which plans safe arcs and checks reach first, and explain the difference.
