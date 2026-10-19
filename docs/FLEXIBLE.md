# Flexible interfaces (G5)

Three things were added to the script and viewer interfaces: task logic as behavior trees and state machines, a gamepad as a second input, and recording, replaying and exporting demonstrations.
Each was run for real: real Menagerie models, the real viewer in headless Chrome, and what follows are the numbers and files of those runs.

## Behavior trees and state machines

`robotforge.flow` is a small library for scripts: `Sequence`, `Fallback`, `Parallel`, `Retry`, `Repeat`, `Inverter`, `Timeout`, `Action`, `Condition`, `sensor_is`, `Wait`, `WaitFor` and `StateMachine` with `State`.
Every node is documented with an example in [EXTENDING.md](EXTENDING.md#behavior-trees-and-state-machines), and every example is run by the test suite.
A tree is shown in the viewer while it runs (the panel at the bottom left): one row per node, the node that runs lit, a note such as `attempt 2 of 3` on a retry.
A call the playground refuses inside an action (out of reach, a bin too small for the cube) is a failed node with the message as its note, so retries and fallbacks can deal with it.

The proof is a Go2 carrying a PiPER that brings a cube to a bin, with a condition on its IMU, a retry and a check that the cube really is in the bin.
Halfway through the check, someone knocks the cube out of the bin (a push on the cube: the only thing staged).

```text
 45.76 s  standing level: success
 45.76 s  fetch the cube, put it in the bin: success
 48.76 s  retry 3x: attempt 1 of 3 failed
 48.76 s  fetch and place: failure
105.41 s  fetch the cube, put it in the bin: success
108.41 s  bring the cube: success
```

The first attempt put the cube in the bin, the check three seconds later found it gone, the retry sent the robot after it and the second attempt worked.
The film (not committed, the files are large) shows the panel following it.

## Gamepad

The viewer reads `navigator.getGamepads()` once a frame (no library) and maps the pad by the kind of the selected robot (`padmap.js`):

| Robot | Left stick | Right stick | Buttons |
|---|---|---|---|
| walks, drives or swims | forward and back, turns | turns | d-pad at half strength |
| aircraft | forward, sideways | turns, climbs | triggers climb and descend |
| underwater robot | forward, turns | heaves up and down | |
| arm | moves the tool over the floor (up on the stick is away from the camera) | raises and lowers it | A or right trigger closes the gripper, B or left trigger opens it |
| robot that goes and has an arm | as the first row | | Y switches between driving and the arm |

Drive reaches the robot as the same `teleop` message the keys send (so the keyboard still works, and the pad wins an axis it is pushing), an arm as `teleop_arm`: the tool is jogged by inverse kinematics at the stick's velocity, held where it is when the stick returns to centre, and held too if the pad goes quiet for half a second.
A pad that goes away, a tab that is hidden and a robot that is deselected each stop what they were driving.

Checked three ways:

- `tests/test_viewer_padmap.py` runs the mapping table in Node.
- `tests/test_viewer_gamepad.py` replaces `navigator.getGamepads` in headless Chrome with a pad the test moves and watches the simulation: the dog walks forward, stops when the stick is let go, turns with the right stick, the arm's tool follows the stick, A closes and B opens the gripper, an unplugged pad stops the dog, the keyboard still drives, a phone layout.
- A film of the same events (not committed) shows the pad driving the Go2.

No physical pad was used: the events are simulated through the browser's own Gamepad API.

## Demonstrations

`demonstration(action="start" | "stop" | "replay" | "export" | "list")` is the one new tool; scripts use `robot.record(name)` and `robot.replay(name)`.
A recording keeps, at 20 frames a simulated second, the joint positions, the actuator targets, the base pose and the sensor readings, and the commands that steered the robot (keyboard, gamepad, `move`, `move_arm`, a script), each with its time.
A fetch's own steps inside the simulation are not recorded as commands: only what was asked from outside.

- Replay `commands` (default) gives the commands again on any robot of the same kind; speeds are scaled for a slower robot, and arm targets and the places of `go` and `pick_and_place` are kept relative to the robot's own frame, so another robot goes the same way from where it stands.
- Replay `joints` puts the robot in the first pose and plays the actuator targets back, for the same actuators; it is open loop, so a robot that walks, drives or flies is refused.
- A second replay on a robot takes over from the first.
- Export writes JSON or NPZ (never over a file that exists or another demonstration, unless `overwrite=true`; a reading missing in some frames is `null` in the JSON and NaN in the NPZ); LeRobot's parquet format needs `pyarrow`, which robotforge does not install (a gate).

Real models:

| Run | Result |
|---|---|
| Go2 driven 5.5 s with `move`, recorded | 112 frames, 12 joints, 4 commands, walked (1.12, 0.54) m |
| replayed on an A1 (a quarter turn to its left) | moved (-0.61, 1.17) m, which is the Go2's walk turned 90 degrees, upright |
| replayed on a template dog | speeds scaled to 0.61 (it is slower), moved (0.72, 0.39) m, upright |
| Panda reach, jogged 0.14 m, gripper closed, recorded | 75 frames, 12 commands; replayed in `joints` mode from a reset the tool ends at (3.398, 0.142, 0.296) m, where the jog left it |
| Go2 driven by simulated pad events 11.6 s, replayed on an A1 | 232 frames, 4 commands; JSON 64.6 kB, NPZ 57.2 kB |

The film of the last row (not committed) shows: the pad drives the Go2 while the page says "Recording pad_lap", then the A1 replays it ("Replaying pad_lap").
