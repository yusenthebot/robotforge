# The open robot catalogue (G12a)

Over a hundred and seventy open robots can be added by name: arms, robot dogs, humanoids, wheeled bases, hands and a few drones, from many makers.
The tuned library (Go2, Panda, G1 and the others in `list_library`) is unchanged and comes first.
The catalogue is what is left of the 208 open robot descriptions once those are taken out: 175 robots (68 arms, 55 humanoids, 21 wheeled, 15 quadrupeds, 11 hands, 4 others, 1 aircraft).

## Use

```
list_library(kind="arm", search="kinova")     # what is there; kind: arm quadruped humanoid wheeled hand aircraft other
add_robot(name="arm", library="j2n6s300")     # downloads its folder once, imports it, says what it is
```

`list_library` without arguments is the tuned library plus a count of the catalogue by kind.
With `kind` or `search` it lists the matches from both, by name, maker or tag, best match first.
A robot that the tuned library has under another name (the Menagerie Go2 is `go2_mj`) is not listed twice: asking for it gets the tuned one.

`add_robot` answers with a `catalogue` block: the kind the structure of the loaded robot says it is, a one-line description, the mass, joints and motors, and which tools work on it.
The kind comes from the compiled model, not from the tags of the list: a hand whose file names a site `tcp` is a hand, two toe-and-heel pairs under a tall body are two feet, a robot on four feet is a dog.
A world remembers `catalogue: <name>` in its robot entry (a new optional field, nothing else in `world.yaml` changed) and loads the robot again from the cache.

## How it works

- `src/robotforge/design/catalogue.json` is a plain copy of what the robot_descriptions package knows: the repository and commit of each robot, its model file, its makers' name, license and tags.
  `scripts/catalogue_gen.py` makes it (`uv run --no-project --with robot_descriptions python scripts/catalogue_gen.py`); the package is not a dependency and nothing is cloned at import time.
- A download takes only the robot's folder at the pinned commit: a shallow sparse git checkout, or a streamed archive of the commit when git is missing.
  Folders are cached in `~/.cache/robotforge/catalogue/<repository>-<commit>/` (`ROBOTFORGE_CACHE` moves it) in the layout of the repository, so a URDF finds its sibling packages.
  A robot whose files name a package outside its folder (the Toyota HSR, the Franka FER) makes the import ask for the whole repository once and go again.
- MJCF files are read as they are, after the repairs below; URDF and xacro go through the import doctor.
  Arms get a `tcp` site pointing the way the wrist points and a bare `flange` site (move_arm and the mount tool steer by them), found from the kinematic chain: motor-driven joints that end in a hand, a gripper or nothing, at least 20 cm of links long, at most two of them.
- Tests stay offline: they put a fake repository in the cache folder (`ROBOTFORGE_REPOS`, `ROBOTFORGE_CACHE`).

## What adding real robots taught the import doctor

Found by adding 20 random robots and then 105 more for real, fixed with a test each:

- An MJCF with numbers written with commas (`quat="0.7, 0, 0.7, 0"`, LimX Oli) did not load: the repaired copy is read, the file is left alone.
- Torque-motor MJCF (unitree_mujoco's A1, Menagerie's torque ToddlerBot) have no way to hold a pose: motors on limited joints become position servos with the torque limit kept (stiff enough to give it at 0.25 rad of error); a model with a `_pos` sibling uses that.
- A dog with no keyframe lies down on its trunk, its zero pose outside its joint limits: the same hip, thigh and knee angles on all four legs are searched, feet under the hips, legs bent.
- Arms from files sagged under their own weight (an iiwa 7 cm short of its point): catalogue arms hold themselves against gravity in `move_arm` as the library's do.
- A xacro macro that hands its own `origin` block to another macro inserted itself for ever (Toyota HSR, Hello Robot Stretch SE3): a block now means what it meant where it was written.
- xacro formulas with `str.find`, `join` and list comprehensions (OpenArm v1) are read.
- An inertia a hair above singular (Stretch RE1's thin rod) was positive until it was written out with nine digits, then MuJoCo refused it: tensors within 1e-8 of singular are rebuilt like the zero ones.
- A humanoid with a leg joint that has no motor of its own (Oli, Cassie: linkages) failed with a bare `5`; it now says so.
- A hand that names a site `tcp` was taken for an arm, a Crazyflie file with no motor for a robot.

## The 20 robots (seed 12)

`uv run python scripts/catalogue_bench.py` draws them (arm 5, quadruped 3, humanoid 4, wheeled 3, hand 3, aircraft 1, other 1, by tag), adds each by name to a fresh world and tries what its kind should do: an arm reaches a point, a legged robot stands for six seconds and walks, a wheeled robot drives, a hand closes.
Standing is judged by how tall the robot's bodies stay, not by where its base origin is.
The pictures are `scripts/screenshots.py` shots `g12_*` of the real viewer; `scripts/catalogue_grid.py` puts them in one grid.

Result: all 20 load, 17 stand, 14 move.

| robot | structure says | loads | stands | moves | why not |
|---|---|---|---|---|---|
| ur7e | arm | yes | yes | reaches a point to 0.0 cm | |
| open_manipulator_x | arm | yes | yes | reaches 0.6 cm | four joints: only some points are reachable |
| yumi | arm (two) | yes | yes | reaches 1.7 cm | |
| sawyer_mj | arm | yes | yes | reaches 0.0 cm | |
| iiwa7_epfl | arm | yes | yes | reaches 0.0 cm | |
| r1_pro | arm (two, on a base) | yes | yes | reaches 0.1 cm | tagged wheeled, but its base has no wheels the drive code finds: it is an arm to the tools |
| b1 | quadruped | yes | yes | walks 2.5 m, then falls | the generic gait is tuned for the library's dogs; a 63 kg B1 tips |
| a1_mj | quadruped | yes | yes | walks 1.2 m upright | needed the stance search and servos |
| anymal_d | quadruped | yes | yes | walks 1.4 m upright | |
| oli_mj | humanoid | yes | no | no | legs are linkages with passive joints: no balance controller fits, it falls over |
| h1_2 | humanoid | yes | no | walks 0.6 m, falls | the humanoid balance code is tuned for G1, T1 and OP3 |
| toddlerbot_2xm_mj | humanoid | yes | yes | no | stands on its servos; no gait fits its body (RL II can learn one) |
| tron2_wf_tron2a_mj | humanoid (wheel-legged) | yes | no | no | needs a balance controller for a wheel-legged biped |
| stretch_3_mj | wheeled | yes | yes | drives, 0.39 m in 4 s | slow: 0.39 m in 4 s at the 0.3 m/s asked |
| tiago_mj | wheeled | yes | yes | drives 1.3 m | |
| shadow_dexee_mj | hand | yes | yes | fingers close | |
| ability_hand | hand | yes | yes | fingers close | |
| aero_hand_open | hand | yes | yes | fingers close | |
| cf2 | other | yes | yes | no | the Crazyflie URDF has no motors: a rigid body (the library's `crazyflie` flies) |
| finger_edu | other | yes | yes | joints move | |

A first add took 5 to 60 seconds, about two minutes for the models with tens of megabytes of meshes (Stretch 3, TIAGo++, R1 Pro); a repeat takes under a few seconds.

## A wider load check

105 more robots drawn at random (two seeds, 45 and 60) were added to see only whether they load.
Eight failed the first time (97 of 105 loaded); after the fixes above 102 do.
What still fails:

- `openarm_v2`: its xacro builds a preset with Python dictionaries the reader does not evaluate.
- `tiago_official`: its xacro needs packages from three other repositories (`pal_urdf_utils` and others).
- `stretch_4`: its xacro includes a file chosen by a xacro argument with no default.

The eight were: four xacro files that needed a sibling package or the block scope fix (Toyota HSR, Franka FER, Stretch SE3, OpenArm v1), `openarm_v2`, `tiago_official`, `stretch_4`, and Stretch RE1, whose URDF has a thin rod with an inertia a rounding error from singular.

## Limits

- Legged robots outside the library walk with a gait found from their legs (`quadruped.for_robot`, `humanoid.for_robot`); it is tuned for the library's robots, so `add_robot` says to check `pose` `upright` after a move.
- Humanoids with linkage legs, wheel-legged robots and robots whose file has no motors do not stand or move by themselves.
- Arms with few joints reach only some points: the planner keeps the tool pointing down.
- The gripper of a catalogue arm is not wired to `gripper` and `pick_and_place`: mount one on the flange.
