# The open robot catalogue (G12a)

Over a hundred and seventy open robots can be added by name: arms, robot dogs, humanoids, wheeled bases, hands and a few drones, from many makers.
The tuned library (Go2, Panda, G1 and the others in `list_library`) is unchanged and comes first.
The catalogue is what is left of the 208 open robot descriptions once those are taken out: 175 robots (68 arms, 55 humanoids, 21 wheeled, 15 quadrupeds, 11 hands, 4 others, 1 aircraft).

## Use

```
list_library(kind="arm", search="kinova")     # what is there; kind: arm quadruped humanoid wheeled hand aircraft other
                                              # (also gripper, dog, drone: the names the tuned library's rows use)
add_robot(name="arm", library="j2n6s300")     # downloads its folder once, imports it, says what it is
```

`list_library` without arguments is the tuned library plus a count of the catalogue by kind.
With `kind` or `search` it lists the matches from both, best match first.
Each word of `search` has to start a word of the robot's id, name, maker or tags (`ur` finds the UR arms, not the file format `urdf`).
The plain call also says where the downloads are and how much room they take (`open_catalogue.cache`).
A robot that the tuned library has under another name (the Menagerie Go2 is `go2_mj`) is not listed twice: asking for it gets the tuned one.

`add_robot` answers with a `catalogue` block: the kind the structure of the loaded robot says it is, a one-line description, the mass, joints and motors, and which tools work on it.
The kind comes from the compiled model, not from the tags of the list: a hand whose file names a site `tcp` is a hand, two toe-and-heel pairs under a tall body are two feet, a robot on four feet is a dog.
A walker or humanoid also gets `checked`: a few seconds of simulated time of the new robot alone in a scratch world, four standing at home and six walking slowly.
It says whether the robot stood, walked upright (how far), fell over (after how far), got nowhere or has no gait, and the description follows it: a robot that collapsed at home is "a four-legged robot that collapses at home", not "a walking robot", and a gait that fell is offered with that said.
The trial is kept next to the downloads, keyed by the robot, its commit and a hash of the code it depends on, so a robot is tried once until that code changes; it costs a fraction of a second.
A world remembers `catalogue: <name>` in its robot entry (a new optional field, nothing else in `world.yaml` changed) and loads the robot again from the cache.

## How it works

- `src/robotforge/design/catalogue.json` is a plain copy of what the robot_descriptions package knows: the repository and commit of each robot, its model file, its makers' name, license and tags.
  `scripts/catalogue_gen.py` makes it (`uv run --no-project --with robot_descriptions python scripts/catalogue_gen.py`); the package is not a dependency and nothing is cloned at import time.
- A download takes only the robot's folder at the pinned commit: a shallow sparse git checkout, or a streamed archive of the commit when git is missing.
  Folders are cached in `~/.cache/robotforge/catalogue/<repository>-<commit>/` (`ROBOTFORGE_CACHE` moves it) in the layout of the repository, so a URDF finds its sibling packages.
  A robot whose files name a package outside its folder (the Toyota HSR, the Franka FER) makes the import ask for the whole repository once and go again.
- MJCF files are read as they are, after the repairs below; URDF and xacro go through the import doctor.
  Arms get a `tcp` site pointing the way the wrist points and a bare `flange` site (move_arm and the mount tool steer by them), found from the kinematic chain: motor-driven joints that end in a hand, a gripper or nothing, at least 20 cm of links long, at most two of them.
- A URDF or xacro robot is imported once per `ROBOTFORGE_HOME`: the doctor's plan and report are kept under `cache/import` (keyed by the file, its size and modification time and the doctor's version), so a repeat add or a reopened world takes 0.1 to 0.2 s.
  The first add in a fresh home runs the doctor and its stability check: 0.1 to 3 s for most robots, up to 10 s for the biggest (Ginger 9 s).
  `scripts/catalogue_bench.py` uses a throwaway home for every robot, which is why its add times are the first-add numbers.
- Nothing deletes a download.
  `uv run python scripts/catalogue_cache.py` lists them by size (4.4 GB on the machine that made the tables); `--remove NAME` or `--older-than DAYS` frees room, and a removed folder comes back on use.
  Folders of MuJoCo Menagerie that the tuned library already fetched are used as they are, not fetched again.
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

Found by the reviewer's second pass, again with a test each:

- Unitree's Aliengo gives hips and calves a range and the thighs none: only the limited joints became servos, so the legs mixed servos and torque motors and no walker took them.
  A joint with no range that carries a limited joint further out (a thigh, not a wheel) is driven over the full turn too.
- Unitree's B2 URDF writes one `<material>` per mesh colour into the same `<visual>`, which MuJoCo refuses (`repeated element: material`): the first is kept.
- `run_fast` waited for ever when a simulation diverged: MuJoCo resets it to time 0 and the clock never reached the end.
  It now ends the run (the trial of H1_2 had sat at full speed for twenty minutes).
- xacro formulas share a budget of loop passes (200 000), so `[[a for a in X] for b in X]` over a long list stops with an error instead of hanging on a hostile file.
- Descriptions: an arm counts only the joints between its base and its tool point (YuMi is two 7-joint arms, not "an 18-joint arm"; the OpenMANIPULATOR-X has 4); wheels count as wheels, not feet, so the wheel-legged TRON robots and Stretch 2 are wheeled, with an honest line that no drive controller fits them; a rigid body is not "a 0-joint robot with 1 legs".

## The 20 robots (seed 12)

`uv run python scripts/catalogue_bench.py` draws them (arm 5, quadruped 3, humanoid 4, wheeled 3, hand 3, aircraft 1, other 1, by tag), adds each by name to a fresh world and tries what its kind should do.
How the table is judged, since the numbers are easy to read as more than they are:

- Stands: a robot with a free base keeps its bodies at 70 % of their height and upright for six seconds (a robot on its side stops the test early); an arm or a hand stands if no joint drifts more than 0.35 rad.
  A one-body robot (cf2) stands trivially.
- An arm tries up to 11 points in order (round where its tool starts, then in front of its base) and passes with the first that it reaches within 3 cm.
  The table says which point that was: "reaches" means some reachable point, not that every point in range is.
- A walker is asked for 0.2 m/s for six seconds (1.2 m) and passes if it ends upright having covered more than 30 % of that.
  A wheeled base is asked for 0.3 m/s for four seconds (1.2 m) with the same rule, so 0.39 m is a pass at a third of the speed asked.
- A hand closes every motor to its limit for two seconds and passes if a joint moved more than 0.1 rad.
- A robot that fell over at home is not sent to walk.

The same behavior runs in the pictures: `scripts/screenshots.py` shots `g12_<name>` (the real viewer, 1920x1080) run the test headless to its end and pause there, so the large picture is the end of the test, with its time on it, and the corner picture (`g12_<name>_start`) is the robot as added.
`scripts/catalogue_grid.py` puts them in one grid; the grid and the pictures of the last run are in `/private/tmp/claude-501/-Users-yusenthebot-Desktop/72aa2bf4-b624-4808-a637-20c7f41214de/scratchpad/g8/out/catalogue/` (not committed).

The table is from the picture run (it is the one the pictures show).
The gait is sensitive to the instant it starts: the table run of the same code let anymal_d walk 1.4 m upright and b1 fall after 2.5 m, the picture run let anymal_d fall after 1.9 m and b1 after 0.6 m.
Read a legged result as "may", not "does".

All 20 load; 17 stand; 13 move in the picture run (14 in the table run: anymal_d).

| robot | structure says | loads | stands | moves | why not |
|---|---|---|---|---|---|
| ur7e | arm | yes | yes | reaches a point to 0.0 cm (6th point) | |
| open_manipulator_x | arm | yes | yes | 0.6 cm (8th) | four joints: only some points are reachable |
| yumi | arm (two) | yes | yes | 1.7 cm (9th) | |
| sawyer_mj | arm | yes | yes | 0.0 cm (1st) | |
| iiwa7_epfl | arm | yes | yes | 0.0 cm (6th) | |
| r1_pro | arm (two, on a base) | yes | yes | 0.1 cm (3rd) | tagged wheeled, but its base has no wheels the drive code finds |
| b1 | quadruped | yes | yes | walks 0.6 m, then falls | the generic gait is tuned for the library's dogs; a 63 kg B1 tips |
| a1_mj | quadruped | yes | yes | walks 1.2 m upright | needed the stance search and servos |
| anymal_d | quadruped | yes | yes | walks 1.9 m, then falls (1.4 m upright in another run) | |
| oli_mj | humanoid | yes | no | no | linkage legs with passive joints: no balance controller fits, it falls over after 2.9 s |
| h1_2 | humanoid | yes | no | no | falls over after 4.9 s (the picture shows it on its side; an earlier picture showed it standing because MuJoCo had reset its diverged simulation) |
| toddlerbot_2xm_mj | humanoid | yes | yes | no | stands on its servos; no gait fits its body (RL II can learn one) |
| tron2_wf_tron2a_mj | wheeled (wheel-legged) | yes | no | no | needs a balance controller for a wheel-legged biped; falls after 1.8 s |
| stretch_3_mj | wheeled | yes | yes | drives 0.39 m of 1.2 m | slow: 0.10 m/s achieved of the 0.3 m/s asked |
| tiago_mj | wheeled | yes | yes | drives 1.32 m | 0.33 m/s |
| shadow_dexee_mj | hand | yes | yes | fingers close (1.5 rad) | |
| ability_hand | hand | yes | yes | fingers close (2.1 rad) | |
| aero_hand_open | hand | yes | yes | fingers close (2.0 rad) | |
| cf2 | other | yes | yes | no | the Crazyflie URDF has no motors: a rigid body (the library's `crazyflie` flies) |
| finger_edu | other | yes | yes | joints move (2.5 rad) | |

A second draw (`--seed 99`, the failures are not the same ones): 20 load, 17 stand, 13 move.

| robot | structure says | stands | moves | why not |
|---|---|---|---|---|
| ur12e, ur10_official, j2s7s300, j2n7s300, omx_f | arm | yes | all reach a point (0.0 to 0.3 cm; the 1st to 6th point) | |
| anymal_b, aliengo_mj | quadruped | yes | walk 1.5 m and 1.2 m upright | aliengo_mj did not move at all before the thigh fix |
| solo | quadruped | yes | walks 0.6 m upright | slow: 0.11 m/s of 0.2 |
| bolt | humanoid | no | no | collapses at home |
| h1 | humanoid | yes | no | stands; no gait fits it (H1_2 gets a gait that falls, H1 none: the generic search finds H1_2's legs and not H1's) |
| tron1_wf_tron1a_mj | wheeled (wheel-legged) | no | no | collapses at home; no balance controller |
| stretch_mj | wheeled | yes | no | its drive is two abstract actuators (`forward`, `turn`), no wheel motors for the drive code |
| stretch_se3 | wheeled | yes | no | drove 0.2 m of 1.2 m: 0.05 m/s |
| luna, dynamixel_2r_mj | other | yes | yes | |
| ginger | other | yes | no | its joints do not move when commanded |
| shadow_hand_mj, robotiq_2f85_v4_mj, aero_hand_open_mj | hand | yes (aero: drifts 2.3 rad at rest) | fingers close | |
| cf2 | other | yes | no | no motors |

## What the trial of the legs shows

`add_robot` tries every walker and humanoid it adds (see Use).
All 55 of those downloaded here (21 would not stand at home, 18 stand with no gait, 8 fell, 6 walked, 2 stalled):

- Walked 1 m or more at 0.2 m/s upright: a1_mj, aliengo_mj, anymal_b, anymal_b_mj, anymal_d (laikago 0.4 m).
- Stood and the gait fell over: aliengo (URDF; its MJCF walks), b1, b2, hyq, solo, mini_cheetah, icub, h1_2.
  Of the twelve dogs tried, six walked and six fell (aliengo, b1, b2, hyq, solo, mini_cheetah); a dog that walks in one run can fall in the next, so the answer says so at add time.
- Stood and the gait got nowhere: ergocub, valkyrie.
- Stand, no gait fits: apollo_mj, atlas_drc, atlas_v4, draco3, h1, luna, luna_mj, minitaur, oli, sigmaban, talos_mj, toddlerbot (3 variants), tron2_dach and sfyg, upkie, wl_p311d.
- Collapse at home within 4 s: bolt, cassie (2), elf2 (2), h1_mj, h1_2_mj, jaxon, jvrc_mj, n1_mj, oli_mj, r2, spryped, and the TRON1 and TRON2 wheel-, point- and straight-foot robots (9).

## A wider load check

105 more robots drawn at random (two seeds, 45 and 60) were added to see only whether they load.
Eight failed the first time (97 of 105 loaded); after the fixes above 102 do.
Unitree's B2 was not in those draws and failed the same way on its first real add.
What still fails:

- `openarm_v2`: its xacro builds a preset with Python dictionaries the reader does not evaluate.
- `tiago_official`: its xacro needs packages from three other repositories (`pal_urdf_utils` and others).
- `stretch_4`: its xacro includes a file chosen by a xacro argument with no default.

The eight were: four xacro files that needed a sibling package or the block scope fix (Toyota HSR, Franka FER, Stretch SE3, OpenArm v1), `openarm_v2`, `tiago_official`, `stretch_4`, and Stretch RE1, whose URDF has a thin rod with an inertia a rounding error from singular.

## Limits

- Legged robots outside the library walk with a gait found from their legs (`quadruped.for_robot`, `humanoid.for_robot`); it is tuned for the library's robots and falls over for about half of the catalogue's dogs, so `add_robot` runs a trial and says what happened.
- Humanoids with linkage legs, wheel-legged robots and robots whose file has no motors do not stand or move by themselves.
- Arms with few joints reach only some points: the planner keeps the tool pointing down.
- The gripper of a catalogue arm is not wired to `gripper` and `pick_and_place`: mount one on the flange.
