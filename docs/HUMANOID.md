# Humanoids: standing on their own

A humanoid you add from the library or the open catalogue stands on its own feet, holds a load and takes a push, whatever its joints are called.
Every two-legged robot stands by whole-body balance except T1 and OP3, which keep their preview-control gait (it stands them as well, with a load and a push).
G1 stands by whole-body balance until its first `move`, then walks on its learned policy: the policy alone falls under a load of a fifth of its mass, the balance holds it, and it takes a push of 0.2 to 0.3 m/s instead of the policy's 0.4.
A G1 with an arm mounted on it stays on the policy, because the arm's controller and the balance would both drive the joints of one tree.

## What decides it

- **Kind from the feet.**
  A floating robot that stands on two feet (and not on wheels) is a humanoid (`behaviors/stance.py`), however it names its joints.
  Before this, Talos and H1_2 read as snakes, iCub as a dog and most others as "other".
  A foot is the end body of a leg that reaches the floor; the toe and heel of one foot, or two shells of it, count once.
- **A stance it can hold.**
  Many keyframes are drawn to look good, not to balance: Menagerie's H1 has its weight 13 cm behind the middle of its feet, on its heels; Fourier's GR1 has its zero pose with the toes up.
  When a robot's file is read, its feet are put flat on the floor where they are and its centre of mass is brought over the middle of the feet (a third of the way toward the ankles, for robots with weak ankles), staying as near the keyframe as that allows (`behaviors/stance_ik.py`, `design/mjcf_stance.py`).
  The robot starts in that pose (the note says "moved N joints to balance").
- **Whole-body balance** (`behaviors/whole_body.py`) holds it there.
  Every 5 ms it reads the real model (mass matrix, Jacobians, bias force) and the contacts MuJoCo finds between the robot and anything fixed in the world.
  It then solves one small problem for the accelerations of all joints and the contact forces together: the feet do not move, the forces push and never pull and never slip (a friction pyramid at every corner of each foot's print), the dynamics of the floating base hold, and what is left goes to three wishes: the centre of mass over the middle of the feet, the trunk level, every joint near its stance.
  The torques that follow are turned into the control of whichever actuator the joint has: a position servo is aimed a little off where its joint is by torque over stiffness (a stiff one also where the balance wants the joint a period on), a torque motor gets torque over gain.
  Nothing about legs is assumed: the number of joints, passive joints, a waist, hands with twenty joints, torque motors or servos all work.
  The servo's own damper is left on at the ankles and feet (cancelled there, a foot rocked on the soft floor) and cancelled elsewhere only as far as the joint's inertia allows (cancelled in full, the small joints rang).
- A push is taken by shifting the weight over the feet, bending at hips and ankles, until still; there are no steps, so the size a robot takes is what its feet can carry.

`robots` says `balance` for a robot held up this way, and `walk_hint` for the ones that cannot walk (no `max_speed` is given for them).
`move`, `pose` and `go_to` refuse these robots with a message that says they stand by balance and that no walk has been shown to carry them.
The gait found from a robot's legs (`humanoid.for_robot`) used to be handed over at the first `move`; measured on these robots it dropped oli, atlas_v4, toddlerbot, talos, tron2_sf_tron2a_mj, gr1 and n1 within one to four seconds (oli went through the floor) and set h1_2's simulation off with a numerical instability warning, so the swap is gone.
Only a gait tuned for the robot is used: our template's, T1's and OP3's, and G1's policy.

## How each one does

`uv run python scripts/balance_bench.py --robots all` puts every humanoid alone in an empty world, leaves it with what stands it up when it appears, and watches:

- six seconds standing: upright (trunk tilt under 20 degrees), no lower than 85 percent of its height, and never fallen over on the way (a whole-body balance that declared the robot fallen and fell back on its servos counts as a failure, however upright the robot ends);
- the same for five seconds with a backpack of 20 percent of its own mass put on over a second and a half;
- a shove: 0.1, 0.2, 0.3 and 0.4 m/s added to the trunk's velocity at once forwards, backwards, to the left and to the right.
  Every speed is tried, not up to the first failure, and the table gives the largest one the robot was still standing after (0.0: none).
  No robot threw a gap in this run (a smaller shove failing where a larger one held); the bench marks one with a star if it does.

What the numbers are, so that they are not read as more:

- A shove is an impulse of velocity on the trunk, a single event, not a force on a spring and not a person's push on a moving robot.
  For scale, a person standing still is thrown off balance by about 0.3 m/s.
- The load is a mass added to the trunk in the model, and the balance reads the mass from the model on every tick.
  It is a load the controller can feel; a load that arrives unannounced and is not in the model would be felt only as a disturbance.
- "Standing" does not say how the robot stands.
  The column `legs` is the height of the trunk over the feet divided by the length of the leg as drawn: 1.00 is straight, under 0.80 is a squat.
  The robots marked squat (OP3, MuJoCo's humanoid at 0.45, ToddlerBot, the Tron2 pair) stand because they are drawn low, and their numbers say less about balance than the others'.
- One run each.
  The runs are deterministic, but a robot at the edge of what it takes may flip with a different start.
- `CPU s per sim s` is the CPU time of the process for one second of simulation, taken while the machine was loaded by other work, so read it as a ratio between robots and not as a speed.

Numbers at 2026-10-20.

51 of the robots read as humanoids; 37 stand; of the 46 held by whole-body balance, 32 stand and 23 of those carry the load.
G1, which stood on its learned policy in the first version of this table (it took 0.4 m/s every way and fell under the load), now stands by balance until its first move: 0.2 to 0.3 m/s and the load.

| robot | mass (kg) | controller | legs | 20% load | largest shove f/b/l/r (m/s) | CPU s per sim s |
|---|---|---|---|---|---|---|
| adam_lite_mj | 58 | whole-body balance | 0.88 | yes | 0.3/0.2/0.3/0.3 | 0.30 |
| atlas_drc | 182 | whole-body balance | 0.88 | yes | 0.3/0.2/0.4/0.4 | 0.34 |
| atlas_v4 | 182 | whole-body balance | 0.88 | yes | 0.2/0.2/0.4/0.4 | 0.45 |
| booster_t1 | 32 | preview-control gait | 0.83 | yes | 0.4/0.4/0.4/0.4 | 0.06 |
| booster_t1_mj | 32 | preview-control gait | 0.83 | yes | 0.4/0.4/0.4/0.4 | 0.09 |
| draco3 | 36 | whole-body balance | 0.97 | no | 0.3/0.3/0.4/0.4 | 0.37 |
| elf2_mj | 31 | whole-body balance | 0.94 | yes | 0.3/0.2/0.2/0.2 | 0.33 |
| ergocub | 57 | whole-body balance | 0.84 | yes | 0.4/0.4/0.4/0.4 | 1.21 |
| g1 | 33 | whole-body balance | 0.92 | yes | 0.2/0.2/0.3/0.3 | 0.51 |
| g1 | 33 | whole-body balance | 0.92 | yes | 0.2/0.2/0.3/0.3 | 0.37 |
| g1_mj | 33 | whole-body balance | 0.92 | yes | 0.2/0.2/0.3/0.3 | 0.37 |
| gene01 | 80 | whole-body balance | 0.90 | yes | 0.3/0.2/0.4/0.4 | 0.85 |
| gr1 | 53 | whole-body balance | 0.91 | yes | 0.4/0.2/0.4/0.4 | 0.53 |
| h1 | 52 | whole-body balance | 0.85 | yes | 0.4/0.2/0.4/0.4 | 0.36 |
| h1_2 | 67 | whole-body balance | 0.91 | yes | 0.3/0.2/0.4/0.4 | 0.56 |
| h1_2_mj | 67 | whole-body balance | 0.91 | yes | 0.3/0.3/0.4/0.4 | 0.62 |
| h1_mj | 51 | whole-body balance | 0.84 | no | 0.3/0.1/0.4/0.4 | 0.31 |
| icub | 33 | whole-body balance | 0.93 | no | 0.2/0.3/0.3/0.3 | 0.63 |
| jvrc_mj | 62 | whole-body balance | 0.88 | yes | 0.3/0.3/0.2/0.2 | 0.39 |
| luna | 56 | whole-body balance | 0.84 | yes | 0.3/0.2/0.3/0.3 | 0.60 |
| mujoco_humanoid_mj | 41 | whole-body balance | 0.45 squat | no | 0.4/0.4/0.4/0.4 | 0.50 |
| n1 | 40 | whole-body balance | 0.90 | yes | 0.4/0.2/0.4/0.4 | 0.56 |
| n1_mj | 40 | whole-body balance | 0.90 | yes | 0.3/0.2/0.4/0.4 | 0.36 |
| oli | 53 | whole-body balance | 0.86 | yes | 0.3/0.3/0.4/0.4 | 0.39 |
| op3 | 3 | preview-control gait | 0.66 squat | yes | 0.4/0.4/0.4/0.4 | 0.07 |
| op3_mj | 3 | preview-control gait | 0.66 squat | yes | 0.4/0.4/0.4/0.4 | 0.03 |
| simple_humanoid | 131 | whole-body balance | 0.88 | no | 0.4/0.2/0.4/0.4 | 0.56 |
| t1 | 32 | preview-control gait | 0.83 | yes | 0.4/0.4/0.4/0.4 | 0.09 |
| talos | 90 | whole-body balance | 0.99 | yes | 0.3/0.2/0.2/0.2 | 0.96 |
| toddlerbot | 4 | whole-body balance | 0.76 squat | yes | 0.2/0.0/0.2/0.2 | 0.91 |
| toddlerbot_2xc_mj | 4 | whole-body balance | 0.75 squat | no | 0.3/0.0/0.3/0.2 | 0.50 |
| toddlerbot_2xm_mj | 4 | whole-body balance | 0.74 squat | no | 0.3/0.0/0.3/0.2 | 0.43 |
| tron1_sf_tron1a_mj | 21 | whole-body balance | 0.81 | yes | 0.3/0.2/0.3/0.4 | 0.37 |
| tron1_sf_tron1b_mj | 21 | whole-body balance | 0.81 | yes | 0.3/0.2/0.3/0.4 | 0.66 |
| tron2_sf_tron2a_mj | 35 | whole-body balance | 0.72 squat | no | 0.3/0.3/0.4/0.4 | 0.79 |
| tron2_sfyg_tron2a_mj | 42 | whole-body balance | 0.72 squat | no | 0.3/0.4/0.4/0.4 | 0.82 |
| valkyrie | 136 | whole-body balance | 0.97 | yes | 0.3/0.2/0.3/0.3 | 0.94 |
| berkeley_humanoid | 16 | whole-body balance | - | - | tilt 91.8 deg, height 0.12 m of 0.58 m after 6 s |
| bolt | 1 | whole-body balance | - | - | tilt 79.9 deg, height 0.09 m of 0.46 m after 6 s |
| cassie | 35 | whole-body balance | - | - | tilt 86.2 deg, height 0.16 m of 0.89 m after 6 s |
| cassie_mj | 33 | whole-body balance | - | - | tilt 68.2 deg, height 0.16 m of 1.12 m after 6 s |
| elf2 | 31 | whole-body balance | - | - | tilt 101.1 deg, height 0.08 m of 0.70 m after 6 s |
| jaxon | 132 | whole-body balance | - | - | tilt 111.9 deg, height 0.67 m of 1.31 m after 6 s |
| jvrc | 62 | whole-body balance | - | - | tilt 20.4 deg, height 0.85 m of 0.86 m after 6 s |
| luna_mj | 57 | whole-body balance | - | - | tilt 88.1 deg, height 0.08 m of 0.82 m after 6 s |
| oli_mj | 54 | whole-body balance | - | - | tilt 92.5 deg, height 0.11 m of 0.91 m after 6 s |
| r2 | 214 | whole-body balance | - | - | tilt 79.3 deg, height 0.41 m of 1.23 m after 6 s |
| spryped | 11 | whole-body balance | - | - | tilt 99.6 deg, height 0.07 m of 0.80 m after 6 s |
| talos_mj | 94 | whole-body balance | - | - | tilt 86.4 deg, height 0.20 m of 1.03 m after 6 s |
| tron1_pf_tron1a_mj | 18 | whole-body balance | - | - | tilt 92.2 deg, height 0.11 m of 0.85 m after 6 s |
| tron1_pf_tron1b_mj | 18 | whole-body balance | - | - | tilt 92.2 deg, height 0.11 m of 0.85 m after 6 s |


## What does not stand, and why

Fourteen robots held by whole-body balance fall within six seconds (the table above lists them with their symptoms).
Some of that is understood, some is not; what follows says which.

- **Linkage legs** (cassie, cassie_mj, oli_mj): the knee and ankle are closed by rods and passive joints, which a joint-space controller cannot drive.
  Oli's URDF twin `oli` stands.
- **Point feet** (tron1_pf, two files): the foot is a single point, so a weight shift cannot hold two of them; it needs steps.
- **Not diagnosed** (berkeley_humanoid, bolt, spryped, jaxon, r2, talos_mj, luna_mj, elf2, jvrc):
  each falls over in six seconds, with a different symptom and no single cause found.
  `elf2` and `jvrc` come from URDF files and their MJCF twins `elf2_mj` and `jvrc_mj` stand, which points at the actuator mapping of the URDF files (gains, torque limits) and not at the controller; `talos_mj` falls where the URDF-built `talos` stands.
  Nothing has been checked beyond that.

Not in the table because they are not two-legged robots on the floor:
apollo_mj has no collision shapes at all (every geom has contype 0), so it falls through the floor;
romeo has a fixed base;
tron1_wf, tron2_wf and tron2_wfyg, rhea, upkie and cookie roll on wheels;
sigmaban and tron2_dach stand by their own servos and are not read as two-footed.

What the controller cannot do, whatever the robot:
it does not step, so a push it cannot take by shifting weight knocks it over;
it does not know a load it is not told of;
and it costs a lot of CPU.

## What it costs

The controller solves a dense non-negative least squares problem and builds the mass matrix 200 times per simulated second.
The last column of the table puts it at 0.3 to 1.2 CPU seconds for one second of simulation (median 0.5) for the robots it holds, against 0.03 to 0.09 for the gait of T1 and OP3.
One standing humanoid probably keeps real time on an unloaded core; four (as in the demo) do not.
Updating at 100 Hz instead of 200 took a third to a half off the CPU time of G1, H1_2, Atlas and ToddlerBot with the same stand, load and shoves within one step (0.1 m/s) of the 200 Hz run, but that was four robots and one run each, and it is not applied; the whole table would have to be measured again for it.

## Walking: tried, not shipped, and not reproducible from this repository

`move` still walks these robots only with a gait found from their legs (`humanoid.for_robot`), and says "stands, but no gait fits" when there is none.
Whole-body balance as a walker was tried and does not work well enough to ship.

What was built: footsteps and a centre of mass path from the preview control of `behaviors/biped.py` (the walk of T1 and OP3), handed to the same balance: the centre of mass follows its path with a spring and damper, the swinging sole follows its own and is kept level, the foot that carries the weight keeps its contact points and the swinging foot's are left out.
Timing was stretched three times over what a pendulum of that height would take, so that the joints have time to follow (a step takes about 3 s, 0.09 m/s at best).

What was found, in the order it was found:

- The centre of mass follows its path closely (a few millimetres) in every run, so the plan and the balance agree.
- The swinging foot does not lift as planned.
  The balance plans the leg's accelerations, the simulation does not deliver them: a servo with a big damper (Talos's knee gives 223 N*m per rad/s, Atlas's 1400) holds the joint back, and cancelling the damper with the velocity of a control period ago makes the loop ring where the joint's inertia is small.
  Cancelling it with a smoothed velocity (50 ms) instead, a little ahead by the planned acceleration, and only as far as stays stable for that joint's inertia, lets the leg move.
- With that Talos walks: 1.5 m in 24 s, six steps, the trunk leaning at most 13 degrees, 3 cm off line.
  It does not turn the way it is told (the turn on the spot came out at 16 degrees for an asked 86) and it is not still after the stop.
- The same settings drop Atlas and H1 within a second, and Gene01 after seven, of the first step, and without them they fall within 4 to 10 s: none of the 28 standing robots in the bench walked 24 s without them.
  Joints held at their limit (Atlas's knee, straight, at 0) cannot be bent by a torque command through a servo with 24 000 N*m per rad of stiffness updated 200 times a second; the stiff servos need position commands that follow the plan, not torques.
- The smoothed cancellation made standing worse for the same robots (H1_2 did not stand, Atlas lost its load), so it would have to apply to a walking robot only.

Where to go from here: command joint positions that integrate the balance's accelerations (a model reference per joint) instead of torques for the stiff servos, a crouched walking stance with the knees bent so that no leg is at a limit, and a swing foot that lifts first and moves after.
The walker, its bench (`walk_bench.py`) and these runs are not in the repository, so nothing in this section can be reproduced from it; the numbers are what the attempt reported.
An earlier note in the roadmap described a first attempt on G1 (it stood four seconds, leaned 15 to 20 degrees and lifted a foot 0.5 cm of the 7 cm it needed); that one is superseded by the findings above, which are of the later walker.

## Where it is in the code

- `behaviors/whole_body.py`: the controller; `tests/test_whole_body.py`.
- `behaviors/stance.py`, `behaviors/stance_ik.py`, `design/mjcf_stance.py`: the kind of a robot from its feet, and the stance a keyframe is moved to.
- `scripts/balance_bench.py`: the table.
- `scripts/humanoid_grid.py`: a picture of every humanoid that stands, from the bench's JSON.
- `scripts/demos.py --only humanoids_stand_push`: four humanoids pushed at 0.2 m/s forwards and backwards; `handover_panda` and `rovers_cross` are the films of docs/MULTI.md.
