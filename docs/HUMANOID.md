# Humanoids: standing on their own

A humanoid you add from the library or the open catalogue stands on its own feet, holds a load and takes a push, whatever its joints are called.
The library's own humanoids keep what they had (G1 its learned walk, T1 and OP3 their preview-control gait); every other two-legged robot stands by whole-body balance.

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

`robots` says `balance` for a robot held up this way, and `walk_hint` when no gait fits its legs.
`move` walks it only when the gait found from its legs works (`humanoid.for_robot`); otherwise it says the robot can stand but not walk.

## How each one does

`uv run python scripts/balance_bench.py --robots all` puts every humanoid alone in an empty world, leaves it with what stands it up when it appears, and watches:
six seconds standing (upright, no lower than 85 percent of its height); the same for five seconds with a backpack of 20 percent of its own mass put on over a second and a half; and a shove, 0.1 to 0.4 m/s added to the trunk's velocity at once forwards, backwards, to the left and to the right, the table giving the largest one it was still standing after (0.0: none).
For scale, a person standing still is thrown off balance by about 0.3 m/s.
Numbers at 2026-10-20, one run each (a push is a single event: a robot at the edge of what it can take may or may not).

36 of the 50 robots read as humanoids stand; 29 of the 43 held by whole-body balance.

| robot | mass (kg) | controller | stands | 20% load | largest shove f/b/l/r (m/s) |
|---|---|---|---|---|---|
| adam_lite_mj | 58 | whole-body balance | yes | yes | 0.3/0.2/0.3/0.3 |
| atlas_drc | 182 | whole-body balance | yes | yes | 0.3/0.2/0.4/0.4 |
| atlas_v4 | 182 | whole-body balance | yes | yes | 0.2/0.2/0.4/0.4 |
| booster_t1 | 32 | preview-control gait | yes | yes | 0.4/0.4/0.4/0.4 |
| booster_t1_mj | 32 | preview-control gait | yes | yes | 0.4/0.4/0.4/0.4 |
| draco3 | 36 | whole-body balance | yes | no | 0.3/0.3/0.4/0.4 |
| elf2_mj | 31 | whole-body balance | yes | yes | 0.3/0.2/0.2/0.2 |
| ergocub | 57 | whole-body balance | yes | yes | 0.4/0.4/0.4/0.4 |
| g1 | 33 | learned policy | yes | no | 0.4/0.4/0.4/0.4 |
| g1_mj | 33 | learned policy | yes | no | 0.4/0.4/0.4/0.4 |
| gene01 | 80 | whole-body balance | yes | yes | 0.3/0.2/0.4/0.4 |
| gr1 | 53 | whole-body balance | yes | yes | 0.4/0.2/0.4/0.4 |
| h1 | 52 | whole-body balance | yes | yes | 0.4/0.2/0.4/0.4 |
| h1_2 | 67 | whole-body balance | yes | yes | 0.4/0.2/0.4/0.4 |
| h1_2_mj | 67 | whole-body balance | yes | yes | 0.3/0.4/0.4/0.4 |
| h1_mj | 51 | whole-body balance | yes | yes | 0.3/0.1/0.4/0.4 |
| icub | 33 | whole-body balance | yes | no | 0.2/0.3/0.3/0.3 |
| jvrc_mj | 62 | whole-body balance | yes | yes | 0.3/0.4/0.2/0.2 |
| luna | 56 | whole-body balance | yes | yes | 0.3/0.2/0.3/0.3 |
| mujoco_humanoid_mj | 41 | whole-body balance | yes | yes | 0.4/0.4/0.4/0.4 |
| n1 | 40 | whole-body balance | yes | yes | 0.4/0.2/0.4/0.4 |
| n1_mj | 40 | whole-body balance | yes | yes | 0.3/0.2/0.4/0.4 |
| oli | 53 | whole-body balance | yes | yes | 0.3/0.3/0.4/0.4 |
| op3 | 3 | preview-control gait | yes | yes | 0.4/0.4/0.4/0.4 |
| op3_mj | 3 | preview-control gait | yes | yes | 0.4/0.4/0.4/0.4 |
| simple_humanoid | 131 | whole-body balance | yes | no | 0.4/0.2/0.4/0.4 |
| t1 | 32 | preview-control gait | yes | yes | 0.4/0.4/0.4/0.4 |
| talos | 90 | whole-body balance | yes | yes | 0.3/0.2/0.2/0.2 |
| toddlerbot | 4 | whole-body balance | yes | yes | 0.2/0.0/0.2/0.2 |
| toddlerbot_2xc_mj | 4 | whole-body balance | yes | no | 0.3/0.0/0.3/0.2 |
| toddlerbot_2xm_mj | 4 | whole-body balance | yes | yes | 0.3/0.0/0.3/0.2 |
| tron1_sf_tron1a_mj | 21 | whole-body balance | yes | yes | 0.3/0.2/0.3/0.4 |
| tron1_sf_tron1b_mj | 21 | whole-body balance | yes | yes | 0.3/0.2/0.3/0.4 |
| tron2_sf_tron2a_mj | 35 | whole-body balance | yes | no | 0.3/0.3/0.4/0.4 |
| tron2_sfyg_tron2a_mj | 42 | whole-body balance | yes | no | 0.3/0.4/0.4/0.4 |
| valkyrie | 136 | whole-body balance | yes | yes | 0.3/0.2/0.3/0.3 |
| berkeley_humanoid | 16 | whole-body balance | no | - | tilt 91.8 deg, height 0.12 m of 0.58 m after 6 s |
| bolt | 1 | whole-body balance | no | - | tilt 79.9 deg, height 0.09 m of 0.46 m after 6 s |
| cassie | 35 | whole-body balance | no | - | tilt 86.2 deg, height 0.16 m of 0.89 m after 6 s |
| cassie_mj | 33 | whole-body balance | no | - | tilt 68.2 deg, height 0.16 m of 1.12 m after 6 s |
| elf2 | 31 | whole-body balance | no | - | tilt 101.1 deg, height 0.08 m of 0.70 m after 6 s |
| jaxon | 132 | whole-body balance | no | - | tilt 111.9 deg, height 0.67 m of 1.31 m after 6 s |
| jvrc | 62 | whole-body balance | no | - | tilt 20.4 deg, height 0.85 m of 0.86 m after 6 s |
| luna_mj | 57 | whole-body balance | no | - | tilt 88.1 deg, height 0.08 m of 0.82 m after 6 s |
| oli_mj | 54 | whole-body balance | no | - | tilt 92.5 deg, height 0.11 m of 0.91 m after 6 s |
| r2 | 214 | whole-body balance | no | - | tilt 79.3 deg, height 0.41 m of 1.23 m after 6 s |
| spryped | 11 | whole-body balance | no | - | tilt 99.6 deg, height 0.07 m of 0.80 m after 6 s |
| talos_mj | 94 | whole-body balance | no | - | tilt 86.4 deg, height 0.20 m of 1.03 m after 6 s |
| tron1_pf_tron1a_mj | 18 | whole-body balance | no | - | tilt 92.2 deg, height 0.11 m of 0.85 m after 6 s |
| tron1_pf_tron1b_mj | 18 | whole-body balance | no | - | tilt 92.2 deg, height 0.11 m of 0.85 m after 6 s |

## What does not stand, and why

Fourteen robots held by whole-body balance fall within six seconds.
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
and the learned G1 policy does not carry a load of a fifth of its mass (it is not whole-body balance).

## Walking: tried, not shipped

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
The walker, its bench (`walk_bench.py`) and these runs are not in the repository.

## Where it is in the code

- `behaviors/whole_body.py`: the controller; `tests/test_whole_body.py`.
- `behaviors/stance.py`, `behaviors/stance_ik.py`, `design/mjcf_stance.py`: the kind of a robot from its feet, and the stance a keyframe is moved to.
- `scripts/balance_bench.py`: the table.
- `scripts/humanoid_grid.py`: a picture of every humanoid that stands, from the bench's JSON.
