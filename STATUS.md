# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9; vision 2026-10-14: any robot, scenes, terrains, algorithm adapters)
- Plan: docs/PLAY_PLAN.md, "第二版" (user 2026-10-15: real robots vs our templates, lessons up to the cart-pole standard, tuning that shows a difference, terrain flicker); phase order A -> B -> C
- Mode: evolving, /loop running; trunk-based on main (only `main` exists)
- Round: 12 done (phase B part 1: experiment bench for PID and RL); round 13 next = phase B part 2 (twins, LQR/MPC/gait knobs)
- Real robots: AgileX PiPER added (Menagerie, MIT); library arms get a tool point + gripper (PiPER, Panda pick and place); Unitree G1 walks on a learned UniLab policy shipped in `walkers/g1_unilab_walk` (trained 62 min here, heading hold)
- Face-offs (Learn -> 真机对比): arms (ours / PiPER / Panda: spec table, workspace clouds, reach-out test, gravity compensation switch, pick race) and humanoids (ours ZMP / G1 RL: walk test scored by size and energy, shove test, quiz)
- Teaching mode otherwise: 4 lessons (PID, IK, sensors, RL) + 3 challenges (maze, shove, obstacle course)
- Experiment bench (engine `Bench`): last 5 runs overlaid with a metrics table (best marked), sweeps of one knob plotted as metric vs setting; PID records every step test and sweeps Kp/Kd; RL has step size / noise / directions knobs, fast-forward, recorded training curves and a "tune" step
- Flicker: measured with `__robotforge.flicker` (pixels that flip back while the camera slides): stairs 144k -> 327, grass 12k -> 0; causes were squeezed box UVs and per-texel grain
- Fixed this round: the release fling (arc turn judged from clipped tool angles), IK stalls on library arms (restarts, collision-free nearest answer), wrist snap out of home poses
- Verified for real: both face-offs and the bench (PID tests + Kp sweep, three RL trainings) run end to end in the browser (screens sent); 509 passed / 20 skipped (default install) + smoke green; CI green through a70fa5d
- Floor status: A1 [x GPT via codex exec] A2 [x] A3 [x] A4 [x] A5 [~ MCP App embed untested] A6 [~ Claude Code blocked by expired CLI login] A7 [x] A8 [x] A9 [x]
- Open: builder MED 6-9; unilab findings 1, 2, 4 + naming A-E; joint-space branch moves do not avoid obstacles (motion planning is a frontier item); G1 is not trained against shoves
- Pending for user: new MCP lesson tools need a yes; license + publish timing; `claude auth login`
- Updated: 2026-10-15
