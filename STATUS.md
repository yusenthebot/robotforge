# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9; vision 2026-10-14: any robot, scenes, terrains, algorithm adapters)
- Plan: docs/ROADMAP.md from phase G (2026-10-18): lightweight, flexible, creative and comprehensive robotics as equal lines, teaching one entry point; milestones G1 to G11 (2026-10-19 priorities: legged, mobile manipulation, mounting, generation; drones parked); phases A to F in docs/PLAY_PLAN.md
- Mode: evolving, /loop running; trunk-based on main (only `main` exists); repo public since 2026-10-15 (user's request)
- Round: 37; phase 1 done: all ten in-flight lanes landed through scripts/land.py (library splicing, mobile picking, sensors, water, mechanisms, drones, straight walks, evals 2, pixel pass, bench); phase 2 running: G8 legged, G9 mobile manipulation, G10 mounting, G11 generation (robots, environments, tasks), each landing after every verified step
- Learned walks: any legged robot learns a rhythm in RL II (built from parts or imported from a URDF, even with a gait), keeps it in `walks/<robot>.json`, then learns to turn on the spot; `move(walker="rhythm")` and WASD drive it
- Real robots: PiPER, Panda, Go2, G1, A1, ANYmal C, H1, Z1, UR5e, Kinova Gen3, xArm 7, T1, Barkour, ALOHA, OP3, Robotiq; arms ride on dogs; mobile pick and place 29 of 29; sensors on any robot; water, mechanisms, drones (parked)
- Teaching mode: 13 lessons (now with imitation and seeing) + 3 challenges + 2 face-offs + 4 benches; `scripts/playtest.py` plays all 22 to the end
- Push recovery: dog, Go2, built dog and hexapod take shoves and 0.4 m drags from load (fresh: 0 falls in 32, before 12) and on their gait (0 in 24, before 8)
- Verified for real: every landing passed ruff, the suite (1446 passed / 29 skipped) and smoke; real Menagerie runs per lane; videos and pictures from the real viewer sent to the user
- Floor status: A1 [x GPT via codex exec, re-verified round 31] A2 [x] A3 [x] A4 [x] A5 [~ MCP App embed untested] A6 [~ Claude Code blocked by expired CLI login] A7 [x] A8 [x] A9 [x]
- Open: a forward drag by the back flips a never-moved dog; idle gaits cost 20 percent headless speed; a 1 s turn straight out of a kept walk reaches about 55% of its rate; the Z1 and xArm 7 library homes are folded keyframes; go2_arm cannot fetch from counter height; template humanoid tips at small drags (by design)
- Pending for user: licence + PyPI/.mcpb release; new deps (ROS 2, LeRobot/pyarrow, MuJoCo WASM); `claude auth login`
- Updated: 2026-10-19
