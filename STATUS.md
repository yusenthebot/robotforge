# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9; vision 2026-10-14: any robot, scenes, terrains, algorithm adapters)
- Plan: docs/ROADMAP.md from phase G (2026-10-18): lightweight, flexible, creative and comprehensive robotics as equal lines, teaching one entry point; milestones G1 to G7; phases A to F in docs/PLAY_PLAN.md
- Mode: evolving, /loop running; trunk-based on main (only `main` exists); repo public since 2026-10-15 (user's request)
- Round: 37 = land all ten in-flight lanes (G1 five, sensors, drones, water, mechanisms, bench) through scripts/land.py, then G8-G11 with lanes that land after every verified step; priorities 2026-10-19: legged, mobile manipulation, free mounting, generating robots/scenes/tasks; drones parked
- Learned walks: any legged robot learns a rhythm in RL II (built from parts or imported from a URDF, even with a gait), keeps it in `walks/<robot>.json`, then learns to turn on the spot; `move(walker="rhythm")` and WASD drive it
- Real robots: AgileX PiPER, Franka Panda, Unitree Go2, G1, A1 (URDF); gallery: ten designs incl. Go2 + PiPER and a Panda on wheels; more Menagerie robots and splicing in round 35
- Teaching mode: 13 lessons (now with imitation and seeing) + 3 challenges + 2 face-offs + 4 benches; `scripts/playtest.py` plays all 22 to the end
- Push recovery: dog, Go2, built dog and hexapod take shoves and 0.4 m drags from load (fresh: 0 falls in 32, before 12) and on their gait (0 in 24, before 8)
- Verified for real: round 34 local 1227 passed / 27 skipped + 5 slow + smoke + playtest 22 of 22; gallery screenshots from the real viewer; agent evals through codex exec (round 34 branch)
- Floor status: A1 [x GPT via codex exec, re-verified round 31] A2 [x] A3 [x] A4 [x] A5 [~ MCP App embed untested] A6 [~ Claude Code blocked by expired CLI login] A7 [x] A8 [x] A9 [x]
- Open: a forward drag by the back flips a never-moved dog; idle gaits cost 20 percent headless speed; learned walks arc and turn slowly out of the walk (round 33); template humanoid tips at small drags (by design)
- Pending for user: `learn_walk(robot)` MCP tool (gate); new MCP lesson/tutor tools; licence + PyPI/.mcpb release; `claude auth login`
- Updated: 2026-10-19
