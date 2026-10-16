# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9; vision 2026-10-14: any robot, scenes, terrains, algorithm adapters)
- Plan: docs/PLAY_PLAN.md; phases A to D done; now phase E (user 2026-10-15: a big overnight goal, multi-agent): every lesson completable, robots that take a push, imported robots that learn to walk, agent mistakes fixable from the error, docs with pictures
- Mode: evolving, /loop running; trunk-based on main (only `main` exists); repo public since 2026-10-15 (user's request)
- Round: 32 done (agent error audit, quadrupeds steady from load, README pictures; straight walks held back); round 33 running: robot gallery, more real robots and splicing, visual dressing, straight walks (user 2026-10-16: better and more unique robot assets)
- Learned walks: any legged robot learns a rhythm in RL II (built from parts or imported from a URDF, even with a gait), keeps it in `walks/<robot>.json`, then learns to turn on the spot; `move(walker="rhythm")` and WASD drive it
- Real robots: AgileX PiPER, Franka Panda, Unitree Go2, Unitree G1 (learned UniLab policy); Unitree A1 imported from its URDF
- Teaching mode: 11 lessons + 3 challenges + 2 real-robot face-offs + 4 tuning benches; `scripts/playtest.py` plays all 20 to the end
- Push recovery: dog, Go2, built dog and hexapod take shoves and 0.4 m drags from load (fresh: 0 falls in 32, before 12) and on their gait (0 in 24, before 8)
- Verified for real: round 31 Codex E2E from the public repo (A1 imported, learned in the browser, walked 1.54 m); round 32 local 877 passed / 20 skipped + 3 slow + smoke + playtest 20 of 20; CI green on 35e2765
- Floor status: A1 [x GPT via codex exec, re-verified round 31] A2 [x] A3 [x] A4 [x] A5 [~ MCP App embed untested] A6 [~ Claude Code blocked by expired CLI login] A7 [x] A8 [x] A9 [x]
- Open: a forward drag by the back flips a never-moved dog; idle gaits cost 20 percent headless speed; learned walks arc and turn slowly out of the walk (round 33); template humanoid tips at small drags (by design)
- Pending for user: `learn_walk(robot)` MCP tool (gate); new MCP lesson/tutor tools; licence + PyPI/.mcpb release; `claude auth login`
- Updated: 2026-10-16
