# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9; vision 2026-10-14: any robot, scenes, terrains, algorithm adapters)
- Plan: docs/PLAY_PLAN.md; phases A to D done; now phase E (user 2026-10-15: a big overnight goal, multi-agent): every lesson completable, robots that take a push, imported robots that learn to walk, agent mistakes fixable from the error, docs with pictures
- Mode: evolving, /loop running; trunk-based on main (only `main` exists); repo public since 2026-10-15 (user's request)
- Round: 31 done (E1 playtester, E2 push recovery, E3 URDF imports learn to walk, E4 hex colors, all merged and reviewed); round 32 running as a Workflow (steady from load, straight walks and quick turns, agent error audit)
- Learned walks: any legged robot learns a rhythm in RL II (built from parts or imported from a URDF, even with a gait), keeps it in `walks/<robot>.json`, then learns to turn on the spot; `move(walker="rhythm")` and WASD drive it
- Real robots: AgileX PiPER, Franka Panda, Unitree Go2, Unitree G1 (learned UniLab policy); Unitree A1 imported from its URDF
- Teaching mode: 11 lessons + 3 challenges + 2 real-robot face-offs + 4 tuning benches; `scripts/playtest.py` plays all 20 to the end
- Push recovery: template dog and Go2 on their gait take 0.7 m/s shoves and 0.4 m drags, 0 flips in 24 cases (before: 8)
- Verified for real (round 31): Codex installed the plugin from the public GitHub repo, imported the A1, then drove the walk it learned in the browser (1.54 m, upright, turned left); local suite 626 passed / 20 skipped + 3 slow + smoke green; CI on 2736c10 failed on Ubuntu (IK arm jam, fixed in 4450579)
- Floor status: A1 [x GPT via codex exec, re-verified round 31] A2 [x] A3 [x] A4 [x] A5 [~ MCP App embed untested] A6 [~ Claude Code blocked by expired CLI login] A7 [x] A8 [x] A9 [x]
- Open: a quadruped dragged before its first move still flips (round 32); learned forward walks arc and a 1 s turn after walking reaches 18 percent of its rate (round 32); template humanoid tips at small drags (by design of the balance lesson)
- Pending for user: `learn_walk(robot)` MCP tool (gate); new MCP lesson/tutor tools; licence + PyPI/.mcpb release; `claude auth login`
- Updated: 2026-10-16
