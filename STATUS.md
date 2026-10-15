# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9; vision 2026-10-14: any robot, scenes, terrains, algorithm adapters)
- Plan: docs/PLAY_PLAN.md, "第二版" (user 2026-10-15: real robots vs our templates, lessons up to the cart-pole standard, tuning that shows a difference, terrain flicker); phases A, B, C done, now D (frontier)
- Mode: evolving, /loop running; trunk-based on main (only `main` exists)
- Round: 30 done (edits keep robots and objects where they are, lesson round trips too; a Codex session found and we fixed a walking Go2 read as buried); round 31 next = Review round
- Learned walks: a robot built from parts learns a rhythm in RL II and keeps it in `walks/<robot>.json`, then learns to turn on the spot each way; `move` and WASD drive it; gait-less robots carry a `walk_hint` and agents are told to offer the lesson first
- Real robots: AgileX PiPER (Menagerie, MIT), Franka Panda, Unitree Go2, Unitree G1 on a learned UniLab policy (`walkers/g1_unilab_walk`)
- Teaching mode: 11 lessons (PID I/II, IK I/II, motion planning, sensors I/II, gaits, balance, RL I cart-pole, RL II any body learns to walk) + 3 challenges + 2 real-robot face-offs + 4 tuning benches; every test lands in an experiment record
- Viewer: works from phones (375 px) through split panes (672 px) to desktops; round 25 fixed the black screen, reloads mid-lesson, plot ticks, tags, favicon
- Security: the viewer refuses other web pages and DNS rebinding (round 25)
- Verified for real: Codex review task (world, Go2 walk, pick and place, screenshot); G1 face-off opened from a world and dragged with the mouse in the browser; 562 passed / 20 skipped + smoke green; CI Ubuntu green on 0156aa7
- Floor status: A1 [x GPT via codex exec, re-verified rounds 24 and 25] A2 [x] A3 [x] A4 [x] A5 [~ MCP App embed untested] A6 [~ Claude Code blocked by expired CLI login] A7 [x] A8 [x] A9 [x]
- Open: template humanoid tips and the Go2 gait flips when dragged hard; learned reflexes unreliable (2 of 6 collapse); Codex's bare errors in round 25 run 1 never reproduced
- Pending for user: a `learn_walk(robot)` MCP tool (new tool, gated; run 4 reached it through viewer internals at 1.29M tokens); new MCP lesson/tutor tools need a yes; license + publish timing; `claude auth login` (claude -p expired)
- Updated: 2026-10-15
