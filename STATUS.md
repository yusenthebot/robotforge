# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9; vision 2026-10-14: any robot, scenes, terrains, algorithm adapters)
- Plan: docs/PLAY_PLAN.md, "第二版" (user 2026-10-15: real robots vs our templates, lessons up to the cart-pole standard, tuning that shows a difference, terrain flicker); phases A, B, C done, now D (frontier)
- Mode: evolving, /loop running; trunk-based on main (only `main` exists)
- Round: 25 done (keep what a robot learned: RL II "Keep this walk for my robot", `move` and WASD use it; verified with real Codex sessions); round 26 next = lessons in a narrow window
- Learned walks: a robot built from parts learns a rhythm in RL II and keeps it in `walks/<robot>.json`; Codex walked the kept tripod walk 1.657 m in 4 s with 6 tool calls; gait-less robots carry a `walk_hint` and agents are told to offer the lesson first
- Real robots: AgileX PiPER (Menagerie, MIT), Franka Panda, Unitree Go2, Unitree G1 on a learned UniLab policy (`walkers/g1_unilab_walk`)
- Teaching mode: 11 lessons (PID I/II, IK I/II, motion planning, sensors I/II, gaits, balance, RL I cart-pole, RL II any body learns to walk) + 3 challenges + 2 real-robot face-offs + 4 tuning benches; every test lands in an experiment record
- Viewer fixed this round: black screen on a page loaded with no size, lesson panel sliding sideways, reloads mid-lesson (camera, stale record, follow offset), plot ticks, name tags, favicon 404
- Security fixed this round: cross-site WebSocket hijacking and DNS rebinding of the viewer (only its own pages may drive it)
- Verified for real: browser (train, keep, W-drive the tripod; reload checks measured frame by frame); 4 Codex sessions (numbers in progress.md round 25); 557 passed / 20 skipped + smoke green; CI green on 60c46c5 (later runs pending)
- Floor status: A1 [x GPT via codex exec, re-verified rounds 24 and 25] A2 [x] A3 [x] A4 [x] A5 [~ MCP App embed untested] A6 [~ Claude Code blocked by expired CLI login] A7 [x] A8 [x] A9 [x]
- Open: lessons unusable below 700 px wide (round 26); Codex's bare observe/run_script errors in run 1 never reproduced (errors now carry their message); learned walks cannot turn; agent token cost (321k input for a 6-call task); learned reflexes unreliable (2 of 6 collapse)
- Pending for user: a `learn_walk(robot)` MCP tool (new tool, gated; run 4 reached it through viewer internals at 1.29M tokens); new MCP lesson/tutor tools need a yes; license + publish timing; `claude auth login` (claude -p expired)
- Updated: 2026-10-15
