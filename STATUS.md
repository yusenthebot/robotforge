# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9; vision 2026-10-14: any robot, scenes, terrains, algorithm adapters)
- Mode: evolving, multi-agent waves with review before every merge
- Round: 5, wave 1 merging
- Merged into main: feat/ci (Ubuntu + Windows CI green), feat/control (tasks, Gym-style env, PD/LQR, sampling MPC, ARS, comparisons), feat/robot-library (Go2/Go1/A1/Spot/Panda/SO-ARM100/G1, git or HTTPS fetch, gait for any quadruped), fix for library quadrupeds in the control layer
- Pending: feat/humanoid (fixing review: fall on re-plan into a stop, NaN guard, 0.6 m limits), feat/import-doctor (rebasing onto main, then review)
- Verified for real by the orchestrator: library Go2 driven with W in the viewer 1.79 m in 5 s, settles on release; Go2 MPPI-on-gait walks 1.41 m in 5 s through a side shove; 147 tests + smoke green on main
- Floor status: A1 [~ works in tests/viewer; real-agent run blocked] A2 [~ same] A3 [~ import doctor pending merge] A4 [x verified in viewer] A5 [~ local viewer; MCP App embed untested] A6 [ blocked] A7 [~ cold start 0.19 s] A8 [x CI green Ubuntu + Windows] A9 [x 147 tests, ruff clean]
- Wave 2 next: free-form parts builder, scenes + terrains, RL tier 1 (SB3 extra), UniLab bridge
- Blockers (need user): `claude` CLI OAuth expired; `codex exec` "workspace routing discovery" failures
- Gates pending: none (pycollada not needed: import doctor ships its own DAE reader)
- Updated: 2026-10-14
