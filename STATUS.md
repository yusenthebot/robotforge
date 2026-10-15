# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9; vision 2026-10-14: any robot, scenes, terrains, algorithm adapters)
- Mode: evolving; trunk-based on main since 2026-10-15 (no long-lived branches; every feature branch is merged and deleted)
- Round: 6, viewer UI redesign done (2c69719)
- On main: CI, humanoid, control layer (PD/LQR/MPC/MPPI/ARS), robot library, import doctor, plugins + lessons, RL (SB3), terrain + 6 scenes, parts builder, UniLab learned walkers
- Viewer (new): minimal light/dark UI, amber selection ring, Robots drawer, and a World panel with Maps (6 scenes + empty), Terrain (9 grounds, 8 terrain types) and Modes (gait, learned walk, joints, hold, MPC stand, MPPI walk, LQR/MPC balance, IK/OSC/MPC reach, MPC drive-to-goal) per robot type
- Verified for real: in the browser, park/maze/empty maps load, stairs land in front of the dog, ground styles switch, MPC stand holds, MPPI walk covers 1.23 m, arm IK reaches a fresh random target each click; 440 tests + smoke green
- Floor status: A1 [x GPT via codex exec] A2 [x] A3 [x raw Unitree A1 URDF] A4 [x] A5 [~ MCP App embed untested] A6 [~ Codex works; Claude Code blocked by expired CLI login] A7 [x] A8 [x] A9 [x]
- Open review findings (fix next, on main):
  builder HIGH 1-4 (two-wheel rover seen as balancer, rangefinder intprm, mirrored slide axis, arm seed angle wrap), MED 6-9;
  unilab 1-4 (Hydra quoting, concurrent jobs, Go2 walker needs onnxruntime: give it a numpy runtime, clip before scale) + naming A-E
- Known gaps: walk_mppi mode walks on forever (a dog left for minutes can tip over); skill + lessons do not mention builder/terrain/unilab tools yet
- Pending for user: showcase film re-render on main; roadmap A->B->C; license + publish timing; G1 2 h training; `claude auth login`
- Blockers (need user): `claude` CLI OAuth expired
- Updated: 2026-10-15
