# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9; vision section 2026-10-14: any robot, scenes, terrains, algorithm adapters)
- Mode: evolving, multi-agent waves (user asked for orderly multi-agent development)
- Round: 5 in progress (wave 1 of parallel agents)
- Last done on main: behaviors (trot gait, diff drive, arm IK pick-and-place, WASD driving, follow camera), agent-written scripts (lockstep runner), 48 tests
- Wave 1 agents (worktrees in ~/worktrees, branches feat/*, orchestrator reviews + merges serially):
  - feat/humanoid: humanoid template, balance, poses, slow walking
  - feat/import-doctor: robust URDF/mesh import with a doctor report (A3)
  - feat/robot-library: robot_descriptions catalog, Go2 walking with the generalized gait (A4)
  - feat/ci: GitHub Actions ubuntu + windows, smoke test, no-GL fallback (A8)
  - feat/control: Gym-style RobotEnv, sampling MPC, PD/LQR, comparison harness
- Wave 2 (after wave 1 merges): free-form parts builder, scenes + terrains, RL tier 1 (SB3 extra), UniLab bridge (external backend, ONNX policy playback)
- Floor status: A1 [~ gait works in tests + viewer; real-agent run blocked] A2 [x in tests via agent-style script; real-agent run blocked] A3 [ ] A4 [ ] A5 [~] A6 [ blocked] A7 [~] A8 [ ] A9 [~ 48 tests green]
- Blockers (need user): `claude` CLI OAuth expired; `codex exec` fails "workspace routing discovery" (proxy?)
- Gates pending: pycollada for DAE meshes may be requested by the import agent
- Updated: 2026-10-14
