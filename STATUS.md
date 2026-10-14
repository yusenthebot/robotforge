# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9; vision 2026-10-14: any robot, scenes, terrains, algorithm adapters)
- Mode: evolving, multi-agent waves with review before every merge
- Round: 5, wave 1 fully merged, wave 2 running
- Merged into main: feat/ci (Ubuntu + Windows CI green), feat/humanoid (ZMP walking, balance on load, poses, review fixes), feat/control (tasks, Gym-style env, PD/LQR, sampling MPC, ARS, comparisons), feat/robot-library (Go2/Go1/A1/Spot/Panda/SO-ARM100/G1, git or HTTPS fetch, gait for any quadruped), fix for library quadrupeds in the control layer
- Also merged: feat/import-doctor (URDF/xacro/mesh import + doctor; security review fixed: bounded xacro, off-loop imports, serialized world edits)
- Wave 2 running (each pushes its branch after every commit): feat/rl (SB3 tier 1), feat/unilab (ONNX playback of UniLab policies, training bridge), feat/builder (free-form parts, mounted templates), feat/terrain (hfield terrains, scene presets, viewer hfield rendering), feat/plugin (Claude Code + Codex plugins, .mcpb, agent skill, lessons)
- Verified for real by the orchestrator: library Go2 driven with W in the viewer 1.79 m in 5 s, settles on release; Go2 MPPI-on-gait walks 1.41 m in 5 s through a side shove; 147 tests + smoke green on main
- Floor status: A1 [x GPT via codex exec: new world + dog walked 1.545 m in 5 s upright; Claude side blocked by CLI login] A2 [x GPT via codex exec: agent wrote behaviors/pick_red_cube.py, run_script ok, cube inside the box, self-verified] A3 [x raw Unitree A1 URDF (package://, DAE) opened in the viewer: 12 actuators, recognized as quadruped, W drove it 1.81 m in 5 s; 209/214 public URDFs import] A4 [x verified in viewer] A5 [~ local viewer; MCP App embed untested] A6 [~ Codex works (default_tools_approval_mode=approve); Claude Code blocked by expired CLI login] A7 [x measured: 49 MB download / 157 MB unpacked, cold start 0.19 s, 120 fps] A8 [x CI green Ubuntu + Windows] A9 [x 313 tests, ruff clean]
- Blockers (need user): `claude` CLI OAuth expired (Codex works again)
- Gates pending: none (pycollada not needed: import doctor ships its own DAE reader)
- Updated: 2026-10-14
