# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9)
- Mode: evolving
- Round: 1 done (sim server + thin viewer)
- Phase: R0 spikes; spike 1 done and verified in the browser pane
- Last done: MuJoCo real-time session, three.js viewer over WebSocket (sliders, play/pause/step/reset, speed, drag with gravity comp), `robotforge open|demo` CLI, 17 tests
- Verified for real: demo scene + a 263-geom Go2 room scene (81 meshes, 15 MB) run at realtime 1.00x; drag lifts a cube to the cursor; sliders move the arm
- Next: round 2 - MCP server (official SDK 2.3) with world_open / robot_add / sim / observe / view, driven by a real `claude -p` session
- Then: spike 3 - MCP App embed in Claude Desktop (localhost WebSocket via connectDomains)
- Floor status: A1 [ ] A2 [ ] A3 [~ MJCF/URDF open works, no doctor yet] A4 [ ] A5 [~ local web viewer works] A6 [ ] A7 [~ cold start 0.19s] A8 [ ] A9 [~ ruff + pytest green]
- Blockers: none
- Gates pending: none
- Updated: 2026-10-14
