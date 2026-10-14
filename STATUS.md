# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9)
- Mode: evolving
- Round: 3 done (design layer); round 2 (MCP server) done except real-agent verification
- Last done: world.yaml + MjSpec compiler, arm/quadruped/rover templates from a real-parts catalog, atomic edits + undo, 18 MCP tools, grouped/collapsible viewer panel, bounds-based camera framing, no-cache assets
- Verified for real: showcase world (dog + arm + rover + cube + bin + ball) in the browser pane at 604 px; quadruped stands 3 s upright; rover drives straight; MCP over real stdio (SDK client) incl. PNG observe
- Next: round 4 - behaviors: quadruped trot gait (A1), arm IK pick-and-place script runner (A2), keyboard teleop
- Floor status: A1 [ ] A2 [ ] A3 [~ open MJCF/URDF; no doctor] A4 [ ] A5 [~ local viewer; MCP App resource exists, host embed untested] A6 [~ blocked, see below] A7 [~ cold start 0.19 s] A8 [ ] A9 [~ 36 tests, ruff clean]
- Blockers (need user): `claude` CLI OAuth expired (run `claude auth login`); `codex exec` fails with "workspace routing discovery failed/timed out" (Codex backend via proxy). One earlier codex run did reach the model and called our tools (approval mode had to be set).
- Gates pending: none
- Updated: 2026-10-14
