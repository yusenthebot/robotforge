# progress

## Current state

Round 0 (setup), 2026-10-14.
Only docs exist: docs/OVERVIEW.md (plan v0.2), docs/GOAL.md (goal, floor A1-A9, gates, frontier).
No code yet.

## What worked

- Planning converged with the user over several turns: beginner playground, Python-authoritative physics, thin three.js viewer, build123d optional.

## What did NOT work

- Nothing attempted yet.

## Next-round seed

R0 spike 1: Python MuJoCo sim server stepping in real time, streaming body poses over WebSocket to a vendored three.js viewer; joint sliders and drag back to the server.
Verify in the built-in browser pane with a Menagerie-style model and a hand-written MJCF.
Measure frame rate and latency.

## Frontier

- Current ceiling: plan only.
- Next frontiers: see docs/GOAL.md Frontier ladder (RL for custom morphologies is the top one after the floor).
- Radical ideas weighed: physics in browser via MuJoCo WASM (rejected as primary because Python controllers and policies must run live; kept for sharing).
- Escalation ladder: open-loop gaits -> CPU RL -> GPU backends (MuJoCo Warp on Windows NVIDIA) -> sim2real with real hardware.
