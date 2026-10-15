# robotforge

Beginner-friendly robot playground: natural language -> any robot -> make it do anything, as an MCP plugin for Claude and GPT.

## Resume stack (read first, every round)

1. docs/GOAL.md - north star, floor A1-A9, gates, approved dependencies.
2. STATUS.md - where we are.
3. progress.md - what worked, what did NOT work, next seed, frontier.
4. `git log --oneline -10`.

Plan of record: docs/OVERVIEW.md (Chinese).

## Conventions

- Python 3.12, uv, ruff, pytest. Run everything via `uv run`.
- Physics is authoritative in Python (MuJoCo); the browser viewer is a thin three.js client over WebSocket.
- No dependencies outside the approved list in docs/GOAL.md without asking.
- Commit email for this repo: GitHub noreply (set in repo-local git config).
- Trunk-based (user request, 2026-10-15): work on `main`. No long-lived feature branches; parallel branches made merging painful.
  If a task truly needs isolation, use one short-lived branch, merge it the same day and delete it. Never run two branches that touch the same shared files.
- Push in real time: push main after every commit. Never force-push.
- Docs: one sentence per line in long Markdown; no em dashes; no emojis.

## Dev notes

- The repo sits in an iCloud-synced Desktop, which sets the macOS `hidden` flag on `.venv/**/*.pth`; Python 3.12 then skips the editable-install `.pth` and `import robotforge` fails.
  Fix in place: `.venv` is a symlink to `.venv.nosync/` (iCloud ignores `*.nosync`). Recreate the same way if the venv is ever deleted.
- Run the viewer: `uv run robotforge demo` or `uv run robotforge open path/to/model.xml` (MJCF or URDF). Browser-pane preview config `robotforge-demo` lives in `~/Desktop/.claude/launch.json`.
- The viewer exposes `window.__robotforge` (`scene`, `time`, `bodyPosition`, `pick`, `screenOf`, `lookAt`, `pixels`, `flicker`) for scripted checks.
  `flicker(eye, target, dir)` counts pixels that flip there and back while the camera slides: pause first, compare before and after a rendering change (2026-10-15: stairs 144k -> 327, grass 12k -> 0).
  The browser pane may be hidden and throttle `requestAnimationFrame`, so drive pointer tests with `screenOf` coordinates (CSS px, not screenshot px).
- Tests: `uv run pytest -q`; lint: `uv run ruff check . && uv run ruff format --check .`.
  Lesson playthroughs that train for a minute or more are marked `slow` and skipped by default: `uv run pytest -q -m slow` (CI runs them on Ubuntu).
  `uv run python scripts/playtest.py` plays every lesson headless and prints which step each one reaches.
- CI (`.github/workflows/ci.yml`): lint, tests and the headless smoke test on Ubuntu and Windows; macOS is tested locally only.
  Read a failed run with `gh run view <id> --log-failed`.
- `uv run python scripts/smoke.py` starts the viewer and an MCP stdio session as separate processes and checks both; run it locally before pushing anything that touches startup, the viewer server or the MCP server.
- Offscreen rendering needs OpenGL. Without it (Windows CI, VMs, servers) `observe` returns the state plus a note instead of an image, and rendering tests skip with the reason.
  `ROBOTFORGE_REQUIRE_GL=1` turns those skips into failures; the Ubuntu job sets it and uses Mesa EGL (`MUJOCO_GL=egl`).
- Portability rules: pass `encoding="utf-8"` to every text read and write, compare paths with `Path` (never `endswith("a/b")`), and do not bind or probe ports with `SO_REUSEADDR` on Windows.
