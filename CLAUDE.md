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
- Docs: one sentence per line in long Markdown; no em dashes; no emojis.

## Dev notes

- The repo sits in an iCloud-synced Desktop, which sets the macOS `hidden` flag on `.venv/**/*.pth`; Python 3.12 then skips the editable-install `.pth` and `import robotforge` fails.
  Fix in place: `.venv` is a symlink to `.venv.nosync/` (iCloud ignores `*.nosync`). Recreate the same way if the venv is ever deleted.
- Run the viewer: `uv run robotforge demo` or `uv run robotforge open path/to/model.xml` (MJCF or URDF). Browser-pane preview config `robotforge-demo` lives in `~/Desktop/.claude/launch.json`.
- The viewer exposes `window.__robotforge` (`scene`, `time`, `bodyPosition`, `pick`, `screenOf`) for scripted checks. The browser pane may be hidden and throttle `requestAnimationFrame`, so drive pointer tests with `screenOf` coordinates (CSS px, not screenshot px).
- Tests: `uv run pytest -q`; lint: `uv run ruff check . && uv run ruff format --check .`.
