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
