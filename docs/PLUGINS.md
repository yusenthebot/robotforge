# Plugins: how robotforge ships to Claude Code, Codex and Claude Desktop

This page is for maintainers.
Users follow the install steps in the [README](../README.md).

## Layout

The repository root is the plugin, so the plugin always carries the exact source and `uv.lock` it runs.

| File | Client | What it does |
|---|---|---|
| `.claude-plugin/plugin.json` | Claude Code | Manifest; declares the MCP server inline |
| `.claude-plugin/marketplace.json` | Claude Code, Codex | Marketplace with one plugin whose source is `./` (Codex reads this file too) |
| `.codex-plugin/plugin.json` | Codex | Manifest: skills, MCP config path, listing metadata |
| `.codex-plugin/mcp.json` | Codex | MCP server config, with `default_tools_approval_mode = "approve"` |
| `skills/robotforge/` | all | The agent skill (open agent-skills format): `SKILL.md` plus `references/` |
| `mcpb/manifest.json` | Claude Desktop | MCP bundle manifest, `uv` server type (manifest 0.4) |
| `scripts/build_desktop.py` | Claude Desktop | Builds `dist/robotforge-<version>.mcpb` and `dist/robotforge-skill.zip` |
| `assets/icon.png` | Codex, Claude Desktop | 512 px icon |

There is deliberately no `.mcp.json` at the root: Claude Code would also load it as this repository's project MCP config, and its Codex-only keys would not belong there.

## The server command

All three clients start the same locked server:

```
uv run --project <plugin root> --frozen --no-dev robotforge mcp
```

- Claude Code: `--project ${CLAUDE_PLUGIN_ROOT}` and `UV_PROJECT_ENVIRONMENT=${CLAUDE_PLUGIN_DATA}/venv`, so the plugin folder stays clean and the environment survives plugin updates.
- Codex: `--project .` with `cwd = "."`, which Codex resolves to the installed plugin folder; the environment lands in its `.venv`. Codex passes only a few environment variables to MCP servers, so `env_vars` forwards `ROBOTFORGE_*`, `MUJOCO_GL`, uv index settings and proxies.
- Claude Desktop: `uv run --directory ${__dirname} ...` in the unpacked bundle.

The tests in `tests/test_plugin_packaging.py` keep the three commands, the versions and the skill's tool names in sync.

### First start

The first start creates the environment and installs about 50 MB of packages (plus Python 3.12 if missing).
Even with uv's cache warm it takes about 30 s on macOS, mostly the system scanning freshly installed native libraries on first load; later starts take 2 to 3 s.
Clients do not always wait that long:

- Claude Code: if the server is not up within `MCP_TIMEOUT`, reconnect it in `/mcp`.
- Codex `exec` does not wait for MCP servers before the first model turn; a session started during the first install may not see the tools. The skill tells the agent to look again and not to work around missing tools.

A faster cold start would need the MCP handshake to answer before mujoco and numpy load (lazy imports in `mcp_server.py`).

## Versions

The plugin version is the package version in `pyproject.toml`; the manifests must match it (a test checks).
Claude Code and Codex keep users on a version until it changes, so bump the version to ship an update.

## Validate and verify

```bash
claude plugin validate .                         # marketplace + plugin; MCP entries checked on Claude Code >= 2.1.281
uv run pytest -q tests/test_plugin_packaging.py tests/test_lessons.py
uv run python scripts/build_desktop.py --check   # packs the .mcpb with the official mcpb CLI (npx) and starts it
```

The only expected validation warning is that `CLAUDE.md` at the plugin root is not loaded as context (it is the developers' file, not plugin context).

Real-client checks, with nothing written to the user's own settings:

- Claude Code: `CLAUDE_CONFIG_DIR=<scratch> claude plugin marketplace add yusenthebot/robotforge@<branch>`, `claude plugin install robotforge@robotforge`, then `claude mcp list` shows `plugin:robotforge:robotforge ... Connected`.
- Codex: a scratch `CODEX_HOME` with a symlink to `~/.codex/auth.json`, then `codex plugin marketplace add yusenthebot/robotforge --ref <branch>`, `codex plugin add robotforge@robotforge`, and `ROBOTFORGE_HOME=<scratch> ROBOTFORGE_NO_BROWSER=1 codex exec --skip-git-repo-check --json "<task>" < /dev/null` (without `< /dev/null`, `codex exec` waits for stdin).

## Not done yet (needs a decision)

- Publishing: a PyPI release (`uvx robotforge`), a GitHub release with the `.mcpb`, and a public repository are gated on the owner's yes.
- License: the repository has no license file; the manifests leave `license` out until one is chosen.
- Optional extras (`rl`): the plugin starts the core install; how plugin users opt into `--extra rl` is open.
