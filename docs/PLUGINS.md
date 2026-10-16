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
- Codex `exec`: of two sessions started on a cold install, one saw the tools after about 40 s and finished the task; the other looked for them too early, concluded they were missing and imported robotforge directly (no viewer). The skill now tells the agent to wait and look again and not to work around missing tools; the second cold run was made with that wording.

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

## 真实 agent 评测（维护者）

`scripts/agent_eval.py` 让 Codex 通过插件完成一组固定的新手任务，用来在每一轮之间比较 agent 做真实任务的能力是变好还是变差。
任务一共 8 个，中英文都有，都是新手真会说的话：模板四足狗在草地上走、模板机械臂把红方块放进盒子、真机 Go2 走几步再左转、导入磁盘上的 URDF 小车并开动、用零件拼六足机器人并走起来、造火星车开到石头旁、想学控制算法（第 6 课）、修一个箱子放得太远的旧世界（改颜色、从失败中恢复、抓放）。
`--list` 列出全部任务和它们覆盖的能力。

运行方法：

```bash
uv run python scripts/agent_eval.py                                  # 全部任务，评测当前 HEAD
uv run python scripts/agent_eval.py --ref 2f614f8 --label baseline   # 评测某个提交
uv run python scripts/agent_eval.py --only arm_cube,fix_workshop     # 只跑几个任务
uv run python scripts/agent_eval.py --source yusenthebot/robotforge@main   # 评测 GitHub 上的插件
uv run python scripts/agent_eval.py --compare docs/evals/agent/A.json docs/evals/agent/B.json
```

脚本用 `git archive` 导出被评测的提交，作为本地 marketplace 装进一个临时的 `CODEX_HOME`，所以未提交的改动不会被评测。
临时的 `CODEX_HOME` 只有最小配置：模型 `gpt-6.1-sol`、reasoning effort `medium`、`approval_policy = "never"`、`sandbox_mode = "danger-full-access"`，`auth.json` 链接到 `~/.codex/auth.json`，不碰用户自己的设置。
浏览器、computer use、生图和 ChatGPT apps 都关掉了，因为它们和 robotforge 无关，只会给每个会话加 token 和随机性。
第一个会话开始前脚本先把插件的环境装好并预热一次导入，冷启动的耗时在上面的 First start 一节单独测，不算进评测。
每个任务一个 `codex exec --json` 会话，用自己全新的 `ROBOTFORGE_HOME`，同时最多 2 个，每个最多 10 分钟；会话用的是用户的 ChatGPT 套餐，所以一次全量运行大约 8 个会话，不要随手重复跑。
stdin 接到 `/dev/null`，否则 `codex exec` 会一直等输入。

成功与否由确定性的检查决定，只读真实发生的事：事件流里 robotforge 工具调用的参数和结果，以及会话结束后磁盘上的 world.yaml。
检查从不读 agent 自己说做成了什么。
比如“走了至少 1 米”来自 `move` 测得的距离或 `robots` 报告的位置，“方块进了箱子”来自工具报告的方块位置和 world.yaml 里箱子的尺寸位置，机器人的种类按保存的世界编译出来判断，和 `robots` 工具的判断一致。
除了通过与否，每个会话还记下 robotforge 工具调用次数、失败的调用和错误原文、shell 命令数、输入 token（含缓存）、输出 token 和墙钟时间。

每次运行在 `--work` 下留一个运行目录（默认在系统临时目录），里面有每个任务的事件流 `events.jsonl`、stderr 和它留下的世界；结果的 JSON 摘要（不含绝对路径）写到 `docs/evals/agent/<日期>-<label>.json`，提交进仓库，供以后的轮次比较。
`--transcript <运行目录>/tasks/<任务>` 把一个会话按顺序打印出来（agent 说的话、每个工具调用和结果、每条 shell 命令），失败或浪费的会话要逐个读。
`--rescore <运行目录>` 不开新会话，用新的检查重新评一遍已有的事件流。
改检查是为了让它更准确地读出真实发生的事，不能为了让任务通过而放宽。

这个脚本不是测试：它需要 Codex、登录和网络。
`tests/test_agent_eval.py` 用 `tests/fixtures/agent_eval/` 里录下来并裁剪过的事件流和世界，测试解析和每条检查本身，CI 里不需要网络。

## Not done yet (needs a decision)

- Publishing: a PyPI release (`uvx robotforge`), a GitHub release with the `.mcpb`, and a public repository are gated on the owner's yes.
- License: the repository has no license file; the manifests leave `license` out until one is chosen.
- Optional extras (`rl`): the plugin starts the core install; how plugin users opt into `--extra rl` is open.
