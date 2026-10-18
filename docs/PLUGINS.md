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
任务一共 13 个，中英文都有，都是新手真会说的话：模板四足狗在草地上走、模板机械臂把红方块放进盒子、真机 Go2 走几步再左转、导入磁盘上的 URDF 小车并开动、用零件拼六足机器人并走起来、造火星车开到石头旁、想学控制算法（第 6 课）、修一个箱子放得太远的旧世界（改颜色、抓放）、打开一个用户手改坏了的花园世界让狗走起来（从失败中恢复）。
后 4 个是阶段 F 加的，考的是那天合进 main 的新能力：在仓库场景里让机器人从仓库另一头开上装卸台（走到一个地点）、造一个半人马放进公园走到长椅旁（设计库、走到一个物体旁）、打开一个有三个编号方块的世界，用摄像头找出红色的那个放进盒子（用颜色找东西）、把背着机械臂的 Go2 放进厨房走到中岛旁（真机加机械臂、走到家具旁）。
花园世界里狗的模板名拼错了，直接 `open_world` 会失败。
检查要求修好之后花园能打开、球还在、狗在花园打开期间走起来；先读文件修好再打开、一次都没失败的会话同样算通过，失败的调用另外记数。
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
每个任务一个 `codex exec --json` 会话，用自己全新的 `ROBOTFORGE_HOME`，同时最多 2 个，每个最多 10 分钟；会话用的是用户的 ChatGPT 套餐，所以一次全量运行就是 9 个会话，不要随手重复跑。
两个会话开始的时间至少隔 10 秒：同时开始的两个会话会弄坏 Codex 往共用的 `CODEX_HOME` 里安装自带技能的过程，约 30 个会话里出过 3 次。
评测 1d20b0d 之前的提交要加 `--jobs 1`：那之前同时启动的两个 robotforge 服务器会抢同一个查看器端口，其中一个会话完全没有 robotforge 工具，基线里 dog_grass 失败就是这个原因。
stdin 接到 `/dev/null`，否则 `codex exec` 会一直等输入。

成功与否由确定性的检查决定，只读真实发生的事：事件流里 robotforge 工具调用的参数和结果，以及会话结束后磁盘上的 world.yaml。
检查从不读 agent 自己说做成了什么。
比如“走了至少 1 米”算的是机器人朝着被派去的方向走了多远：每次 `move` 按它的速度、转向和出发时的朝向算出该去的方向，量位移在这个方向上的长度；另一种量法是最早和最晚报告的位置在它最初朝向上相距多远（脚本、`simulate` 和 `run_controller` 也能让它动），取两者中大的。
往前派却倒着走的机器人不算走了：一台舵机太弱的六足蜘蛛往前派却倒退了 0.21 米，按距离算的旧检查判它通过。
机器人的种类按保存的世界编译出来判断，和 `robots` 工具的判断一致。
机器人和物体去了哪里，只从它所在的世界打开期间的调用里读（robotforge 的世界类结果都写明当前打开的是哪个世界）：在另一个世界里走了的同名狗不算花园里的狗。
“方块进了箱子”要同时满足三条：方块在 world.yaml 里的起点不在箱子里，有一次成功的 `pick_and_place` 点名了它（或者手动合上过夹爪），工具最后报告的方块位置在箱子里。
只有 `bin` 算箱子，`box` 形状是实心的长方体，方块放在它上面不算放进去。
“是否直立”取最后一次报告的值，`move`、`robots` 和带 IMU 的机器人在 `observe` 里的姿态传感器都算，必须是直立。
火星车离石头多近，从车的外轮廓（俯视的包围盒，含轮子）量到石头表面，不从车的中心量，所以长车和短车用同一个标准：0.5 米以内算开到了旁边。
走到柜台、长椅这类东西旁边也这样量：从机器人的外轮廓量到它俯视的矩形（石头、球、柱子是圆），0.5 米以内算到了旁边，robotforge 的 `move(to=...)` 自己停在 0.15 到 0.45 米处。
这些任务的起点取 world.yaml 里保存的位置：开始时已经在旁边的不算走过去，用 `update_robot` 把机器人直接摆到终点也不算，因为那样保存的位置就在终点。
“开上装卸台”要求机器人的中心最后落在坡道顶上的平台里（按 world.yaml 里坡道的位置、长度和平台长度算），起点离平台至少 3 米。
半人马要是零件拼的，`robots` 认作四足，并且带两只手；背机械臂的 Go2 要在 build 里装了机器人库的 go2，并且 `robots` 列出它带的机械臂。
找红方块要求 `observe(find=...)` 的结果点名了红方块，而且在机械臂第一次碰它之前；之后红方块进了盒子，另外两个方块没有。
`move(to=...)` 的“派去的方向”是从出发点朝目的地的方向，不是出发时朝的方向。
“学控制”要求打开第 6 课，并且用 `run_controller` 在实时世界里跑过一次平衡控制。
`compare_controllers` 在后台跑，查看器里什么都不动，所以只记在旁边，不算数。
除了通过与否，每个会话还记下 robotforge 工具调用次数、失败的调用和错误原文、shell 命令数、输入 token（含缓存）、输出 token、墙钟时间，以及 Codex 自己在 stderr 里记下的错误。
摘要统计出过这类错误的会话数，模型列表刷新超时不算，因为 Codex 会继续用配置好的模型，会话不受影响。

每次运行在 `--work` 下留一个运行目录（默认在系统临时目录），里面有每个任务的事件流 `events.jsonl`、stderr 和它留下的世界；结果的 JSON 摘要（不含绝对路径）写到 `docs/evals/agent/<日期>-<label>.json`，提交进仓库，供以后的轮次比较。
没跑全部任务时文件名带上任务名，已经有同名文件时文件名再加上开始时间，脚本从不覆盖已有的结果；`--results` 给的是一个 .json 文件时会有意写到这个文件上，重评旧结果时用。
每个任务的结果都记下会话收到的提示词，`--compare` 发现前后两边的提示词不同（或者有一边没记）会提出警告。
`--transcript <运行目录>/tasks/<任务>` 把一个会话按顺序打印出来（agent 说的话、每个工具调用和结果、每条 shell 命令），失败或浪费的会话要逐个读。
`--rescore <运行目录>` 不开新会话，用新的检查重新评一遍已有的事件流；改了检查以后要把以前的结果也重评一遍，前后才可比。
`--fixture <运行目录>/tasks/<任务> tests/fixtures/agent_eval/<名字>` 把一个会话裁剪成测试用的录像：去掉图片、shell 输出和检查不读的长文本，零件列表换成个数，其余参数原样保留，路径换成 `RUN` 和 `~`。
只重跑了几个任务时，`--compare` 和 `--after` 都可以给多个结果文件，后面文件里的任务替换前面文件里的同名任务，例如 `--compare 全量.json 重跑.json --after 修复后.json 修复后重跑.json`。
改检查是为了让它更准确地读出真实发生的事，不能为了让任务通过而放宽。
每个任务每次只跑一个会话，模型本身有随机性，一次的通过或失败、几次调用或几万 token 的差别都可能是噪声；结论要看反复出现的问题，小的差别要多跑几次才可信。

已有的结果（`docs/evals/agent/`，都按现在的检查评过）：
- `20261017-baseline.json`、`-baseline-urdf.json`、`-baseline-garden.json`：基线，提交 2f614f8；urdf_rover 换了提示词后重跑过一次，fix_garden 是后来加的任务。
- `20261017-head.json`：提交 15fddbb，9 个任务各跑一次；按现在的检查 8 个通过，learn_control 只在后台比较过控制器，不算。
- `20261017-head-arm.json`：提交 102c652（修了机械臂够不着时的提示），只重跑了两个抓放任务。
- `20261017-round3.json`：提交 ae450e3，9 个任务各跑一次，8 个通过；hexapod_build 的蜘蛛往前派却倒退了 0.21 米，由此有了“按派去的方向算”的检查、建造时的余量提示和 `move` 的倒退提示。
- `20261017-round3b-hexapod_build+learn_control.json`：提交 a3a8c88，重跑这两个任务，都通过；这次的 agent 一开始就选了最强的舵机，所以新的余量提示没有被触发。
- `20261017-main.json`：main 的提交 c2e37cd（导航、设计库、用颜色找东西合进来之后），13 个任务各跑一次，全部通过；102 次 robotforge 调用，2 次失败（都是任务故意设下的：箱子够不着、模板名拼错）。
  浪费的地方：6 个用了 `move(to=...)` 的会话里有 2 个在到达后又调了 `stop`（结果里没写已经停下，紧接着的 observe 还看到轮子在转）；仓库的提示还写着 `move('rover', 0.4)`，agent 先开到坡道脚下的一个点再开上去，还读了整个 world.yaml。
- `20261017-fixed-rover_rock+warehouse_dock.json`：提交 c9b1579（到达后机器人停稳、结果开头写 `"stopped": true`；场景提示改成 `move(robot, to=...)`），重跑这两个任务，都通过，都没有再调 `stop`；warehouse_dock 一次 `move(to="dock")` 直接开上装卸台，调用从 8 次降到 5 次，输入 token 从 403k 降到 220k，用时从 89 秒降到 67 秒。

这个脚本不是测试：它需要 Codex、登录和网络。
`tests/test_agent_eval.py` 用 `tests/fixtures/agent_eval/` 里录下来并裁剪过的事件流和世界，测试解析和每条检查本身，CI 里不需要网络。
录下来的会话里有一个第 6 课只在后台比较过控制器（提交 15fddbb），必须判不通过，另一个实时跑过（提交 f281b69），必须判通过。
还有往前派却倒退了 0.21 米的六足蜘蛛（提交 ae450e3），必须判不通过。
它还手工编了几个“走捷径”的会话（方块直接建在箱子里、方块放在实心盒子上、改世界把方块挪进箱子、四足狗走到石头旁、翻倒的火星车、在另一个世界里走的同名狗），检查必须判它们不通过；一个先修好文件、没有失败过的花园会话必须判通过。
main（c2e37cd）上录下的 4 个新任务的会话也在里面，必须判通过；从它们改出来的捷径必须判不通过：直接摆到装卸台上的小车、停在坡道半路的小车、一开始就摆在长椅旁的半人马、没有手的零件狗、没用摄像头或者抓完才看的会话、放错方块的会话、保存在中岛里面的 Go2、不带机械臂的 Go2。
世界里的 Go2 和 PiPER 用 `tests/fixtures/menagerie` 里的离线替身编译，测试不用下载。

## Not done yet (needs a decision)

- Publishing: a PyPI release (`uvx robotforge`), a GitHub release with the `.mcpb`, and a public repository are gated on the owner's yes.
- License: the repository has no license file; the manifests leave `license` out until one is chosen.
- Optional extras (`rl`): the plugin starts the core install; how plugin users opt into `--extra rl` is open.
