# robotforge Goal

## North star

robotforge 是新手入门具身智能最好的启动选择。
用户在 Claude 或 GPT 里用自然语言设计任何机器人，让它做任何事情，也能打开外部 URDF 和网格。
它轻量、好上手，支持 macOS 和 Windows。
完整方案见 [OVERVIEW.md](OVERVIEW.md)。

## Mode

Evolving。
下面的 Floor 是第一阶段的地板，不是终点。
地板全部真实验证通过后，进入 Frontier 阶梯，持续抬高标准，直到用户叫停或遇到 Gate。

## Floor：MVP 验收（每条都要真实运行验证）

| ID | 验收项 | 怎么验证 |
|---|---|---|
| A1 | “给我一只四足狗，让它在草地上走” | 真实 agent 会话（`claude -p` 或 `codex exec` 挂载本插件）完成；仿真中 10 秒内机身前进超过 1 米且不摔倒；查看器截图 |
| A2 | “做一个机械臂，把红方块放进盒子” | 真实 agent 会话完成；agent 写的 `behaviors/*.py` 运行后方块质心落在盒子内；查看器截图 |
| A3 | 丢进一个外部 URDF 文件夹 | 至少 3 个真实公开 URDF（含 `package://` 路径和 DAE 网格的）诊断修复后能加载；滑条和键盘可控 |
| A4 | “给我一台 Go2” | 从机器人库加载；键盘遥控能站立并移动 |
| A5 | 查看器 | 本地网页可用；Claude Desktop 对话内嵌可用，或记录清楚为什么不行并提供退回方案 |
| A6 | 双客户端 | Claude Code 和 Codex 都能挂载插件并完成 A1 |
| A7 | 轻量化 | 实测并记录：核心安装体积、冷启动到看到机器人的时间、重编译时间、查看器帧率 |
| A8 | Windows | GitHub Actions 的 Windows runner 上测试和无头冒烟测试通过 |
| A9 | 工程质量 | `ruff check` 干净，`pytest` 全绿，核心逻辑（编译器、导入诊断、模板、工具处理）有测试 |

## Frontier 阶梯（地板之后，按 ambition x feasibility 排序，每轮可重排）

1. 自定义形态的强化学习：CPU 训练基准，然后让用户设计的四足学会走路。
2. 学习路线：内置任务线和教学口吻的 Skill，新手从挥手走到训练策略。
3. 评测集：约 15 个机器人需求，Claude 和 GPT 自动跑分对比。
4. 一键安装：Claude Desktop 的 .mcpb 包，冷启动体验打磨。
5. 传感器套件：相机、深度、激光雷达、IMU、触觉的读数和查看器曲线。
6. 分享：MuJoCo WASM 静态页面，导出一段仿真或一个策略。
7. CAD 扩展：build123d 精确零件，导出 STEP 和打印文件。
8. sim2real：和 LeRobot SO-100 等真实硬件打通。
9. 远程部署：接入 ChatGPT 和 claude.ai 网页版。

## 预先批准的依赖

运行时：mujoco、numpy、mcp（官方 Python SDK）、pydantic、pyyaml、manifold3d、trimesh、mink、robot_descriptions、coacd、websockets 或 starlette 加 uvicorn（二选一，取更轻的）、Pillow。
前端：three.js（vendored ES module，不引入前端构建链）。
开发：pytest、ruff。
扩展（可选 extra，后期）：build123d。

超出这个列表的新依赖属于 Gate。

## Gates：暂停并询问用户

- 引入上面列表以外的依赖，或启用新的 MCP server。
- 仓库改为公开、发布到 PyPI、发布 .mcpb 或任何对外发布与部署。
- 宣布 world.yaml 或 MCP 工具接口为稳定版之后的不兼容修改。
- 任何花钱的操作、删除用户数据、修改安全或凭据策略。

推送到私有仓库 origin 的 main 分支属于日常 RECORD 的一部分，用户已在建立仓库时授权。

## 每轮协议

遵循全局 CLAUDE.md 的 Autonomous Loop：ORIENT（冷启动，读 GOAL、STATUS、progress、git log）、PLAN、EXECUTE、VERIFY（真实运行并观察）、RECORD（提交、推送、覆盖 STATUS 和 progress、写 heartbeat）。
重型任务（完整测试、仿真批量、浏览器验证）串行执行，用完即关。
