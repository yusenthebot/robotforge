# robotforge 方案大览

> 状态：v0.2，2026-10-14。
> 项目名 robotforge 为暂名。
> v0.2 相对 v0.1 的变化：定位改为新手游乐场；物理以 Python 端为准，浏览器只做薄显示；build123d 降为可选扩展；新增行为层、场景层、导入诊断和机器人库。

## 1. 定位

robotforge 是新手入门具身智能的第一个游乐场。
用户用自然语言设计任何机器人，让它做任何事情，也能直接打开外部的 URDF 和网格。
它以插件形式运行在 Claude 和 GPT 里，安装轻、启动快，支持 macOS 和 Windows。

## 2. 四根支柱：造、开、动、学

| 支柱 | 用户说的话 | 系统做的事 |
|---|---|---|
| 造 | “做一只六条腿的蜘蛛机器人” | 从原型模板或零件组装生成机器人 |
| 开 | “打开这个 URDF”“给我一台 Go2” | 自动诊断修复后导入；从机器人库一键加载 |
| 动 | “让它挥手”“把红方块放进盒子”“在草地上走” | 关键帧、遥控、控制器、agent 写的控制脚本 |
| 学 | “它是怎么抓起来的？” | agent 边做边讲概念；所有代码可读，可脱离 agent 单独运行 |

## 3. 架构

```
用户 <── 对话 ──> Agent (Claude / GPT) + Skill (建模规范, 教学口吻)
                  │ MCP tools
                  v
robotforge 本地进程 (Python)
  │
  ├── 设计层: world.yaml (机器人 + 场景) -> MJCF
  │     ├── 几何: 基本体 / 程序网格 / 外部网格 / CAD 扩展
  │     └── 质量惯量从几何计算, 执行器参数来自真实舵机规格
  │
  ├── 导入层: URDF / xacro / MJCF / 网格 + 诊断修复, 机器人库
  │
  ├── 仿真层: MuJoCo 实时步进, 传感器, 相机, 录制
  │
  ├── 行为层: 关键帧, 遥控, 控制器库, 用户与 agent 的脚本
  │
  └── 查看器服务: WebSocket 推送状态, 接收交互
        v
浏览器薄客户端 (three.js)
  ├── 嵌在对话里 (MCP App)
  └── 本地网页
```

关键决策：

- 物理以 Python 端为准。
  控制器、传感器、相机和以后的策略都在 Python 里跑，因为具身智能的入门语言是 Python。
- 浏览器只做薄显示。
  它通过 WebSocket 接收每个刚体的位姿，把滑条、拖拽、键盘操作传回去。
- 只有一套物理运行时，不存在两边版本不一致的问题。
- 对话内嵌依赖 MCP Apps 在 `connectDomains` 里声明 localhost 的 WebSocket，需要实测；连不上就退回本地网页。
- MuJoCo WASM 改做后期的分享功能：把一段仿真或训练好的策略导出成静态网页。
- 没有 AI 也能用：`robotforge open robot.urdf` 直接打开游乐场。

## 4. 设计层：设计任何机器人

- 原型模板覆盖八成需求：机械臂、夹爪、差速小车、阿克曼小车、四足、六足、双足、人形、蛇形、无人机。
  每个模板是带参数的生成器，agent 调参数几秒就能得到一个靠谱的机器人。
- 零件组装覆盖剩下两成：基本体、关节、执行器（舵机、电机、轮子、推进器）、传感器（相机、深度、激光雷达、IMU、触觉）像乐高一样拼。
- 精度的含义是结构和物理正确：关节位姿由配合特征推导，质量和惯量从几何计算，执行器参数来自真实舵机规格，碰撞体合理。
- 默认几何用 MuJoCo 基本体和程序网格（manifold3d），核心安装保持轻量。
- CAD 级精度（build123d、STEP、打印公差）是可选扩展 `robotforge[cad]`，留给想把机器人造出来的用户。
- 一个工程就是一个目录，用 git 管理，agent 的每一步修改都可撤销、可对比：

```
my_world/
  world.yaml        # 机器人 + 场景描述, 唯一真实来源
  parts/            # 自定义零件 (程序网格或 CAD 脚本)
  behaviors/*.py    # 控制脚本, 可脱离 agent 运行
  build/            # 编译产物: MJCF, 网格, 导出的 URDF
```

## 5. 行为层：让它做任何事

| 级别 | 内容 | 可靠性 | 新手学到什么 |
|---|---|---|---|
| 1 姿态与关键帧 | 挥手、点头、跳舞 | 高 | 关节、自由度 |
| 2 遥控 | 键盘或手柄开车、控制机械臂 | 高 | 控制回路 |
| 3 控制器库 | 逆运动学抓取（mink）、差速导航、PD、四足开环步态 | 机械臂和小车高；自定义腿式尽力而为 | 逆运动学、PD、步态 |
| 4 agent 写脚本 | “把红方块放进盒子”写成 `behaviors/pick.py` | 中，靠自检迭代 | 读代码、改代码 |
| 5 策略 | 库里的机器人加载预训练策略；自定义机器人用强化学习训练 | 后期 | 强化学习、奖励设计 |

场景层：桌子、方块、球、盒子、坡道、楼梯、迷宫、起伏地形，都可以一句话生成。

脚本在独立进程里运行，有超时，只开放受限 API。
它不是安全沙箱，首次运行 agent 写的脚本前需要用户确认。

## 6. 导入层：打开外部机器人

自动诊断并修复 URDF 的常见问题，然后给出报告：

- `package://` 路径解析。
- DAE 等 MuJoCo 不支持的网格自动转换为 OBJ。
- URDF 没有执行器时自动添加位置执行器。
- 惯量缺失或明显不合理时修正。
- 碰撞网格过重时自动凸分解。
- 浮动基座识别。
- xacro 展开。

机器人库：通过 robot_descriptions 按需下载 125 个以上的知名机器人（含 MuJoCo Menagerie 的 Go2、Franka、SO-100 等），逐个标注许可，不打包进安装包。

## 7. 新手友好

- 内置学习路线，agent 带着走：让机械臂挥手，用逆运动学抓方块，自己写 PD 控制器，给机器人装相机，训练四足走路。
- 不做死胡同的玩具：导出标准 MJCF、URDF 和普通 Python 代码，可以带去 MuJoCo Playground、Isaac Lab、LeRobot、ROS2。
- 一键安装：Claude Desktop 用 .mcpb 扩展包（uv 类型，自动管理 Python 环境）；Claude Code 和 Codex 一行配置。

## 8. 轻量化指标（目标值，需实测）

- 核心安装不超过 100MB（不含 CAD 和强化学习扩展）。
- 从启动到看到机器人不超过 3 秒。
- 基本体机器人改参数后重编译不超过 1 秒。
- 普通笔记本集成显卡上，常见机器人查看器 60 帧。

## 9. MVP 验收：四个标志性体验

1. “给我一只四足狗，让它在草地上走”：模板四足配开环步态，2 分钟内看到它走起来。
2. “做一个机械臂，把红方块放进盒子”：机械臂加场景，agent 写逆运动学抓取脚本并成功完成。
3. 丢进一个 URDF 文件夹：诊断修复后能打开，滑条和键盘可以控制。
4. “给我一台 Go2”：从机器人库加载，键盘遥控它站立和移动。

MVP 不做：CAD 扩展、强化学习训练、WASM 分享、远程部署、BOM、有限元分析。

## 10. MCP 工具接口（草案）

| 工具 | 作用 |
|---|---|
| `world_open` | 新建或打开工程 |
| `robot_add` | 从模板、零件、机器人库或外部文件添加机器人（含导入诊断） |
| `robot_edit` | 修改机器人描述，提交前做结构校验 |
| `scene_edit` | 添加或修改场景物体和地形 |
| `behavior_set` | 设置姿态、关键帧或控制器 |
| `script_write` / `script_run` | 写入并运行控制脚本 |
| `sim` | 播放、暂停、重置、单步、录制 |
| `observe` | 相机图像、状态、传感器读数、接触信息 |
| `view` | 打开查看器（内嵌或本地链接） |
| `export` | 导出 MJCF、URDF、脚本、视频 |

## 11. 技术选型

| 领域 | 选择 | 理由 |
|---|---|---|
| 物理 | MuJoCo（Python） | 快、准、跨平台，生态最好 |
| 程序网格 | manifold3d + trimesh | 体积小，布尔运算稳健，可算质量属性 |
| 逆运动学 | mink | 基于 MuJoCo 的轻量逆运动学库 |
| 机器人库 | robot_descriptions | 按需下载，覆盖 Menagerie |
| MCP | 官方 Python MCP SDK | Claude 和 Codex 都兼容 |
| 查看器 | three.js + WebSocket | 薄客户端，可内嵌 |
| CAD 扩展 | build123d | 精确 B-rep，能出 STEP |
| 环境与分发 | uv、.mcpb | 一行或一键安装 |

## 12. 主要风险

| 风险 | 应对 |
|---|---|
| 自定义腿式机器人不用强化学习很难走好 | MVP 只保证模板四足开环步态；强化学习是后续重点，先做 CPU 训练基准 |
| 对话内嵌窗口连本地 WebSocket，各宿主行为不同 | R0 第一项验证；失败退回本地网页 |
| agent 写的脚本在用户本机执行 | 独立进程、超时、受限 API、首次运行确认 |
| 新手首次上手门槛（安装、配置、订阅） | 一键安装包作为硬需求，尽早做 |
| agent 建模质量随模型不同差异大 | 评测集量化，Claude 和 GPT 对比 |

## 13. 参考

- [MCP Apps 官方公告](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/)
- [MCP Apps CSP 与 CORS](https://apps.extensions.modelcontextprotocol.io/api/documents/csp-and-cors.html)
- [MCP Bundles (mcpb)](https://github.com/modelcontextprotocol/mcpb)
- [OpenAI Codex Agent Skills](https://developers.openai.com/codex/skills)
- [robot_descriptions](https://pypi.org/project/robot_descriptions)
- [MuJoCo JavaScript 绑定](https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.9.0/README.md)
- [build123d](https://build123d.readthedocs.io/en/latest/installation.html)
- [RLE-Bench](https://seas.harvard.edu/news/can-your-ai-engineer-robot)
