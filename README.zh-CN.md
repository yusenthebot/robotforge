# robotforge

**真实机器人模型、自由组合、轻量仿真。**

[![CI](https://github.com/yusenthebot/robotforge/actions/workflows/ci.yml/badge.svg)](https://github.com/yusenthebot/robotforge/actions/workflows/ci.yml)
[English](README.md) | [简体中文](README.zh-CN.md)

![查看器里的真实机器人模型组合：Spot、Go2、ANYmal C 与挂载的机械臂](docs/images/grid_splice.jpg)

robotforge 把机器人模型、场景、传感器与控制放进同一个本地仿真工作区。
按名字加载 Go2、Panda、UR5e、PiPER 或 G1，导入自己的 URDF、MJCF，也可以把底盘、机械臂和夹爪组合成新的设计。
通过 MCP 在 Claude 或 GPT 中操作，编写 Python 脚本，或者直接打开浏览器查看器探索。

[MuJoCo](https://mujoco.org/) 在 Python 中运行物理，轻量 three.js 查看器通过 WebSocket 显示场景。
项目持续改善共享材质、加载速度和创造玩法，复用场景资源，机器人模型按需下载。
支持 macOS、Windows 和 Linux。

## 从真实机器人模型开始

- **按名字加载模型。** 精选库包括宇树 Go2、波士顿动力 Spot、ANYmal C、Franka Panda、UR5e、AgileX PiPER、宇树 G1 和 Booster T1，还有更大的[开源机器人目录](docs/CATALOGUE.md)。
- **导入自己的模型。** 支持 URDF、xacro、MJCF 和网格，提供导入体检报告；世界可导出为带网格的 MJCF，单台机器人可导出为 MJCF 或 URDF。保留执行器请使用 MJCF（[导入与导出](docs/EXTENDING.md)）。
- **自由组合。** 在已有模型上挂载机械臂、夹爪和传感器，检查质量、平衡与够取范围；需要新结构时，也能从零件搭建（[拼装指南](docs/BUILDER.md)）。
- **搭建场景并试玩。** 在仓库、公园和不同地形中驾驶，放置和复制已配置的道具，调整机器人起点与朝向，保存布局后恢复重玩（[创造玩法](docs/CREATIVE_PLAY.md)、[场景](docs/SCENES.md)）。
- **看见传感器读数。** 查看相机、深度、分割、激光雷达、IMU 和接触信息，探索建图、定位与导航（[传感器](docs/SENSORS.md)、[感知](docs/PERCEPTION.md)）。
- **控制、对比和扩展。** 使用键盘、手柄、脚本和插件，在共享任务与实验记录下比较控制器或训练策略；PPO、SAC 使用可选的 `rl` 依赖（[算法](docs/ALGORITHMS.md)、[训练](docs/TRAINING.md)、[插件](docs/EXTENDING.md)）。

模型可加载与控制器可用是两件事：目录分别记录加载、站立和运动检查的结果。
具体适用范围见目录中的逐模型结果，以及[足式](docs/LEGGED.md)、[人形](docs/HUMANOID.md)和[移动操作](docs/MOBILE.md)说明。

已有 Zeno Malo 描述包时，可按 [Malo 本地工作区指南](docs/MALO.md)导入，使用独立转向驾驶、关节控制和可复用的保存设计。
Malo 网格保留在本地；控制器和模拟传感器的适用范围见指南。

## 查看器实景

以下都是机器人模型在仿真查看器中的实际截图。

<table>
  <tr>
    <td width="33%"><img src="docs/images/import_a1.jpg" alt="从 URDF 导入并显示在查看器中的宇树 A1"><br><sub><b>模型导入。</b>从厂商 URDF 加载的宇树 A1。</sub></td>
    <td width="33%"><img src="docs/images/grid_sensors.jpg" alt="仓库里的 Go2、激光雷达点云和深度相机画面"><br><sub><b>传感器。</b>Go2 的雷达点云与实时深度画面。</sub></td>
    <td width="33%"><img src="docs/images/g8_stairs_go2.jpg" alt="台阶地形场景中的 Go2"><br><sub><b>地形探索。</b>在台阶路线中查看和操控 Go2 模型。</sub></td>
  </tr>
</table>

![开源目录中的二十台机器人模型与各自的检查结果](docs/images/catalogue_grid.jpg)

*目录抽样包括机械臂、四足、人形、移动底盘和灵巧手。
图中同时保留站立、运动检查失败的情况；模型能加载，不等于已有适配它的控制器。*

## 安装与启动

robotforge 用 [uv](https://docs.astral.sh/uv/) 管理 Python 和依赖，先安装 uv：

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 不用 agent，直接打开自己的模型

```bash
uvx --from git+https://github.com/yusenthebot/robotforge robotforge open path/to/robot.urdf
```

支持 URDF、xacro 和 MJCF。
想先看内置示例场景，可以将 `open path/to/robot.urdf` 换成 `demo`。
首次启动下载 Python 依赖；机器人库的模型在首次使用时单独下载并缓存。
国内网络慢时可以设置镜像：`export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`。

### Claude Code

在 Claude Code 里输入：

```
/plugin marketplace add yusenthebot/robotforge
/plugin install robotforge@robotforge
```

装好后运行 `/reload-plugins` 或重开一个会话。
插件带来 robotforge 的全部工具、一个帮助 agent 加载和搭建机器人的技能，以及可选的课程提示词。
第一次启动时如果工具还没出现（依赖还在下载），稍等片刻后在 `/mcp` 里重新连接 robotforge。

### Codex

在终端里运行：

```bash
codex plugin marketplace add yusenthebot/robotforge
codex plugin add robotforge@robotforge
```

然后开一个新的 Codex 会话（也可以在 Codex 里用 `/plugins` 浏览和安装）。
插件已经把 robotforge 的工具设为免确认，所以 `codex exec` 这样的非交互运行也能直接用。

### Claude Desktop

双击 `robotforge-0.0.1.mcpb`，在弹出的窗口里点“安装”，Claude Desktop 会自己用 uv 准备好环境。
目前还没有发布版本，安装包需要在本地构建：

```bash
uv run python scripts/build_desktop.py   # 生成 dist/robotforge-0.0.1.mcpb 和 dist/robotforge-skill.zip
```

如需为 Claude Desktop 添加可选技能，在“设置 > 功能 > 技能”里上传 `dist/robotforge-skill.zip`。

## 可以对 agent 说什么

- “加载一台宇树 Go2，放在仓库里，我想用键盘遥控。”
- “导入这个 URDF：`/path/to/robot.urdf`，解释体检报告。”
- “在 Go2 上挂载 PiPER 机械臂，给我看载荷和够取范围检查。”
- “给 Go2 加激光雷达和深度相机，显示传感器读数。”
- “加载一台 Panda，展示关节和末端控制。”
- “搭一条障碍路线，保存布局，让我反复驾驶。”
- “这台机器人支持哪些任务和控制器？”

查看器支持模型检查、关节滑条和拖拽；有适配控制器的机器人还可以使用键盘或手柄。
世界保存在 `~/robotforge/worlds/`，脚本位于各世界的 `behaviors/` 文件夹。
模板和互动课程仍可用于快速实验，见[零件拼装](docs/BUILDER.md)和[课程](docs/lessons/README.md)。

## 材质与模型保真

查看器在不同场景中复用共享表面和贴图。
导入的图片纹理，以及非地面 MJCF 材质显式定义的粗糙度和金属度会保留；注册资产表面与内置材质预设继续具有原有优先级。
实现与实测成本见[资产引擎](docs/ASSETS.md)、[材质验证](docs/bench/2026-10-24-native-pbr.md)和[性能预算](docs/bench/budget.json)。
上方截图展示模型和传感器；材质改动的前后对照见验证报告。

## 架构

![架构：MCP 调用本地 Python，MuJoCo 运行物理，浏览器显示位姿](docs/diagrams/architecture.png)

agent 通过 MCP 调用本地工具；Python 管理仿真、机器人能力、任务、算法与实验记录。
浏览器通过 WebSocket 接收刚体位姿并回传交互。
架构图附有可编辑的 [HTML 源文件](docs/diagrams/architecture.html)。

## 文档

- [目标与验收](docs/GOAL.md) · [路线图](docs/ROADMAP.md) · [项目方案](docs/OVERVIEW.md)
- [机器人目录](docs/CATALOGUE.md) · [零件拼装](docs/BUILDER.md) · [资产引擎](docs/ASSETS.md)
- [场景与地形](docs/SCENES.md) · [创造玩法](docs/CREATIVE_PLAY.md)
- [传感器](docs/SENSORS.md) · [感知与建图](docs/PERCEPTION.md) · [导航](docs/NAVIGATION.md)
- [算法](docs/ALGORITHMS.md) · [训练](docs/TRAINING.md) · [策略回放与 UniLab](docs/UNILAB.md)
- [足式机器人](docs/LEGGED.md) · [人形机器人](docs/HUMANOID.md) · [操作](docs/MANIPULATION.md) · [移动操作](docs/MOBILE.md)
- [脚本、插件与导出](docs/EXTENDING.md) · [插件打包](docs/PLUGINS.md)
- [当前状态](STATUS.md) · [开发进度](progress.md) · [0.0.1 快照说明](docs/RELEASE_0.0.1.md)

<details>
<summary>早期演示视频</summary>

[观看原有的一分钟演示（MP4，1080p）](docs/media/promo.mp4)。
其中包含真实机器人模型、早期生成原型和教学示例，由 `scripts/demos.py` 录制、`scripts/promo.py` 剪辑。
README 截图由 `scripts/screenshots.py` 生成。

</details>
