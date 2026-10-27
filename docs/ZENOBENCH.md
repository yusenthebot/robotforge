# ZenoBench on MuJoCo：在 Mac 上跑 Zeno Malo 的家务基准，不装 Isaac Sim

[ZenoBench](https://github.com/CMU-Agentic-Library/ZenoBench) 是 CMU Agentic Library 为 Zeno Malo 移动双臂机器人做的家务基准：一栋 Infinigen 生成的房子，十一个任务场景，71 个动词技能（SkillNode），两套任务目录（35 个技能任务和 41 个只给指令的 benchmark 任务）。
它原本只能跑在 Isaac Sim 5.1 加 Isaac Lab 2.3 上，需要 Linux 或 Windows 和 NVIDIA RTX 显卡，Mac 装不了。
robotforge 的 `zenobench` 模块把它搬到 MuJoCo 上：ZenoBench 自己的 Python 代码一行不改，照样运行。

## 怎么做到的

ZenoBench 和 Isaac 打交道的地方很少，只有三个文件：`zeno_skills/runtime.py` 打开场景、建仿真和相机，`zeno_skills/rig.py` 读写关节和物体，`zeno_skills/skills.py` 有一处给电器电机下目标。
技能、谓词、规划器、碰撞模型和逆运动学都是纯 Python。
所以 robotforge 不改 ZenoBench，而是提供它 import 的那几个 Isaac 模块，背后换成 MuJoCo：

- `usd_mjcf` 用 usd-core 读 USD 场景，按 PhysX 的语义建 MuJoCo 模型：刚体、关节（转动、滑动、固定）、关节限位、驱动、质量、摩擦材质、碰撞过滤、关闭重力的连杆（变成重力补偿）。
- `hulls` 处理碰撞网格：MuJoCo 只按凸包碰撞，PhysX 还能用三角网格和凸分解。网格先拆成连通部件，近似凸的部件保留凸包，凹的小部件用 CoACD 分解，大的凹部件（房间的地板和墙）用精确的薄棱柱。结果按网格内容缓存在磁盘上。
- `world` 按 PhysX 的方式推进仿真：一步 1/120 秒拆成 4 个 MuJoCo 子步，关节驱动用 PhysX 的隐式公式，底座跟随场景里那个被 ZenoBench 改写的固定关节锚点。
- `shim` 把 `isaacsim.core.prims`、`isaaclab.sim`、`isaaclab.sensors`、`isaaclab.app`、`omni.usd` 和 `pxr.PhysxSchema` 注册成真实的模块名，接口只做 ZenoBench 用到的部分。没装 PyTorch 时，一个很小的 NumPy 替身负责 ZenoBench 用到的几个 torch 调用。

ZenoBench 没有自己的许可证，所以 robotforge 不附带它的任何代码和资产：`robotforge zenobench fetch` 在你的电脑上从上游仓库下载一个固定的、验证过的 commit（含 Git LFS 里约 800 MB 的场景）。
机器人描述是 Apache-2.0，房子来自 Infinigen（BSD-3），物体来自 EmbodiedGen V2（Apache-2.0）。

## 安装和运行

```bash
uv sync --extra zenobench          # usd-core、scipy、opencv、imageio
brew install git-lfs               # 场景放在 Git LFS 里
uv run robotforge zenobench fetch  # 下载 ZenoBench（约 1 GB）
uv run robotforge zenobench check
```

已经有 ZenoBench 的 checkout 时，设置 `ZENOBENCH_ROOT` 指向它，不用再下载。

跑一个任务（参数和 ZenoBench 的 `benchmark/run_episode.py` 一样，写在 `--` 后面）：

```bash
uv run robotforge zenobench episode -- --tasks skill_library/tasks.json --task collect_fruits --planner oracle --video
```

跑 ZenoBench 的任意脚本，或者给外部规划器起 HTTP 服务：

```bash
uv run robotforge zenobench python -- tools/run_skills.py --scene tasks/collect_fruits/scene.usd \
    --ann tasks/collect_fruits/annotation.json --out runs/demo --plan "pick orange" "place orange in:fruit_basket"
uv run robotforge zenobench serve -- --port 8800
```

`serve` 会自动把 worker 的 Python 换成 robotforge 的启动器，原来 README 里的 `$ISAACLAB_PYTHON` 不用设置。

第一次打开一个场景时，碰撞网格的分解要几分钟（并行跑在所有核心上），之后从缓存读，几秒钟就能建好模型。

## 和 Isaac Sim 的差别

两个物理引擎的接触模型不同，同一个任务的成功率不会和 ZenoBench 公布的 Isaac Sim 数字完全一样，这里的结果要单独统计。
已知的差别：

- 驱动：PhysX 的 acceleration 型驱动按关节的有效惯量缩放，这里用 MuJoCo 质量矩阵对角线上的值近似。
- 摩擦：MuJoCo 只有一个摩擦系数，这里用 USD 材质的静摩擦系数；两个物体取较大的值，PhysX 多数材质取平均。
- 底座轮子和转向连杆不参与碰撞：底座本来就由锚点拖动，轮子只是被拖着走。
- 接触偏移（contactOffset、restOffset）、CCD 和休眠没有对应。

逐项的测量结果见下文。
