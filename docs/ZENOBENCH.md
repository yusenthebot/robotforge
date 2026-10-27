# ZenoBench on MuJoCo：在 Mac 上跑 Zeno Malo 的家务基准，不装 Isaac Sim

[ZenoBench](https://github.com/CMU-Agentic-Library/ZenoBench) 是 CMU Agentic Library 为 Zeno Malo 移动双臂机器人做的家务基准：一栋 Infinigen 生成的房子，十一个任务场景，71 个动词技能（SkillNode），两套任务目录（35 个技能任务和 41 个只给指令的 benchmark 任务）。
它原本只能跑在 Isaac Sim 5.1 加 Isaac Lab 2.3 上，需要 Linux 或 Windows 和 NVIDIA RTX 显卡，Mac 装不了。
robotforge 的 `zenobench` 模块把它搬到 MuJoCo 上：ZenoBench 自己的 Python 代码一行不改，照样运行。

## 怎么做到的

ZenoBench 和 Isaac 打交道的地方很少，只有三个文件：`zeno_skills/runtime.py` 打开场景、建仿真和相机，`zeno_skills/rig.py` 读写关节和物体，`zeno_skills/skills.py` 有一处给电器电机下目标。
技能、谓词、规划器、碰撞模型和逆运动学都是纯 Python。
所以 robotforge 不改 ZenoBench，而是提供它 import 的那几个 Isaac 模块，背后换成 MuJoCo：

- `usd_mjcf` 用 usd-core 读 USD 场景，按 PhysX 的语义建 MuJoCo 模型：刚体、关节（转动、滑动、固定）、关节限位、驱动、质量、摩擦材质、碰撞过滤、关闭重力的连杆（变成重力补偿）。
- `hulls` 处理碰撞网格：MuJoCo 只按凸包碰撞，PhysX 还能用三角网格和凸分解。
  网格先拆成连通部件，近似凸的部件保留凸包，凹的小部件用 CoACD 分解，大的凹部件（房间的地板和墙）用精确的薄棱柱。
  判断凸不凸时除了看顶点是否都在凸包上，还要看网格表面是否盖满了凸包表面：没有屋顶的房间所有顶点都在凸包上，凸包却会把整个房间填实。
  结果按网格内容缓存在磁盘上。
- `world` 按 PhysX 的方式推进仿真：一步 1/120 秒拆成 4 个 MuJoCo 子步，关节驱动用 PhysX 的隐式公式，底座跟随场景里那个被 ZenoBench 改写的固定关节锚点。
- `shim` 把 `isaacsim.core.prims`、`isaaclab.sim`、`isaaclab.sensors`、`isaaclab.app`、`omni.usd` 和 `pxr.PhysxSchema` 注册成真实的模块名，接口只做 ZenoBench 用到的部分。
  没装 PyTorch 时，一个很小的 NumPy 替身负责 ZenoBench 用到的几个 torch 调用。
  ZenoBench 是对着 Isaac Sim 自带的 NumPy 1.26 写的，有几处用二维向量做 `np.cross`，NumPy 2 已经不支持，shim 在 ZenoBench 进程里恢复了旧行为。
- 画面：可视网格按材质（GeomSubset）拆开，漫反射贴图缩到 512 像素后带 UV 贴上；Infinigen 家具动辄十万个面，渲染用的网格按尺寸的 1/250 做顶点聚类，碰撞仍用原网格。

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
- 接触偏移（contactOffset、restOffset）和 CCD 没有对应。
- 时间步长：PhysX 每步 1/120 秒，这里每步拆成 8 个 MuJoCo 子步（1/960 秒）。
  只拆 4 步时，手指捏住的易拉罐会在手臂上抬转腕时从指垫间慢慢滑出去（实测每秒约 3 厘米），8 步时不会。
- 关节限速（physxJoint:maxJointVelocity）：每个 MuJoCo 子步之后把关节速度截在限值内，和 PhysX 求解器的做法一致；手指是 0.1 米每秒。
- 捏住的物体：PhysX 的摩擦是静摩擦，MuJoCo 的是软约束，所以这里加了"夹持静摩擦锁"，见后文。

## 物理设置和调节项

默认值写在 `src/robotforge/zenobench/world.py` 顶部，每一项都可以用环境变量临时覆盖，方便做对比实验：

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `ZENOBENCH_SUBSTEPS` | 8 | 每个 1/120 秒的 PhysX 步拆成几个 MuJoCo 子步 |
| `ZENOBENCH_SOLREF` | 0.003 | 接触的时间常数（秒），越小越硬，至少两个子步 |
| `ZENOBENCH_SOLIMP0` / `ZENOBENCH_SOLIMP1` | 0.95 / 0.99 | 接触阻抗 |
| `ZENOBENCH_IMPRATIO` | 10 | 摩擦相对法向的硬度 |
| `ZENOBENCH_NOSLIP` | 0 | noslip 迭代次数；能消除蠕滑，但在厨房场景里让每步慢 4 倍 |
| `ZENOBENCH_SLEEP` | 1 | 静止物体休眠到被碰为止；厨房场景每步从 2.2 毫秒降到 0.5 毫秒 |
| `ZENOBENCH_GRASP_LOCK` | 1 | 夹持静摩擦锁（见后文）；设为 0 关闭 |
| `ZENOBENCH_VMAX` | 1 | 按 maxJointVelocity 截断关节速度；设为 0 关闭 |

扭转摩擦按"滑动摩擦系数乘以 15 毫米接触半径"设置：PhysX 的平面接触是一块接触面，能抵抗扭转，MuJoCo 默认的 0.005 太小，捏住的物体会在指间打转。

## 调试：接触轨迹

`ZENOBENCH_TRACE` 写一个或几个刚体的 prim 路径（逗号分隔），仿真时每隔 `ZENOBENCH_TRACE_EVERY` 步（默认 4）把它们的位姿、所有接触（对方 geom、距离、法向力）和手指关节写成 JSON 行，文件名由 `ZENOBENCH_TRACE_OUT` 指定。
定位"物体为什么从手里掉了"这类问题时，先记下物体和夹爪的轨迹，再算物体在夹爪坐标系里的位移，就能分清是碰撞、蠕滑还是被甩出去。
夹持静摩擦锁上锁、松开和松开的原因（滑动还是扭转超载、手指松开）会以 INFO 日志打印，激活中的锁和它承受的力也会写进轨迹。

`ZENOBENCH_RECORD_AT` 写一个 tick 号：仿真到这一步时，把编译好的模型（.mjb）和完整状态存下来，再记录之后 `ZENOBENCH_RECORD_TICKS` 个 tick（默认 1200）的控制量，文件名前缀由 `ZENOBENCH_RECORD_OUT` 指定。
离线回放这一段（加载模型、恢复状态、逐 tick 套用控制量）只要几秒钟，可以在出问题的那一刻反复试不同的物理设置，而不用每次重跑几分钟的整个回合。

## 和 Isaac Sim 的结果对比

ZenoBench 自带物理验证工具 `tools/verify_skills.py`，35 个场景逐步检查技能的前置和后置条件，Isaac Sim 上的结果记录在 `skill_library/verification/STATUS.md`。
在 MuJoCo 上跑同一套场景：

```bash
uv run robotforge zenobench python -- tools/verify_skills.py --all --jobs 4 --out runs/mj_all
```

2026-10-27 在 M5 Max 上的结果（ZenoBench `d573467`，默认设置，6 个并行任务，35 个场景共 31 分钟）：

- MuJoCo 通过 24/35，Isaac Sim 通过 29/35，两边都通过的 23 个。
- k_arrange_sort 只在 MuJoCo 上通过；Isaac 上它在第 1 步取物时失败。
- Isaac 也失败的另外 5 个，MuJoCo 同样失败；其中 k_fridge_chill 和 k_soup_stove 停在和 Isaac 相同的那一步，k_soup_stove 的失败原因也相同（倒汤时 0/3 落进锅里）。
- 只在 MuJoCo 上失败的 6 个：
  - 找不到能把书滑过去的桌边（k_book、g_expose_then_pick_book）。
    这一步只用标注和物体位姿做几何判断，不经过物理引擎；书在 MuJoCo 里一动没动，每条边被排除的理由都能复现（路径上压着马克杯和番茄）。
    所以更可能是 Isaac 的验证记录早于当前布局，需要在 Isaac 上重跑才能确认。
  - 推、拖之后的位置和 Isaac 不同（k_pin、k_restore_swap、k_separate），以及 b_books 最后一步双手推书时书没有动。
- 第一版（还没有夹持静摩擦锁）是 18/35，差距主要是 6 个场景里物体在搬运中从手里滑掉，下一节讲原因和办法。

| Scenario | Isaac Sim | MuJoCo | MuJoCo failure |
|---|---|---|---|
| b_books | pass 7/7 | fail 5/6 | POLICY_FAILED: push book_green: object did not move path two_hand_flat |
| c_approach_moving | pass 2/2 | pass 2/2 |  |
| c_tray_moving | pass 5/5 | pass 5/5 |  |
| g_expose_then_pick_book | pass 5/5 | fail 1/2 | slide to edge: no graspable free edge of kitchen_island/top |
| g_heat_on_stove_instead_of_microwave | pass 5/5 | pass 5/5 |  |
| g_search_then_fetch | pass 5/5 | pass 5/5 |  |
| k_arrange_sort | fail 1/2 | pass 2/2 |  |
| k_blocks | pass 6/6 | pass 6/6 |  |
| k_body | pass 5/5 | pass 5/5 |  |
| k_book | pass 5/5 | fail 1/2 | POLICY_FAILED: slide to edge: no graspable free edge of kitchen_island/top path slide_to_edge |
| k_bottle | pass 3/3 | pass 3/3 |  |
| k_brace | pass 4/4 | pass 4/4 |  |
| k_cabinet | pass 8/8 | pass 8/8 |  |
| k_empty_hide | fail 2/3 | fail 2/3 | POLICY_FAILED: pick cherry_tomato_1: not held (lift 0.006 m) path pick_each_inside |
| k_fetch_collect | pass 2/2 | pass 2/2 |  |
| k_floor | pass 4/4 | pass 4/4 |  |
| k_fridge_chill | fail 5/6 | fail 5/6 | POLICY_FAILED: carry_in_via: soda_can slipped out of the hand while carrying it path cabinet_or_fridge_shelf |
| k_handle_mug | pass 2/2 | pass 2/2 |  |
| k_hands | pass 8/8 | pass 8/8 |  |
| k_microwave | fail 12/13 | fail 5/6 | POLICY_FAILED: joint move to microwave_front: every joint-space path collides path microwave_staged |
| k_mug_on_burner | pass 8/8 | pass 8/8 |  |
| k_perceive | pass 12/12 | pass 14/14 |  |
| k_pin | pass 4/4 | fail 3/4 | POLICY_FAILED: rolling_pin: not resting on an annotated support path push_inward |
| k_posture | pass 9/9 | pass 9/9 |  |
| k_present | pass 6/6 | pass 6/6 |  |
| k_restore_swap | pass 3/3 | fail 2/3 | POSTCONDITION_FAILED: POSTCONDITION_FAILED: positions_swapped(a=wooden_block_a, b=soda_can) is false (errors 0 |
| k_separate | pass 5/5 | fail 3/4 | PRECONDITION_FAILED: PRECONDITION_FAILED: not grasp_clearance(object=wooden_block_a) is false (pinch at [7.22, |
| k_soup_stove | fail 6/7 | fail 6/7 | POLICY_FAILED: pour: 0/3 items reached handled_cooking_pot path tilt_over_rim |
| k_stir_cover | pass 13/13 | pass 13/13 |  |
| k_sweep | pass 3/3 | pass 3/3 |  |
| k_trash | pass 10/10 | pass 10/10 |  |
| k_wipe | pass 5/5 | pass 5/5 |  |
| o_clear | pass 1/1 | pass 1/1 |  |
| r_bimanual_box | fail 1/2 | fail 1/2 | POLICY_FAILED: bimanual_box_lift: two-hand lift/contact check failed path two_hand_box |
| t_floor_moving | pass 5/5 | pass 5/5 |  |

## 捏住的物体：夹持静摩擦锁

PhysX 的摩擦是静摩擦：切向力没超过摩擦上限，捏住的物体在指间纹丝不动。
MuJoCo 的接触是软约束，持续的受力会让物体以很小的速度蠕滑。
最初的结果里，有 6 个场景是物体在搬运中从手里滑掉。

排查过程（工具见上一节）：

1. 接触轨迹显示 k_trash 的易拉罐在手里匀速滑动，没有碰到任何东西；把子步从 4 拆到 8 后 k_trash 通过，但其余 5 个场景还在掉。
2. k_bottle 的瓶子在一个 tick 内被指垫弹开 13 毫米：碰撞接触点集合切换时产生的冲量，把 48 克的手指甩开了。PhysX 有 0.1 米每秒的手指限速，补上关节限速后跳动降到 1.7 毫米，但瓶子仍然滑出。
3. 在 multiccd、原生 CCD、凸包顶点上限、接触余量、胶囊指垫这些碰撞设置上逐一对比，结果都一样。
4. 用快照回放在出问题的那一刻反复试：机器人完全静止、关掉重力，瓶子照样漂移。原因是两侧接触法向差了约 7.7 度（瓶身有锥度），夹紧力把瓶子往外挤；真实世界和 PhysX 里静摩擦挡得住，MuJoCo 的软摩擦挡不住。noslip 加到 200 次迭代也只能减慢，消除不了。

所以这里给捏持单独加了静摩擦：

- 同一只手的两根手指都以至少 2 N 压在同一个物体上、物体相对手掌基本静止时，激活一个建模时就声明好（默认不激活）的焊接约束，把物体锁在手掌上当前的位置。
- 每个 tick 计算夹持必须对外提供的力：物体受到的全部约束力（焊接和指垫加在一起，彼此的内力抵消）减去它和其他东西的接触（比如它压着的桌面），按 0.1 秒做平滑。
  这个力超过"摩擦系数乘以两侧夹紧力"，或力矩超过它再乘以 2 厘米接触半径时，锁就松开，物体照常滑动，静止后可以再锁上。
- 任一根手指连续 50 毫秒不再压着物体（张开手、放下物体）时松开。
- 关闭焊接约束不会唤醒正在休眠的物体，所以松开时给物体一个 tick 的极小推力，免得它悬在手离开的位置。

单元测试覆盖了上锁、抬起时零漂移、张手后掉落，以及摩擦拿不住的重物会滑脱而不是被焊住提起来。
