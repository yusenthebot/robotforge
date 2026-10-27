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
- 接触偏移（contactOffset、restOffset）、CCD 和休眠没有对应。

- 时间步长：PhysX 每步 1/120 秒，这里每步拆成 8 个 MuJoCo 子步（1/960 秒）。
只拆 4 步时，手指捏住的易拉罐会在手臂上抬转腕时从指垫间慢慢滑出去（实测每秒约 3 厘米），8 步时不会。
- 关节限速（physxJoint:maxJointVelocity）没有对应：规划器本身已经按 URDF 速度的六成给路点。

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

扭转摩擦按"滑动摩擦系数乘以 15 毫米接触半径"设置：PhysX 的平面接触是一块接触面，能抵抗扭转，MuJoCo 默认的 0.005 太小，捏住的物体会在指间打转。

## 调试：接触轨迹

`ZENOBENCH_TRACE` 写一个或几个刚体的 prim 路径（逗号分隔），仿真时每隔 `ZENOBENCH_TRACE_EVERY` 步（默认 4）把它们的位姿、所有接触（对方 geom、距离、法向力）和手指关节写成 JSON 行，文件名由 `ZENOBENCH_TRACE_OUT` 指定。
定位"物体为什么从手里掉了"这类问题时，先记下物体和夹爪的轨迹，再算物体在夹爪坐标系里的位移，就能分清是碰撞、蠕滑还是被甩出去。

## 和 Isaac Sim 的结果对比

ZenoBench 自带物理验证工具 `tools/verify_skills.py`，35 个场景逐步检查技能的前置和后置条件，Isaac Sim 上的结果记录在 `skill_library/verification/STATUS.md`。
在 MuJoCo 上跑同一套场景：

```bash
uv run robotforge zenobench python -- tools/verify_skills.py --all --jobs 4 --out runs/mj_all
```

2026-10-27 在 M5 Max 上的结果（ZenoBench `d573467`，默认设置，4 个并行任务，35 个场景共 21 分钟）：

- MuJoCo 通过 18/35，Isaac Sim 通过 29/35；MuJoCo 通过的每一个场景在 Isaac 上也通过。
- Isaac 也失败的 6 个场景，MuJoCo 同样失败。
- 剩下 11 个只在 MuJoCo 上失败，分三类：
  - 物体在搬运中从手里滑出（6 个：b_books、k_bottle、k_floor、k_hands、k_stir_cover、o_clear）。
    这是现在最主要的差距，见下一节。
  - 找不到能把书滑过去的桌边（2 个：k_book、g_expose_then_pick_book）。
    这一步只用标注和物体位姿做几何判断，不经过物理引擎；书在 MuJoCo 里一动没动，每条边被排除的理由都能复现（路径上压着马克杯和番茄）。
    所以更可能是 Isaac 的验证记录早于当前布局，需要在 Isaac 上重跑才能确认。
  - 推、拖之后的位置和 Isaac 不同（3 个：k_pin、k_restore_swap、k_separate）。

| Scenario | Isaac Sim | MuJoCo | MuJoCo failure |
|---|---|---|---|
| b_books | pass 7/7 | fail 2/3 | POLICY_FAILED: carry: book_red slipped out of the hand while carrying it path carry |
| c_approach_moving | pass 2/2 | pass 2/2 |  |
| c_tray_moving | pass 5/5 | pass 5/5 |  |
| g_expose_then_pick_book | pass 5/5 | fail 1/2 | slide to edge: no graspable free edge of kitchen_island/top |
| g_heat_on_stove_instead_of_microwave | pass 5/5 | pass 5/5 |  |
| g_search_then_fetch | pass 5/5 | pass 5/5 |  |
| k_arrange_sort | fail 1/2 | fail 1/2 | SUBSKILL_FAILED: SUBSKILL_FAILED: fetch_object(object=soda_can, receptacle=trash_can): SUBSKILL_FAILED: naviga |
| k_blocks | pass 6/6 | pass 6/6 |  |
| k_body | pass 5/5 | pass 5/5 |  |
| k_book | pass 5/5 | fail 1/2 | POLICY_FAILED: slide to edge: no graspable free edge of kitchen_island/top path slide_to_edge |
| k_bottle | pass 3/3 | fail 2/3 | POLICY_FAILED: upright_stage: juice_bottle slipped out of the hand while carrying it path pick_orient_place |
| k_brace | pass 4/4 | pass 4/4 |  |
| k_cabinet | pass 8/8 | pass 8/8 |  |
| k_empty_hide | fail 2/3 | fail 2/3 | POLICY_FAILED: pick cherry_tomato_1: not held (lift 0.008 m) path pick_each_inside |
| k_fetch_collect | pass 2/2 | pass 2/2 |  |
| k_floor | pass 4/4 | fail 3/4 | POLICY_FAILED: place_over_neighbours: floor_block slipped out of the hand while carrying it path surface |
| k_fridge_chill | fail 5/6 | fail 4/5 | POLICY_FAILED: carry_in: soda_can slipped out of the hand while carrying it path carry |
| k_handle_mug | pass 2/2 | pass 2/2 |  |
| k_hands | pass 8/8 | fail 6/7 | POLICY_FAILED: left_carry: rolling_pin slipped from the left hand path left_holds_load |
| k_microwave | fail 12/13 | fail 5/6 | POLICY_FAILED: joint move to microwave_front: every joint-space path collides path microwave_staged |
| k_mug_on_burner | pass 8/8 | pass 8/8 |  |
| k_perceive | pass 12/12 | pass 14/14 |  |
| k_pin | pass 4/4 | fail 3/4 | POLICY_FAILED: rolling_pin: not resting on an annotated support path push_inward |
| k_posture | pass 9/9 | pass 9/9 |  |
| k_present | pass 6/6 | pass 6/6 |  |
| k_restore_swap | pass 3/3 | fail 2/3 | POSTCONDITION_FAILED: POSTCONDITION_FAILED: positions_swapped(a=wooden_block_a, b=soda_can) is false (errors 0 |
| k_separate | pass 5/5 | fail 3/4 | PRECONDITION_FAILED: PRECONDITION_FAILED: not grasp_clearance(object=wooden_block_a) is false (pinch at [7.22, |
| k_soup_stove | fail 6/7 | fail 5/6 | POLICY_FAILED: carry: mug slipped out of the hand while carrying it path above_target |
| k_stir_cover | pass 13/13 | fail 6/7 | POLICY_FAILED: stir: breakfast_spoon slipped out of the hand while carrying it path circle_below_rim |
| k_sweep | pass 3/3 | pass 3/3 |  |
| k_trash | pass 10/10 | pass 10/10 |  |
| k_wipe | pass 5/5 | pass 5/5 |  |
| o_clear | pass 1/1 | fail 0/1 | SUBSKILL_FAILED: SUBSKILL_FAILED: fetch_object(object=snack_carton, receptacle=bookcase_2/top): SUBSKILL_FAILE |
| r_bimanual_box | fail 1/2 | fail 1/2 | POLICY_FAILED: bimanual_box_lift: two-hand lift/contact check failed path two_hand_box |
| t_floor_moving | pass 5/5 | pass 5/5 |  |

## 已知差距：捏住的物体会滑出

PhysX 的摩擦是"静摩擦"：没超过摩擦力上限就完全不动。
MuJoCo 的接触是软约束，持续的切向力会让物体以很小的速度蠕滑；指垫捏住的物体在手臂转腕时，会沿指垫一点点滑出去。
用接触轨迹（见上文）定位到的 k_trash 过程：易拉罐在 carry_up 期间沿指垫宽度方向以约 3 厘米每秒的速度滑动 1.5 秒，没有碰到任何东西，夹紧力约 35 N，远大于需要的摩擦力。

试过的办法和结果：

| 办法 | 结果 |
|---|---|
| 4 个子步，提高 impratio（50、100、200） | k_trash 仍然掉落 |
| 4 个子步，noslip 3 或 10 次迭代 | k_trash 仍然掉落，而且每步慢 4 倍 |
| 更硬的接触（solref 0.0045，solimp 0.95/0.99），4 个子步 | 仍然掉落 |
| 8 个子步（默认接触或更硬的接触）；6 个子步加更硬的接触 | k_trash 通过，定为默认 |
| 8 个子步加 noslip 5 次 | k_floor 通过，k_stir_cover 从第 6 步推进到第 11 步；k_bottle、k_hands 仍然掉落；b_books 反而推不动书了 |

所以默认是 8 个子步、不开 noslip。
下一步是把抓取接触建得更像 PhysX 的接触面：比如指垫用多个小盒子组成，让每个接触都有一块面积；或者只对指垫和被抓物体之间的接触做 noslip，而不是整个场景。
