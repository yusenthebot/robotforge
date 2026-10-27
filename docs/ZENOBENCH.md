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
  会动的刚体（物体和机器人）的凸包再按 PhysX 的顶点上限处理：PhysX 烹制凸包时最多保留 64 个顶点（场景里写了 `hullVertexLimit` 就用写的值），超出时把包围盒按凸包的面一刀刀切下去，见后文。
- `world` 按 PhysX 的方式推进仿真：一步 1/120 秒拆成 8 个 MuJoCo 子步，关节驱动用 PhysX 的隐式公式，底座跟随场景里那个被 ZenoBench 改写的固定关节锚点。
- `shim` 把 `isaacsim.core.prims`、`isaaclab.sim`、`isaaclab.sensors`、`isaaclab.app`、`omni.usd` 和 `pxr.PhysxSchema` 注册成真实的模块名，接口只做 ZenoBench 用到的部分。
  没装 PyTorch 时，一个很小的 NumPy 替身负责 ZenoBench 用到的几个 torch 调用。
  ZenoBench 是对着 Isaac Sim 自带的 NumPy 1.26 写的，有几处用二维向量做 `np.cross`，NumPy 2 已经不支持，shim 在 ZenoBench 进程里恢复了旧行为。
- 画面：可视网格按材质（GeomSubset）拆开，漫反射贴图缩到 512 像素后带 UV 贴上；Infinigen 家具动辄十万个面，渲染用的网格按尺寸的 1/250 做顶点聚类，碰撞仍用原网格。
  投影子的"太阳"跟着机器人底座走（方向在世界里不变），阴影贴图只覆盖机器人周围 5 米：整栋房子共用一张贴图时，机器人的影子边缘是锯齿状的色块。

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
- 摩擦：MuJoCo 只有一个摩擦系数，这里用 USD 材质的静摩擦系数；两个物体取较大的值。
  指垫材质在 USD 里写的就是取较大值（PhysX 里 max 的优先级最高），所以夹持和 PhysX 一致；物体之间、物体和台面之间 PhysX 取平均，这里大约高 15%。
- 底座轮子和转向连杆不参与碰撞：底座本来就由锚点拖动，轮子只是被拖着走。
- 接触偏移（contactOffset、restOffset）和 CCD 没有对应。
- 时间步长：PhysX 每步 1/120 秒，这里每步拆成 8 个 MuJoCo 子步（1/960 秒）。
  只拆 4 步时，手指捏住的易拉罐会在手臂上抬转腕时从指垫间慢慢滑出去（实测每秒约 3 厘米），8 步时不会。
- 关节限速（physxJoint:maxJointVelocity）：每个 MuJoCo 子步之后把关节速度截在限值内，和 PhysX 求解器的做法一致；手指是 0.1 米每秒。
- 刚体阻尼：PhysX 按速率（每秒）衰减每个刚体的速度，角速度默认 0.05，ZenoBench 给樱桃番茄和木块写了 3；这里给每个自由物体的关节加上"速率乘以质量或转动惯量"的阻尼，效果相同。
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
| `ZENOBENCH_DAMPING` | 1 | PhysX 的刚体阻尼（见上）；设为 0 关闭 |
| `ZENOBENCH_SEED` | 0 | 大于 0 时，第一步之前把每个自由物体挪一点点（同一个种子挪法相同），见"重复测量" |
| `ZENOBENCH_JITTER` | 1,0.5 | 上面那一挪的上限：水平毫米数、绕竖直轴的度数 |
| `ZENOBENCH_HULL_LIMIT` | all | PhysX 凸包顶点上限用在哪些刚体上：`all`（物体和机器人）、`objects`（只有自由物体）、`off` |

扭转摩擦按"滑动摩擦系数乘以 15 毫米接触半径"设置：PhysX 的平面接触是一块接触面，能抵抗扭转，MuJoCo 默认的 0.005 太小，捏住的物体会在指间打转。

## 调试：接触轨迹

`ZENOBENCH_TRACE` 写一个或几个刚体的 prim 路径（逗号分隔），仿真时每隔 `ZENOBENCH_TRACE_EVERY` 步（默认 4）把它们的位姿、所有接触（对方 geom、距离、法向力）和手指关节写成 JSON 行，文件名由 `ZENOBENCH_TRACE_OUT` 指定。
定位"物体为什么从手里掉了"这类问题时，先记下物体和夹爪的轨迹，再算物体在夹爪坐标系里的位移，就能分清是碰撞、蠕滑还是被甩出去。
夹持静摩擦锁上锁、松开和松开的原因（滑动还是扭转超载、手指松开）会以 INFO 日志打印，激活中的锁和它承受的力也会写进轨迹。

`ZENOBENCH_RECORD_AT` 写一个 tick 号：仿真到这一步时，把编译好的模型（.mjb）和完整状态存下来，再记录之后 `ZENOBENCH_RECORD_TICKS` 个 tick（默认 1200）的控制量，文件名前缀由 `ZENOBENCH_RECORD_OUT` 指定。
离线回放这一段（加载模型、恢复状态、逐 tick 套用控制量）只要几秒钟，可以在出问题的那一刻反复试不同的物理设置，而不用每次重跑几分钟的整个回合。

ZenoBench 自己的录像机位在机器人身后，手在做什么常被身体挡住。
`ZENOBENCH_FILM` 写一个 mp4 路径，会在真实运行时另录一路导演机位：每秒 30 帧（仿真时间），镜头跟着 `ZENOBENCH_FILM_TARGET` 里的刚体（prim 路径，逗号分隔时取中点，默认机器人底座），从机器人坐标系里一个固定的角度看过去。
`ZENOBENCH_FILM_VIEW` 是方位角、仰角（度）和距离（米），默认 `35,25,2.2`，方位角 0 是机器人正前方，90 是它的左侧；`ZENOBENCH_FILM_SIZE` 默认 `960x540`。
墙和固定的家具挡在中间时，镜头绕到离设定角度最近的、看得见的方向；四周都挡着（走廊里）才往近处拉。
例如看 k_pin 里的擀面杖：

```bash
ZENOBENCH_FILM=runs/k_pin_film.mp4 ZENOBENCH_FILM_TARGET=/World/Tasks/rolling_pin/Asset/rolling_pin ZENOBENCH_FILM_VIEW=35,25,1.6 \
    uv run robotforge zenobench python -- tools/verify_skills.py --id k_pin --jobs 1 --video --out runs/k_pin
```

## 和 Isaac Sim 的结果对比

ZenoBench 自带物理验证工具 `tools/verify_skills.py`，35 个场景逐步检查技能的前置和后置条件，Isaac Sim 上的结果记录在 `skill_library/verification/STATUS.md`。
在 MuJoCo 上跑同一套场景：

```bash
uv run robotforge zenobench python -- tools/verify_skills.py --all --jobs 4 --out runs/mj_all
```

2026-10-27 在 M5 Max 上的结果（ZenoBench `d573467`，默认设置，6 个并行任务，35 个场景共 49 分钟，期间测试套件也在跑）：

- MuJoCo 通过 23/35，Isaac Sim 通过 29/35，两边都通过的 22 个。
- k_arrange_sort 只在 MuJoCo 上通过；Isaac 上它在第 1 步取物时失败。
- Isaac 也失败的另外 5 个，MuJoCo 同样失败；k_soup_stove 停在和 Isaac 相同的那一步，原因也相同（倒汤时 0/3 落进锅里），k_fridge_chill 比 Isaac 多走了一步。
- 只在 MuJoCo 上失败的 7 个：
  - 找不到能把书滑过去的桌边（k_book、g_expose_then_pick_book）。
    这一步只用标注和物体位姿做几何判断，不经过物理引擎；书在 MuJoCo 里一动没动，每条边被排除的理由都能复现（路径上压着马克杯和番茄）。
    所以更可能是 Isaac 的验证记录早于当前布局，需要在 Isaac 上重跑才能确认。
  - k_separate 和 k_restore_swap：下一节，没找到移植上的问题。
  - k_pin：滚和拉两步现在都通过，第 4 步把擀面杖往里推时手从它旁边擦过去。
    ZenoBench 估计物体底部时把"原点到底面中心"的偏移跟着物体一起转；擀面杖滚到 -84 度时，这个估计横向偏了 2.8 厘米（半径乘以滚动角的正弦），手正好对着偏掉的位置。
    Isaac 记录里它滚到 174.5 度，偏差只有 3 毫米；滚多少度两边都会有随机性，MuJoCo 这几轮在 54 到 179 度之间。
  - b_books：两次推书都推到了位，最后一步双手拿书时 ZenoBench 的规划器找不到两只手共用的无碰撞底座位姿。
  - k_bottle：上一轮通过。
    瓶子的凸包按 PhysX 削到 64 个顶点后，推倒的瓶子朝向差了约 1 度，ZenoBench 换了一个抓取点、绕到另一侧，扶正以后找不到能把瓶子立回去的位置。
    设 `ZENOBENCH_HULL_LIMIT=off` 时它照样通过；这是规划对微小姿态的敏感，不是瓶子从手里滑掉。
- 历次结果：第一版 18/35，差距主要是物体在搬运中从手里滑掉；加上夹持静摩擦锁后 24/35；这一轮修正推、拖、滚的物理后 23/35，丢的是上面的 k_bottle，换来的是推和滚的距离接近 Isaac（见下一节）。
- 下表是单次运行；加上刚体阻尼以后每个场景跑 5 次的通过率见下一节。

| Scenario | Isaac Sim | MuJoCo | MuJoCo failure |
|---|---|---|---|
| b_books | pass 7/7 | fail 5/6 | POLICY_FAILED: bimanual_flat_pick: no collision-free shared base pose path two_hand_flat |
| c_approach_moving | pass 2/2 | pass 2/2 |  |
| c_tray_moving | pass 5/5 | pass 5/5 |  |
| g_expose_then_pick_book | pass 5/5 | fail 1/2 | slide to edge: no graspable free edge of kitchen_island/top |
| g_heat_on_stove_instead_of_microwave | pass 5/5 | pass 5/5 |  |
| g_search_then_fetch | pass 5/5 | pass 5/5 |  |
| k_arrange_sort | fail 1/2 | pass 2/2 |  |
| k_blocks | pass 6/6 | pass 6/6 |  |
| k_body | pass 5/5 | pass 5/5 |  |
| k_book | pass 5/5 | fail 1/2 | POLICY_FAILED: slide to edge: no graspable free edge of kitchen_island/top path slide_to_edge |
| k_bottle | pass 3/3 | fail 2/3 | POLICY_FAILED: upright juice_bottle: no base pose reaches the top end path pick_orient_place |
| k_brace | pass 4/4 | pass 4/4 |  |
| k_cabinet | pass 8/8 | pass 8/8 |  |
| k_empty_hide | fail 2/3 | fail 2/3 | POLICY_FAILED: pick cherry_tomato_2: not held (lift -0.000 m) path pick_each_inside |
| k_fetch_collect | pass 2/2 | pass 2/2 |  |
| k_floor | pass 4/4 | pass 4/4 |  |
| k_fridge_chill | fail 5/6 | fail 6/7 | POLICY_FAILED: tuck arm: posture did not settle path handle_push |
| k_handle_mug | pass 2/2 | pass 2/2 |  |
| k_hands | pass 8/8 | pass 8/8 |  |
| k_microwave | fail 12/13 | fail 5/6 | POLICY_FAILED: joint move to microwave_front: every joint-space path collides path microwave_staged |
| k_mug_on_burner | pass 8/8 | pass 8/8 |  |
| k_perceive | pass 12/12 | pass 14/14 |  |
| k_pin | pass 4/4 | fail 3/4 | POLICY_FAILED: push rolling_pin: object did not move path push_inward |
| k_posture | pass 9/9 | pass 9/9 |  |
| k_present | pass 6/6 | pass 6/6 |  |
| k_restore_swap | pass 3/3 | fail 2/3 | SUBSKILL_FAILED: SUBSKILL_FAILED: place_object(object=soda_can, receptacle=kitchen_island/top, hint_xy=[7.6787 |
| k_separate | pass 5/5 | fail 3/4 | PRECONDITION_FAILED: PRECONDITION_FAILED: not grasp_clearance(object=wooden_block_a) is false (pinch at [7.219 |
| k_soup_stove | fail 6/7 | fail 6/7 | POLICY_FAILED: pour: 0/3 items reached handled_cooking_pot path tilt_over_rim |
| k_stir_cover | pass 13/13 | pass 13/13 |  |
| k_sweep | pass 3/3 | pass 3/3 |  |
| k_trash | pass 10/10 | pass 10/10 |  |
| k_wipe | pass 5/5 | pass 5/5 |  |
| o_clear | pass 1/1 | pass 1/1 |  |
| r_bimanual_box | fail 1/2 | fail 1/2 | POLICY_FAILED: bimanual_box_lift: two-hand lift/contact check failed path two_hand_box |
| t_floor_moving | pass 5/5 | pass 5/5 |  |

Isaac 29/35 pass, MuJoCo 23/35 pass, both 22

## 重复测量：每个场景跑 5 次

MuJoCo 是确定性的，同一个场景跑一次只能看到一种结果，而有些结果取决于很小的差别（推倒的瓶子朝向差一度、擀面杖滚多远）。
`sweep` 每个种子跑一遍 `tools/verify_skills.py`。
种子 0 是原样的场景；种子大于 0 时，第一步之前把每个自由物体水平挪最多 1 毫米、绕竖直轴转最多 0.5 度（`ZENOBENCH_SEED`、`ZENOBENCH_JITTER`），同一个种子挪法相同。
已经跑完的种子会跳过，中断以后重跑同一条命令就接着跑。
结果和 Isaac 的单次记录并排写进 `<out>/summary.md`：

```bash
uv run robotforge zenobench sweep -- --seeds 5 --jobs 4 --out runs/sweep5
uv run robotforge zenobench sweep -- --summary --out runs/sweep5   # 只汇总已有结果
```

2026-10-28 的结果（robotforge `c1b5466`，默认设置，已加上 PhysX 刚体阻尼；175 次运行约 2.5 小时，同时还在跑下一节的基准）：

- 按通过率算，MuJoCo 期望通过 22.4 个场景，Isaac 单次是 29/35。
- 21 个场景 5 次都通过，11 个 5 次都失败，只有 3 个时过时不过。
  1 毫米的扰动改变不了其余 32 个场景的结果，所以和 Isaac 差的那几个场景不是运气，每次都会复现。
- 5 个种子各自的通过数是 23、21、22、23、23。
- 5 次都失败的 11 个，正是上一节表里失败的 12 个去掉 b_books。
- 时过时不过的 3 个：
  - b_books 2/5：失败的 3 次都停在最后一步，ZenoBench 找不到两只手共用的无碰撞底座位姿，和上一节单次运行的失败相同；两次推书各推多远，决定了书最后停在哪。
  - k_hands 4/5：有一次左手拿着东西去开门时，东西从左手滑了出去。
  - k_brace 1/5：加上刚体阻尼以后才这样。
    ZenoBench 给樱桃番茄写了 3 的角阻尼，番茄落定时只滚 5 毫米（不加阻尼时 13 毫米），ZenoBench 据此选了另一个停放手臂的位置，收左臂时关节误差 0.41 弧度。
    设 `ZENOBENCH_DAMPING=0` 重跑时它通过。
- 种子 0 上 b_books 现在通过、k_brace 失败，所以和上一节相比总数还是 23/35，换了一个场景。

| Scenario | Isaac Sim | MuJoCo passes | Steps passed | Most common failure |
|---|---|---|---|---|
| b_books | pass 7/7 | 2/5 | 5/6, 7/7 | bimanual_flat_pick: no collision-free shared base pose |
| c_approach_moving | pass 2/2 | 5/5 | 2/2 |  |
| c_tray_moving | pass 5/5 | 5/5 | 5/5 |  |
| g_expose_then_pick_book | pass 5/5 | 0/5 | 1/2 | slide to edge: no graspable free edge of kitchen_island/top |
| g_heat_on_stove_instead_of_microwave | pass 5/5 | 5/5 | 5/5 |  |
| g_search_then_fetch | pass 5/5 | 5/5 | 5/5 |  |
| k_arrange_sort | fail 1/2 | 5/5 | 2/2 |  |
| k_blocks | pass 6/6 | 5/5 | 6/6 |  |
| k_body | pass 5/5 | 5/5 | 5/5 |  |
| k_book | pass 5/5 | 0/5 | 1/2 | slide to edge: no graspable free edge of kitchen_island/top |
| k_bottle | pass 3/3 | 0/5 | 2/3 | upright juice_bottle: no base pose reaches the top end |
| k_brace | pass 4/4 | 1/5 | 3/4, 4/4 | fold left arm: joint error 0.412 rad |
| k_cabinet | pass 8/8 | 5/5 | 8/8 |  |
| k_empty_hide | fail 2/3 | 0/5 | 2/3 | pick cherry_tomato_1: not held (lift 0.000 m) |
| k_fetch_collect | pass 2/2 | 5/5 | 2/2 |  |
| k_floor | pass 4/4 | 5/5 | 4/4 |  |
| k_fridge_chill | fail 5/6 | 0/5 | 6/7 | tuck arm: posture did not settle |
| k_handle_mug | pass 2/2 | 5/5 | 2/2 |  |
| k_hands | pass 8/8 | 4/5 | 6/7, 8/8 | open door while left holds: left grasp slipped |
| k_microwave | fail 12/13 | 0/5 | 10/11, 5/6 | joint move to microwave_front: every joint-space path collides |
| k_mug_on_burner | pass 8/8 | 5/5 | 8/8 |  |
| k_perceive | pass 12/12 | 5/5 | 14/14 |  |
| k_pin | pass 4/4 | 0/5 | 3/4 | push rolling_pin: object did not move |
| k_posture | pass 9/9 | 5/5 | 9/9 |  |
| k_present | pass 6/6 | 5/5 | 6/6 |  |
| k_restore_swap | pass 3/3 | 0/5 | 2/3 | SUBSKILL_FAILED: place_object(object=soda_can, receptacle=kitchen_island/top) |
| k_separate | pass 5/5 | 0/5 | 3/4 | PRECONDITION_FAILED: not grasp_clearance(object=wooden_block_a) is false |
| k_soup_stove | fail 6/7 | 0/5 | 6/7 | pour: 0/3 items reached handled_cooking_pot |
| k_stir_cover | pass 13/13 | 5/5 | 13/13 |  |
| k_sweep | pass 3/3 | 5/5 | 3/3 |  |
| k_trash | pass 10/10 | 5/5 | 10/10 |  |
| k_wipe | pass 5/5 | 5/5 | 5/5 |  |
| o_clear | pass 1/1 | 5/5 | 1/1 |  |
| r_bimanual_box | fail 1/2 | 0/5 | 1/2 | bimanual_box_lift: two-hand lift/contact check failed |
| t_floor_moving | pass 5/5 | 5/5 | 5/5 |  |

## 整任务基准：和 Isaac 的 benchmark/RESULTS.md 对比

上面的验证逐个检查技能；ZenoBench 的基准跑的是整个任务：机器人从回合起点出发，规划器一个接一个调用技能，最后按任务目标判成败。
Isaac 上的结果记录在 `benchmark/RESULTS.md`，分三批：

- accept：10 个任务的 60 条脚本路线，包括"先失败一次再换办法"的路线，失败的那一步也要按预期失败才算通过。
- probes：21 个单项探针，每个只测一类技能。
- baselines：三种规划器各跑 10 个回合；fixed 按固定计划执行，retry 失败了就重试，oracle 知道真实世界的状态。

`run_episode.py --batch` 最多同时开 2 个进程（Isaac 的限制），想跑得快就把批次拆成几份分别启动，输出写进同一个目录。
`compare` 按"任务、种子、变体、规划器、路线"逐个回合对上 Isaac 的记录，后写的目录覆盖前面的同一回合，方便在整批结果上叠加重跑：

```bash
uv run robotforge zenobench episode -- --batch benchmark/batches/accept.json --jobs 2 --no-eval --out runs/bench/accept
uv run robotforge zenobench episode -- --batch benchmark/batches/probes.json --jobs 2 --no-eval --out runs/bench/probes
uv run robotforge zenobench episode -- --batch benchmark/batches/baselines.json --jobs 2 --no-eval --out runs/bench/baselines
uv run robotforge zenobench compare -- runs/bench/accept runs/bench/probes runs/bench/baselines
```

2026-10-28 的结果（124 个回合，没有崩溃或超时，回合时间加起来 7.2 小时）：

| 批次 | MuJoCo | Isaac Sim（同样的回合） |
|---|---|---|
| accept（路线） | 43/60 | 60/60 |
| probes | 13/21 | 14/21 |
| baselines：fixed | 2/10 | 2/10 |
| baselines：retry | 2/10 | 3/10 |
| baselines：oracle | 9/10 | 9/10 |

基准的主要结论在 MuJoCo 上照样成立：世界和计划不符时，固定计划和简单重试大多失败，知道真实状态的 oracle 几乎都能完成。

accept 里只在 MuJoCo 上失败的 17 条路线，按原因分：

- T07 热燕麦（6 条）：放进微波炉时 ZenoBench 的规划器说"每条关节空间路径都会碰撞"；改用灶台的路线，按灶台电源键时差了 3.9 厘米。
  这和验证场景 k_microwave 失败在同一处，probes 里的 P13 和 oracle 的 T07 也是。
- T01 书（4 条）：边缘捏书失败后，改成推书露出书边，书一动不动；验证场景 k_book、g_expose_then_pick_book 和探针 P05 翻书，问题也都出在把书往桌边滑的这一步。
- T02 叠木块（4 条）：路线预期"挤在一起的木块拿不起来"，再走先分开的办法；MuJoCo 里第一次就拿起来了，预期的失败没有发生。
  oracle 的 T02 反而只在 MuJoCo 上通过，和这一条是同一回事；retry 的 T02 是搬运时木块从手里滑掉。
- T09 番茄汤（2 条）：搅拌只转了 0.37 圈。
- T05 铅笔（1 条）：要找的铅笔在搜索的几个位置上都没看见。

另外 4 条起初也失败，已经修正：T08 和 T10 的 hard 路线从别的房间开始。
ZenoBench 把机器人摆到回合起点时，以每秒 2.4 米拖着底座锚点直线走过去，中间穿墙；MuJoCo 里底座被墙挡住，回合从错的地方开始，路上还撞开过冰箱门。
ZenoBench 自己导航最快每秒 0.35 米、0.8 弧度，所以现在锚点一步走得比每秒 1.5 米或 3 弧度还快时，直接把底座放到锚点上，并关掉机器人的碰撞，直到锚点慢下来；正常导航不受影响。
修正后重跑了 13 个从别处开始的回合，这 4 条通过，原来通过的 5 条仍然通过（上表已经是修正后的结果）。

probes 里 P03 擀面杖（推到台面中间时离边只有 4.7 厘米，要求 6 厘米）、P05、P13 只在 MuJoCo 上失败，P06 双手和 P09 扫木块只在 MuJoCo 上通过。

## 推、拖和滚：凸包顶点上限和锁的误判

k_pin、k_restore_swap、k_separate 和 b_books 最后一步的差距都出在推、拖、滚上。
逐个用快照回放查下来，有两处是移植和 PhysX 不一致，已经修正：

1. 夹持锁的误判。
   推、拖、滚的时候夹爪是合拢的，两根指尖一起从上面压在物体上，每根都超过 2 N，旧规则把这当成捏住，物体被焊在手上跟着走。
   b_books 里绿书因此被拖出 13.2 厘米（要求 11.5），第二次推的时候手就没碰到书；k_pin 滚擀面杖时也锁住过。
   现在要求两根手指的推力方向相反。
2. 凸包太细。
   PhysX 烹制凸包时最多保留 64 个顶点（Isaac 里 `hullVertexLimit` 的取值范围是 8 到 64），这里原来直接用网格的完整凸包。
   擀面杖的凸包有 995 个顶点，中段截面是 149 边形，几乎是光滑的圆柱，回退的推一下就滚出 31 厘米、掉下台面（要求 8 厘米）；PhysX 里它只有 64 个顶点，滚起来有颠簸和损耗。
   现在会动的刚体按 PhysX 的默认做法处理：凸包不超过上限就原样保留，超过时从点集的定向包围盒开始，每次用当前形状超出最多的那个凸包面切一刀，直到再切就超过上限。
   结果仍然包住所有原始点；厨房、水果两个场景和机器人上被截的 115 个凸包，体积最多是原来的 1.11 倍，中位数 1.04 倍。
   擀面杖中段截面变成 24 边形，改完后同一下推让它走了 12.9 厘米（要求 8），没有再掉下台面。
   起初按面积从大到小选面来切，机器人躯干的凸包大了 25%，b_books 里的书被推着撞上机器人自己的躯干，被挤得飞出去 35 厘米；改成先切超出最多的面后躯干只大 9%，两次推书都推到了位（12 厘米推了 11.4，11.5 厘米推了 12.6）。

另外两个场景没找到移植上的问题：

- k_separate：放 wooden_block_b 时要求的位置 (7.22, 1.69) 被 wooden_block_a 占着，ZenoBench 在 3 厘米网格上找最近的空位，得到 (7.230, 1.725)，这只取决于标注和物体位姿，和物理引擎无关。
  MuJoCo 把木块放在了离这个点 4 毫米的地方，此时 wooden_block_a 两侧的手指槽都是空的，"先分开"这一步的前置条件不成立。
  Isaac 记录里它通过了，说明那次 wooden_block_a 离得更近；和 k_book 一样需要在 Isaac 上重跑确认。
- k_restore_swap：去抓易拉罐时，手掌的碰撞凸包（底面在 TCP 上方 3.6 厘米）以 68 N 压在罐顶，手停在抓取点上方 2.4 厘米。
  PhysX 用的是同一个网格的凸包，这一步应该和 Isaac 一样。
  之后手臂转向放置点，罐子撞上机器人自己的头（最大 258 N），超过摩擦能提供的力，夹持锁松开，罐子在手里移位，放下时偏了 7.4 厘米；头的凸包按 PhysX 削顶点后大了 9%，这一轮罐子干脆从手里滑了出去。
  回放里如果锁一直不松，罐子会落在目标 5 毫米以内；撞头的接触点离头部真实碰撞网格不到 1 毫米，头的姿态也和目标一致。
  Isaac 记录里这个罐子的误差是 1.8 厘米，是这组里最大的，那边可能也碰到了头，只是偏得没这么多。

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

- 同一只手的两根手指都以至少 2 N 从相对的两侧压在同一个物体上（两侧推力的方向至少差 120 度）、物体相对手掌基本静止时，激活一个建模时就声明好（默认不激活）的焊接约束，把物体锁在手掌上当前的位置。
  "相对的两侧"这条是后来加的：推、拖、滚的时候两根并拢的指尖从上面压住物体，两侧推力同向，不是捏，之前也被锁住，物体就被手带着走。
- 每个 tick 计算夹持必须对外提供的力：物体受到的全部约束力（焊接和指垫加在一起，彼此的内力抵消）减去它和其他东西的接触（比如它压着的桌面），按 0.1 秒做平滑。
  这个力超过"摩擦系数乘以两侧夹紧力"，或力矩超过它再乘以 2 厘米接触半径时，锁就松开，物体照常滑动，静止后可以再锁上。
- 任一根手指连续 50 毫秒不再压着物体（张开手、放下物体）时松开。
- 关闭焊接约束不会唤醒正在休眠的物体，所以松开时给物体一个 tick 的极小推力，免得它悬在手离开的位置。

单元测试覆盖了上锁、抬起时零漂移、张手后掉落、摩擦拿不住的重物会滑脱而不是被焊住提起来，以及指尖从上面压住物体时不会上锁。
