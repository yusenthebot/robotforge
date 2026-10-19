# 算法适配层：同一个机器人，换算法对比

robotforge 把“让机器人做到一件事”拆成两半：任务（要做到什么）和算法（怎么做到）。
同一个任务可以交给不同的算法，跑完给出同一套指标，新手可以直接看到每种方法的长处和代价。
代码在 `src/robotforge/control/`，不需要额外安装任何依赖。
神经网络强化学习（PPO、SAC）例外：它需要可选依赖组 `rl`，见下文“神经网络强化学习”一节。

## 一句话怎么用

在 Claude 或 GPT 里直接说：

- “有哪些控制算法？” 对应工具 `list_algorithms`，列出任务、适用的算法和一句话解释。
- “用 LQR 让倒立摆保持平衡，2 秒时推它一下” 对应 `run_controller(robot="pole", task="balance", method="lqr", push_at=2)`。
- “用 MPC 让机械臂把末端移到方块那里” 对应 `run_controller(robot="arm", task="reach", method="mpc", task_params={"target": "cube"})`。
- “对比几种算法谁更好” 对应 `compare_controllers(robot="arm", task="reach")`，返回一张对比表。
- “训练倒立摆保持平衡” 对应 `train_policy(task="balance", robot="pole")`，在后台训练，立刻返回。
- “训练我的机器狗走得更快” 对应 `train_policy(task="walk", robot="dog", task_params={"speed": 0.6})`。

查看器打开时，`run_controller` 按真实时间播放，用户能看到机器人怎么动。
跑完后控制器继续工作（比如继续扶住摆杆），用 `stop(robot)` 释放。
`compare_controllers` 在后台从当前状态出发分别跑每种算法，不会改动正在显示的世界。

在 Python 里：

```python
from robotforge.control import compare_methods, make_env

print(compare_methods("balance", ["lqr", "pd", "mpc"], seconds=6, push_at=3).table)

env = make_env("reach")  # Gym 风格环境
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## 五个内置任务

| 任务 | 要做到什么 | 适用的机器人 | 可选算法 |
|---|---|---|---|
| reach | 把机械臂末端（tcp 点）移到目标点并停住，误差小于 1 厘米算成功 | 机械臂模板，库里的机械臂（Panda、UR5e 等） | ik, pd, cartesian, joint_impedance, cartesian_impedance, osc, mpc, mppi |
| balance | 推小车让摆杆保持竖直，小车尽量待在中间，倾角超过 46 度算失败 | 倒立摆模板 cartpole | lqr, pd, mpc, mppi, ars, none |
| stand | 在指定高度站稳，不漂移，被推之后能恢复 | 四足模板 | hold, mpc, mppi |
| walk | 按指定速度往前走（可带转向），不能摔倒 | 四足模板 | gait, mpc, mppi |
| drive_to | 开到地面上的目标点并停下，误差小于 5 厘米算成功 | 小车模板 | pd, mpc, mppi |

目标可以写坐标，也可以写物体名字，例如 `task_params={"target": "cube"}` 或 `{"goal": [1.0, 0.5]}`。
不写目标时会随机选一个肯定够得着的点。
倒立摆是新加的机器人模板，用 `add_robot(name="pole", template="cartpole")` 放进任何世界。

## 每种算法是什么，什么时候用

### ik：逆运动学加舵机位置控制

先算出“末端要到那个点，每个关节该转多少度”，再把角度直接发给舵机。
这是绝大多数业余机械臂的做法，简单可靠。
舵机本质上是一根弹簧，重力会把手臂往下拉，所以会差一两厘米（实测 1.9 厘米）。

### pd：比例微分反馈

误差越大推得越用力（P），误差变化越快越要刹车（D）。
机械臂上的 pd 还加了重力补偿：先算出抵消重力需要的力矩再叠加上去，所以能消掉 ik 的下垂，实测误差 1.2 毫米。
倒立摆上的 pd 是一条凭直觉写的规则：“摆杆往哪边倒，就把小车往哪边推”。
小车上的 pd 是“先转向目标，再按距离开过去，越近越慢”。

### cartesian：操作空间控制

不解逆运动学，而是想象末端被一根看不见的弹簧直接拉向目标，再把这股拉力换算成各关节力矩。
适合要求末端走直线、或者要控制末端力的场合。

### 柔顺控制：关节阻抗、笛卡尔阻抗、操作空间控制

这三种是给“会碰到东西”的机械臂用的，代码在 `src/robotforge/control/impedance.py`。
它们都先拿到任务每一步给出的目标（末端该在哪、多快、朝哪、要不要往某个面上压多大的力），再算关节力矩，由 `TorqueMap` 换成舵机或电机能收的指令。

- joint_impedance：每个关节是一根弹簧加阻尼，拉向逆运动学给出的角度，再加重力补偿。
  关节很软，碰到东西会让；压出多大的力等于关节刚度乘以关节被拦下的角度，所以刚度写在数字里（N·m/rad），不随姿态变。
- cartesian_impedance：末端本身是弹簧阻尼拉向目标，`torque = J^T F + 重力`，用不到的关节只有一根弱弹簧拉回休息姿态。
  末端像一个挂在弹簧上的质量，不管手臂摆什么姿态，它在空间里的软硬都一样（N/m，可以每个轴不同：沿桌面软，沿桌面法向更软）。
- osc：操作空间控制。把末端的误差换成末端该有的加速度，再乘上末端空间的惯量，科氏力和重力全部精确抵消，所以末端在任何姿态下都像一个单位质量。
  多余的关节在零空间里动，不会干扰末端。
  压在面上时，法向不控位置，而是对测到的接触力做闭环（力位混合控制）。

力是这样测的：手和手里拿着的东西受到的接触力，等价于腕上一个力矩传感器（`ToolForce`），再过一个 40 Hz 的低通。
带重力补偿的模型（Menagerie 的 UR5e 自己抵消了连杆重力）不会被补偿两次。

Panda 和 UR5e 上的实测（reach 4 秒，往前 0.1 米、往侧 0.1 米、往下 0.12 米）：4 秒结束时三种的误差都在 1 毫米上下（实测 0 到 1.0 毫米）。

### lqr：线性二次型调节器

先让 MuJoCo 在平衡点附近把物理“线性化”（`mjd_transitionFD` 给出 x' = A x + B u），再解 Riccati 方程，得到一组最优反馈增益 K。
你只需要说“哪个误差更重要”（Q）和“用力有多贵”（R），增益是算出来的，不是调出来的。
它在平衡点附近又快又稳，计算量几乎为零，但离平衡点太远就不灵了。

### mpc：预测采样模型预测控制

每 20 毫秒做一次：把上一次的计划往后平移，随机做几十个变体，用物理引擎把每个变体往未来模拟零点几秒，按任务的代价打分，执行最好的那个。
不需要梯度，也不需要训练，任何物理引擎能模拟的机器人都能用，这是它最大的优点。
代价是计算量大，而且只能看到很短的未来。
模拟用的是 `mujoco.rollout`，多线程 C 代码，Python 不参与每一步。

### mppi：模型预测路径积分

和 mpc 一样采样很多未来，但不是只选最好的一个，而是按好坏加权平均所有方案。
动作通常更平滑，更省力，但在需要“果断转一大圈”的目标上可能犹豫。

### ars：强化学习（增强随机搜索）

不给任何控制规则，让机器人自己试错：随机微调一个线性策略（动作等于观测的加权和），保留得分高的方向，反复几十轮。
它只用 numpy，不需要安装 PyTorch，在倒立摆上大约 3 秒就学会平衡（相当于在仿真里练了 800 多秒）。
`run_controller` 和 `compare_controllers` 里选 ars 时，会先当场训练再运行，结果里有训练用时。
目前只对 balance 开放；更难的任务（随机目标的 reach、走路）需要神经网络策略，也就是可选 `rl` 依赖组里的 PPO、SAC，见下文。
在 Python 里也可以直接训练：`train_ars(make_env("balance"))`，返回的策略能 `save` 成一个小文件。

### hold、gait、none

hold 是让所有舵机保持初始姿态，也就是“没人控制时机器人在做什么”。
gait 是内置的小跑步态：对角腿一起迈，脚走预设的弧线，用腿部逆运动学算关节角。
none 是完全不控制，用来看物理本身会发生什么（摆杆会倒）。

### 叠加模式：在已有行为上做修正

走路时，单纯的采样 MPC 在 0.4 秒的预测里发明不出步态，实测只挪了 25 厘米。
所以 walk 任务里的 mpc 和 mppi 默认叠加在内置步态上：步态负责迈腿，MPC 只学“每条腿该多迈一点还是少迈一点”。
想试纯 MPC，用 `method_params={"base": None}`。
Gym 环境也有同样的叠加模式，见下文的 `action_mode="residual"`。

## 触觉任务：压和擦

机械臂碰到东西之后，要紧的是力。
`press` 和 `wipe` 两个任务把“力”当成要做到的事，所有控制方法在同一个任务、同一份计划上比。

| 任务 | 要做到什么 | 主要参数 | 可选算法 |
|---|---|---|---|
| press | 在工具下面的面上压：`mode="hold"` 压住并保持这个力，`mode="stop"` 往下走，力一到限值就停（受力限制的接触） | `force`（牛，默认 5）、`mode`、`depth`、`surface` | ik, pd, joint_impedance, cartesian_impedance, osc |
| wipe | 在桌面上来回擦几遍，同时保持压力 | `force`（8）、`length` 0.2 米、`width` 0.1 米、`passes` 3、`speed` 0.08 米每秒、`depth`、`center`、`surface` | 同上 |

两个任务都用机械臂末端的 tcp 点，和 reach 一样；机械臂要在一个面（桌子）上方。
还有第三个触觉任务 `insert`（把夹着的销插进孔里）和抓取规划、用力拧阀门，见 [MANIPULATION.md](MANIPULATION.md)。
`surface` 写物体的名字，不写就用工具下面的东西；桌子不在手臂够得着的地方时，开始前就会告诉你哪一点够不着。

### 计划：所有方法拿到同一份

计划每一步给出目标：工具该在哪、多快、朝哪（指向面里面）、压多大的力。
流程是：移到起点上方，往下走（先快后慢，最后 5 厘米以 2 厘米每秒），碰到（滤波后的力到 1.5 牛）就停，0.5 秒内把力升到目标值，然后沿路径走，最后抬起。
`wipe` 的路径是 `passes` 道来回的线，拐角处放慢，结束时抬起 6 厘米。

位置控制不知道力，所以计划也给它们一个位置：碰到的点再往下 `depth`（默认 3 毫米）。
这是给“估计表面高度”的误差：真机的桌面高度也只能知道到几毫米。
位置控制把几毫米变成多大的力，取决于手臂有多硬：这就是对比要看的。

- ik、pd：只往那个位置走。
- joint_impedance：关节是软弹簧，位置目标加上一个前馈的压力（雅可比转置）。
- cartesian_impedance：末端在空间里是弹簧，沿面的方向硬（2000 牛每米，摩擦拖着工具时还能走直），沿面的法向软（400 牛每米），再加前馈压力。
- osc：沿面的方向像阻抗一样硬；法向由一根软弹簧把工具按在面上，目标位置不带 `depth`，压力由测到的力做 PI 闭环修正，再加一个沿法向的阻尼（一个控制步长能承受的最大阻尼的 60%）。

法向的阻尼是擦的时候必须有的：手臂拖着一个有摩擦的工具滑过去，不加阻尼，工具会在面上弹跳，接触力在零和目标值的两三倍之间来回跳。

### 擦头

擦要用一块平的软垫：`update_robot("panda", end_effector="wiper")`，在任何带 tcp 点的机械臂上装一块黄色海绵（6 x 5 厘米，8 牛压下去约 1 毫米，比硬的垫子软，滑动时力的抖动小得多），tcp 点移到海绵面上。
手指原来的 `gripper` 改名成 `fingers` 并张开；海绵什么也拿不住，所以带擦头的手臂不能 `pick_and_place`。
没装擦头也能擦：这时手会先合上手指，用指尖的小方块擦；接触是几个小方块，力的噪声比海绵大得多。

### 实测（Panda 和 UR5e，Menagerie 模型，擦头，M5 Max）

力用手腕力传感器的方式读：手和擦头受到的接触力，过 40 Hz 低通。
`wipe`，目标 8 牛，三道线，每道 20 厘米，共约 14 秒：

| 机械臂 | method | 平均力 (N) | 标准差 (N) | 丢失接触 | 路径误差 (mm) | 擦到 | 成功 |
|---|---|---|---|---|---|---|---|
| Panda | ik | 130.8 | 54.7 | 0% | 93.3 | 100% | no |
| Panda | pd | 13.6 | 12.1 | 17.3% | 15.9 | 72.9% | no |
| Panda | joint_impedance | 22.1 | 12.7 | 1.3% | 66.8 | 100% | no |
| Panda | cartesian_impedance | 9.5 | 1.8 | 0.1% | 5.4 | 100% | yes |
| Panda | osc | 8.1 | 2.1 | 0.5% | 5.7 | 100% | yes |
| UR5e | ik | 43.9 | 3.4 | 0% | 18.7 | 100% | no |
| UR5e | pd | 3.6 | 1.7 | 21.5% | 3.3 | 97.1% | no |
| UR5e | joint_impedance | 11.0 | 1.7 | 0.1% | 24.8 | 100% | no |
| UR5e | cartesian_impedance | 9.3 | 1.5 | 0% | 4.5 | 100% | yes |
| UR5e | osc | 8.0 | 1.1 | 0% | 4.7 | 100% | yes |

“成功”是：擦到的路径不少于 90%，平均力和目标差不到四分之一，95% 的时间力不到目标的两倍。
读表：

- ik 把手臂自己的舵机当成很硬的弹簧，几毫米的深度变成几十到一百多牛，力是手臂的硬度乘上位置误差。
- pd 的增益更高、没有力的前馈：压下去又被弹起，接触断断续续。
- joint_impedance 在关节里软，在空间里什么方向都一样软：要让工具沿着路径走直，就得硬，但法向的力就跟着变大；两头顾不了，路径误差 25 到 67 毫米。
- cartesian_impedance 的法向软、沿面硬，力只比目标多出 `depth` 乘 400 牛每米，也就是 1.2 牛。
- osc 用测到的力闭环，不靠位置，平均力正好是目标，UR5e 上标准差 1.1 牛。

`press`，目标 6 牛：

| 机械臂 | method | hold 的稳定力 (N) | hold 的峰值 (N) | stop 的峰值 (N) |
|---|---|---|---|---|
| Panda | ik | 41.7 | 56.4 | 37.8 |
| Panda | pd | 1.9 | 73.8 | 6.0 |
| Panda | joint_impedance | 6.6 | 38.0 | 6.0 |
| Panda | cartesian_impedance | 6.5 | 6.5 | 6.0 |
| Panda | osc | 6.0 | 6.0 | 6.0 |
| UR5e | ik | 30.4 | 30.4 | 25.3 |
| UR5e | pd | 1.5 | 55.4 | 6.0 |
| UR5e | joint_impedance | 6.5 | 36.8 | 6.0 |
| UR5e | cartesian_impedance | 6.5 | 6.5 | 6.0 |
| UR5e | osc | 6.0 | 6.0 | 6.1 |

stop 模式的计划看到力到限值就停下来，所以会控制的方法峰值正好是 6 牛；ik 靠自己的舵机，下沉量没有重力补偿，停下来的位置已经压出 25 到 38 牛。
hold 模式里 joint_impedance 和 pd 的峰值是碰到的那一下：手臂的惯量撞在面上，软的方法也要等力传感器看到才反应。

```text
run_controller(robot="panda", task="wipe", method="osc", task_params={"force": 8})
compare_controllers(robot="panda", task="wipe", methods=["ik", "joint_impedance", "cartesian_impedance", "osc"])
run_controller(robot="panda", task="press", method="osc", task_params={"force": 6, "mode": "stop"})
```

在 Python 里，每一步的力曲线在任务的 `tally.trace`（时间，牛）里。

## 实测对比（M5 Max，MuJoCo 3.15，默认参数）

列的含义：reward 越高越好；success 是结束时是否达到目标；effort 是电机做的机械功（焦耳）；wall/sim 是算 1 秒仿真花几秒真实时间，小于 1 就比实时快。
每秒规划次数（plans/s）和电脑当时有多忙有关，同一台机器上空闲时能快一倍左右。

reach，目标 [0.25, 0.1, 0.15]，4 秒：

| method | reward | success | distance_m | effort (J) | wall/sim | notes |
|---|---|---|---|---|---|---|
| ik | -0.60 | no | 0.0186 | 0.33 | 0.01 |  |
| pd | -0.35 | yes | 0.0012 | 0.26 | 0.02 | goal reached at 0.73s |
| cartesian | -0.42 | yes | 0.0044 | 0.25 | 0.04 | goal reached at 1.33s |
| mpc | -0.40 | yes | 0.0014 | 1.97 | 0.11 | goal reached at 0.61s; 642 plans/s |
| mppi | -0.43 | yes | 0.0029 | 1.79 | 0.10 | goal reached at 0.64s; 662 plans/s |

balance，起始倾角随机，3 秒时推摆杆一下，6 秒：

| method | reward | success | pole_angle_deg | effort (J) | wall/sim | notes |
|---|---|---|---|---|---|---|
| lqr | 11.72 | yes | 0.34 | 1.03 | 0.01 |  |
| pd | 11.38 | yes | 0.45 | 0.71 | 0.01 |  |
| mpc | 11.67 | yes | -2.64 | 1.10 | 0.08 | 854 plans/s |
| mppi | 11.65 | yes | 2.64 | 1.29 | 0.08 | 885 plans/s |
| ars | 11.69 | yes | -1.19 | 1.32 | 0.01 | learned in 3.8s |
| none | -25.73 | no | 117.14 | 0.00 | 0.01 | failed at 0.84s |

stand，2 秒时侧向猛推（1.2 米每秒），5 秒：

| method | reward | success | height_m | effort (J) | wall/sim | notes |
|---|---|---|---|---|---|---|
| hold | -23.84 | no | 0.038 | 0.19 | 0.02 | failed at 2.36s |
| mpc | 9.45 | yes | 0.174 | 7.20 | 0.11 | 636 plans/s |
| mppi | 9.44 | yes | 0.177 | 6.78 | 0.11 | 608 plans/s |

walk，0.3 米每秒，2.5 秒时侧向猛推（1.5 米每秒），6 秒：

| method | reward | success | distance_m | effort (J) | wall/sim | notes |
|---|---|---|---|---|---|---|
| gait | -24.28 | no | 0.853 | 254.63 | 0.05 | failed at 2.84s |
| mpc | 9.47 | no | 1.323 | 160.22 | 0.38 | failed at 5.76s; 152 plans/s |
| mppi | 10.22 | yes | 1.358 | 150.87 | 0.38 | goal reached at 1.00s; 155 plans/s |

drive_to，目标 [-0.8, 0.3]，6 秒：

| method | reward | success | distance_m | effort (J) | wall/sim | notes |
|---|---|---|---|---|---|---|
| pd | -9.49 | yes | 0.024 | 3.51 | 0.01 | goal reached at 3.93s |
| mpc | -4.06 | yes | 0.003 | 4.55 | 0.10 | goal reached at 1.72s; 607 plans/s |
| mppi | -5.30 | yes | 0.005 | 4.19 | 0.11 | goal reached at 2.85s; 561 plans/s |

能读出来的规律：

- 经典控制几乎不花算力，在它擅长的场景里（平衡点附近、已知运动学）又准又省力。
- MPC 不用为每个任务专门设计控制律，换任务只换代价函数，代价是每秒几百到上千次物理模拟。
- 被猛推时，只会“保持姿态”的 hold 和只会“按节奏迈腿”的 gait 都会摔，会预测未来的 MPC 能救回来。
- MPC 的动作有采样噪声，所以 effort 往往比经典控制高。
- 强化学习（ars）不需要懂物理，靠试错学出来的线性策略和 LQR 的效果几乎一样，但要先花时间训练。

## 统一的环境接口（Gym 风格）

`RobotEnv` 完全按 Gymnasium 的约定写，但不依赖 gymnasium：

- `reset(seed=None, options=None)` 返回 `(obs, info)`，同一个 seed 得到完全相同的结果。
- `step(action)` 返回 `(obs, reward, terminated, truncated, info)`。
- `observation_space` 和 `action_space` 是带 `low`、`high`、`shape`、`dtype`、`sample()` 的 Box，dtype 都是 float32。
- 动作永远在 [-1, 1] 之间，每个维度对应任务控制的一个电机，自动换算到电机自己的范围（舵机角度、轮速或推力）。
- 每步奖励等于（存活奖励减去任务代价）乘以这一步的时长，所以一局的总回报和对比表里的 reward 是同一个量。
- 会失败的任务（balance、stand、walk）有存活奖励，避免学习算法发现“早点摔倒反而扣分少”。
- terminated 表示失败（摔倒、摆杆倒下、仿真数值发散），truncated 表示到了时间上限。
- `action_mode="residual"` 时，动作是叠加在经典控制器输出上的修正量（reach 叠在 ik 上，balance 叠在 lqr 上，walk 叠在 gait 上），学习可以从一个已经基本能用的策略出发。
- `render_mode="rgb_array"` 时 `render()` 返回画面。

`make_env("reach")` 会自动搭一个小世界；用 `make_env("reach", model=model, robot="arm")` 可以用任何编译好的世界。

接入强化学习库只需要一行：

```python
from robotforge.control.gym_compat import as_gymnasium

env = as_gymnasium(make_env("balance"))  # 得到真正的 gymnasium.Env
```

gymnasium 只在调用这一行时才导入，核心安装不受影响；没装时会提示 `pip install gymnasium`，或者直接装可选依赖组 `rl`。

## 神经网络强化学习：PPO 和 SAC（可选依赖组 rl）

ars 只能学“观测的加权和”这种线性策略，平衡倒立摆够用，走路和随机目标的 reach 就不够了。
PPO 和 SAC 学的是一个小神经网络，能表达复杂得多的动作规则。
它们来自 Stable-Baselines3，需要 PyTorch，所以放在可选依赖组 `rl` 里，核心安装不包含。

### 安装

- 在 robotforge 的仓库里：`uv sync --extra rl`。
- 用 pip 安装时：`pip install "robotforge[rl]"`。
- macOS（Apple 芯片）和 Windows 上，PyPI 默认的 PyTorch 就是 CPU 版：macOS 下载约 130 MB（torch 本身 127 MB），装好后约 580 MB（torch 占 526 MB），核心安装是 180 MB。
- Linux 上，PyPI 默认的 PyTorch 是带 CUDA 的版本：torch 本身 555 MB，还会拉进十几个 nvidia 包，合计好几个 GB。
- 只用 CPU 的 Linux 机器，先装 CPU 版再装 robotforge：`pip install torch --index-url https://download.pytorch.org/whl/cpu`，然后 `pip install "robotforge[rl]"`。
- 用 uv 的项目也一样：`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`，或者在 `pyproject.toml` 里用 `[tool.uv.sources]` 把 torch 指到这个 CPU 源。
- 没装 `rl` 时调用 `train_policy`，会直接告诉你上面的安装命令，不会出现看不懂的报错。

### 一句话怎么用

- “训练倒立摆保持平衡” 对应 `train_policy(task="balance", robot="pole")`，立刻返回策略名字和预计时间，训练在后台进行。
- “训练进度怎么样？” 对应 `training_status(name="pole_balance", wait_seconds=60)`，返回进度、平均奖励、成功率和剩余时间；等待期间会持续发送 MCP 进度通知。
- “先停下，用现在学到的” 对应 `stop_training(name)`，它会保留到目前为止最好的策略并做一次最终测试。
- “用训练好的策略跑一下” 对应 `run_controller(robot="pole", task="balance", method="policy:pole_balance", push_at=4)`，查看器打开时实时播放。
- “和 LQR、MPC 比一比” 对应 `compare_controllers(robot="pole", task="balance", methods=["lqr", "mpc", "policy:pole_balance"], push_at=3)`。
- “有哪些训练好的策略？” 对应 `list_policies()`。
- 脚本里用 `sim.run_policy("dog", "dog_walk")` 把机器人交给训练好的策略，直到 `sim.stop("dog")`。

训练在单独的进程里跑（`python -m robotforge.control.rl job <文件夹>`），MCP 服务和查看器不会卡，训练进程的输出只写进 `train.log`，不会混进 MCP 的通信。
训练进程的优先级调低了一档，查看器和 agent 照常流畅。
关掉 Claude 之后训练会继续跑完并保存，下次打开时 `list_policies` 能看到它。

### PPO 和 SAC 是什么（大白话）

PPO（近端策略优化）：同时开几个模拟世界，让机器人在里面反复尝试；每攒够一批经验，神经网络就朝“得分更高的动作”挪一小步，每次挪得不会太远，所以训练稳定。
它是最常用的入门算法，适合能并行开很多模拟的任务，比如平衡和走路。

SAC（软演员评论家）：只开一个模拟世界，但把试过的每一步都记下来，反复从记忆里学；另一个网络专门评估“在这个状态做这个动作值多少分”，策略照着它改进，同时保持一点好奇心（随机性），避免过早认定一种做法。
它更省样本，适合动作需要精细连续调整的任务，比如机械臂 reach，代价是每一步都要更新网络，墙钟时间更长。

两种算法学到的都是同一样东西：一个输入观测、输出电机指令的小网络（默认两层，每层 64 到 256 个神经元）。

### 直接模式和叠加模式

直接模式：网络的输出就是电机指令，从零学起，balance 和 reach 默认这样。
叠加模式：内置控制器照常工作，网络只学“在它的指令上加多少修正”，walk 叠在小跑步态上，stand 叠在保持姿态上。
叠加模式起点就能走，学得快，也不容易学出奇怪的动作；修正幅度默认是每个电机行程的 20%，机械臂是 5%。

### 训练时会推它

balance、stand、walk 训练时，机器人每隔 1.5 到 4 秒会被随机推一下（方向和力度随机），和对比测试里 `push_at` 用的是同一种推法。
只在平静环境里练过的策略，第一次被推就会倒；练的时候挨过推的策略学会了恢复。
推得越狠，策略越保守：walk 用最大 2.5 倍力度训练后，走得比步态本身还慢（见下表），所以 walk 默认只轻推（最大 1.2 倍）。
`train_policy(..., shoves=0)` 关掉推，`shoves=1.5` 这样的数值调力度。

### 训练出来的东西

每个策略是一个文件夹：打开世界时在 `<世界>/policies/<名字>/`，没有打开世界时在 `$ROBOTFORGE_HOME/policies/<名字>/`。

- `policy.json`：任务、机器人、算法、观测的含义、训练步数、最终测试成绩。
- `status.json`：训练进度，训练时每一两秒更新一次，包括学习曲线。
- `policy.npz`：训练好的网络，存成普通的 numpy 数组。
- `model.zip` 和 `vecnormalize.pkl`：Stable-Baselines3 的原始模型和观测归一化，想继续训练或者直接用 SB3 时用。
- `train.log`：训练进程打印的所有内容。

运行策略只用 `policy.npz`，几次矩阵乘法，不需要 PyTorch，所以 MCP 服务运行策略时不加载 PyTorch，只装了核心依赖的电脑也能运行别人训练好的策略。
测试保证导出的网络和 Stable-Baselines3 自己算出的动作一致，误差小于十万分之一。
训练时每隔八分之一的进度用固定的测试回合检查一次，保留成绩最好的那一版，所以训练后期变差也不会丢掉好策略。
机器人按世界里的描述（模板、参数、位置、地面）在训练进程里重新搭一份，单独训练；世界里的其他物体不参与训练。

### 在笔记本上要多久（实测）

测量环境：M5 Max（18 核），同时还有另外三个 agent 在跑仿真和测试，负载很高，所以下面的时间偏保守，空闲的电脑会更快。

| 任务 | 算法和模式 | 训练量 | 用时 | 结果 |
|---|---|---|---|---|
| balance | PPO，直接 | 15 万步 | 50 秒（通过 MCP 训练时 98 秒） | 平静 10/10 成功，被推 9/10 成功；3.7 万步时就已经学会 |
| walk，0.6 米每秒 | PPO，叠加在步态上，不推 | 100 万步 | 5.2 分钟 | 实际 0.61 米每秒，10/10 不摔；步态本身最快只到 0.34 米每秒 |
| walk，0.3 米每秒 | PPO，叠加，推力最大 2.5 | 60 万步 | 4.7 分钟 | 2.0 倍推力下 10 次活下来 2 次（步态 0 次）；2.5 倍都会倒；走得更慢（0.22 对 0.31 米每秒） |
| walk，0.3 米每秒 | PPO，叠加，推力最大 1.2（默认） | 60 万步 | 15 分钟（机器负载最高时） | 平静时任务得分 11.83，比步态的 11.63 高；速度 0.29 对 0.31 米每秒；2.0 倍推力下和步态一样会倒 |
| reach，随机目标 | SAC，直接 | 15 万步 | 10.7 分钟 | 最终平均误差 5.3 厘米，没有达到 1 厘米的成功线 |
| reach，随机目标 | SAC，直接 | 16.5 万步（中途停下） | 33.6 分钟（机器负载最高时） | 平均误差 4.5 厘米，还在慢慢变好，但离 2 厘米很远 |
| reach，随机目标 | PPO，直接 | 100 万步 | 3.4 分钟 | 平均误差 5.5 厘米 |

SAC 每走一步就更新一次网络，在负载高的机器上只有每秒 100 到 250 步，所以同样的步数比 PPO 慢得多。
PPO 用 8 个模拟进程并行，每个进程只加载 MuJoCo、numpy 和 gymnasium，不加载 PyTorch，单个进程大约 55 MB（加载 PyTorch 的话每个要 230 MB）；训练主进程 PPO 约 270 MB，SAC 带经验池约 380 MB。
训练时间很受电脑忙不忙影响：同样 60 万步的 walk 训练，负载一般时用了 4.7 分钟，CPU 被其他程序占满时用了 15 分钟。

### 和 MPC、LQR 对比（实测）

balance，3 秒时推摆杆一下，6 秒，同一个起点：

| method | reward | success | pole_angle_deg | effort (J) | wall/sim | notes |
|---|---|---|---|---|---|---|
| lqr | 11.72 | yes | 0.34 | 1.03 | 0.02 |  |
| pd | 11.38 | yes | 0.45 | 0.71 | 0.01 |  |
| mpc | 11.67 | yes | -2.64 | 1.10 | 0.09 | 727 plans/s |
| mppi | 11.65 | yes | 2.64 | 1.29 | 0.09 | 738 plans/s |
| ars | 11.69 | yes | -1.19 | 1.32 | 0.01 | learned in 5.1s |
| policy:pole_ppo | 11.54 | yes | 1.36 | 0.69 | 0.01 | learned in 49.6s |

换 10 个随机起点重复，五种方法都是 10/10 成功；PPO 用的力气最小（0.69 焦耳），运行时和 LQR 一样几乎不花算力，但要先花 50 秒训练。

walk，命令速度 0.6 米每秒，6 秒，10 个随机起点：

| method | success | fell | mean reward | mean avg_speed_m_s | wall/sim |
|---|---|---|---|---|---|
| gait | 8/10 | 0/10 | 9.11 | 0.341 | 0.04 |
| mppi | 0/10 | 7/10 | 4.54 | 0.279 | 0.43 |
| policy:dog_fast | 10/10 | 0/10 | 11.83 | 0.607 | 0.04 |

这是强化学习最能体现价值的地方：内置步态的步幅到上限后只能走 0.34 米每秒，叠加在步态上的 MPC 想走快反而会摔，PPO 花 5 分钟学会了调整步幅，稳稳走到 0.61 米每秒，运行时的算力和步态本身一样。

reach，随机目标，4 秒，10 个随机起点：

| method | success | mean reward | mean distance_m | wall/sim |
|---|---|---|---|---|
| ik | 1/10 | -5.96 | 0.059 | 0.05 |
| pd | 7/10 | -5.90 | 0.047 | 0.12 |
| cartesian | 10/10 | -4.77 | 0.007 | 0.15 |
| mpc | 9/10 | -3.07 | 0.033 | 0.27 |
| SAC 策略 | 0/10 | -3.84 | 0.056 | 0.06 |
| PPO 策略 | 0/10 | -4.45 | 0.055 | 0.07 |

这里强化学习输给了经典控制，原因值得讲给新手：

- 机械臂的运动学是精确已知的，操作空间控制不用训练就能做到 7 毫米。
- 策略要从零学会“任意目标对应的关节角”，还要自己学会抵消重力下垂；舵机角度范围是正负 3.14 弧度，网络输出差 1%，末端就差 1 到 2 厘米。
- 策略很快学会“先快速靠近”（所以 reward 比 ik、pd 高），但最后几厘米的精度需要多得多的训练；它的动作也更抖，effort 比经典控制大一百倍。
- 叠加在 ik 上训练也没有明显帮助：随机目标里有些点 ik 本身就解不好（最大误差 18 厘米），小幅修正救不回来。

结论：已知模型、要求精度的任务用经典控制；模型复杂、规则写不出来的任务（走得更快、被推后恢复）才是强化学习的强项。

### 在 Python 里

```python
from robotforge.control import compare_methods
from robotforge.control.rl import train

policy = train("balance", minutes=2)  # 在单独的进程里训练，打印进度，训练完返回
print(policy.meta["final"])  # 最终测试：平均回报、成功率

method = f"policy:{policy.name}"
where = {method: {"path": str(policy.folder)}}
table = compare_methods("balance", ["lqr", "mpc", method], seconds=6, push_at=3, method_params=where).table
print(table)
```

命令行也可以：`python -m robotforge.control.rl train walk --task-params '{"speed": 0.6}' --timesteps 1000000`。
`train` 在单独的进程里训练，所以普通脚本不用写 `if __name__ == "__main__":`。

## 怎么调参数

- MPC：`method_params={"samples": 64, "horizon": 0.6, "noise": 0.1}`。样本越多越准但越慢，预测越远越有远见但越不稳。
- LQR：`method_params={"q_weights": [5, 50, 0.5, 1], "r_weight": 0.5}`，四个数依次是小车位置、摆杆角度、小车速度、摆杆角速度有多重要。
- PD（机械臂）：`method_params={"omega": 20, "zeta": 1.0}`，omega 越大越硬越快，zeta 等于 1 时不超调。

## 目前的局限

- 机械臂的任务需要一个名为 tcp 的末端点，机械臂模板自带；导入的机械臂要先加一个。
- MPC 预测时会模拟整个世界，世界里机器人越多越慢；同时在一个世界里跑多个 MPC 会互相抢 CPU。
- LQR 目前只支持没有浮动基座的机器人（倒立摆这类），四足的 LQR 需要处理接触，留到后面。
- 纯采样 MPC 学不会走路，walk 默认叠加在内置步态上。
- 神经网络策略只能用在和训练时同一种机器人（电机名字和数量一致）、同一个任务上，换了机器人要重新训练。
- 训练时世界里只有这一个机器人和地面，桌子、障碍物这些物体不参与训练。
- walk 策略是在固定的命令速度上训练的，运行时换一个差很多的速度，效果没有保证。
- 同一时间只训练一个策略：每个训练已经会占满大部分 CPU 核。
- reach 的强化学习还达不到 2 厘米以内，原因见上面的对比。
- 小跑步态的腿部逆运动学是纯 Python 写的，占了走路训练 75% 的时间；换成解析解能让走路训练快好几倍。
