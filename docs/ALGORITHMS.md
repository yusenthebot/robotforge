# 算法适配层：同一个机器人，换算法对比

robotforge 把“让机器人做到一件事”拆成两半：任务（要做到什么）和算法（怎么做到）。
同一个任务可以交给不同的算法，跑完给出同一套指标，新手可以直接看到每种方法的长处和代价。
代码在 `src/robotforge/control/`，不需要额外安装任何依赖。

## 一句话怎么用

在 Claude 或 GPT 里直接说：

- “有哪些控制算法？” 对应工具 `list_algorithms`，列出任务、适用的算法和一句话解释。
- “用 LQR 让倒立摆保持平衡，2 秒时推它一下” 对应 `run_controller(robot="pole", task="balance", method="lqr", push_at=2)`。
- “用 MPC 让机械臂把末端移到方块那里” 对应 `run_controller(robot="arm", task="reach", method="mpc", task_params={"target": "cube"})`。
- “对比几种算法谁更好” 对应 `compare_controllers(robot="arm", task="reach")`，返回一张对比表。

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
| reach | 把机械臂末端（tcp 点）移到目标点并停住，误差小于 1 厘米算成功 | 机械臂模板 | ik, pd, cartesian, mpc, mppi |
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
目前只对 balance 开放；更难的任务（随机目标的 reach、走路）需要神经网络策略，也就是可选的 `rl` 依赖组里的 PPO、SAC。
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

## 实测对比（M5 Max，MuJoCo 3.15，默认参数）

列的含义：reward 越高越好；success 是结束时是否达到目标；effort 是电机做的机械功（焦耳）；wall/sim 是算 1 秒仿真花几秒真实时间，小于 1 就比实时快。

reach，目标 [0.25, 0.1, 0.15]，4 秒：

| method | reward | success | distance_m | effort (J) | wall/sim | notes |
|---|---|---|---|---|---|---|
| ik | -0.60 | no | 0.0186 | 0.33 | 0.01 | |
| pd | -0.35 | yes | 0.0012 | 0.26 | 0.02 | goal reached at 0.73s |
| cartesian | -0.42 | yes | 0.0044 | 0.25 | 0.03 | goal reached at 1.33s |
| mpc | -0.38 | yes | 0.0019 | 1.71 | 0.06 | goal reached at 0.65s; 1158 plans/s |
| mppi | -0.39 | yes | 0.003 | 1.70 | 0.07 | goal reached at 0.61s; 969 plans/s |

balance，起始倾角随机，3 秒时推摆杆一下，6 秒：

| method | reward | success | pole_angle_deg | effort (J) | wall/sim | notes |
|---|---|---|---|---|---|---|
| lqr | 11.72 | yes | 0.34 | 1.03 | 0.01 | |
| pd | 11.38 | yes | 0.45 | 0.71 | 0.01 | |
| mpc | 11.67 | yes | -2.64 | 1.10 | 0.04 | 1756 plans/s |
| mppi | 11.65 | yes | 2.64 | 1.29 | 0.04 | 1603 plans/s |
| ars | 11.69 | yes | -1.19 | 1.32 | 0.01 | learned in 3.2s |
| none | -25.73 | no | 117.14 | 0.00 | 0.01 | failed at 0.84s |

stand，2 秒时侧向猛推（1.2 米每秒），5 秒：

| method | reward | success | height_m | effort (J) | wall/sim | notes |
|---|---|---|---|---|---|---|
| hold | -23.84 | no | 0.038 | 0.19 | 0.02 | failed at 2.36s |
| mpc | 9.45 | yes | 0.174 | 7.20 | 0.12 | 543 plans/s |
| mppi | 9.44 | yes | 0.177 | 6.78 | 0.13 | 495 plans/s |

walk，0.3 米每秒，2.5 秒时侧向猛推（1.5 米每秒），6 秒：

| method | reward | success | distance_m | effort (J) | wall/sim | notes |
|---|---|---|---|---|---|---|
| gait | -24.28 | no | 0.853 | 254.63 | 0.04 | failed at 2.84s |
| mpc | 9.47 | no | 1.323 | 160.22 | 0.35 | failed at 5.76s; 165 plans/s |
| mppi | 10.22 | yes | 1.358 | 150.87 | 0.35 | 165 plans/s |

drive_to，目标 [-0.8, 0.3]，6 秒：

| method | reward | success | distance_m | effort (J) | wall/sim | notes |
|---|---|---|---|---|---|---|
| pd | -9.49 | yes | 0.024 | 3.51 | 0.01 | goal reached at 3.93s |
| mpc | -4.06 | yes | 0.003 | 4.55 | 0.08 | goal reached at 1.72s; 837 plans/s |
| mppi | -5.30 | yes | 0.005 | 4.19 | 0.08 | goal reached at 2.85s; 803 plans/s |

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

gymnasium 只在调用这一行时才导入，核心安装不受影响；没装时会提示安装可选的 `rl` 依赖组。

## 怎么调参数

- MPC：`method_params={"samples": 64, "horizon": 0.6, "noise": 0.1}`。样本越多越准但越慢，预测越远越有远见但越不稳。
- LQR：`method_params={"q_weights": [5, 50, 0.5, 1], "r_weight": 0.5}`，四个数依次是小车位置、摆杆角度、小车速度、摆杆角速度有多重要。
- PD（机械臂）：`method_params={"omega": 20, "zeta": 1.0}`，omega 越大越硬越快，zeta 等于 1 时不超调。

## 目前的局限

- 机械臂的任务需要一个名为 tcp 的末端点，机械臂模板自带；导入的机械臂要先加一个。
- MPC 预测时会模拟整个世界，世界里机器人越多越慢；同时在一个世界里跑多个 MPC 会互相抢 CPU。
- LQR 目前只支持没有浮动基座的机器人（倒立摆这类），四足的 LQR 需要处理接触，留到后面。
- 纯采样 MPC 学不会走路，walk 默认叠加在内置步态上。
- 神经网络强化学习（PPO、SAC）还没接入，环境接口已经按 Stable-Baselines3 的要求准备好；目前能用的是只依赖 numpy 的 ars。
