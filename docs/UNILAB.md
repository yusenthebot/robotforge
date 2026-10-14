# 学习型策略与 UniLab：让机器狗用神经网络走路

robotforge 自带的走路方式是手写的步态（gait）：脚按固定的弧线抬起落下。
强化学习（RL）走的是另一条路：让成千上万只仿真机器狗反复试错，学出一个神经网络，输入身体的感觉，输出每个关节该去的角度。
这一页讲三件事：怎么直接玩现成的学习型策略，怎么用 UniLab 训练新的策略，以及把别处训练的策略搬进 robotforge 时要注意什么。

## UniLab 是什么

[UniLab](https://github.com/unilabsim/UniLab) 是一个开源（Apache-2.0）的机器人强化学习框架，论文发表在 CoRL 2026。
它用 CPU 并行跑上千个 MuJoCo 仿真，用 GPU（CUDA、Apple 的 MPS）更新神经网络，在一台笔记本上几分钟就能学出一个会走的 Go2。
任务用 Hydra 配置文件描述（观测、动作、奖励、终止条件都是配置项），训练完可以导出 ONNX 文件，拿到别的程序里运行。

UniLab 有自己锁定的 torch 和 MuJoCo 版本（目前是 mujoco 3.11 加它自己编译的 mjbatch），robotforge 用的是 mujoco 3.15。
两者不能装在同一个 Python 环境里。
所以 UniLab 放在自己的文件夹和虚拟环境里，robotforge 只通过命令行子进程和文件跟它打交道，从不 import 它。

## 先玩现成的：机器狗 Go2 的学习型步态

robotforge 自带一个用 UniLab 训练好的 Go2 策略 `go2_unilab_walk`（PPO 算法，1000 轮迭代，2460 万步仿真，在 M5 Max 上训练 13.5 分钟）。
运行它只需要一个可选依赖 onnxruntime（约 76 MB），不需要安装 UniLab：

```bash
uv sync --extra policy          # 在 robotforge 仓库里
pip install "robotforge[policy]"  # 或者用 pip
```

在 Claude 或 GPT 里直接说：

- “有哪些学会走路的神经网络？” 对应工具 `list_policies`，列出策略、适用的机器人和它学过的速度范围。
- “给我一台 Go2，用强化学习的步态往前走” 对应 `add_robot(name="go2", library="go2")`，然后 `move(robot="go2", speed=0.5, walker="rl")`。
- “换回原来的步态” 对应 `move(robot="go2", speed=0.3, walker="gait")`。

`walker` 选好以后会一直保留，查看器里用 WASD 键盘驾驶时也用同一个。
查看器顶部的驾驶栏里有一个 “gait / learned” 下拉框，只在机器人有学习型策略时出现，切到 learned 就换成神经网络走路。
没装 onnxruntime 时选 learned 会弹出提示，告诉你要装哪个依赖，机器人继续用原来的步态。

在 Python 里：

```python
from robotforge.bridge.onnx_policy import walker_for

walker = walker_for(session.model, "go2", "go2", "rl")  # 机器人名、库里的型号、策略
session.set_controller("go2", walker)
walker.command(0.5, 0.0)  # 前进 0.5 m/s，不转弯
```

这个策略学过的指令范围是：前进 -0.6 到 1.0 m/s，转弯 -0.8 到 0.8 rad/s，超出的指令会被裁到这个范围里。
停下（`stop` 或松开按键）时策略继续运行，指令为零，机器狗原地站稳；摔倒以后策略会暂停，把它拖起来或者重置世界就能继续。

## 平台支持

| 做什么 | macOS（Apple Silicon） | Linux | Windows |
|---|---|---|---|
| 在 robotforge 里运行学习型策略 | 支持，本机实测 | 支持（onnxruntime 有官方安装包） | 支持（onnxruntime 有官方安装包），尚未在 Windows 上实测 |
| 用 UniLab 训练 | 支持，本机实测（M5 Max，MuJoCo 后端，学习器在 MPS 上） | UniLab 官方支持 x86_64 和 aarch64，有 NVIDIA 显卡时学习器用 CUDA；robotforge 的桥接按同样的方式工作，尚未实测 | UniLab 的 MuJoCo 后端依赖的 mjbatch 没有 Windows 安装包，训练请在 WSL2 的 Linux 里进行 |

UniLab 的文档里写了 Windows 的安装方式，但它的 MuJoCo 训练后端在 Windows 上不可用，这一点以 UniLab 的安装文档为准。
robotforge 一侧的桥接代码在 Windows 上会找 `.venv\Scripts\python.exe`，用独立的进程组启动训练，取消时用 `taskkill` 结束整棵进程树，但这些路径还没有在 Windows 上跑过。

## 安装 UniLab（用来训练新策略）

需要 [uv](https://docs.astral.sh/uv/) 和一个 C++ 编译环境（macOS 上运行 `xcode-select --install`）。
在 robotforge 之外的任意文件夹里：

```bash
git clone https://github.com/unilabsim/UniLab.git
cd UniLab
uv sync --extra mujoco --extra uni_rl
export ROBOTFORGE_UNILAB_DIR="$PWD"   # 写进 shell 配置文件，以后一直有效
```

本机实测：克隆 37 MB，`uv sync` 用时 81 秒（uv 缓存里已经有 torch 时），虚拟环境 1.3 GB，第一次训练会从 Hugging Face 下载 Go2 的网格文件（UniLab 的资源目录约 85 MB，含 G1）。
Hugging Face 访问慢时，UniLab 的文档建议设置 `HF_ENDPOINT=https://hf-mirror.com`。

检查是否装好：

```bash
robotforge unilab check
```

它会用 UniLab 自己的 Python 报告 UniLab、torch、MuJoCo 的版本和可用的计算设备（本机是 cpu 和 mps），用时约 3 秒。
对应的 MCP 工具是 `unilab_check`。

## 训练一个新策略

```bash
robotforge unilab train --iterations 1000 --name go2_walk_mine --wait
```

或者在 Claude 里说 “用 UniLab 训练一个 Go2 走路的策略”，对应 `unilab_train(iterations=1000, name="go2_walk_mine")`，再用 `unilab_job` 查看进度。

训练在后台的独立进程里进行，robotforge 和 MCP 会话可以照常使用，查看器也不会卡。
整个流程是：UniLab 训练，UniLab 自带的导出器生成 ONNX，`unilab_tool.py` 从一个真实的 UniLab 环境里读出观测和动作的约定写成策略卡，最后安装到 `~/robotforge/policies/<名字>/`。
进度写在 `~/robotforge/unilab/jobs/<任务号>/status.json` 里，训练日志在同一个文件夹的 `train.log`，都不会写到 MCP 的标准输出。

本机实测（M5 Max，1024 个并行环境）：

| 迭代次数 | 训练 | 导出和安装 | 效果 |
|---|---|---|---|
| 5 | 17 秒 | 20 秒 | 只验证流程，机器狗几乎不动（5 秒走 0.01 米） |
| 20 | 34 秒 | 25 秒 | 只验证流程，还不会走 |
| 151（UniLab 的默认值） | 2 分钟 | 约 20 秒 | 能往前走，转弯很弱（指令 0.8 rad/s 只转出 0.17） |
| 1000 | 13.5 分钟 | 约 20 秒 | 前进、后退、转弯都跟得上，就是自带的 go2_unilab_walk |

每一轮迭代让 1024 只机器狗各走 24 步（0.48 秒），一共约 2.5 万步仿真，再用这些经验更新一次网络。
导出的 ONNX 文件约 0.78 MB。

已经用 UniLab 训练过的结果，也可以直接导出安装：

```bash
robotforge unilab export <UniLab 的运行目录> --name my_go2
```

运行目录是 UniLab `logs/` 下面带 `run_config.json` 的那一层。

## 策略卡 policy.json：为什么要把约定写下来

神经网络只认数字的位置。
训练时第 7 个输入是左前腿髋关节的角度，运行时也必须是它，换成别的关节，策略就会乱动。
所以每个策略旁边有一张卡片 `policy.json`，写清楚这些约定：

- `joints`：动作的第 i 个数控制哪个关节，关节类的观测也按这个顺序排列。
- `default_joint_pos`、`action_scale`：目标角度等于默认姿态加上网络输出乘以缩放系数。
- `pd`：训练时关节的 PD 参数（刚度 kp、阻尼 kd、关节本身的被动阻尼、力矩上限）。
- `observations`：观测由哪些项按什么顺序拼起来，比如角速度、重力方向、关节角、上一步动作、速度指令、步态时钟。
- `policy_dt`：多久跑一次网络（Go2 是 0.02 秒，即 50 Hz）。
- `commands`：训练时见过的速度指令范围。

robotforge 按名字找关节，不按顺序：UniLab 的 Go2 执行器顺序是 FR、FL、RR、RL，动作顺序却是关节顺序 FL、FR、RL、RR，这两者容易弄混。
如果机器人的关节名字不一样，可以传一个改名表 `joint_map`；找不到的关节、没有执行器的关节、默认姿态超出关节限位，都会直接报错并说明原因，不会悄悄地错位。

卡片不是手写的：`unilab_tool.py export` 在 UniLab 自己的环境里搭起一个真实的训练环境，从动作管理器、观测管理器、事件配置和 MuJoCo 模型里读出这些数值。
测试 `tests/test_onnx_policy.py` 里有一段在 UniLab 里录下的 40 步轨迹，robotforge 用同样的状态重建观测，关节、动作、指令和步态时钟逐位一致。

## 从 UniLab 到 robotforge：仿真到仿真的差距

同一个策略换一个仿真器，常常就走不动了，这叫 sim-to-sim 差距。
UniLab 训练用的 Go2 模型和 robotforge 库里的 Go2（来自 MuJoCo Menagerie）有这些不同：

| | UniLab 训练时 | robotforge 库里的 Go2 |
|---|---|---|
| 执行器 | 位置伺服，kp 35，kd 0.5（每次重置在 31.5 到 38.5、0.45 到 0.55 之间随机） | 位置伺服，kp 100，无 kd |
| 关节被动阻尼 | 0.5 | 2.0 |
| 足底摩擦 | 0.4 | 0.8 |
| 仿真步长 | 0.01 秒，Euler 积分 | 0.002 秒，implicitfast 积分 |
| MuJoCo 版本 | 3.11（mjbatch） | 3.15 |
| 重力观测 | IMU 的 z 轴在世界坐标里的方向取反（framezaxis 传感器） | 由 robotforge 按同样定义计算 |

robotforge 的做法：策略每 0.02 秒给出目标角度，之间每个物理步都按训练时的 kp、kd 算出关节力矩，再换算成 robotforge 自己执行器能接受的控制量；关节阻尼的差值也在这一步补上。
这样不用修改机器人本身，内置步态照样可以用原来的伺服参数。

实测结果（自带策略，每项 10 秒，从第 2 秒起取平均速度）：

| 指令 | robotforge 库里的 Go2 | UniLab 自己的仿真 |
|---|---|---|
| 前进 0.5 m/s | 0.49 m/s，10 秒走 4.8 米 | 0.48 m/s |
| 前进 1.0 m/s | 0.97 m/s | 0.98 m/s |
| 后退 0.4 m/s | -0.41 m/s | -0.44 m/s |
| 左转 0.8 rad/s | 0.68 rad/s | 0.65 rad/s |
| 右转 0.8 rad/s | -0.77 rad/s | -0.79 rad/s |
| 原地站立 | 10 秒漂移 0.11 米 | 0.12 米 |

全部测试里机身始终直立。
在查看器里用键盘驾驶（按住 W 10 秒，指令 0.6 m/s）走了 5.84 米，按住 A 2 秒转了约 62 度。

逐项排除的实验（前进 0.5 m/s 和左转 0.8 rad/s 两项，每次只改一处）：

| 改动 | 前进 | 左转 | 结论 |
|---|---|---|---|
| 完整做法 | 0.49 | 0.68 | 基准 |
| 直接把目标角度交给 robotforge 的 kp 100 伺服 | -0.14，还在原地打转 | -0.14 | PD 参数不同是最大的差距，必须按训练时的 PD 回放 |
| 不补关节阻尼的差值 | 0.43 | 0.53 | 阻尼多了 4 倍，速度和转弯都打折扣 |
| 步长改成 0.01 或 0.005 秒 | 0.50 | 0.68 | 步长影响可以忽略 |
| 重力观测改用机身坐标系 | 0.49 | 0.67 | 平地上影响可以忽略，但仍按训练时的定义计算 |
| 关节顺序错成执行器顺序 | 立刻翻倒 | 立刻翻倒 | 关节映射必须按名字、显式校验 |

还有一处已知的细小差别：UniLab 的角速度和重力来自 MuJoCo 传感器，读到的是上一个物理步（10 毫秒前）的值。
robotforge 读到的是 2 毫秒前的值，实测这点差别对走路没有可见影响。

## 限制和下一步

- 目前能在 robotforge 里训练的 UniLab 任务是 `go2_joystick_flat`；G1 人形机器人的 `g1_walk_flat` 在 UniLab 里默认要训练 2200 轮（本机约 2 小时）。
- 策略只在平地上训练过，robotforge 的草地、坡道和台阶对它是没见过的情况。
- 用 robotforge 自己设计的机器人去 UniLab 训练，需要把它导出成 UniLab 的任务配置（MJCF、关节名单、传感器和奖励），这一步还没有做。
