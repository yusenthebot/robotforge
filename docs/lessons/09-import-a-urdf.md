# 第 9 课 带上你自己的机器人：导入 URDF，读懂体检报告

目标：导入一个 URDF 机器人（你自己的，或者下面这个小例子），读懂导入体检报告的每一行，再让导入的关节动起来。

## 你可以这样说

> 我有一个 URDF 机器人文件，帮我导入进来，告诉我体检报告说了什么，再让它动一动。

## 你会看到

agent 新建一个世界，把你的 URDF 加进来（`add_robot(file=...)`）。
导入时会自动做一次体检，结果里的 `doctor` 一栏写着修好了什么、修不好什么。
用下面的小例子，作者电脑上的体检报告是：

```
Import doctor: The doctor fixed 4 problems.
  [fixed] some links had no mass or inertia
  [note] the robot is bolted down (fixed base) ... like a robot arm on a table
  [fixed] part of the robot is 2.0 cm below its base origin, which would sink into the floor
  [fixed] the URDF describes joints but no motors, so sliders and controls would do nothing
  [fixed] 1 pair(s) of links overlap in their collision shapes (common in URDFs), which makes a robot jitter
  [note] hold test passed: with every motor at its starting pose the robot stays put for 3 s
```

然后 agent 把肩关节转到 0.6 弧度，导入的机械臂就动起来了。

## 一个可以直接用的小例子

没有自己的 URDF 时，把下面的内容存成 `two_link.urdf`，再告诉 agent 文件路径。
在 Claude Code 或 Codex 里，agent 也可以替你写这个文件。

```xml
<?xml version="1.0"?>
<!-- A tiny two-joint arm: a base, an upper arm and a forearm. -->
<robot name="two_link">
  <link name="base">
    <visual><geometry><cylinder radius="0.05" length="0.04"/></geometry></visual>
    <collision><geometry><cylinder radius="0.05" length="0.04"/></geometry></collision>
    <inertial><mass value="0.5"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="upper_arm">
    <visual><origin xyz="0 0 0.1"/><geometry><box size="0.03 0.03 0.2"/></geometry></visual>
    <collision><origin xyz="0 0 0.1"/><geometry><box size="0.03 0.03 0.2"/></geometry></collision>
  </link>
  <link name="forearm">
    <visual><origin xyz="0 0 0.08"/><geometry><box size="0.025 0.025 0.16"/></geometry></visual>
    <collision><origin xyz="0 0 0.08"/><geometry><box size="0.025 0.025 0.16"/></geometry></collision>
    <inertial><origin xyz="0 0 0.08"/><mass value="0.1"/><inertia ixx="0.0002" iyy="0.0002" izz="0.00002" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="upper_arm"/><origin xyz="0 0 0.02"/><axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="2" velocity="3"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="upper_arm"/><child link="forearm"/><origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="1" velocity="3"/>
  </joint>
</robot>
```

注意 `upper_arm` 故意没写 `<inertial>`，整个文件也没写电机，体检会把这两处补上。

## 背后的道理

URDF 是描述机器人最常见的文件格式：link 是刚体零件（形状和质量），joint 把两个 link 连起来（转轴和转动范围）。
复杂的外形放在单独的网格文件里（.stl、.dae、.obj），URDF 用路径指向它们，经常写成 `package://` 开头。
网上的 URDF 大多是给可视化或 ROS 写的，拿来仿真常常缺东西：没有质量和惯量、没有电机、网格路径只在作者电脑上有效。
导入体检会修好能修的，修不好的写成 NOT FIXED 并说明会怎么处理。
机器人要么固定在地上（固定基座，像桌上的机械臂），要么自由活动（浮动基座，像机器狗）。
体检会自己猜，猜错了可以用 `params={"base": "fixed"}` 或 `{"base": "floating"}` 重新导入。
如果文件里写的扭矩太小、撑不起机器人，体检会适当放宽并报告；想严格按文件来，用 `{"respect_effort_limits": true}`。
除了 URDF，也能导入 xacro、MJCF，和单个网格文件（比如一个 .stl 零件）。

## 自己动手

用 `{"base": "floating"}` 重新导入这条机械臂，看看它会怎样。
从 GitHub 下载一个带网格文件夹的 URDF 机器人，导入后和这份体检报告比一比。

## 怎么算成功

`add_robot` 成功，体检报告里 `unfixed` 是 0；如果不是 0，agent 把每一条都给你讲清楚。
关节转到命令的角度附近（误差不超过 0.1 弧度），截图里机器人动了。

## 挑战

找一个真实的开源机器人 URDF（比如一条开源机械臂），导入它，用滑条把每个关节都试一遍。

上一课：[第 8 课](08-train-a-policy.md)　回到 [学习路线](README.md)
