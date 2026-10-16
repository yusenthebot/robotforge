# 零件拼装：用零件造任何机器人

模板只是起点。
用 `build` 可以像搭乐高一样，从一个底座和一串零件造出任何机器人，也可以把模板机器人当成一个零件装上去。
agent 先调用 `builder_guide` 读速查表，再用 `add_robot(name, build={...})` 创建，之后用 `add_part`、`update_part`、`remove_part` 一块一块地改。

## 基本概念

- 单位：米、千克、角度（度）。
- 方向：机器人的前方是 +x，左边是 +y，上方是 +z。
- 每个零件都有自己的坐标系，原点在它的关节上；零件里写的数字都在父零件的坐标系里。
- 零件的名字同时是它的刚体、关节和执行器的名字，在世界里会加上机器人名前缀，例如 `dog/left_thigh_1`。

## 结构

```yaml
robots:
  - name: dog
    build:
      color: orange          # 主色（底座），零件默认白色
      material: pla          # pla、aluminum、carbon、rubber，决定密度和质量
      servo: sts3215         # 铰链默认用的舵机
      motor: tt_motor        # 轮子和连续旋转关节默认用的电机
      linear: l16            # 滑动关节默认用的直线执行器
      base: {shape: box, size: [0.3, 0.14, 0.06], fixed: false, sensors: [imu]}
      parts: [...]
```

底座的 `fixed: true` 表示固定在地面上（机械臂、转台），默认是能自由移动的。
颜色（机器人、底座、零件都有 `color`）可以写颜色名（orange、brown 等）、十六进制 `"#9c6b3f"` 或 `"#963"`，或者 0 到 1 的 `[r, g, b]`；在 world.yaml 里十六进制要加引号，不然 `#` 会被当成注释。

## 零件字段

| 字段 | 含义 |
|---|---|
| `name` | 小写字母、数字和下划线 |
| `parent` | 挂在哪个零件上，默认 `base` |
| `at` | 挂载点：`[x, y, z]`，或锚点 `top`、`bottom`、`front`、`back`、`left`、`right`、`center`、`start`、`end`，或父零件的 `mounts` 名字 |
| `shift` | 在 `at` 之上再平移 |
| `rotate` | `[横滚, 俯仰, 偏航]`，单位度 |
| `joint` | `hinge`（默认）、`slide`、`fixed`，或 `{type, axis, range, home, continuous, damping, stiffness}` |
| `actuator` | 零件库里的型号，`none` 表示无动力，或 `{model, control, gear}` |
| `shape` | `box`、`sphere`、`capsule`、`cylinder`、`wheel`、`caster`、`none` |
| `size` | box 是 `[长, 宽, 高]`，sphere 和 caster 是 `[直径]`，capsule 和 cylinder 是 `[直径, 长度]`，wheel 是 `[直径, 宽度]` |
| `direction` | 连杆从关节朝哪个方向长出去（腿用 `down`）；不写就以关节为中心 |
| `foot` | `true` 在末端加一个橡胶脚 |
| `sensors` | `imu`、`touch`、`camera`、`rangefinder`、`lidar`，或带参数的写法 |
| `mirror` | `true` 在另一侧加一个镜像副本，名字里的 left 和 right 互换 |
| `repeat` | `{count, around}` 环形排列，`{count, along, spacing}` 一排，`{count, chain: true}` 一节挂一节 |
| `robot` | 把整个模板或机器人库里的机器人装在这个零件上，例如 `{template: arm}` |

`end` 锚点是连杆的末端，所以腿、机械臂、蛇身都可以写成“下一节挂在上一节的 end 上”。
质量由形状体积和材料密度算出，舵机和电机的质量加在关节处。
外观是自动加上的，不影响物理：方块是带分模线和螺丝的外壳，会摆动的胶囊画成连杆，不摆动的胶囊（比如胶囊形的机身）照原样画成光滑的胶囊，每个有动力的铰链画出舵机壳和随零件转动的舵盘，轮子背后有减速电机，脚是橡胶球，相机、激光雷达、IMU 和测距仪都画成能认出来的模块。
写 `servo_caps: false` 时关节上不画舵机和电机，它们的质量照算。
装在某个面上的传感器模块贴在那个面上。
装在顶面上、但不朝顶面外看的相机（比如装在 `top`、朝前看）会站到面上，相机点比锚点高出约 1 厘米，所以画面里看得到机器人自己的顶面，就像真车顶上的相机。
装在底面或侧面上的相机留在锚点上，模块一半嵌进面里，画面和没有外观时基本一样。
相机默认装在零件中心。
相机在零件里面或表面上时，`observe` 从这个相机看出去的画面里，它所在的零件按物理形状画、不画外观，就像测距仪的射线也不碰外观一样，所以画面里不会出现零件内部、关节罩或挤在一起的其他模块。
零件保持自己的颜色，`material` 决定表面：`pla` 是亮面塑料，`aluminum` 是阳极氧化铝，`carbon` 是带编织纹的碳纤维，`rubber` 是哑光橡胶；没写颜色的铝件和碳纤维件显示材料本色。

![用零件拼出来的机器狗、带激光雷达和云台相机的铝合金小车、碳纤维六足和背着机械臂的四足](images/built_robots.jpg)

![同一个小机器人的四种材料：PLA、阳极氧化铝、碳纤维和橡胶](images/built_materials.jpg)

## 自动检查

每次创建或修改都会先把机器人放到地面上试一秒，再决定是否保存：

- 零件互相穿插、或者有零件插进地面，会被拒绝，并告诉你是哪两个零件、穿进去多少毫米、怎么改。
- 关节范围写反、初始角度超出范围、尺寸个数不对、父零件写错（会提示最接近的名字），都会直接报错。
- 舵机光是撑住初始姿态就用掉全部力矩时，会在 notes 里提示换更强的舵机或加减速比 `gear`。
- 出生高度自动计算，脚或轮子刚好落在地面上。

## 造好之后能做什么

造出来的机器人和模板机器人一样，有滑条、`set_controls`、脚本和算法层，并且按形状自动获得内置行为：

- 4 条或 6 条腿、每条腿 3 个铰链：`move` 和 WASD 行走（四足小跑步态，六足三角步态）。
- 两侧都有速度控制的轮子：像小车一样开。
- 只有一根轴上的两个轮子着地：自动用 LQR 保持平衡，并且能开；`run_controller(task="balance")` 可以对比 LQR、MPC 和不控制。
- 4 个以上左右摆动的关节串成一条链：蛇形蜿蜒前进。
- 装了机械臂或夹爪：`move_arm`、`pick_and_place`，机械臂的名字写 `rover` 或 `rover/arm` 都行，底盘开到别处后也能用。

## 例子 1：六足机器人

```json
{
  "color": "teal",
  "base": {"shape": "cylinder", "size": [0.26, 0.05], "sensors": ["imu"]},
  "parts": [
    {"name": "coxa", "at": [0.1126, 0.065, 0], "rotate": [0, 0, 30],
     "joint": {"axis": "z", "range": [-35, 35]}, "shape": "box", "size": [0.05, 0.03, 0.025],
     "direction": "forward", "repeat": {"count": 6, "around": "z"}},
    {"name": "femur", "parent": "coxa", "at": "end", "joint": {"range": [-80, 80], "home": -20},
     "size": [0.02, 0.08], "direction": "forward"},
    {"name": "tibia", "parent": "femur", "at": "end", "joint": {"range": [0, 150], "home": 110},
     "size": [0.016, 0.12], "direction": "forward", "foot": true}
  ]
}
```

一条腿写一次，`repeat` 绕 z 轴每隔 60 度复制一条，得到 `coxa_1` 到 `coxa_6`。

## 例子 2：移动机械臂（小车加机械臂模板）

```json
{
  "color": "yellow",
  "motor": "jga25",
  "base": {"shape": "box", "size": [0.34, 0.22, 0.06], "mass": 3.0, "sensors": ["imu", {"type": "camera", "at": "front"}]},
  "parts": [
    {"name": "left_wheel", "at": [0, 0.13, -0.01], "shape": "wheel", "size": [0.1, 0.03], "mirror": true},
    {"name": "front_caster", "at": [0.14, 0, -0.04], "shape": "caster", "size": [0.04]},
    {"name": "rear_caster", "at": [-0.14, 0, -0.04], "shape": "caster", "size": [0.04]},
    {"name": "arm", "at": "top", "shift": [0.06, 0, 0],
     "robot": {"template": "arm", "params": {"reach": 0.45, "color": "yellow"}}}
  ]
}
```

先 `move(robot="mm", speed=0.3, seconds=3)` 开过去，再 `pick_and_place(robot="mm", object="cube", target="bin")`。

## 例子 3：蛇形机器人

```json
{
  "color": "green",
  "servo": "mg996r",
  "base": {"shape": "box", "size": [0.09, 0.06, 0.045], "sensors": ["imu", {"type": "camera", "at": "front"}]},
  "parts": [
    {"name": "segment", "at": "back", "joint": {"axis": "z", "range": [-70, 70]},
     "shape": "box", "size": [0.08, 0.05, 0.04], "direction": "back", "repeat": {"count": 8, "chain": true}},
    {"name": "left_wheel", "parent": "segment", "at": [-0.04, 0.031, -0.004],
     "shape": "wheel", "size": [0.036, 0.008], "actuator": "none", "mirror": true}
  ]
}
```

`chain` 让每一节挂在上一节的 `back` 上。
每节两侧的被动小轮让身体容易前后滑、不容易横着滑，这正是蛇形运动需要的。

## 例子 4：两轮自平衡机器人

```json
{
  "color": "red",
  "motor": "jga25",
  "base": {"shape": "box", "size": [0.07, 0.14, 0.22], "sensors": ["imu"]},
  "parts": [
    {"name": "left_wheel", "at": [0, 0.095, -0.11], "shape": "wheel", "size": [0.12, 0.025], "mirror": true},
    {"name": "head", "at": "top", "joint": "fixed", "shape": "sphere", "size": [0.07], "direction": "up",
     "color": "white", "sensors": [{"type": "camera", "at": "front"}]}
  ]
}
```

它一出现就开始平衡。
`compare_controllers(robot="seg", task="balance", push_at=2)` 会给出 LQR、MPC、MPPI 和不控制的对比表。

## 例子 5：背着机械臂的四足

把 `builder_guide` 里的四足例子拿来，再加一个零件：

```json
{"name": "arm", "at": "top", "shift": [0.06, 0, 0],
 "robot": {"template": "arm", "params": {"reach": 0.4, "color": "purple"}}}
```

腿的识别只看“3 个铰链、脚着地”的链，机械臂有 5 个关节，不会被当成腿，所以它照样能走，到了地方再用 `move_arm(robot="spot", ...)`。

## 自己做机械臂

底座设 `fixed: true`，用几个铰链零件串成手臂，在最后一节的 `end` 上装 `{"robot": {"template": "gripper"}}`。
夹爪的手指朝它所在零件坐标系的 +z 方向，`move_arm` 和 `pick_and_place` 都能直接用。
