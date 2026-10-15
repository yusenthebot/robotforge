"""Lesson: how a robot senses the world.

A small rover in a walled arena carries the sensors real robots use: wheel encoders (how far the wheels
turned), an IMU (acceleration and turning speed), a ring of laser range finders (a lidar), a bumper switch
and a camera. The learner drives it and watches each sensor answer a different question.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import mujoco
import numpy as np

from robotforge.learn.engine import Lesson, LessonContext, Plot, Quiz, Series, Step

WHEEL_R = 0.035
TRACK = 0.18  # distance between the wheels, m
MAX_SPEED = 0.45  # m/s
MAX_TURN = 2.2  # rad/s
RAYS = 24
LIDAR_Z = 0.12  # height of the lidar head above the chassis origin, m
ARENA = 1.5  # half the arena's side, m
BALL = (-0.95, 0.95)
CRATE = (0.85, 0.55)
SAMPLE_DT = 0.02
ACCEL_TAU = 0.12  # s, low-pass filter on the accelerometer
RANGE_DIST = 1 << int(mujoco.mjtRayDataField.mjRAYDATA_DIST)


def _quat_z_out(angle: float) -> list[float]:
    """Orientation whose z axis points horizontally at `angle` (a rangefinder measures along its site's z)."""
    qy = np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])  # z -> x
    qz = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qz, qy)
    return [float(x) for x in out]


def _camera_quat() -> list[float]:
    """A camera looking along the rover's +x with +z up (cameras look down their own -z)."""
    mat = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)  # columns: camera x, y, z in the body
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    return [float(x) for x in q]


class Rover:
    """Differential drive on wheel-speed servos, plus the sensor bookkeeping the lesson shows."""

    rate_hz = 100.0

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext, prefix: str = "") -> None:
        """`prefix` names one rover among several (see sensors_spec and the autopilot lesson)."""
        self.ctx = ctx
        self.body = model.body(f"{prefix}rover").id
        self.acts = [model.actuator(f"{prefix}{n}").id for n in ("left", "right")]
        self.adr = {
            s: (int(model.sensor(f"{prefix}{s}").adr[0]), int(model.sensor(f"{prefix}{s}").dim[0]))
            for s in ("enc_l", "enc_r", "accel", "gyro", "bumper")
        }
        self.lidar = [int(model.sensor(f"{prefix}ray{i}").adr[0]) for i in range(RAYS)]
        self.speed = 0.0
        self.turn = 0.0
        self.last_enc: tuple[float, float] | None = None
        self.last_pos: np.ndarray | None = None
        self.odom = 0.0  # path length from the encoders
        self.true = 0.0  # path length from the simulator, which a real robot never knows
        self.yaw_gyro = 0.0  # heading from integrating the gyro
        self.spin = 0.0  # total turning from the gyro, for the "turn a full circle" step
        self.ax_smooth = 0.0  # raw accelerometers buzz with every bump; robots low-pass filter them
        self.last_t: float | None = None
        self.next_sample = 0.0
        self.samples: deque[dict[str, float]] = deque(maxlen=1024)

    def read(self, data: mujoco.MjData, name: str) -> np.ndarray:
        adr, dim = self.adr[name]
        return np.array(data.sensordata[adr : adr + dim])

    def ranges(self, data: mujoco.MjData) -> list[float]:
        return [float(data.sensordata[a]) for a in self.lidar]

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        left = (self.speed - self.turn * TRACK / 2) / WHEEL_R
        right = (self.speed + self.turn * TRACK / 2) / WHEEL_R
        data.ctrl[self.acts[0]], data.ctrl[self.acts[1]] = left, right
        enc = (float(self.read(data, "enc_l")[0]), float(self.read(data, "enc_r")[0]))
        pos = np.array(data.xpos[self.body][:2])
        dt = 0.0 if self.last_t is None else data.time - self.last_t
        self.last_t = data.time
        if self.last_enc is not None and self.last_pos is not None:
            self.odom += abs((enc[0] - self.last_enc[0]) + (enc[1] - self.last_enc[1])) / 2 * WHEEL_R
            self.true += float(np.linalg.norm(pos - self.last_pos))
        self.last_enc, self.last_pos = enc, pos
        gyro_z = float(self.read(data, "gyro")[2])
        self.ax_smooth += (float(self.read(data, "accel")[0]) - self.ax_smooth) * (1 - math.exp(-dt / ACCEL_TAU))
        self.yaw_gyro += gyro_z * dt
        self.spin += abs(gyro_z) * dt
        if data.time >= self.next_sample:
            self.next_sample = data.time + SAMPLE_DT
            ranges = self.ranges(data)
            front = [r for i, r in enumerate(ranges) if (i <= 1 or i >= RAYS - 1) and r >= 0]
            self.samples.append(
                {
                    "t": data.time,
                    "odom": self.odom,
                    "true": self.true,
                    "ax": self.ax_smooth,
                    "gz": math.degrees(gyro_z),
                    "front": min(front) if front else 4.0,
                    "touch": float(self.read(data, "bumper")[0]),
                }
            )

    def status(self) -> dict[str, Any]:
        return {"controller": "lesson-rover", "speed": round(self.speed, 3), "turn": round(self.turn, 3)}


def _drove(ctx: LessonContext) -> bool:
    return ctx.results.get("odom", 0.0) >= 0.8


def _spun(ctx: LessonContext) -> bool:
    return ctx.results.get("spin_deg", 0.0) >= 330


def _stopped_short(ctx: LessonContext) -> bool:
    """Close to something ahead, and no bump in the last two seconds (back off and try again after one)."""
    recent = ctx.recent(2.0)
    return bool(recent) and recent[-1]["front"] < 0.35 and all(s["touch"] < 0.05 for s in recent)


def _bumped(ctx: LessonContext) -> bool:
    return any(s["touch"] > 0.5 for s in ctx.recent(0.5))


def _found_ball(ctx: LessonContext) -> bool:
    ball = ctx.results.get("ball", {})
    return ball.get("bearing", 180.0) < 12 and ball.get("distance", 9.0) < 1.2


STEPS = (
    Step(
        id="encoders",
        title={"zh": "编码器：轮子转了多少", "en": "Encoders: how far the wheels turned"},
        body={
            "zh": "每个轮子的电机上都有一个编码器，数它转了多少角度。走过的距离 = 轮子半径 × 转过的角度。\n"
            "这种只靠自己轮子推算位置的办法叫“里程计”。\n"
            "按住 W 往前开 1 米左右，看曲线里“编码器算的距离”和“真实距离”。",
            "en": "Each wheel motor has an encoder that counts how far it has turned. "
            "Distance = wheel radius * angle.\n"
            "Working out where you are from your own wheels is called odometry.\n"
            "Hold W and drive about a meter. Compare 'from the encoders' with 'true distance' in the plot.",
        },
        done={
            "zh": "在平地上不打滑时，两条线几乎重合：编码器很准。但它只知道轮子转了多少，不知道轮子有没有打滑。",
            "en": "On flat ground without slipping the two lines almost match: encoders are precise. "
            "But they only know how far the wheels turned, not whether the wheels slipped.",
        },
        check=_drove,
        show=("rays",),
    ),
    Step(
        id="imu",
        title={"zh": "IMU：感觉加速和转弯", "en": "The IMU: feeling acceleration and turns"},
        body={
            "zh": "IMU 里有两种传感器：加速度计（感觉被推了多猛，还会感觉到重力）和陀螺仪（感觉转得多快）。\n"
            "按住 A 或 D 原地转一整圈，看陀螺仪曲线；再猛地起步、刹车，看加速度曲线。",
            "en": "An IMU holds two sensors: an accelerometer (how hard you are pushed, gravity included) "
            "and a gyroscope (how fast you turn).\n"
            "Hold A or D to spin a full circle and watch the gyro; "
            "then start and stop sharply and watch the accelerometer.",
        },
        done={
            "zh": "把陀螺仪的转速加起来（积分）就得到朝向。但每一点小误差都会被加进去，"
            "时间一长朝向就会“漂”，所以 IMU 要和别的传感器一起用。",
            "en": "Adding up (integrating) the gyro gives the heading. Every small error gets added too, "
            "so over time the heading drifts. That is why an IMU is combined with other sensors.",
        },
        check=_spun,
        show=("rays", "imu"),
    ),
    Step(
        id="lidar",
        title={"zh": "激光雷达：一圈距离", "en": "Lidar: distances all around"},
        body={
            "zh": "车顶的激光雷达向 24 个方向打出激光，测量每个方向离障碍物多远（射线越红越近）。\n"
            "开到离墙 30 厘米以内停下，但不要撞上。",
            "en": "The lidar on top fires lasers in 24 directions and measures how far each one travels (redder means "
            "closer).\n"
            "Drive to within 30 cm of a wall and stop without touching it.",
        },
        done={
            "zh": "激光雷达给出每个方向的距离，自动驾驶避障、扫地机器人建地图都靠它。",
            "en": "A lidar gives the distance in every direction. Self-driving cars avoid obstacles with it, and robot "
            "vacuums map rooms with it.",
        },
        check=_stopped_short,
        show=("rays",),
    ),
    Step(
        id="bumper",
        title={"zh": "碰撞开关：最后一道保险", "en": "The bumper: the last line of defence"},
        body={
            "zh": "车头的保险杠里有一个触碰传感器，碰到东西就会感觉到压力。\n轻轻开过去碰一下那个木箱。",
            "en": "The front bumper has a touch sensor that feels the push when it hits something.\n"
            "Drive gently into the wooden crate.",
        },
        done={
            "zh": "碰撞开关最可靠，但也最晚：撞上了才知道。所以它是最后一道保险，平时靠激光雷达和相机提前发现。",
            "en": "A bumper is the most reliable sensor and the latest: it only knows once you have hit something. "
            "So it is the last line of defence; lidar and cameras see things coming.",
        },
        check=_bumped,
        show=("rays", "bumper"),
    ),
    Step(
        id="camera",
        title={"zh": "相机：看见颜色和形状", "en": "The camera: colours and shapes"},
        body={
            "zh": "右边的小画面是机器人相机看到的世界。\n转动小车，把红球放到画面正中间，再开到离它 1 米以内。",
            "en": "The small picture on the right is what the robot's camera sees.\n"
            "Turn until the red ball is in the middle of the picture, then drive within a meter of it.",
        },
        done={
            "zh": "相机的信息最丰富，颜色、形状、文字都有，"
            "但要靠视觉算法（比如找出红色像素在哪）才能变成机器人能用的数字。",
            "en": "A camera carries the most information (colour, shape, even text), but it takes a vision algorithm, "
            "such as finding where the red pixels are, to turn it into numbers a robot can act on.",
        },
        check=_found_ball,
        show=("rays", "camera"),
    ),
    Step(
        id="drift",
        title={"zh": "想一想", "en": "Think about it"},
        body={
            "zh": "只靠编码器（里程计）推算位置，为什么开得越久越不准？",
            "en": "Why does a position worked out from the encoders alone get worse the longer you drive?",
        },
        done={
            "zh": "所以真正的机器人会做“传感器融合”：编码器、IMU、激光雷达、相机互相校正，取长补短。",
            "en": "That is why real robots fuse their sensors: encoders, IMU, lidar and camera correct each other.",
        },
        quiz=Quiz(
            question={"zh": "选一个最好的解释", "en": "Pick the best explanation"},
            options=(
                {"zh": "编码器的分辨率太低", "en": "The encoders are too coarse"},
                {
                    "zh": "轮子会打滑、地面不平，每一步的小误差会一直累积下去",
                    "en": "Wheels slip and floors are uneven, and every small error keeps adding up",
                },
                {"zh": "电机转得太慢", "en": "The motors are too slow"},
            ),
            answer=1,
            explain={
                "zh": "对：里程计的误差只会累积不会消失，需要其他传感器（激光雷达、相机）时不时把它拉回来。",
                "en": "Right: odometry errors only accumulate, so other sensors (lidar, camera) must pull it back now "
                "and then.",
            },
        ),
    ),
)


def sensors_spec(name: str = "sensors_lesson", floor: bool = True) -> mujoco.MjSpec:
    """The walled arena with its rover and sensors, not yet compiled (the autopilot attaches two)."""
    spec = mujoco.MjSpec()
    spec.modelname = name
    spec.compiler.degree = False
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    G = mujoco.mjtGeom
    w = spec.worldbody
    w.add_light(pos=[0.5, -1.0, 3.0], dir=[-0.2, 0.3, -1.0], diffuse=[0.8, 0.8, 0.8])
    spec.add_material(name="rf_floor", rgba=[0.66, 0.7, 0.74, 1])
    for name in ("rf_dark", "rf_metal", "rf_shell", "rf_rubber", "rf_glass", "rf_paint", "rf_wood"):
        spec.add_material(name=name)
    if floor:
        w.add_geom(name="floor", type=G.mjGEOM_PLANE, size=[ARENA + 0.5, ARENA + 0.5, 0.05], material="rf_floor")
    wall = [0.86, 0.87, 0.88, 1]
    for name, pos, size in (
        ("wall_n", [0, ARENA, 0.1], [ARENA, 0.025, 0.1]),
        ("wall_s", [0, -ARENA, 0.1], [ARENA, 0.025, 0.1]),
        ("wall_e", [ARENA, 0, 0.1], [0.025, ARENA, 0.1]),
        ("wall_w", [-ARENA, 0, 0.1], [0.025, ARENA, 0.1]),
    ):
        w.add_geom(name=name, type=G.mjGEOM_BOX, pos=pos, size=size, rgba=wall, material="rf_paint")
    w.add_geom(
        name="crate",
        type=G.mjGEOM_BOX,
        pos=[*CRATE, 0.12],
        size=[0.16, 0.16, 0.12],
        rgba=[0.72, 0.53, 0.32, 1],
        material="rf_wood",
    )
    w.add_geom(
        name="ball",
        type=G.mjGEOM_SPHERE,
        pos=[*BALL, 0.09],
        size=[0.09, 0, 0],
        rgba=[0.88, 0.16, 0.14, 1],
        material="rf_shell",
    )
    w.add_geom(
        name="post",
        type=G.mjGEOM_CYLINDER,
        pos=[0.9, -0.85, 0.15],
        size=[0.06, 0.15, 0],
        rgba=[0.2, 0.45, 0.85, 1],
        material="rf_shell",
    )

    dark, shell, metal, rubber = (
        [0.16, 0.18, 0.21, 1],
        [0.95, 0.96, 0.97, 1],
        [0.8, 0.82, 0.85, 1],
        [0.08, 0.08, 0.09, 1],
    )
    rover = w.add_body(name="rover", pos=[-0.6, -0.5, WHEEL_R + 0.012])
    rover.add_freejoint(name="rover_free")
    rover.add_geom(
        name="chassis",
        type=G.mjGEOM_BOX,
        size=[0.11, 0.07, 0.025],
        pos=[0, 0, 0.02],
        rgba=shell,
        material="rf_shell",
        mass=0.6,
    )
    rover.add_geom(
        name="deck",
        type=G.mjGEOM_BOX,
        size=[0.08, 0.055, 0.012],
        pos=[-0.01, 0, 0.055],
        rgba=dark,
        material="rf_dark",
        mass=0.05,
    )
    rover.add_geom(
        name="bumper",
        type=G.mjGEOM_BOX,
        size=[0.012, 0.075, 0.018],
        pos=[0.122, 0, 0.018],
        rgba=[0.2, 0.22, 0.25, 1],
        material="rf_rubber",
        mass=0.02,
    )
    rover.add_site(
        name="bumper_zone", type=G.mjGEOM_BOX, size=[0.03, 0.085, 0.03], pos=[0.125, 0, 0.018], rgba=[0, 0, 0, 0]
    )
    rover.add_geom(
        name="caster",
        type=G.mjGEOM_SPHERE,
        size=[0.012, 0, 0],
        pos=[-0.085, 0, -0.023],
        rgba=metal,
        material="rf_metal",
        friction=[0.05, 0.005, 0.0001],
        mass=0.01,
    )
    rover.add_geom(
        name="mast",
        type=G.mjGEOM_CYLINDER,
        size=[0.012, 0.02, 0],
        pos=[0, 0, 0.085],
        rgba=dark,
        material="rf_dark",
        mass=0.01,
    )
    rover.add_geom(
        name="lidar",
        type=G.mjGEOM_CYLINDER,
        size=[0.03, 0.014, 0],
        pos=[0, 0, LIDAR_Z],
        rgba=dark,
        material="rf_dark",
        mass=0.03,
    )
    rover.add_geom(
        name="lidar_window",
        type=G.mjGEOM_CYLINDER,
        size=[0.031, 0.006, 0],
        pos=[0, 0, LIDAR_Z],
        rgba=[0.25, 0.55, 0.95, 0.85],
        material="rf_glass",
        contype=0,
        conaffinity=0,
        mass=0,
    )
    rover.add_geom(
        name="camera_body",
        type=G.mjGEOM_BOX,
        size=[0.012, 0.02, 0.012],
        pos=[0.095, 0, 0.068],
        rgba=dark,
        material="rf_dark",
        mass=0.01,
    )
    rover.add_geom(
        name="lens",
        type=G.mjGEOM_CYLINDER,
        size=[0.008, 0.004, 0],
        pos=[0.108, 0, 0.068],
        quat=[0.7071068, 0, 0.7071068, 0],
        rgba=[0.1, 0.15, 0.25, 1],
        material="rf_glass",
        contype=0,
        conaffinity=0,
        mass=0,
    )
    rover.add_camera(name="eye", pos=[0.115, 0, 0.068], quat=_camera_quat(), fovy=60)
    rover.add_site(name="imu", pos=[0, 0, 0.02], size=[0.01, 0, 0])
    for i in range(RAYS):
        rover.add_site(
            name=f"ray{i}",
            pos=[0, 0, LIDAR_Z],
            quat=_quat_z_out(2 * math.pi * i / RAYS),
            size=[0.005, 0, 0],
            rgba=[0, 0, 0, 0],
        )
    for side, y in (("left", 0.09), ("right", -0.09)):
        wheel = rover.add_body(name=f"{side}_wheel", pos=[0.02, y, 0])
        wheel.add_joint(name=side, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=0.001)
        wheel.add_geom(
            name=f"{side}_tyre",
            type=G.mjGEOM_CYLINDER,
            size=[WHEEL_R, 0.012, 0],
            quat=[0.7071068, 0.7071068, 0, 0],
            rgba=rubber,
            material="rf_rubber",
            mass=0.05,
            friction=[1.2, 0.01, 0.001],
        )
        wheel.add_geom(
            name=f"{side}_hub",
            type=G.mjGEOM_CYLINDER,
            size=[0.016, 0.0135, 0],
            quat=[0.7071068, 0.7071068, 0, 0],
            rgba=metal,
            material="rf_metal",
            contype=0,
            conaffinity=0,
            mass=0,
        )
        motor = spec.add_actuator(name=side, target=side, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        motor.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        motor.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        motor.gainprm[0] = 0.08
        motor.biasprm[:3] = [0.0, 0.0, -0.08]
        motor.ctrllimited = True
        motor.ctrlrange = [-20, 20]
        motor.forcelimited = True
        motor.forcerange = [-0.5, 0.5]
        spec.add_sensor(
            name=f"enc_{side[0]}",
            type=mujoco.mjtSensor.mjSENS_JOINTPOS,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname=side,
        )
    spec.add_sensor(
        name="accel", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu"
    )
    spec.add_sensor(name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(
        name="bumper", type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="bumper_zone"
    )
    for i in range(RAYS):
        ray = spec.add_sensor(
            name=f"ray{i}",
            type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=f"ray{i}",
            cutoff=4.0,
        )
        ray.intprm[0] = RANGE_DIST  # MuJoCo 3.15 asks which ray data to report: just the distance
    return spec


class SensorsLesson(Lesson):
    id = "sensors"
    chapter = {"zh": "感知", "en": "Sensing"}
    title = {"zh": "感知（一）：机器人怎么感知世界", "en": "Sensing I: how a robot senses the world"}
    summary = {
        "zh": "开一台小车，看编码器、IMU、激光雷达、碰撞开关和相机各自回答什么问题。",
        "en": "Drive a rover and see what encoders, the IMU, lidar, a bumper and a camera each tell it.",
    }
    minutes = 12
    drive = {"max_speed": MAX_SPEED, "turn": MAX_TURN}
    plots = (
        Plot(
            "distance",
            {"zh": "走过的距离", "en": "Distance travelled"},
            "m",
            (
                Series("true", {"zh": "真实", "en": "true"}, "muted", dashed=True),
                Series("odom", {"zh": "编码器算的", "en": "from encoders"}, "ink"),
            ),
            window=10.0,
        ),
        Plot(
            "imu",
            {"zh": "IMU", "en": "IMU"},
            "",
            (
                Series("ax", {"zh": "前后加速度 m/s²（滤波后）", "en": "forward accel m/s² (filtered)"}, "blue"),
                Series("gz", {"zh": "转速 °/s ÷ 50", "en": "turn rate °/s ÷ 50"}, "violet"),
            ),
        ),
        Plot(
            "range",
            {"zh": "正前方距离（激光雷达）", "en": "Distance ahead (lidar)"},
            "m",
            (Series("front", {"zh": "前方", "en": "ahead"}, "amber"),),
        ),
    )
    steps = STEPS

    def __init__(self) -> None:
        self.rover: Rover | None = None

    def build(self) -> mujoco.MjModel:
        return sensors_spec().compile()

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.rover = Rover(model, data, ctx)
        return {"lesson": self.rover}

    def on_drive(self, speed: float, turn: float, ctx: LessonContext) -> None:
        assert self.rover is not None
        self.rover.speed = float(np.clip(speed, -MAX_SPEED, MAX_SPEED))
        self.rover.turn = float(np.clip(turn, -MAX_TURN, MAX_TURN))

    def drain_samples(self) -> list[dict[str, float]]:
        if self.rover is None:
            return []
        out = list(self.rover.samples)
        self.rover.samples.clear()
        for s in out:
            s["gz"] = s["gz"] / 50  # on the same axis as acceleration
        return out

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        assert self.rover is not None
        r = self.rover
        body = r.body
        pos = np.array(data.xpos[body][:2])
        heading = math.atan2(data.xmat[body][3], data.xmat[body][0])
        to_ball = np.array(BALL) - pos
        bearing = abs(
            math.degrees(
                math.atan2(
                    math.sin(math.atan2(to_ball[1], to_ball[0]) - heading),
                    math.cos(math.atan2(to_ball[1], to_ball[0]) - heading),
                )
            )
        )
        touch = float(r.read(data, "bumper")[0])
        ctx.results["odom"] = round(r.odom, 3)
        ctx.results["spin_deg"] = round(math.degrees(r.spin), 1)
        ctx.results["ball"] = {"bearing": round(bearing, 1), "distance": round(float(np.linalg.norm(to_ball)), 2)}
        accel = r.read(data, "accel")
        return {
            "ranges": [round(x, 3) for x in r.ranges(data)],
            "touch": round(touch, 3),
            "accel": [round(float(x), 2) for x in accel],
            "gyro_z": round(float(r.read(data, "gyro")[2]), 3),
            "odom": round(r.odom, 3),
            "true": round(r.true, 3),
            "heading_gyro": round(math.degrees(r.yaw_gyro), 1),
            "heading_true": round(math.degrees(heading), 1),
            "ball": ctx.results["ball"],
        }

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        cam = model.camera("eye")
        return {
            "kind": "rover",
            "body": int(model.body("rover").id),
            "lidar": [0, 0, LIDAR_Z],
            "rays": [2 * math.pi * i / RAYS for i in range(RAYS)],
            "bumper": [0.125, 0, 0.018],
            "camera": {"pos": [float(x) for x in cam.pos], "quat": [float(x) for x in cam.quat], "fovy": 60},
        }

    def camera(self) -> dict[str, list[float]]:
        return {"eye": [-1.35, -1.95, 1.45], "target": [-0.25, -0.2, 0.0]}
