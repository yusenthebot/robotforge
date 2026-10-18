"""Face-offs: our template robots next to real ones, on the same task, in the same physics.

A face-off puts its robots side by side in one world, fills a spec table from the models themselves
(joints, mass, reach, how they hold themselves up) and runs one task on every robot at once with the
same planner, so a difference the learner sees comes from the robot and not from a different test.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors.manager import BehaviorError
from robotforge.design.world import World, compile_world
from robotforge.learn.engine import Control, Lesson, LessonContext, Plot, Quiz, Series, Step, Text

WORKSPACE_SAMPLES = 900  # random joint settings per arm for the workspace cloud and the reach figure
DWELL = 0.4  # s an arm holds still on a target before its error is measured
REACHED = 0.01  # m: closer than this counts as reaching a target
STIFF_SERVO = 1000.0  # N*m/rad: position gains this high barely sag under an arm's own weight


def _text(zh: str, en: str) -> Text:
    return {"zh": zh, "en": en}


@dataclass(frozen=True)
class Entrant:
    name: str  # robot name in the world
    label: Text
    robot: dict[str, Any]  # RobotEntry fields: {"template": ...} or {"library": ...}
    real: bool  # a real product (its MuJoCo Menagerie model) rather than one of our templates
    color: str  # plot and overlay colour key (see engine.Series)


class FaceOff(Lesson):
    """Robots in a row along y, facing +x toward the camera; subclasses add the task."""

    kind = "compare"
    entrants: tuple[Entrant, ...] = ()
    spacing = 1.1  # m between robots

    def __init__(self) -> None:
        self.playground = None  # set by the runner: face-offs drive the real behaviors
        self.world: World | None = None  # the behaviors find this lesson's objects here

    def base(self, index: int) -> tuple[float, float]:
        return 0.0, (index - (len(self.entrants) - 1) / 2) * self.spacing

    def objects(self, index: int, x: float, y: float) -> list[dict[str, Any]]:
        return []

    def build(self) -> mujoco.MjModel:
        robots, objects = [], []
        for i, e in enumerate(self.entrants):
            x, y = self.base(i)
            robots.append({"name": e.name, **e.robot, "pos": [x, y]})
            objects += self.objects(i, x, y)
        self.world = World.model_validate(
            {"name": f"faceoff_{self.id}", "floor": {"style": "concrete"}, "robots": robots, "objects": objects}
        )
        return compile_world(self.world).model

    def columns(self) -> list[dict[str, Any]]:
        return [{"name": e.name, "label": e.label, "real": e.real, "color": e.color} for e in self.entrants]


def robot_bodies(model: mujoco.MjModel, name: str) -> list[int]:
    return [b for b in range(1, model.nbody) if (model.body(b).name or "").startswith(f"{name}/")]


def root_body(model: mujoco.MjModel, name: str) -> int:
    """The robot's body attached to the world (its base, or its floating trunk)."""
    return next(b for b in robot_bodies(model, name) if model.body_parentid[b] == 0)


# ---- arms: ours, AgileX PiPER, Franka Panda -----------------------------------------------------------

CUBE = (0.25, -0.12)  # m from each arm's base, toward the camera and to its right
TRAY = (0.12, 0.26)  # both well inside the smallest arm's straight-down reach
TARGET_Z = 0.05  # m: picking height, just above the floor
# Every arm gets the same six targets in its own frame, further and further out, alternately left and right.
TARGETS = tuple((0.2 + 0.1 * k, 0.1 if k % 2 == 0 else -0.1, TARGET_Z) for k in range(6))


def arm_facts(model: mujoco.MjModel, data: mujoco.MjData, arm: Any, name: str, seed: int = 0) -> dict[str, Any]:
    """Joints, mass, reach, how the arm copes with gravity, and a cloud of reachable tool points.

    Everything comes from the compiled model, so the table says what the simulation actually has.
    """
    bodies = robot_bodies(model, name)
    mass = float(sum(model.body_mass[b] for b in bodies))
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    rng = np.random.default_rng(seed)
    lo, hi = arm.chain.lower, arm.chain.upper
    cloud = []
    for q in rng.uniform(lo, hi, size=(WORKSPACE_SAMPLES, len(lo))):
        scratch.qpos[arm.chain.qpos_adr] = q
        mujoco.mj_kinematics(model, scratch)
        cloud.append(scratch.site_xpos[arm.site].copy())
    points = np.array(cloud)
    axis = scratch.xpos[arm.base_body]  # the base joint turns about this body's vertical axis
    reach = float(np.max(np.linalg.norm(points[:, :2] - axis[:2], axis=1)))
    gains = [float(model.actuator_gainprm[a, 0]) for a in arm.actuators]
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies]
    top = max(float(data.geom_xpos[g][2] + model.geom_rbound[g]) for g in geoms) if geoms else 0.5
    return {
        "top": top,
        "joints": len(lo),
        "mass": mass,
        "reach": reach,
        "gravcomp": any(model.body_gravcomp[b] > 0 for b in bodies),
        "kp": max(gains),
        "cloud": np.round(points[points[:, 2] > 0.005], 3).tolist(),
    }


class ArmFaceOff(FaceOff):
    id = "arm_faceoff"
    chapter = _text("真机对比", "Real robots")
    title = _text("机械臂：我们的、PiPER、Panda", "Arms: ours, PiPER and Panda")
    summary = _text(
        "我们的 5 轴舵机臂和两台真机并排：同样的目标、同样的规划器，看臂展、关节数和电机控制带来的差别。",
        "Our 5-joint servo arm next to two real arms: same targets, same planner. See what reach, joint count "
        "and motor control change.",
    )
    minutes = 10
    entrants = (
        Entrant("ours", _text("我们的臂", "Our arm"), {"template": "arm"}, False, "amber"),
        Entrant("piper", _text("AgileX PiPER", "AgileX PiPER"), {"library": "piper"}, True, "blue"),
        Entrant("panda", _text("Franka Panda", "Franka Panda"), {"library": "panda"}, True, "violet"),
    )
    controls = (
        Control("workspace", "toggle", _text("显示工作空间", "Show workspace")),
        Control("reach", "button", _text("开始抓取距离测试", "Run the reach test")),
        Control("gravcomp", "toggle", _text("给每台臂加重力补偿", "Gravity compensation on every arm")),
        Control("pick", "button", _text("开始抓放", "Run pick and place")),
    )
    plots = (
        Plot(
            "track",
            _text("跟踪误差：手爪离规划位置多远", "Tracking error: gripper vs. its plan"),
            "mm",
            tuple(Series(e.name, e.label, e.color) for e in entrants),
            window=20.0,
            floor=0.0,
        ),
    )
    steps = (
        Step(
            id="meet",
            title=_text("认识三台机械臂", "Meet the three arms"),
            body=_text(
                "左边是我们的模板臂：5 个舵机关节，3D 打印的零件。中间是 AgileX PiPER，6 个关节，常见的桌面和教学臂。"
                "右边是 Franka Panda，7 个关节，研究实验室里最常见的机械臂。\n"
                "下面的规格表都是从模型本身算出来的。\n"
                "打开「显示工作空间」：每个点都是手爪能到的一个位置。谁够得最远？",
                "On the left, our template arm: 5 servo joints, 3D-printed parts. In the middle the AgileX PiPER, "
                "6 joints, a common desktop and teaching arm. On the right the Franka Panda, 7 joints, the arm most "
                "research labs have.\nThe spec table below is worked out from the models themselves.\n"
                "Turn on Show workspace: each dot is a place the gripper can get to. Which arm reaches furthest?",
            ),
            done=_text(
                "工作空间就是“手能到哪儿”。臂越长、关节越多，点云越大：Panda 的点云比另外两台大一圈。"
                "但能到那儿，不代表能用合适的姿势到那儿，下一步来测。",
                "The workspace is everywhere the hand can get to. Longer links and more joints give a bigger cloud, "
                "and the Panda's is clearly the biggest. Getting there is not the same as getting there in a useful "
                "pose; that is the next test.",
            ),
            check=lambda ctx: ctx.values.get("workspace", 0) >= 1,
            unlock=("workspace",),
            show=("labels",),
        ),
        Step(
            id="reach",
            title=_text("能抓多远？", "How far out can it pick?"),
            body=_text(
                "点「开始抓取距离测试」：每台臂把手爪垂直朝下，依次伸到 6 个琥珀色小球，从 0.2 米到 0.7 米越来越远。"
                "三台用同一个规划器、同样的速度。\n"
                "看表里“够到的目标”“朝下最远”和“平均误差”，再看曲线里谁跟得最紧。",
                "Press Run the reach test: each arm points its gripper straight down and visits the six amber balls, "
                "from 0.2 m out to 0.7 m. All three use the same planner at the same speed.\nWatch 'targets "
                "reached', 'furthest straight down' and 'mean error' in the table, and who follows its plan most "
                "closely in the plot.",
            ),
            done=_text(
                "抓东西时手爪要垂直朝下，手腕得弯够角度，所以能抓的距离比臂展短：PiPER 的臂展超过 0.7 米，"
                "朝下却只到 0.45 米左右。我们的臂还有个问题：误差有 2 到 3 厘米，而且曲线一直不归零。下一步看为什么。",
                "To pick something the gripper points straight down, and the wrist has to bend far enough for that, "
                "so picking reach is shorter than the arm's span: the PiPER spans over 0.7 m but reaches only about "
                "0.45 m pointing down. Our arm has another problem: it misses by 2 to 3 cm and its curve never "
                "returns to zero. The next step shows why.",
            ),
            check=lambda ctx: "reach" in ctx.results,
            unlock=("reach",),
            show=("labels", "targets"),
        ),
        Step(
            id="gravity",
            title=_text("重力补偿", "Gravity compensation"),
            body=_text(
                "我们的舵机只有 P 控制：差得越多才使越大的劲，所以举着自己的重量时总是差一点，臂会往下垂"
                "（PID 课里的稳态误差）。PiPER 的控制器先算出“重力要多少力矩”，直接加上去，这叫重力补偿。"
                "Panda 的电机非常硬，平时下垂很小，可伸到 0.7 米时也差了 1 厘米。\n"
                "打开「给每台臂加重力补偿」，再测一次，看误差和曲线。",
                "Our servos only use P control: they push harder the further off they are, so holding up their "
                "own weight always leaves them a little short, and the arm sags (the steady-state error from the "
                "PID lesson). The PiPER's controller works out how much torque gravity needs and adds it up "
                "front, which is called gravity compensation. The Panda's motors are very stiff, so it sags "
                "little, yet stretched out to 0.7 m it still misses by a centimetre.\nTurn on Gravity "
                "compensation on every arm, run the test again and compare the errors and the curve.",
            ),
            done=_text(
                "同样的电机，我们的臂误差从约 2 厘米降到不到 1 毫米，Panda 伸远时的 1 厘米也没了："
                "变的只是控制器知道了重力。这就是前馈：能提前算出来的力，不要等误差出现了再去纠正。"
                "真实的 Panda 和 PiPER 控制器里都有这一步。",
                "Same motors, and our arm's error drops from about 2 cm to under a millimetre, and the Panda's "
                "centimetre at full stretch is gone too: all that changed is that the controller knows about "
                "gravity. That is feedforward: a force you can work out in advance should not wait for an error "
                "to appear before it is applied. The real Panda and PiPER controllers both do this.",
            ),
            check=lambda ctx: ctx.results.get("reach", {}).get("ours", {}).get("gravcomp", False),
            unlock=("reach", "gravcomp"),
            show=("labels", "targets"),
        ),
        Step(
            id="why7",
            title=_text("为什么是 6 个、7 个关节？", "Why 6 or 7 joints?"),
            body=_text(
                "手爪在空间里有 6 个自由度：3 个管位置（前后、左右、上下），3 个管朝向。",
                "A gripper in space has 6 degrees of freedom: 3 for where it is (forward, sideways, up) and 3 for "
                "which way it points.",
            ),
            done=_text("", ""),
            quiz=Quiz(
                question=_text(
                    "Panda 有 7 个关节，比需要的多一个。多出来的那个有什么用？",
                    "The Panda has 7 joints, one more than it needs. What is the extra one for?",
                ),
                options=(
                    _text("让它动得更快", "It makes the arm faster"),
                    _text(
                        "手爪不动时手肘还能换位置，可以绕开障碍物",
                        "The elbow can move while the gripper stays put, to get around obstacles",
                    ),
                    _text("让它能举更重的东西", "It lets the arm lift heavier things"),
                ),
                answer=1,
                explain=_text(
                    "对。6 个关节刚好能定下手爪的位置和朝向；第 7 个是“冗余”的，同一个手爪姿态可以配无数种手肘位置，"
                    "用来避开障碍和关节极限。5 个关节的臂少了一个朝向，所以有些姿态它做不到。",
                    "Right. Six joints are exactly enough to set the gripper's position and direction; the seventh "
                    "is redundant, so one gripper pose allows endless elbow positions, which helps avoid obstacles "
                    "and joint limits. A 5-joint arm is one short, so some gripper directions are impossible for it.",
                ),
            ),
        ),
        Step(
            id="pick",
            title=_text("抓放比赛", "Pick-and-place race"),
            body=_text(
                "点「开始抓放」：每台臂把自己面前的红方块放进蓝盒子。规划器一样，比的是抓得稳、放得准。",
                "Press Run pick and place: each arm moves its red cube into its blue tray. Same planner; what "
                "differs is whether the grip holds and the cube lands in the tray.",
            ),
            done=_text(
                "三台都能完成：在臂展以内的简单任务，便宜的臂也够用。真机贵在更大的工作空间、更高的精度和负载，"
                "以及长时间工作的可靠性。选机器人要看任务需要什么。",
                "All three can do it: for a simple task inside its reach, an inexpensive arm is enough. Real arms "
                "earn their price with a bigger workspace, more accuracy and payload, and reliability over long "
                "shifts. Choose the robot for what the task needs.",
            ),
            check=lambda ctx: "pick" in ctx.results,
            unlock=("pick",),
            show=("labels",),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.facts: dict[str, dict[str, Any]] = {}
        self.native: tuple[np.ndarray, np.ndarray] | None = None  # the models' own gravity compensation
        self.runs: dict[str, dict[str, Any]] = {}  # robot -> the running test's progress
        self.test: str | None = None  # "reach" or "pick" while one runs
        self.samples: deque[dict[str, float]] = deque(maxlen=2048)
        self.next_sample = 0.0

    def objects(self, index: int, x: float, y: float) -> list[dict[str, Any]]:
        name = self.entrants[index].name
        return [
            {"name": f"cube_{name}", "shape": "box", "size": [0.04, 0.04, 0.04], "pos": [x + CUBE[0], y + CUBE[1]]},
            {
                "name": f"tray_{name}",
                "shape": "bin",
                "size": [0.14, 0.14, 0.05],
                "pos": [x + TRAY[0], y + TRAY[1]],
                "color": "blue",
            },
        ]

    def targets(self, index: int) -> list[list[float]]:
        x, y = self.base(index)
        return [[round(x + tx, 3), round(y + ty, 3), tz] for tx, ty, tz in TARGETS]

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.runs, self.test = {}, None
        if not self.facts:
            self.native = (model.body_gravcomp.copy(), model.jnt_actgravcomp.copy())
            for e in self.entrants:
                self.facts[e.name] = arm_facts(model, data, self._arm(e.name), e.name)
        self._gravcomp(model, ctx.values.get("gravcomp", 0) >= 1)
        for e in self.entrants:
            self._arm(e.name)  # every arm holds its home pose from the start
        return {}

    def _arm(self, name: str) -> Any:
        return self.playground.behaviors._arm(name)

    def _gravcomp(self, model: mujoco.MjModel, on: bool) -> None:
        """Cancel gravity on every arm's links through its motors, as a controller's feedforward would.

        Off brings back what the models have themselves (the PiPER's controller compensates gravity).
        MuJoCo only computes gravity compensation when some body had it at compile time; the PiPER's
        model does, which is what makes switching the others at run time take effect.
        """
        assert self.native is not None
        gravcomp, actgravcomp = self.native
        model.body_gravcomp[:] = gravcomp
        model.jnt_actgravcomp[:] = actgravcomp
        if on:
            bodies = [b for e in self.entrants for b in robot_bodies(model, e.name)]
            model.body_gravcomp[bodies] = 1.0
            joints = [j for j in range(model.njnt) if model.jnt_bodyid[j] in bodies]
            model.jnt_actgravcomp[joints] = 1  # applied by the actuators, not as a free force

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        return {
            "kind": "faceoff",
            "entrants": [
                {
                    **c,
                    "base": [*self.base(i), 0.0],
                    "body": root_body(model, c["name"]),
                    "top": round(self.facts.get(c["name"], {}).get("top", 0.5), 3),
                    "cloud": self.facts.get(c["name"], {}).get("cloud", []),
                }
                for i, c in enumerate(self.columns())
            ],
            "targets": [self.targets(i) for i in range(len(self.entrants))],
        }

    def camera(self) -> dict[str, Any]:
        # From the front and to the side of our (smallest) arm, so the row foreshortens between the panels.
        return {"eye": [3.15, -2.85, 1.9], "target": [-0.11, -0.1, 0.25], "fit": True}

    # ---- the tests ------------------------------------------------------------------------------

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        if control == "gravcomp":
            self._gravcomp(self.playground.session.model, value >= 1)
            return
        if control not in ("reach", "pick") or self.test is not None:
            return
        self.test = control
        self.runs = {}
        for e in self.entrants:
            # Every run starts from the home pose, so no arm is helped or hurt by where the last test left it.
            self._arm(e.name).go_home()
            self.runs[e.name] = {"homing": True, "start": None, "end": None, "track_max": 0.0}
            self.runs[e.name]["gravcomp"] = ctx.values["gravcomp"] >= 1

    def _begin(self, name: str, run: dict[str, Any], data: mujoco.MjData) -> None:
        """The arm is home: start its test and its clock."""
        run["homing"] = False
        run["start"] = float(data.time)
        if self.test == "reach":
            run.update(index=-1, errors=[], hold_until=None)
            return
        try:
            _, info = self.playground.behaviors.pick_and_place(name, f"cube_{name}", f"tray_{name}")
            run["region"] = info["region"]
        except BehaviorError:
            run["end"], run["success"] = run["start"], False

    def _step_reach(self, i: int, name: str, run: dict[str, Any], data: mujoco.MjData) -> None:
        arm = self._arm(name)
        targets = self.targets(i)
        if run["index"] >= 0:
            if run["hold_until"] is None:
                if not arm.done:
                    return
                run["hold_until"] = float(data.time) + DWELL
            if data.time < run["hold_until"]:
                return
            run["errors"].append(float(np.linalg.norm(data.site_xpos[arm.site] - targets[run["index"]])))
        run["index"] += 1
        run["hold_until"] = None
        while run["index"] < len(targets):
            try:
                self.playground.behaviors.move_arm(name, targets[run["index"]])
                return
            except BehaviorError:  # out of reach with the gripper pointing down
                run["errors"].append(None)
                run["index"] += 1
        run["end"] = float(data.time)

    def _step_pick(self, name: str, run: dict[str, Any], data: mujoco.MjData) -> None:
        if not self._arm(name).done:
            return
        if run.get("settle") is None:
            run["settle"] = float(data.time) + 0.5
        elif data.time >= run["settle"]:
            run["end"] = float(data.time)
            run["success"] = self.playground.behaviors.placed(f"cube_{name}", run["region"])["success"]

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        point: dict[str, float] = {"t": float(data.time)}
        for i, e in enumerate(self.entrants):
            arm = self._arm(e.name)
            track = float(np.linalg.norm(data.site_xpos[arm.site] - arm.ik.point(arm.chain, arm.q))) * 1000
            point[e.name] = track
            run = self.runs.get(e.name)
            if run is None or run["end"] is not None:
                continue
            if run["homing"]:
                if arm.done:
                    self._begin(e.name, run, data)
                continue
            run["track_max"] = max(run["track_max"], track)
            if self.test == "reach":
                self._step_reach(i, e.name, run, data)
            else:
                self._step_pick(e.name, run, data)
        if data.time >= self.next_sample:
            self.next_sample = float(data.time) + 0.05
            self.samples.append(point)
        if self.test is not None and all(r["end"] is not None for r in self.runs.values()):
            ctx.results[self.test] = {name: self._result(i, run) for i, (name, run) in enumerate(self.runs.items())}
            self.test = None
        return {"table": self.table(ctx), "active": self._active_targets(), "readout": self._readout()}

    def _result(self, index: int, run: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "time": round(run["end"] - run["start"], 1),
            "track_max": round(run["track_max"], 1),
            "gravcomp": run["gravcomp"],
        }
        if "errors" not in run:
            out["success"] = bool(run.get("success"))
            return out
        hit = [k for k, e in enumerate(run["errors"]) if e is not None and e < REACHED]
        out["reached"] = len(hit)
        out["of"] = len(TARGETS)
        out["furthest"] = round(max((math.hypot(*TARGETS[k][:2]) for k in hit), default=0.0), 2)
        tried = [e for e in run["errors"] if e is not None]
        out["mean_mm"] = round(1000 * sum(tried) / len(tried), 1) if tried else None
        return out

    def _active_targets(self) -> list[int]:
        """Which target each arm is on during the reach test (-1: none)."""
        if self.test != "reach":
            return [-1] * len(self.entrants)
        return [self.runs[e.name].get("index", -1) for e in self.entrants]

    def _readout(self) -> Text:
        if self.test == "reach":
            return _text("正在测抓取距离……", "Reach test running...")
        if self.test == "pick":
            return _text("正在抓放……", "Pick and place running...")
        return _text("", "")

    def drain_samples(self) -> list[dict[str, float]]:
        out = list(self.samples)
        self.samples.clear()
        return out

    # ---- the table ------------------------------------------------------------------------------

    def table(self, ctx: LessonContext) -> dict[str, Any]:
        names = [e.name for e in self.entrants]
        facts = [self.facts.get(n, {}) for n in names]
        gravcomp_on = ctx.values.get("gravcomp", 0) >= 1
        rows = [
            {"heading": _text("规格（从模型算出）", "Specs (from the models)")},
            _row(_text("关节", "Joints"), [_joints(f) for f in facts], [f.get("joints", 0) for f in facts]),
            _row(_text("质量", "Mass"), [f"{f.get('mass', 0):.1f} kg" for f in facts], None),
            _row(_text("臂展", "Reach"), [f"{f['reach']:.2f} m" for f in facts], [f["reach"] for f in facts]),
            _row(
                _text("对抗重力", "Against gravity"),
                [_holds_up(f, gravcomp_on) for f in facts],
                None,
            ),
            _row(
                _text("来源", "Source"),
                [_source(e) for e in self.entrants],
                None,
            ),
        ]
        reach = ctx.results.get("reach")
        if reach:
            got = [reach[n] for n in names]
            on = reach[names[0]]["gravcomp"]
            rows += [
                {
                    "heading": _text("抓取距离测试（开了重力补偿）", "Reach test (gravity compensation on)")
                    if on
                    else _text("抓取距离测试", "Reach test")
                },
                _row(
                    _text("够到的目标", "Targets reached"),
                    [f"{r['reached']} / {r['of']}" for r in got],
                    [r["reached"] for r in got],
                ),
                _row(
                    _text("朝下最远", "Furthest straight down"),
                    [f"{r['furthest']:.2f} m" if r["reached"] else "-" for r in got],
                    [r["furthest"] for r in got],
                ),
                _row(
                    _text("平均误差", "Mean error"),
                    ["-" if r["mean_mm"] is None else f"{r['mean_mm']:.1f} mm" for r in got],
                    [math.inf if r["mean_mm"] is None else r["mean_mm"] for r in got],
                    high=False,
                ),
            ]
        pick = ctx.results.get("pick")
        if pick:
            got = [pick[n] for n in names]
            rows += [
                {"heading": _text("抓放比赛", "Pick-and-place race")},
                _row(
                    _text("抓放", "Pick and place"),
                    [_text("成功", "done") if r["success"] else _text("失败", "failed") for r in got],
                    None,
                ),
                _row(
                    _text("抓放用时", "Pick-and-place time"),
                    [f"{r['time']:.1f} s" for r in got],
                    [r["time"] if r["success"] else math.inf for r in got],
                    high=False,
                ),
            ]
        return {"columns": self.columns(), "rows": rows}


# ---- humanoids: ours and the Unitree G1 ------------------------------------------------------------

WALK_SECONDS = 6.0
START_X = -1.4  # m: the G1 walks about 3 m in the test, which keeps it between the panels
G1_TEST_SPEED = 0.5  # m/s: well inside what the policy learned, and it stays in view
PUSH_EVERY = 2.5  # s between shoves
PUSH_STEP = 0.15  # m/s of sideways kick added per level, the same for both robots
PUSH_LEVELS = 10
FALLEN = 0.5  # cosine of the torso's tilt below which a humanoid is down
GRAVITY = 9.81


class HumanoidFaceOff(FaceOff):
    id = "humanoid_faceoff"
    spacing = 1.8  # m: wide lanes, so from the camera the 1.3 m G1 does not stand on top of ours
    chapter = _text("真机对比", "Real robots")
    title = _text("人形：我们的、Unitree G1", "Humanoids: ours and the Unitree G1")
    summary = _text(
        "手写的经典控制器对上强化学习：我们的小人形用 ZMP 预观控制走路，G1 用在仿真里练出来的神经网络。",
        "A hand-written classic controller against reinforcement learning: our small humanoid walks with ZMP "
        "preview control, the G1 with a neural network that practised in simulation.",
    )
    minutes = 8
    spacing = 1.6
    entrants = (
        Entrant("ours", _text("我们的人形", "Our humanoid"), {"template": "humanoid"}, False, "amber"),
        Entrant("g1", _text("Unitree G1", "Unitree G1"), {"library": "g1"}, True, "blue"),
    )
    controls = (
        Control("walk", "button", _text("开始行走测试", "Run the walking test")),
        Control("push", "button", _text("开始推", "Start shoving")),
    )
    plots = (
        Plot(
            "speed",
            _text("前进速度", "Forward speed"),
            "m/s",
            tuple(Series(e.name, e.label, e.color) for e in entrants),
            window=12.0,
        ),
        Plot(
            "tilt",
            _text("身体倾斜", "Body tilt"),
            "°",
            tuple(Series(f"{e.name}_tilt", e.label, e.color) for e in entrants),
            window=12.0,
            floor=0.0,
        ),
    )
    steps = (
        Step(
            id="meet",
            title=_text("两种走路的办法", "Two ways to walk"),
            body=_text(
                "前面是我们的模板人形：0.45 米高、不到 2 千克、20 个舵机。它的控制器是人写的：先规划脚落在哪，"
                "再用“倒立摆”模型和 ZMP（零力矩点）预观控制算出身体该怎么晃，脚才不会翻。\n"
                "后面是 Unitree G1：1.3 米、33 千克、29 个电机。它的控制器是一个神经网络，没有人告诉它怎么迈步："
                "UniLab 让 2048 个仿真的 G1 同时练习，练了 2200 轮（这台电脑上约 1 小时），每走稳一步就加分。\n"
                "看规格表，然后点「下一步」。",
                "In front, our template humanoid: 0.45 m tall, under 2 kg, 20 servos. Its controller was written "
                "by people: it plans where each foot lands, then uses an inverted pendulum model and ZMP (zero moment "
                "point) preview control to work out how the body must sway so the foot never tips.\nBehind it, the "
                "Unitree G1: 1.3 m, 33 kg, 29 motors. Its controller is a neural network that nobody told how to "
                "step: UniLab had 2048 simulated G1s practise at once for 2200 rounds (about an hour on this "
                "computer), scoring points for every steady step.\nRead the spec table, then press Next.",
            ),
            done=_text("", ""),
            show=("labels",),
        ),
        Step(
            id="walk",
            title=_text("谁走得好？", "Who walks better?"),
            body=_text(
                f"点「开始行走测试」：两台都往前走 {WALK_SECONDS:g} 秒，"
                f"我们的用它最快的速度，G1 用 {G1_TEST_SPEED:g} 米/秒。\n"
                "只比“米/秒”不公平：腿长的天生走得快。看表里的“每秒走几个身高”和“弗劳德数”"
                "（速度²÷(重力×腿长)），它把大小不同的走路者放到同一把尺子上，猫和马都在它约等于 0.5 时从走换成跑。\n"
                "再看“每米能耗”：同样走一米，谁更省力。",
                f"Press Run the walking test: both walk forward for {WALK_SECONDS:g} s, ours at its top speed "
                f"and the G1 at {G1_TEST_SPEED:g} m/s.\n"
                "Comparing metres per second alone is unfair: long legs walk faster. Look at 'body heights "
                "per second' and the 'Froude number' (speed squared over gravity times leg length): it puts walkers "
                "of any size on one scale, and cats and horses alike switch from walking to running near 0.5.\nThen "
                "look at 'energy per metre': who walks a metre for less effort.",
            ),
            done=_text(
                "按身高折算后，差距比“米/秒”小得多。我们的手写控制器每一步都要先算好再走，所以只敢走得很慢；"
                "强化学习的策略在几千次摔倒里学会了一边走一边接住自己，所以走得又快又自然。代价是：要先在仿真里练一个小时，"
                "而且它为什么这么迈步，没有人能一行一行地解释。",
                "Scaled by size, the gap is much smaller than in metres per second. Our hand-written controller "
                "plans every step before taking it, so it only dares to walk slowly; the reinforcement learning "
                "policy learned over thousands of falls to catch itself as it goes, so it walks faster and more "
                "naturally. The price: an hour of practice in simulation first, and nobody can explain line by "
                "line why it steps the way it does.",
            ),
            check=lambda ctx: "walk" in ctx.results,
            unlock=("walk",),
            show=("labels",),
        ),
        Step(
            id="push",
            title=_text("推一推", "Give them a shove"),
            body=_text(
                "点「开始推」：每 2.5 秒，两台被同样大小的侧向速度踢一下，一次比一次猛，倒了就出局。"
                "看谁扛过的级数多，再看倾斜曲线：谁晃得少、回得快。",
                "Press Start shoving: every 2.5 s both get the same sideways kick, harder each time; a fall puts a "
                "robot out. See who survives more levels, and in the tilt plot who sways less and recovers faster.",
            ),
            done=_text(
                "两台都只扛了几级，因为谁都没专门练过被推。我们的控制器只会用脚踝和髋关节把身体扭回来，不会临时迈步；"
                "G1 的策略训练时没被中途推过，只是每一局开始时带着随机的初速度（最多 0.5 米/秒），"
                "所以学会了从晃动里站稳，"
                "但没学会接住猛推。想让它更抗推，就在训练里加上随机推力：这叫域随机化，是让仿真里学到的本事在现实中也管用的关键。",
                "Neither lasts many levels, because neither practised being shoved. Our controller only twists its "
                "body back with its ankles and hips and never takes an unplanned step; the G1's policy was never "
                "pushed mid-run in training, it only started each run with a random velocity (up to 0.5 m/s), so it "
                "learned to steady itself from a wobble but not to catch a hard shove. To make it tougher, add random "
                "shoves to training: that is domain randomization, the key to skills learned in simulation working "
                "in the real world.",
            ),
            check=lambda ctx: "push" in ctx.results,
            unlock=("push",),
            show=("labels",),
        ),
        Step(
            id="how",
            title=_text("它是怎么学会的？", "How did it learn?"),
            body=_text("想一想 G1 的走路是怎么来的。", "Think about where the G1's walk came from."),
            done=_text("", ""),
            quiz=Quiz(
                question=_text("G1 的行走策略是怎么得到的？", "How did the G1 get its walking policy?"),
                options=(
                    _text("工程师写好了每一步怎么迈", "Engineers wrote down how to take every step"),
                    _text(
                        "在仿真里反复试错，做得好就加分", "Trial and error in simulation, scoring points for good moves"
                    ),
                    _text("看了很多人类走路的视频", "It watched lots of videos of people walking"),
                ),
                answer=1,
                explain=_text(
                    "对，这就是强化学习：奖励“按指令的速度走、身体不倒、动作别太猛”，几千个仿真机器人一起试，"
                    "网络慢慢学会了什么动作能拿高分。前面强化学习那一课的倒立摆，用的是同一个思路。",
                    "Right, that is reinforcement learning: reward walking at the commanded speed, staying upright and "
                    "moving smoothly, let thousands of simulated robots try at once, and the network slowly learns "
                    "which moves score well. The cart-pole in the reinforcement learning lesson used the same idea.",
                ),
            ),
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.facts: dict[str, dict[str, Any]] = {}
        self.missing: dict[str, str] = {}  # robot -> why it has no walking controller
        self.runs: dict[str, dict[str, Any]] = {}
        self.test: str | None = None  # "walk" or "push" while one runs
        self.samples: deque[dict[str, float]] = deque(maxlen=2048)
        self.next_sample = 0.0
        self.last_time = 0.0

    def base(self, index: int) -> tuple[float, float]:
        _, y = super().base(index)
        return START_X, y  # they walk toward +x, so the test happens in the middle of the view

    def camera(self) -> dict[str, Any]:
        # From above the near lane: the far lane (the tall G1) shows above our small humanoid, not behind it.
        return {"eye": [-0.2, -5.0, 3.2], "target": [-0.3, 0.0, 0.4], "fit": True}

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        return {
            "kind": "faceoff",
            "entrants": [
                {
                    **c,
                    "base": [*self.base(i), 0.0],
                    "body": root_body(model, c["name"]),
                    "top": round(self.facts.get(c["name"], {}).get("top", 1.0), 3),
                    "cloud": [],
                }
                for i, c in enumerate(self.columns())
            ],
            "targets": [],
        }

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.runs, self.test = {}, None
        for e in self.entrants:
            if e.name not in self.facts:
                self.facts[e.name] = self._facts(model, data, e.name)
            try:
                self.playground.behaviors.move(e.name, 0.0, 0.0)  # stand, balancing, from the first moment
            except BehaviorError as exc:
                self.missing[e.name] = str(exc)
        return {}

    def _facts(self, model: mujoco.MjModel, data: mujoco.MjData, name: str) -> dict[str, Any]:
        mujoco.mj_forward(model, data)
        bodies = robot_bodies(model, name)
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies]
        top = max(_geom_top(model, data, g) for g in geoms)
        root = root_body(model, name)
        return {
            "top": top,
            "height": top,  # standing on the floor, its highest point is its height
            "mass": float(sum(model.body_mass[b] for b in bodies)),
            "motors": sum(1 for a in range(model.nu) if (model.actuator(a).name or "").startswith(f"{name}/")),
            "leg": float(data.xpos[root][2]),  # hip height: the leg length the Froude number uses
            "root": root,
        }

    def _top_speed(self, name: str) -> float:
        if name == "g1":
            return G1_TEST_SPEED
        info = next((r for r in self.playground.behaviors.robots() if r["name"] == name), {})
        return float(info.get("max_speed") or 0.1)

    # ---- the tests ------------------------------------------------------------------------------

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        if control not in ("walk", "push") or self.test is not None:
            return
        self.test = control
        self.runs = {}
        for e in self.entrants:
            root = self.facts[e.name]["root"]
            run: dict[str, Any] = {"start": float(data.time), "end": None, "x0": float(data.xpos[root][0])}
            if e.name in self.missing:
                run.update(end=float(data.time), skipped=self.missing[e.name])
            elif control == "walk":
                run.update(speed=self._top_speed(e.name), energy=0.0, fell=False)
                self.playground.behaviors.move(e.name, run["speed"], 0.0)
            else:
                run.update(level=0, survived=0)
            self.runs[e.name] = run
        if control == "push":
            self.next_push = float(data.time) + 1.0
            self.level = 0
        self.last_time = float(data.time)

    def _tilt(self, data: mujoco.MjData, root: int) -> float:
        return math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[root][8])))))

    def _step_walk(self, name: str, run: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData) -> None:
        dt = float(data.time) - self.last_time
        acts = [a for a in range(model.nu) if (model.actuator(a).name or "").startswith(f"{name}/")]
        power = float(np.sum(np.abs(data.actuator_force[acts] * data.actuator_velocity[acts])))
        run["energy"] += power * max(0.0, dt)
        root = self.facts[name]["root"]
        run["fell"] = run["fell"] or float(data.xmat[root][8]) < FALLEN
        if data.time - run["start"] >= WALK_SECONDS or run["fell"]:
            run["end"] = float(data.time)
            run["distance"] = float(data.xpos[root][0]) - run["x0"]
            self.playground.behaviors.move(name, 0.0, 0.0)

    def _step_push(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for name, run in self.runs.items():
            if run["end"] is None and float(data.xmat[self.facts[name]["root"]][8]) < FALLEN:
                run["end"] = float(data.time)
                run["survived"] = max(0, self.level - 1)
        if data.time < self.next_push or all(r["end"] is not None for r in self.runs.values()):
            return
        if self.level >= PUSH_LEVELS:
            for run in self.runs.values():
                if run["end"] is None:
                    run["end"], run["survived"] = float(data.time), PUSH_LEVELS
            return
        self.level += 1
        side = 1.0 if self.level % 2 else -1.0
        for name, run in self.runs.items():
            if run["end"] is None:
                dof = int(model.jnt_dofadr[model.body_jntadr[self.facts[name]["root"]]])
                data.qvel[dof + 1] += side * PUSH_STEP * self.level  # the same sideways kick for both
                run["level"] = self.level
        self.next_push = float(data.time) + PUSH_EVERY

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        point: dict[str, float] = {"t": float(data.time)}
        for e in self.entrants:
            root = self.facts[e.name]["root"]
            point[e.name] = _forward_speed(model, data, root)
            point[f"{e.name}_tilt"] = self._tilt(data, root)
        if self.test == "walk":
            for name, run in self.runs.items():
                if run["end"] is None:
                    self._step_walk(name, run, model, data)
        elif self.test == "push":
            self._step_push(model, data)
        self.last_time = float(data.time)
        if data.time >= self.next_sample:
            self.next_sample = float(data.time) + 0.05
            self.samples.append(point)
        if self.test is not None and all(r["end"] is not None for r in self.runs.values()):
            ctx.results[self.test] = {name: self._result(name, run) for name, run in self.runs.items()}
            self.test = None
        return {"table": self.table(ctx), "readout": self._readout()}

    def _result(self, name: str, run: dict[str, Any]) -> dict[str, Any]:
        if "skipped" in run:
            return {"skipped": run["skipped"]}
        if "survived" in run:
            return {"levels": run["survived"]}
        f = self.facts[name]
        seconds = max(1e-6, run["end"] - run["start"])
        distance = max(0.0, run.get("distance", 0.0))
        speed = distance / seconds
        return {
            "distance": round(distance, 2),
            "speed": round(speed, 3),
            "heights_per_s": round(speed / f["height"], 2),
            "froude": round(speed**2 / (GRAVITY * f["leg"]), 3),
            "cost_of_transport": round(run["energy"] / (f["mass"] * GRAVITY * distance), 2)
            if distance > 0.05
            else None,
            "fell": run["fell"],
        }

    def _readout(self) -> Text:
        if self.test == "walk":
            return _text("正在走……", "Walking...")
        if self.test == "push":
            return _text(f"第 {self.level} 级", f"level {self.level}")
        return _text("", "")

    def drain_samples(self) -> list[dict[str, float]]:
        out = list(self.samples)
        self.samples.clear()
        return out

    def table(self, ctx: LessonContext) -> dict[str, Any]:
        names = [e.name for e in self.entrants]
        facts = [self.facts.get(n, {}) for n in names]
        rows: list[dict[str, Any]] = [
            {"heading": _text("规格（从模型算出）", "Specs (from the models)")},
            _row(_text("身高", "Height"), [f"{f.get('height', 0):.2f} m" for f in facts], None),
            _row(_text("质量", "Mass"), [f"{f.get('mass', 0):.1f} kg" for f in facts], None),
            _row(_text("电机", "Motors"), [str(f.get("motors", 0)) for f in facts], None),
            _row(_text("走路的控制器", "Walking controller"), [self._controller(n) for n in names], None),
            _row(_text("来源", "Source"), [_source(e) for e in self.entrants], None),
        ]
        walk = ctx.results.get("walk")
        if walk:
            got = [walk[n] for n in names]
            ok = ["skipped" not in r for r in got]
            rows += [
                {"heading": _text(f"行走测试（{WALK_SECONDS:g} 秒）", f"Walking test ({WALK_SECONDS:g} s)")},
                _row(
                    _text("走了多远", "Distance"),
                    [f"{r['distance']:.2f} m" if k else "-" for r, k in zip(got, ok, strict=True)],
                    None,
                ),
                _row(
                    _text("速度", "Speed"),
                    [f"{r['speed']:.2f} m/s" if k else "-" for r, k in zip(got, ok, strict=True)],
                    [r["speed"] if k else -1 for r, k in zip(got, ok, strict=True)],
                ),
                _row(
                    _text("每秒几个身高", "Body heights per second"),
                    [f"{r['heights_per_s']:.2f}" if k else "-" for r, k in zip(got, ok, strict=True)],
                    [r["heights_per_s"] if k else -1 for r, k in zip(got, ok, strict=True)],
                ),
                _row(
                    _text("弗劳德数", "Froude number"),
                    [f"{r['froude']:.3f}" if k else "-" for r, k in zip(got, ok, strict=True)],
                    None,
                ),
                _row(
                    _text("每米能耗", "Energy per metre"),
                    [_cot(r) if k else "-" for r, k in zip(got, ok, strict=True)],
                    [
                        r["cost_of_transport"] if k and r["cost_of_transport"] is not None else math.inf
                        for r, k in zip(got, ok, strict=True)
                    ],
                    high=False,
                ),
                _row(
                    _text("摔倒", "Fell"),
                    [
                        (_text("摔了", "yes") if r["fell"] else _text("没有", "no")) if k else "-"
                        for r, k in zip(got, ok, strict=True)
                    ],
                    None,
                ),
            ]
        push = ctx.results.get("push")
        if push:
            got = [push[n] for n in names]
            rows += [
                {"heading": _text("推的测试", "Shove test")},
                _row(
                    _text("扛过的级数", "Levels survived"),
                    [str(r["levels"]) if "levels" in r else "-" for r in got],
                    [r.get("levels", -1) for r in got],
                ),
            ]
        return {"columns": self.columns(), "rows": rows}

    def _controller(self, name: str) -> Text:
        if name in self.missing:
            return _text(
                "没有行走策略（用 UniLab 训练 g1_walk_flat）", "no walking policy (train g1_walk_flat in UniLab)"
            )
        if name == "g1":
            return _text("强化学习策略（UniLab PPO）", "reinforcement learning policy (UniLab PPO)")
        return _text("ZMP 预观控制（手写）", "ZMP preview control (hand-written)")


def _forward_speed(model: mujoco.MjModel, data: mujoco.MjData, root: int) -> float:
    """The root body's velocity along its own forward axis (its x), in m/s."""
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root, vel, 0)  # world frame
    forward = data.xmat[root].reshape(3, 3)[:, 0].copy()
    forward[2] = 0.0
    norm = float(np.linalg.norm(forward))
    return float(vel[3:] @ forward / norm) if norm > 1e-6 else 0.0


def _geom_top(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> float:
    """Highest world z of a geom's bounding box (tighter than its bounding sphere)."""
    center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + center
    world = corners @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    return float(world[:, 2].max())


def _cot(result: dict[str, Any]) -> Text | str:
    if result["cost_of_transport"] is None:
        return "-"
    return f"{result['cost_of_transport']:.2f}"


def _source(entrant: Entrant) -> Text:
    return _text("真机模型", "real robot model") if entrant.real else _text("我们的模板", "our template")


def _joints(facts: dict[str, Any]) -> Text:
    n = facts.get("joints", 0)
    return _text(f"{n} + 夹爪", f"{n} + gripper")


def _holds_up(facts: dict[str, Any], switched_on: bool) -> Text:
    if facts.get("gravcomp"):
        return _text("重力补偿（控制器自带）", "gravity compensation (built in)")
    if switched_on:
        return _text("重力补偿（你打开的）", "gravity compensation (you turned it on)")
    if facts.get("kp", 0) >= STIFF_SERVO:
        return _text(f"很硬的电机（kp {facts['kp']:.0f}）", f"very stiff motors (kp {facts['kp']:.0f})")
    return _text(f"只有 P 控制（kp {facts.get('kp', 0):.0f}）", f"P control only (kp {facts.get('kp', 0):.0f})")


def _row(label: Text, cells: list[Any], values: list[float] | None, high: bool = True) -> dict[str, Any]:
    """One table row; `values` marks the best cell (highest, or lowest with high=False) unless all tie."""
    best = None
    if values is not None:
        target = max(values) if high else min(values)
        if math.isfinite(target) and values.count(target) < len(values):
            best = values.index(target)
    return {"label": label, "cells": cells, "best": best}
