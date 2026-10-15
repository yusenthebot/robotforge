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
                    "top": round(self.facts.get(c["name"], {}).get("top", 0.5), 3),
                    "cloud": self.facts.get(c["name"], {}).get("cloud", []),
                }
                for i, c in enumerate(self.columns())
            ],
            "targets": [self.targets(i) for i in range(len(self.entrants))],
        }

    def camera(self) -> dict[str, list[float]]:
        # From the front and to the side of our (smallest) arm, so the row foreshortens between the panels.
        return {"eye": [3.15, -2.85, 1.9], "target": [-0.11, -0.1, 0.25]}

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
