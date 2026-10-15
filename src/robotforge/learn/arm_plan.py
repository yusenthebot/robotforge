"""Lesson: motion planning. Getting an arm round an obstacle with RRT-Connect, then smoothing the path.

Our six-joint template arm and the AgileX PiPER each have a fence in front of them and a target on either
side of it. "Straight" turns every joint at once from one target's configuration to the other's: the hand
sweeps through the fence. "Plan" grows two trees of collision-free configurations (behaviors/motion_plan.py)
until they meet, optionally shortcuts the path, then the arm follows it. The trees, the path and the
straight sweep are drawn at the hand; every run lands in the record as the hand's distance to the goal.
"""

from __future__ import annotations

import copy
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors import motion_plan as mp
from robotforge.behaviors.arm import ArmMotion
from robotforge.design.world import World, compile_world
from robotforge.learn.engine import Bench, Control, Lesson, LessonContext, Metric, Plot, Quiz, Series, Step, Text

ARMS = ("ours", "piper")
LANE = 0.7  # m from the middle to each arm's base
JOINT_SPEED = 1.2  # rad/s: how fast the fastest joint turns along a path
# The path's clock stops while any joint is further than this behind it: hobby servos lag under load (up
# to 47 degrees on a fast base turn), and a lagging arm leaves the planned, collision-free corridor.
MAX_LAG = 0.1  # rad
STEPS_RAD = (0.1, 0.2, 0.4, 0.8)  # the longest hop a tree may make, rad
SAMPLE_DT = 0.02
SETTLE = 0.6  # s the arm is given after the path ends, to come to rest on the target
SETUP = {
    # target either side of a fence (x, y, z relative to the arm's base) and the fence (centre x, length, height)
    "ours": {"target": (0.25, 0.2, 0.05), "fence": (0.40, 0.24, 0.18)},
    "piper": {"target": (0.30, 0.25, 0.08), "fence": (0.48, 0.24, 0.25)},
}


def _text(zh: str, en: str) -> Text:
    return {"zh": zh, "en": en}


def _runs(ctx: LessonContext, **want: Any) -> list[dict[str, Any]]:
    return [r for r in ctx.runs if all(test(r["params"].get(k, 0.0)) for k, test in want.items())]


def _hit_it(ctx: LessonContext) -> bool:
    return any(r["metrics"]["hit"] for r in _runs(ctx, planned=lambda v: v == 0))


def _went_round(ctx: LessonContext) -> bool:
    return any(not r["metrics"]["hit"] for r in _runs(ctx, planned=lambda v: v == 1))


def _smoothing_compared(ctx: LessonContext) -> bool:
    planned = _runs(ctx, planned=lambda v: v == 1, arm=lambda v: v == 0)
    return {r["params"]["smooth"] for r in planned} >= {0.0, 1.0}


def _steps_compared(ctx: LessonContext) -> bool:
    planned = _runs(ctx, planned=lambda v: v == 1, arm=lambda v: v == 0)
    return len({r["params"]["step"] for r in planned}) >= 2


def _piper_planned(ctx: LessonContext) -> bool:
    return any(not r["metrics"]["hit"] for r in _runs(ctx, planned=lambda v: v == 1, arm=lambda v: v == 1))


STEPS = (
    Step(
        id="straight",
        title=_text("直接转过去", "Straight across"),
        body=_text(
            "机械臂的手在挡板左边，要去右边的黄球。最省事的走法是让每个关节同时、匀速地从现在的角度转到目标角度。\n"
            "先猜，再点「直接走」。",
            "The hand is left of the fence and has to get to the amber ball on the right. The easiest move turns "
            "every joint at once, at an even pace, from where it is to where the target needs it.\n"
            "Guess, then press Straight.",
        ),
        done=_text(
            "撞上了。关节角是直线变化的，手走的却是一条弧，正好扫过挡板（紫色那条线，红点是撞上的地方）。"
            "关节空间里的直线，在真实空间里不一定是直线，更不知道哪里有东西挡着。",
            "It hits. The joint angles change in a straight line, but the hand moves on an arc that sweeps "
            "straight through the fence (the violet line; the red dot is where it hits). A straight line in "
            "joint space is no straight line in the room, and it knows nothing of what is in the way.",
        ),
        quiz=Quiz(
            question=_text("直接转过去，会撞到挡板吗？", "Straight across: does it hit the fence?"),
            options=(_text("不会，手会自己绕开", "No, the hand goes round"), _text("会", "Yes")),
            answer=1,
            explain=_text("它不知道挡板在那里。", "It does not know the fence is there."),
            predict=True,
        ),
        check=_hit_it,
        unlock=("straight",),
        show=("labels", "straight", "f_straight"),
    ),
    Step(
        id="rrt",
        title=_text("两棵树：随机撒点", "Two trees of random tries"),
        body=_text(
            "规划器从起点和终点各长一棵树。每一轮随便挑一个关节角组合，让一棵树朝它迈一小步（中途不碰东西才算），"
            "再让另一棵树朝新长出来的点一路追过去。两棵树接上，路就找到了。蓝色和紫色的线是两棵树上每个点对应的手的位置。\n"
            "先猜，再点「规划并走」。",
            "The planner grows a tree from the start and one from the goal. Each round it picks any set of joint "
            "angles at random, grows one tree a small step toward it (only if nothing is touched on the way), "
            "then lets the other tree chase the new node as far as it can. When the trees join, there is a path. "
            "The blue and violet lines show where the hand would be at every node of each tree.\n"
            "Guess, then press Plan and go.",
        ),
        done=_text(
            "十几毫秒就找到了（我们测到 8 到 23 毫秒，树上 25 到 73 个点），手从挡板旁边绕了过去，一下都没碰。"
            "规划时离挡板留了 4.5 厘米：真舵机总会比计划慢一点、偏一点，贴着走就会擦上。"
            "这就是 RRT-Connect，工业机械臂和 MoveIt 里最常用的规划器之一。",
            "It found a way in a dozen milliseconds (8 to 23 ms in our runs, 25 to 73 nodes), and the hand went "
            "round the fence without touching it. The plan keeps 4.5 cm from the fence: real servos always lag "
            "and stray a little, and a path that only just clears it would scrape. This is RRT-Connect, one of the "
            "planners most used on industrial arms and in MoveIt.",
        ),
        quiz=Quiz(
            question=_text("规划器要多久才能找到路？", "How long does the planner take to find a way?"),
            options=(
                _text("几十毫秒以内", "Under a tenth of a second"),
                _text("几秒", "A few seconds"),
                _text("几分钟", "Minutes"),
            ),
            answer=0,
            explain=_text("每次碰撞检查只要几微秒，几百次也很快。", "Each collision check takes microseconds."),
            predict=True,
        ),
        check=_went_round,
        unlock=("plan",),
        show=("trees", "path", "f_plan"),
    ),
    Step(
        id="smooth",
        title=_text("抄近路", "Taking shortcuts"),
        body=_text(
            "随机长出来的路弯弯绕绕。平滑的办法很简单：在路上随便挑两个点，如果它们之间能直接连上又不碰东西，就把中间那段删掉；"
            "重复 60 次。\n先猜，再分别关掉和打开「抄近路」各规划一次，比比关节总共转了多少。",
            "A path grown at random zigzags. Smoothing is simple: pick two random points on the path, and if they "
            "can be joined directly without touching anything, cut out what lies between; repeat 60 times.\n"
            "Guess, then plan once with Shortcuts off and once with them on, and compare how far the joints turn "
            "in total.",
        ),
        done=_text(
            "抄近路以后，关节总共转的角度少了两到五成（我们规划了 8 次，省下 18% 到 48%），手走得也更快、更顺。"
            "真实的规划器都会在找到路以后再加一道平滑。",
            "With shortcuts the joints turn 20 to 50% less in total (18 to 48% over 8 of our plans), and the move "
            "is quicker and smoother. Real planners always smooth a path once they find one.",
        ),
        quiz=Quiz(
            question=_text("抄近路以后，路径会短多少？", "How much shorter does smoothing make the path?"),
            options=(
                _text("差不多", "About the same"),
                _text("短两到五成", "20 to 50% shorter"),
                _text("短九成", "90% shorter"),
            ),
            answer=1,
            explain=_text("随机路径绕的弯很多，能删掉不少。", "Random paths take many detours, and many can go."),
            predict=True,
        ),
        check=_smoothing_compared,
        unlock=("smooth",),
    ),
    Step(
        id="step",
        title=_text("每一步迈多大", "How big a step"),
        body=_text(
            "「每步最多转多少」是树每次能伸出去多远。"
            "先猜，再分别用 0.1 和 0.8 弧度规划，看树上有多少个点、原始路径有多长。",
            "Step size is how far a tree may reach at a time. Guess, then plan with 0.1 and with 0.8 rad, and look "
            "at how many nodes the trees have and how long the raw path is.",
        ),
        done=_text(
            "步子小，树上的点多（0.1 弧度时我们测到 69 到 145 个），路走得细；"
            "步子大，几个点就连上了（0.8 弧度时 8 到 13 个），"
            "可原始路径常常更绕，靠抄近路救回来。步子太大还会跨过窄缝里本该绕开的地方，所以检查碰撞时要在每一段上细细地查。",
            "Small steps grow many nodes (69 to 145 at 0.1 rad in our runs) and a fine path; big steps join up in "
            "a handful (8 to 13 at 0.8 rad) but the raw path often detours more, and shortcuts rescue it. Steps "
            "that are too big would "
            "also jump past tight spots, which is why every segment is checked finely for collisions.",
        ),
        quiz=Quiz(
            question=_text("步子从 0.1 加到 0.8，树上的点会？", "From 0.1 to 0.8 rad steps, the trees get?"),
            options=(_text("更多", "More nodes"), _text("更少", "Fewer nodes"), _text("一样", "The same")),
            answer=1,
            explain=_text("每步走得远，几步就能接上。", "Longer hops join up in fewer steps."),
            predict=True,
        ),
        check=_steps_compared,
        unlock=("step",),
    ),
    Step(
        id="piper",
        title=_text("换成真机：PiPER", "Now the PiPER"),
        body=_text(
            "「哪条机械臂」选 PiPER，它前面也有一块挡板。规划器一行都不用改，先直接走一次，再规划一次。",
            "Set Which arm to the PiPER; it has a fence in front too. The planner needs no change at all: go "
            "straight once, then plan.",
        ),
        done=_text(
            "同一个规划器，PiPER 也绕过去了。规划器只需要两样东西：关节有多少、各能转多少，以及“这个姿势撞没撞”。"
            "所以它对任何机械臂都管用。",
            "The same planner gets the PiPER round too. A planner needs only two things: the joints and their "
            "ranges, and an answer to 'does this pose hit anything'. That is why it works for any arm.",
        ),
        check=_piper_planned,
        unlock=("arm",),
    ),
    Step(
        id="quiz",
        title=_text("想一想", "Think about it"),
        body=_text(
            "为什么不把每个关节的角度都切成 100 份，挨个试一遍？",
            "Why not cut every joint's range into 100 steps and try them all?",
        ),
        done=_text("", ""),
        quiz=Quiz(
            question=_text("六个关节各 100 份，一共要试多少个？", "Six joints, 100 steps each: how many to try?"),
            options=(_text("600 个", "600"), _text("一百万个", "A million"), _text("一万亿个", "A million million")),
            answer=2,
            explain=_text(
                "100 的 6 次方是一万亿。维度一高，挨个试就不可能了，随机撒点只要几十上百次。",
                "100 to the 6th is a million million. In many dimensions trying everything is hopeless; random "
                "tries need tens to hundreds.",
            ),
        ),
    ),
)


class PathFollower:
    """Moves an arm along a list of joint configurations, the fastest joint at JOINT_SPEED, waiting
    whenever the arm falls more than MAX_LAG behind."""

    rate_hz = 100.0

    def __init__(self, actuators: np.ndarray, qadr: np.ndarray, q: np.ndarray) -> None:
        self.actuators = actuators
        self.qadr = qadr
        self.path: list[np.ndarray] = [q]
        self.index, self.progress = 0, 0.0
        self.last_t: float | None = None

    def follow(self, path: list[np.ndarray]) -> None:
        self.path, self.index, self.progress, self.last_t = path, 0, 0.0, None

    @property
    def done(self) -> bool:
        return self.index >= len(self.path) - 1

    def target(self) -> np.ndarray:
        if self.done:
            return self.path[-1]
        a, b = self.path[self.index], self.path[self.index + 1]
        return a + (b - a) * self.progress

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        dt = 0.0 if self.last_t is None else max(0.0, float(data.time) - self.last_t)
        self.last_t = float(data.time)
        budget = JOINT_SPEED * dt  # this tick's turn of the fastest joint
        if float(np.max(np.abs(data.qpos[self.qadr] - self.target()))) > MAX_LAG:
            budget = 0.0  # let the arm catch up before the path moves on
        while budget > 0 and not self.done:
            a, b = self.path[self.index], self.path[self.index + 1]
            hop = float(np.max(np.abs(b - a))) or 1e-9
            left = (1.0 - self.progress) * hop
            if budget >= left:
                budget -= left
                self.index, self.progress = self.index + 1, 0.0
            else:
                self.progress += budget / hop
                budget = 0.0
        data.ctrl[self.actuators] = self.target()

    def status(self) -> dict[str, Any]:
        return {"controller": "path", "done": self.done}


class Watch:
    """Looks for the arm touching anything while it moves, and samples the hand's distance to the goal."""

    rate_hz = 500.0

    def __init__(self, lesson: ArmPlanLesson) -> None:
        self.lesson = lesson
        self.next_sample = 0.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.lesson.measure(model, data)
        if data.time >= self.next_sample:
            self.next_sample = data.time + SAMPLE_DT
            self.lesson.sample(data)

    def status(self) -> dict[str, Any]:
        return {"controller": "arm-plan-watch"}


class ArmPlanLesson(Lesson):
    id = "arm_planning"
    chapter = _text("运动学", "Kinematics")
    title = _text("运动规划：机械臂绕开挡板", "Motion planning: getting an arm round an obstacle")
    summary = _text(
        "直接转过去会撞；RRT-Connect 长两棵随机树找路，再抄近路平滑。先用我们的六轴臂，再换 PiPER。",
        "Straight across hits; RRT-Connect grows two random trees to find a way, then shortcuts smooth it. "
        "Our six-joint arm first, then the PiPER.",
    )
    minutes = 12
    controls = (
        Control("straight", "button", _text("直接走", "Straight")),
        Control("plan", "button", _text("规划并走", "Plan and go")),
        Control("smooth", "toggle", _text("抄近路（平滑）", "Shortcuts (smoothing)"), value=1.0),
        Control(
            "step",
            "choice",
            _text("每步最多转多少", "Step size"),
            options=tuple(_text(f"{s:g} rad", f"{s:g} rad") for s in STEPS_RAD),
            value=1.0,
        ),
        Control(
            "arm",
            "choice",
            _text("哪条机械臂", "Which arm"),
            options=(_text("我们的六轴臂", "Our six-joint arm"), _text("AgileX PiPER", "AgileX PiPER")),
        ),
    )
    plots = (
        Plot(
            "goal",
            _text("手离目标多远", "Hand to target"),
            "m",
            (Series("distance", _text("距离", "distance"), "ink"),),
            window=6.0,
        ),
    )
    steps = STEPS
    bench = Bench(
        title=_text("实验记录：手离目标多远", "Experiment record: hand to target"),
        x=_text("开始走后", "since the start"),
        x_unit="s",
        y=_text("距离", "distance"),
        y_unit="m",
        params=("arm", "planned", "smooth", "step"),
        short=(_text("机械臂", "arm"), _text("规划", "planned"), _text("抄近路", "shortcuts"), _text("步长", "step")),
        metrics=(
            Metric("hit", _text("撞了", "hit"), "", ranked=False),
            Metric("nodes", _text("树上的点", "nodes"), "", ranked=False),
            Metric("plan_ms", _text("规划", "planning"), "ms", ranked=False),
            Metric("turned", _text("关节总转角", "joints turned"), "rad"),
        ),
    )

    def __init__(self) -> None:
        self.ctx: LessonContext | None = None
        self.model: mujoco.MjModel | None = None
        self.arms: dict[str, ArmMotion] = {}
        self.spaces: dict[str, mp.ArmSpace] = {}
        self.ends: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # (start q, goal q) per arm
        self.goals: dict[str, np.ndarray] = {}  # the goal targets in the world
        self.followers: dict[str, PathFollower] = {}
        self.fences: dict[str, int] = {}  # the fence's geom per arm
        self.run: dict[str, Any] | None = None
        self.drawing: dict[str, Any] | None = None
        self.drawing_version = 0
        self.drawing_sent = -1
        self.samples: list[dict[str, float]] = []

    # ---- the world -------------------------------------------------------------------------------------

    def build(self) -> mujoco.MjModel:
        robots, objects = [], []
        for name, y in zip(ARMS, (-LANE, LANE), strict=True):
            entry = (
                {"template": "arm", "params": {"dof": 6, "color": "orange"}} if name == "ours" else {"library": "piper"}
            )
            robots.append({"name": name, **entry, "pos": [0.0, y]})
            x, length, height = SETUP[name]["fence"]
            objects.append(
                {
                    "name": f"fence_{name}",
                    "shape": "box",
                    "size": [length, 0.03, height],
                    "pos": [x, y],
                    "fixed": True,
                    "color": "blue",
                }
            )
        world = World.model_validate(
            {"name": "arm_planning", "floor": {"style": "concrete"}, "robots": robots, "objects": objects}
        )
        self.model = compile_world(world).model
        return self.model

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.ctx = ctx
        home = model.key("home").qpos.copy()
        private = copy.deepcopy(model)  # planning and IK never touch the live world
        controllers: dict[str, Any] = {}
        for name, y in zip(ARMS, (-LANE, LANE), strict=True):
            arm = ArmMotion(private, name, home)
            self.arms[name] = arm
            self.spaces[name] = mp.ArmSpace(private, arm.chain, home)
            tx, ty, tz = SETUP[name]["target"]
            start_q, _, _ = arm._solve(np.array([tx, y + ty, tz]))
            arm.q = arm.home.copy()
            goal_q, _, _ = arm._solve(np.array([tx, y - ty, tz]))
            self.ends[name] = (start_q, goal_q)
            self.goals[name] = np.array([tx, y - ty, tz])
            self.fences[name] = int(model.geom(f"fence_{name}").id)  # a fixed prop is a geom on the world
            data.qpos[arm.chain.qpos_adr] = start_q  # every arm starts beside its fence, the target over the top
            data.qvel[arm.chain.dof_adr] = 0.0
            self.followers[name] = PathFollower(arm.actuators, arm.chain.qpos_adr, start_q)
            controllers[name] = self.followers[name]
        # Both arms compensate gravity through their motors, as a real arm's controller does: hobby servos
        # alone sag about 5 degrees under the arm's weight, enough to leave the planned corridor. (MuJoCo
        # switches this at run time because the PiPER's model compiled with gravity compensation.)
        for name in ARMS:
            bodies = [b for b in range(model.nbody) if model.body(b).name.startswith(f"{name}/")]
            model.body_gravcomp[bodies] = 1.0
            model.jnt_actgravcomp[[j for j in range(model.njnt) if model.jnt_bodyid[j] in bodies]] = 1
        mujoco.mj_forward(model, data)
        self.run = None
        self._draw(straight=None)
        return {**controllers, "watch": Watch(self)}

    @property
    def active(self) -> str:
        assert self.ctx is not None
        return ARMS[int(self.ctx.values["arm"])]

    # ---- runs ------------------------------------------------------------------------------------------

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        if control not in ("straight", "plan") or (self.run is not None and not self.run.get("finished")):
            if control == "arm":
                self._draw(straight=None)
            return
        name = self.active
        start, goal = self.ends[name]
        space = self.spaces[name]
        arm = self.arms[name]
        model = self.model
        assert model is not None
        data.qpos[arm.chain.qpos_adr] = start  # back beside the fence, still
        data.qvel[arm.chain.dof_adr] = 0.0
        mujoco.mj_forward(model, data)
        settings = {
            "arm": float(ctx.values["arm"]),
            "planned": 1.0 if control == "plan" else 0.0,
            "smooth": float(ctx.values["smooth"]),
            "step": float(ctx.values["step"]),
        }
        clash = space.first_clash(start, goal)
        info: dict[str, Any] = {"nodes": 0, "plan_ms": 0.0}
        if control == "straight":
            path = [start, goal]
            self._draw(straight=(start, goal, clash))
        else:
            plan = mp.rrt_connect(space, start, goal, step=STEPS_RAD[int(settings["step"])], seed=ctx.run_count + 1)
            info = {"nodes": plan.nodes, "plan_ms": round(plan.seconds * 1000, 1), "raw": None}
            if plan.path is None:
                ctx.results["summary"] = _text(
                    f"{plan.nodes} 个点也没接上，换个步长再试", f"No way found in {plan.nodes} nodes: try another step"
                )
                return
            path = plan.path
            info["raw"] = round(mp.joint_length(path), 2)
            if settings["smooth"]:
                path = mp.shortcut(space, path, seed=ctx.run_count + 1)
            self._draw(straight=(start, goal, clash), plan=plan, path=path)
        self.followers[name].follow(mp.resample(path, 0.02))
        self.run = {
            "arm": name,
            "t0": float(data.time),
            "settings": settings,
            "info": info,
            "turned": round(mp.joint_length(path), 2),
            "hit": False,
            "times": [],
            "distances": [],
            "finished": False,
        }

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        run = self.run
        if run is None or run["finished"]:
            return
        moving = self.spaces[run["arm"]].moving
        fence = self.fences[run["arm"]]
        for con in data.contact[: data.ncon]:
            g1, g2 = int(con.geom1), int(con.geom2)
            if fence in (g1, g2) and int(model.geom_bodyid[g2 if g1 == fence else g1]) in moving:
                run["hit"] = True
                run.setdefault("hit_at", float(data.time))
        follower = self.followers[run["arm"]]
        if follower.done and run.get("arrived") is None:
            run["arrived"] = float(data.time)
        settled = run.get("arrived") is not None and float(data.time) - run["arrived"] > SETTLE
        # Straight into the fence the arm only pushes on (the path waits for it): a second of that is enough.
        stuck = run.get("hit_at") is not None and float(data.time) - run["hit_at"] > 1.0
        if settled or stuck or float(data.time) - run["t0"] > 12.0:
            self._finish()

    def _finish(self) -> None:
        run = self.run
        assert run is not None and self.ctx is not None
        run["finished"] = True
        info = run["info"]
        metrics = {
            "hit": float(run["hit"]),
            "nodes": float(info["nodes"]),
            "plan_ms": info["plan_ms"],
            "turned": run["turned"],
        }
        self.ctx.results["arm_run"] = {**metrics, "raw": info.get("raw"), "arm": run["arm"]}
        hit = _text("撞上了挡板", "hit the fence") if run["hit"] else _text("一下都没碰", "never touched it")
        if run["settings"]["planned"]:
            raw = info.get("raw")
            smoothed = (
                _text(f"，抄近路前 {raw:g} rad", f", {raw:g} rad before shortcuts")
                if raw and run["settings"]["smooth"]
                else _text("", "")
            )
            self.ctx.results["summary"] = {
                "zh": f"规划 {info['plan_ms']:g} ms、树上 {info['nodes']} 个点；"
                f"关节共转 {run['turned']:g} rad{smoothed['zh']}；{hit['zh']}",
                "en": f"Planned in {info['plan_ms']:g} ms with {info['nodes']} nodes; "
                f"joints turned {run['turned']:g} rad{smoothed['en']}; {hit['en']}",
            }
        else:
            self.ctx.results["summary"] = {"zh": f"直接走：{hit['zh']}", "en": f"Straight: {hit['en']}"}
        self.ctx.record_run(run["settings"], metrics, run["times"], run["distances"])

    def sample(self, data: mujoco.MjData) -> None:
        name = self.run["arm"] if self.run is not None else self.active
        hand = data.site_xpos[self.arms[name].site]
        distance = float(np.linalg.norm(hand - self.goals[name]))
        run = self.run
        if run is not None and not run["finished"]:
            run["times"].append(round(float(data.time) - run["t0"], 3))
            run["distances"].append(round(distance, 4))
        self.samples.append({"t": float(data.time), "distance": distance})

    def drain_samples(self) -> list[dict[str, float]]:
        out, self.samples = self.samples, []
        return out

    # ---- what is drawn ----------------------------------------------------------------------------------

    def _draw(
        self,
        straight: tuple[np.ndarray, np.ndarray, float | None] | None,
        plan: mp.Plan | None = None,
        path: list[np.ndarray] | None = None,
    ) -> None:
        """Hand positions for the overlay: the straight sweep (and where it hits), both trees, the path."""
        name = self.active if self.ctx is not None else ARMS[0]
        space = self.spaces[name]
        out: dict[str, Any] = {"arm": name, "goal": self.goals[name].round(3).tolist()}
        if straight is not None:
            a, b, clash = straight
            sweep = [space.tool(a + (b - a) * (k / 40)).round(3).tolist() for k in range(41)]
            out["straight"] = sweep
            if clash is not None:
                out["clash"] = space.tool(a + (b - a) * clash).round(3).tolist()
        if plan is not None:
            out["trees"] = [
                [
                    [space.tool(t.nodes[p]).round(3).tolist(), space.tool(q).round(3).tolist()]
                    for q, p in zip(t.nodes, t.parents, strict=True)
                    if p >= 0
                ]
                for t in plan.trees
            ]
        if path is not None:
            out["path"] = [space.tool(q).round(3).tolist() for q in mp.resample(path, 0.05)]
        self.drawing = out
        self.drawing_version += 1

    def describe(self, model: mujoco.MjModel) -> dict[str, Any]:
        self.drawing_sent = -1  # a viewer that (re)connects gets the drawing with its next update
        return super().describe(model)

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        out: dict[str, Any] = {"view": self.view(self.active), "formula": self._formula(ctx)}
        if self.drawing_version != self.drawing_sent:  # the drawing goes out only when it changed
            self.drawing_sent = self.drawing_version
            out["drawing"] = self.drawing
        if self.run is not None and not self.run["finished"]:
            out["readout"] = _text("在走……", "Moving...")
        return out

    def _formula(self, ctx: LessonContext) -> list[dict[str, Any]]:
        last = ctx.results.get("arm_run", {})
        return [
            {
                "show": "f_straight",
                "label": _text("直接走", "Straight"),
                "expr": _text(
                    "每个关节 q = 起点 + (终点 − 起点) × s，s 从 0 到 1",
                    "every joint q = start + (goal - start) * s, s from 0 to 1",
                ),
                "value": "",
                "color": "violet",
            },
            {
                "show": "f_plan",
                "label": _text("上次规划", "Last plan"),
                "expr": _text("树上的点 · 用时", "nodes · time"),
                "value": f"{int(last.get('nodes', 0))} · {last.get('plan_ms', 0):g} ms",
                "color": "blue",
            },
            {
                "show": "f_plan",
                "label": _text("关节总转角", "Joints turned"),
                "expr": _text("路径上每一步关节变化加起来", "every hop's joint change, added up"),
                "value": f"{last.get('turned', 0):g} rad",
                "color": "amber",
            },
        ]

    @staticmethod
    def view(name: str) -> dict[str, list[float]]:
        y = -LANE if name == "ours" else LANE
        far = 1.0 if name == "ours" else 1.3
        # From above and in front of the fence: both sides of it, and the hand's way round, in view.
        return {"eye": [0.3 + 0.55 * far, y - 0.5 * far, 1.0 * far], "target": [0.28, y, 0.05]}

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        return {
            "kind": "plan",
            "entrants": [
                {
                    "name": name,
                    "label": label,
                    "color": color,
                    "real": name == "piper",
                    "base": [0.0, y, 0.0],
                    "top": top,
                    "cloud": [],
                }
                for name, label, color, y, top in (
                    ("ours", _text("我们的六轴臂", "Our six-joint arm"), "amber", -LANE, 0.5),
                    ("piper", _text("AgileX PiPER", "AgileX PiPER"), "blue", LANE, 0.6),
                )
            ],
            "targets": [],
        }

    def camera(self) -> dict[str, list[float]]:
        return self.view("ours")
