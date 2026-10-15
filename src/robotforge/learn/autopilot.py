"""Sensing II: a robot vacuum that drives itself on its lidar, without bumping into things.

Two identical arenas side by side, a rover in each, the same simple rule: if anything is closer ahead than
the stop distance, turn toward the more open side; otherwise drive on. Each rover has its own stop
distance, speed, sensor noise and filter, and the floor it has covered is painted in its colour. The
lesson's numbers come from measurements (see progress.md, round 16).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import mujoco
import numpy as np

from robotforge.learn import sensors
from robotforge.learn.engine import Control, LessonContext, Plot, Quiz, Series, Step
from robotforge.learn.twins import Knob, Twin, TwinsLesson, Watch, _num, _row, _text

OFFSET = 1.75  # m from the middle to each arena's centre, A in front and B behind
# The space between the lesson panels is taller than wide, so the arenas stack front to back.
CENTRES = ((0.0, -OFFSET), (0.0, OFFSET))
CELL = 0.25  # m: the floor is counted in squares this size
CELLS = round(2 * sensors.ARENA / CELL)  # per side
RUN_SECONDS = 30.0
PILOT_HZ = 50.0
TURN_TICKS = 10  # once it starts turning away, it keeps turning for this many ticks (0.2 s)
FILTER = 5  # samples in the median filter
FRONT = (0, 1, 2, sensors.RAYS - 2, sensors.RAYS - 1)  # rays within 30 degrees of straight ahead
LEFT = range(2, 7)
RIGHT = range(sensors.RAYS - 7, sensors.RAYS - 2)
BUMP_GAP = 0.5  # s: touches closer together than this are one bump

STOPS = (0.15, 0.3, 0.6, 0.9)
SPEEDS = (0.15, 0.3, 0.45)
NOISES = (0.0, 0.1, 0.3)

ROVERS = (
    Twin("a", _text("A", "A"), "amber", {"stop": 0, "speed": 1, "noise": 0, "filter": 0}),
    Twin("b", _text("B", "B"), "blue", {"stop": 1, "speed": 1, "noise": 0, "filter": 0}),
)


class Pilot:
    """The rule, 50 times a second: read the rays (with noise, maybe filtered), then go or turn away."""

    rate_hz = PILOT_HZ

    def __init__(self, lesson: AutopilotLesson, twin: Twin, rover: sensors.Rover, seed: int) -> None:
        self.lesson = lesson
        self.twin = twin
        self.rover = rover
        self.rng = np.random.default_rng(seed)
        self.history: deque[np.ndarray] = deque(maxlen=FILTER)
        self.turning = 0
        self.direction = 1.0
        self.front = 4.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        s = self.lesson.settings(self.twin)
        rays = np.array(self.rover.ranges(data))
        rays = np.where(rays < 0, 4.0, rays) + self.rng.normal(0.0, s["noise"], rays.size)
        self.history.append(rays)
        if s["filter"]:
            rays = np.median(np.array(self.history), axis=0)  # a lone wild reading cannot fool it
        self.front = float(min(rays[i] for i in FRONT))
        if not self.lesson.running:
            self.rover.speed = self.rover.turn = 0.0
            return
        if self.turning > 0 or self.front < s["stop"]:
            if self.turning <= 0:
                self.turning = TURN_TICKS
                left, right = np.mean(rays[list(LEFT)]), np.mean(rays[list(RIGHT)])
                self.direction = 1.0 if left >= right else -1.0
            self.rover.speed, self.rover.turn = 0.0, self.direction * 0.8 * sensors.MAX_TURN
            self.turning -= 1
        else:
            self.rover.speed, self.rover.turn = s["speed"], 0.0

    def status(self) -> dict[str, Any]:
        return {"controller": "lesson-pilot", "rover": self.twin.name}


def _ran(ctx: LessonContext) -> bool:
    return "clean" in ctx.results


def _ran_with(key: str, ctx: LessonContext, want: Any) -> bool:
    run = ctx.results.get("clean", {})
    return any(run.get(t.name, {}).get(key) == want for t in ROVERS)


STEPS = (
    Step(
        id="stop",
        title=_text("停车距离", "The stop distance"),
        body=_text(
            "两台扫地机器人，规则一模一样：激光雷达看到正前方有东西比“停车距离”近，就原地转向更空的那边；否则往前开。"
            "扫过的地面会涂上它的颜色。\nA 的停车距离是 0.15 米，B 是 0.3 米。先猜谁会撞墙，然后点「开扫 30 秒」"
            "（可以用下面的 2× 加速）。",
            "Two robot vacuums follow exactly the same rule: if the lidar sees anything ahead closer than the stop "
            "distance, turn on the spot toward the more open side; otherwise drive on. The floor each has covered "
            "is painted in its colour.\nA stops at 0.15 m, B at 0.3 m. Guess who hits the walls, then press Clean "
            "for 30 s (the 2x button below speeds it up).",
        ),
        done=_text(
            "A 常常顶着墙推上一二十秒（我们测到 30 秒里有 13 到 22 秒）："
            "激光雷达装在车中间，车头还往前伸出 12 厘米，0.15 米其实只剩 3 厘米的余量。"
            "传感器读数是从传感器本身量的，用它做决定时要算上车身。",
            "A often spends many seconds pushing against a wall (13 to 22 of the 30 s in our runs): "
            "the lidar sits in the middle of the rover and "
            "the nose sticks out 12 cm further, so 0.15 m leaves only 3 cm to spare. Sensor readings are measured "
            "from the sensor itself; decisions must allow for the body around it.",
        ),
        check=_ran,
        unlock=("run",),
        show=("labels", "floor"),
    ),
    Step(
        id="timid",
        title=_text("太谨慎也不行", "Too careful is no good either"),
        body=_text(
            "把 A 的停车距离调到 0.9 米，再扫一次。比比两台扫过的面积。",
            "Set A's stop distance to 0.9 m and clean again. Compare how much floor each covers.",
        ),
        done=_text(
            "A 不撞了，却扫得少（我们测到比 0.3 米少扫两到三成）：离墙老远就掉头，"
            "墙边、角落和箱子旁边都扫不到。停车距离要刚好够刹车，再多就是浪费。",
            "A no longer bumps but covers less (20 to 30% less than at 0.3 m in our runs): it "
            "turns away far from every wall, so edges, corners and the space beside the crate stay dirty. The stop "
            "distance should be just enough to stop in time; more is wasted.",
        ),
        check=lambda ctx: _ran_with("stop", ctx, 0.9),
        unlock=("stop_a", "stop_b"),
        show=("labels", "floor"),
    ),
    Step(
        id="speed",
        title=_text("开快一点", "A little faster"),
        body=_text(
            "两台的停车距离都设成 0.3 米，把 A 的速度调到 0.45 米/秒，再扫一次。",
            "Set both stop distances to 0.3 m and A's speed to 0.45 m/s, then clean again.",
        ),
        done=_text(
            "快的扫得多（我们测到 0.45 米/秒比 0.3 米/秒多扫两成到五成），但偶尔会撞：速度越快，刹车距离越长，"
            "同样的停车距离就不够了。速度和安全要一起调。",
            "The faster one covers more (20 to 50% more in our runs at 0.45 m/s against 0.3 m/s) but "
            "bumps now and then: the faster it goes, the longer it takes to stop, so the same stop distance is no "
            "longer enough. Speed and safety are tuned together.",
        ),
        check=lambda ctx: _ran_with("speed", ctx, 0.45),
        unlock=("speed_a", "speed_b"),
        show=("labels", "floor"),
    ),
    Step(
        id="noise",
        title=_text("传感器会骗人", "Sensors lie a little"),
        body=_text(
            "真的激光雷达读数会跳。把两台的雷达噪声都设成 0.3 米，只给 B 打开「中值滤波」（取最近 5 次读数的中间值），"
            "两台速度都设回 0.3，再扫一次。",
            "Real lidar readings jitter. Set both rovers' lidar noise to 0.3 m, turn on the median filter (the "
            "middle of the last 5 readings) for B only, set both speeds back to 0.3, and clean again.",
        ),
        done=_text(
            "噪声大时，A 会被假的“前面有东西”吓得乱转，扫得少；B 的滤波把单次的离谱读数过滤掉了，通常扫得更多。"
            "不过滤波也有代价：它要等几次读数，反应慢一点，噪声小的时候反而略亏。",
            "With heavy noise A keeps turning away from phantom obstacles and covers less; B's filter drops the "
            "odd wild reading and usually covers more. The filter has a price too: it waits for several readings, "
            "so it reacts a little later, and with little noise it can come out slightly behind.",
        ),
        check=lambda ctx: _ran_with("noise", ctx, 0.3) and _ran_with("filter", ctx, True),
        unlock=("noise_a", "filter_a", "noise_b", "filter_b"),
        show=("labels", "floor"),
    ),
    Step(
        id="quiz",
        title=_text("想一想", "Think about it"),
        body=_text(
            "真的扫地机器人有激光雷达，前面却还装着一圈碰撞开关。",
            "Real robot vacuums have lidar, yet still carry a bumper.",
        ),
        done=_text("", ""),
        quiz=Quiz(
            question=_text("为什么有了激光雷达还要碰撞开关？", "Why a bumper when there is lidar?"),
            options=(
                _text("为了好看", "For looks"),
                _text(
                    "激光雷达也有看不见的东西（玻璃、很细的椅子腿、比雷达低的障碍），碰撞开关是最后一道保险",
                    "Lidar misses some things (glass, thin chair legs, obstacles below the lidar); the bumper is the "
                    "last line of defence",
                ),
                _text("碰撞开关比激光雷达更准", "A bumper is more accurate than lidar"),
            ),
            answer=1,
            explain=_text(
                "对。每种传感器都有盲区，真机器人会用好几种互相补位，这叫多传感器融合。",
                "Right. Every sensor has blind spots, so real robots combine several that cover for each other: "
                "sensor fusion.",
            ),
        ),
    ),
)


def _ran_with_text(run: dict[str, Any]) -> dict[str, str]:
    """A run's settings in one short line: stop distance, speed, noise and the filter."""
    zh = f"{run['stop']:g} m · {run['speed']:g} m/s · 噪声 {run['noise']:g}" + (" · 滤波" if run["filter"] else "")
    en = f"{run['stop']:g} m · {run['speed']:g} m/s · noise {run['noise']:g}" + (" · filter" if run["filter"] else "")
    return {"zh": zh, "en": en}


class AutopilotLesson(TwinsLesson):
    id = "lidar_autopilot"
    kind = "lesson"
    chapter = _text("感知", "Sensing")
    title = _text("感知（二）：扫地机器人，用激光雷达不撞墙", "Sensing II: a robot vacuum that steers by lidar")
    summary = _text(
        "两台扫地机器人用同一条规则自己开：调停车距离、速度、噪声和滤波，看谁撞墙、谁扫得多。",
        "Two robot vacuums drive themselves by one rule: tune the stop distance, speed, noise and filtering, and "
        "see who bumps and who cleans more.",
    )
    minutes = 12
    twins = ROVERS
    knobs = (
        Knob(
            "stop",
            "choice",
            _text("停车距离（米）", "Stop distance (m)"),
            options=tuple(_text(f"{v:g}", f"{v:g}") for v in STOPS),
        ),
        Knob(
            "speed",
            "choice",
            _text("速度（米/秒）", "Speed (m/s)"),
            options=tuple(_text(f"{v:g}", f"{v:g}") for v in SPEEDS),
        ),
        Knob(
            "noise",
            "choice",
            _text("雷达噪声（米）", "Lidar noise (m)"),
            options=tuple(_text(f"{v:g}", f"{v:g}") for v in NOISES),
        ),
        Knob("filter", "toggle", _text("中值滤波", "Median filter")),
    )
    shared = (Control("run", "button", _text("开扫 30 秒", "Clean for 30 s")),)
    plots = (
        Plot(
            "covered",
            _text("扫过的地面", "Floor covered"),
            "%",
            tuple(Series(f"covered_{t.name}", t.label, t.color) for t in ROVERS),
            window=float(RUN_SECONDS),
        ),
        Plot(
            "front",
            _text("正前方的距离（雷达，含噪声）", "Distance ahead (lidar, with its noise)"),
            "m",
            tuple(Series(f"front_{t.name}", t.label, t.color) for t in ROVERS),
            window=8.0,
        ),
    )
    steps = STEPS

    def __init__(self) -> None:
        super().__init__()
        self.rovers: dict[str, sensors.Rover] = {}
        self.pilots: dict[str, Pilot] = {}
        self.start: np.ndarray | None = None
        self.running = False
        self.t0 = 0.0
        self.trial = 0
        self.runs: dict[str, dict[str, Any]] = {}

    def settings(self, twin: Twin) -> dict[str, Any]:
        v = self.view(twin)
        return {
            "stop": STOPS[int(v["stop"])],
            "speed": SPEEDS[int(v["speed"])],
            "noise": NOISES[int(v["noise"])],
            "filter": bool(v["filter"]),
        }

    def build(self) -> mujoco.MjModel:
        spec = mujoco.MjSpec()
        spec.modelname = "lidar_autopilot"
        spec.compiler.degree = False
        spec.option.timestep = 0.002
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for twin, (x, y) in zip(self.twins, CENTRES, strict=True):
            frame = spec.worldbody.add_frame(pos=[x, y, 0])
            spec.attach(sensors.sensors_spec(floor=False), prefix=f"{twin.name}/", frame=frame)
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[sensors.ARENA + 0.8, 2 * OFFSET + 1.0, 0.05],
            material="a/rf_floor",
        )
        return spec.compile()

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        self.ctx = ctx
        self.running = False
        self.start = data.qpos.copy()
        self.rovers = {t.name: sensors.Rover(model, data, ctx, prefix=f"{t.name}/") for t in self.twins}
        self.pilots = {t.name: Pilot(self, t, self.rovers[t.name], seed=i) for i, t in enumerate(self.twins)}
        return {
            **{f"rover_{n}": r for n, r in self.rovers.items()},
            **{f"pilot_{n}": p for n, p in self.pilots.items()},
            "watch": Watch(self, model),
        }

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        if control != "run" or self.running:
            return
        assert self.start is not None
        model = self.playground.session.model
        data.qpos[:] = self.start  # both back on their start spot, still
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        self.trial += 1
        for i, twin in enumerate(self.twins):
            self.pilots[twin.name].rng = np.random.default_rng(10 * self.trial + i)  # fresh noise every run
            self.pilots[twin.name].history.clear()
            rover = self.rovers[twin.name]
            rover.last_enc = rover.last_pos = None  # the jump back to the start is not driving
        self.runs = {
            t.name: {
                **self.settings(t),
                "cells": set(),
                "bumps": 0,
                "pressed": 0.0,
                "last_touch": -9.0,
                "odom0": self.rovers[t.name].odom,
            }
            for t in self.twins
        }
        self.running, self.t0 = True, float(data.time)

    def _cell(self, data: mujoco.MjData, twin: Twin, centre: tuple[float, float]) -> tuple[int, int]:
        p = data.xpos[self.rovers[twin.name].body]
        return int((p[0] - centre[0] + sensors.ARENA) // CELL), int((p[1] - centre[1] + sensors.ARENA) // CELL)

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not self.running:
            return
        for twin, centre in zip(self.twins, CENTRES, strict=True):
            run, rover = self.runs[twin.name], self.rovers[twin.name]
            run["cells"].add(self._cell(data, twin, centre))
            if float(rover.read(data, "bumper")[0]) > 0.5:
                if data.time - run["last_touch"] > BUMP_GAP:
                    run["bumps"] += 1
                run["last_touch"] = float(data.time)
                run["pressed"] += model.opt.timestep
        if data.time - self.t0 >= RUN_SECONDS:
            self._finish()

    def _finish(self) -> None:
        assert self.ctx is not None
        self.running = False
        self.ctx.results["clean"] = {
            name: {
                **{k: run[k] for k in ("stop", "speed", "noise", "filter", "bumps")},
                "covered": round(100 * len(run["cells"]) / CELLS**2, 1),
                "pressed": round(run["pressed"], 1),
                "distance": round(self.rovers[name].odom - run["odom0"], 1),
            }
            for name, run in self.runs.items()
        }

    def sample(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
        out = {}
        for twin in self.twins:
            run = self.runs.get(twin.name)
            out[f"covered_{twin.name}"] = 100 * len(run["cells"]) / CELLS**2 if run else 0.0
            out[f"front_{twin.name}"] = min(self.pilots[twin.name].front, 4.0)
        return out

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        cells = {name: sorted(run["cells"]) for name, run in self.runs.items()}
        left = max(0.0, RUN_SECONDS - (float(data.time) - self.t0)) if self.running else 0.0
        readout = _text(f"还剩 {left:.0f} 秒", f"{left:.0f} s to go") if self.running else _text("", "")
        return {"cells": cells, "trial": self.trial, "table": self._table(ctx), "readout": readout}

    def _table(self, ctx: LessonContext) -> dict[str, Any]:
        now = [self.settings(t) for t in self.twins]
        on, off = _text("开", "on"), _text("关", "off")
        rows: list[dict[str, Any]] = [
            {"heading": _text("现在的设置", "Settings now")},
            _row(_text("停车距离", "Stop distance"), [f"{s['stop']:g} m" for s in now]),
            _row(_text("速度", "Speed"), [f"{s['speed']:g} m/s" for s in now]),
            _row(_text("雷达噪声", "Lidar noise"), [f"{s['noise']:g} m" for s in now]),
            _row(_text("中值滤波", "Median filter"), [on if s["filter"] else off for s in now]),
        ]
        clean = ctx.results.get("clean")
        if clean:
            got = [clean[t.name] for t in self.twins]
            rows += [
                {"heading": _text("上次 30 秒", "Last 30 s")},
                _row(_text("当时的设置", "Ran with"), [_ran_with_text(r) for r in got]),
                _row(
                    _text("扫过的地面", "Floor covered"),
                    [f"{r['covered']:.0f}%" for r in got],
                    [r["covered"] for r in got],
                    high=True,
                ),
                _row(_text("撞了几下", "Bumps"), [str(r["bumps"]) for r in got], [r["bumps"] for r in got]),
                _row(
                    _text("顶着东西的时间", "Time pushing on something"),
                    [_num(r["pressed"], "s") for r in got],
                    [r["pressed"] for r in got],
                ),
                _row(
                    _text("开了多远", "Distance driven"),
                    [_num(r["distance"], "m") for r in got],
                    [r["distance"] for r in got],
                    high=True,
                ),
            ]
        return {"columns": self.columns(), "rows": rows}

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        return {
            "kind": "cleaner",
            "cell": CELL,
            "cells": CELLS,
            "arenas": {
                t.name: [x - sensors.ARENA, y - sensors.ARENA] for t, (x, y) in zip(self.twins, CENTRES, strict=True)
            },
            "entrants": [
                {
                    **c,
                    "base": [x, y, 0.0],
                    "body": int(model.body(f"{c['name']}/rover").id),
                    "top": 0.3,
                    "cloud": [],
                }
                for c, (x, y) in zip(self.columns(), CENTRES, strict=True)
            ],
            "targets": [],
        }

    def camera(self) -> dict[str, list[float]]:
        return {"eye": [0.0, -2.1, 9.4], "target": [0.0, -0.2, 0.0]}  # from above: A in front, B behind
