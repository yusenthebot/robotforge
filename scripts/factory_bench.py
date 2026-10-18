"""Sort the factory's boxes by color, as an agent would with the tools: look with the overhead camera (observe find),
stop the belt for the pick, pick_and_place the box at the stop into the bin of its color, start the belt again.

    uv run python scripts/factory_bench.py [--boxes 16] [--speed 0.1] [--seed 0] [--minutes 4]

Prints each box as it is sorted, then the success rate and the rate per minute of sim time. The demo video
(scripts/demos.py factory_sorting) runs the same Sorter in real time. Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Callable
from typing import Any

VIEW = "top"  # the overhead camera: it sees the whole belt, the stop and the bins
COLORS = ("red", "green", "blue")  # the boxes' colors, each a bin
NEAR_STOP = 0.06  # m: a box whose middle is this close to where boxes wait against the stop is the next to pick
LOOK_EVERY = 0.5  # s of sim time between looks while the belt brings the next box


class Sorter:
    """One pick at a time. `wait(until, seconds)` lets sim time pass until `until()` is true or `seconds` have gone:
    as fast as possible headless, in real time in front of the viewer."""

    def __init__(self, pg: Any, wait: Callable[[Callable[[], bool], float], None], speed: float) -> None:
        from robotforge.design import scenes

        self.pg, self.wait, self.speed = pg, wait, speed
        self.pick_x = scenes.PICK_X
        self.sorted: list[dict[str, Any]] = []
        self.skip: set[str] = set()  # boxes the arm refused: left where they are
        self.running = True
        self.filled: dict[str, int] = {}  # bin -> boxes put in it
        self.box = scenes.FACTORY_BOX

    def belt(self, on: bool) -> None:
        self.running = on
        self.pg.set_controls({"conveyor/speed": self.speed if on else 0.0})

    def look(self) -> list[dict[str, Any]]:
        """The boxes the camera sees on the belt, nearest the stop first: name, color and position. One look per
        color: looking for all colors at once, the list stopped at its twelve biggest finds, and the boxes waiting
        at the stop, furthest from the camera, were not among them. Touching boxes of one color are one find that
        names them all; they came in single file, so the lowest number is the one in front."""
        boxes = []
        for color in COLORS:
            summary, _ = self.pg.find(VIEW, color, image=False)
            for found in summary.get("objects", []):
                names = [found.get("object"), *found.get("merged", [])]
                names = sorted((n for n in names if str(n).startswith("box_")), key=lambda n: int(n.split("_")[1]))
                if names:
                    boxes.append({"object": names[0], "color": found["color"], "position": found["position"]})
        on_belt = [
            o for o in boxes if abs(o["position"][1]) < 0.1 and o["position"][2] > 0.6 and o["object"] not in self.skip
        ]
        return sorted(on_belt, key=lambda o: -o["position"][0])

    def waiting(self) -> bool:
        """Whether the camera sees a box waiting at the stop."""
        seen = self.look()
        return bool(seen) and seen[0]["position"][0] > self.pick_x - NEAR_STOP

    def carry_on(self, arm: Any, seconds: float) -> None:
        """Let the arm finish its move, stopping the belt as soon as the next box reaches the stop."""
        data = self.pg.session.data
        start = float(data.time)
        while not arm.done and float(data.time) - start < seconds:
            self.wait(lambda: arm.done, LOOK_EVERY)
            if self.running and self.waiting():
                self.belt(False)

    def next_box(self, limit: float) -> dict[str, Any] | None:
        """Run the belt until the camera sees a box waiting at the stop (or none is left on the belt)."""
        data = self.pg.session.data
        start = float(data.time)
        while float(data.time) - start < limit:
            seen = self.look()
            if not seen:
                return None
            if seen[0]["position"][0] > self.pick_x - NEAR_STOP:
                if self.running:
                    self.belt(False)
                    self.wait(lambda: False, 0.3)  # settled against the stop
                    seen = self.look()
                return seen[0] if seen else None
            if not self.running:
                self.belt(True)
            self.wait(lambda: False, LOOK_EVERY)
        return None

    def slot(self, bin_name: str) -> list[float]:
        """Where the next box goes in a bin: one of four places on its floor, then a layer up. Dropped all in its
        middle, seven green boxes made a tower that toppled out of the bin."""
        entry = next(o for o in self.pg.store.load().objects if o.name == bin_name)
        n = self.filled.get(bin_name, 0)
        dx, dy = ((-1, -1), (1, -1), (-1, 1), (1, 1))[n % 4]
        step = (entry.size[0] - 2 * 0.005) / 4
        floor = entry.pos[2] + 0.005 + (n // 4) * self.box
        return [round(entry.pos[0] + dx * step, 4), round(entry.pos[1] + dy * step, 4), round(floor, 4)]

    def in_bin(self, box: str, bin_name: str) -> bool:
        entry = next(o for o in self.pg.store.load().objects if o.name == bin_name)
        at = self.pg.session.data.body(box).xpos
        half = entry.size[0] / 2
        inside = abs(at[0] - entry.pos[0]) < half and abs(at[1] - entry.pos[1]) < entry.size[1] / 2
        return bool(inside and entry.pos[2] < at[2] < entry.pos[2] + entry.size[2] + 2 * self.box)

    def cycle(self) -> dict[str, Any] | None:
        """Sort one box: the result, or None when the belt has no box left."""
        box = self.next_box(30.0)
        if box is None:
            return None
        data = self.pg.session.data
        start = float(data.time)
        name, color = box["object"], box["color"]
        target = f"{color}_bin"
        try:
            arm, info = self.pg.behaviors.pick_and_place("arm", name, self.slot(target))
        except ValueError as exc:
            self.skip.add(name)
            result = {"box": name, "seen": color, "success": False, "refused": str(exc)[:200]}
            self.sorted.append(result)
            return result
        # The belt runs again once the box is off it: the next one comes up while this one is carried, and the belt
        # stops when it gets to the stop.
        # (Not "idle": until its first move starts, the arm still reads idle from the last one, and the belt carried
        # the box out from under the cup coming down for it.)
        self.wait(lambda: arm.current_label in ("carry", "lower", "release", "retreat"), 20.0)
        self.belt(True)
        self.carry_on(arm, 30.0)
        self.wait(lambda: False, 0.3)
        inside = self.in_bin(name, target)
        if inside:
            self.filled[target] = self.filled.get(target, 0) + 1
        truth = next(o.color for o in self.pg.store.load().objects if o.name == name)
        result = {
            "box": name,
            "seen": color,
            "truth": truth,
            "bin": target,
            "success": bool(inside and color == truth),
            "cycle_s": round(float(data.time) - start, 1),
        }
        self.sorted.append(result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--boxes", type=int, default=16)
    parser.add_argument("--speed", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minutes", type=float, default=4.0, help="sim minutes at most")
    args = parser.parse_args()
    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="rf-factory-")
    from robotforge.playground import Playground

    pg = Playground.with_demo()
    pg.load_scene("factory", seed=args.seed, options={"boxes": args.boxes, "speed": args.speed})
    sorter = Sorter(pg, lambda until, seconds: pg.run_fast(until, seconds), args.speed)
    t0, wall = float(pg.session.data.time), time.monotonic()
    while float(pg.session.data.time) - t0 < args.minutes * 60:
        result = sorter.cycle()
        if result is None:
            break
        print(json.dumps(result), flush=True)
    sim = float(pg.session.data.time) - t0
    good = sum(1 for r in sorter.sorted if r.get("success"))
    print(
        json.dumps(
            {
                "boxes": args.boxes,
                "sorted": len(sorter.sorted),
                "right_bin": good,
                "success_rate": round(good / max(len(sorter.sorted), 1), 3),
                "sim_s": round(sim, 1),
                "per_minute": round(good / sim * 60, 2) if sim > 0 else 0.0,
                "wall_s": round(time.monotonic() - wall, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
