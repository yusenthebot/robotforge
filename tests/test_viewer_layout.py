"""The viewer's layout in a real browser, at the sizes where the pixel pass (scripts/viewer_qa.py) found faults.

Each test serves a viewer in this process, opens it in headless Google Chrome over the DevTools protocol at one
window size, and measures the page. Skipped without Chrome; slow, since lessons are played in real time."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "viewer_qa.py"
FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


def _load_qa() -> Any:
    spec = importlib.util.spec_from_file_location("viewer_qa", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["viewer_qa"] = module
    spec.loader.exec_module(module)
    return module


qa = _load_qa()


def _chrome() -> str | None:
    try:
        return qa.screenshots._chrome()
    except SystemExit:
        return None


pytestmark = [pytest.mark.slow, pytest.mark.skipif(_chrome() is None, reason="Google Chrome is not installed")]

Scenario = Callable[[Any, Any], Awaitable[Any]]


@pytest.fixture
def viewer(tmp_path, monkeypatch) -> Callable[..., Any]:
    """Run `scenario(tab, server)` against a viewer of a world with a robot dog, in Chrome at `size` and `lang`;
    with `also`, more windows ((size, lang) each) watch the same viewer and the scenario gets them all."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))

    def run(size: str, lang: str, scenario: Scenario, also: tuple[tuple[str, str], ...] = ()) -> Any:
        world = qa.build_world(tmp_path, lambda pg: pg.add_robot("dog", template="quadruped"))

        async def main() -> Any:
            port = qa.screenshots._free_port()
            with qa.Server(world, port) as server:
                tabs = [qa.Tab(lang, size, port, tmp_path), *(qa.Tab(la, si, port, tmp_path) for si, la in also)]
                try:
                    await asyncio.gather(*(tab.open() for tab in tabs))
                    returning = "".join(f"localStorage.setItem({k!r}, {v!r});" for k, v in qa.RETURNING.items())
                    for tab in tabs:
                        await tab.run(returning)
                        await tab.navigate()
                    return await scenario(tabs if also else tabs[0], server)
                finally:
                    await asyncio.gather(*(tab.close() for tab in tabs))

        return asyncio.run(main())

    return run


async def _start(tab: Any, lesson: str) -> None:
    await tab.run(qa.START % lesson)
    await tab.until(f"window.__robotforge.lesson()?.id === {lesson!r}")
    await asyncio.sleep(2.0)


def test_the_lesson_plots_end_on_the_lesson_panels_line_above_the_dock(viewer):
    """At 1280 x 800 in English the plots ran 72 px lower than the lesson beside them and onto the end of the dock."""

    async def scenario(tab, _server):
        await _start(tab, "pid_twins")
        return await tab.eval("""(() => {
            const bottom = (id) => {  // as tall as the panel may grow: a tall height, held down by its max-height
                const el = document.getElementById(id);
                el.style.height = "10000px";
                const edge = el.getBoundingClientRect().bottom;
                el.style.height = "";
                return edge;
            };
            return { panel: bottom("lesson-panel"), plots: bottom("lesson-plots"),
                     dock: document.querySelector(".dock").getBoundingClientRect().top };
        })()""")

    edges = viewer("laptop", "en", scenario)
    assert edges["plots"] == pytest.approx(edges["panel"]), edges
    assert edges["plots"] < edges["dock"], edges


CAMERA_DISTANCE = """(() => {
    const rf = window.__robotforge;
    const [x, y, z] = rf.cameraTarget();
    return rf.three().camera.position.distanceTo({ x, y, z });
})()"""


def test_a_window_that_joins_a_lesson_frames_its_rig_as_a_reload_does(viewer):
    """A page that opened the lesson from the scene the server sent fitted the rig before the robots' first poses,
    every body still at the origin: on a phone the LQR twins came out 13% further off than after a reload (and the
    third PID twin, or the Panda, off the screen in the pixel pass with the real models)."""

    async def scenario(tabs, _server):
        lead, phone = tabs
        settled = "window.__robotforge.lesson()?.id === 'lqr_twins' && !window.__robotforge.tweening()"
        await _start(lead, "lqr_twins")
        await phone.until(settled)
        await asyncio.sleep(1.0)
        joined = await phone.eval(CAMERA_DISTANCE)
        await phone.navigate()  # a reload frames the lesson's camera once the poses are in
        await phone.until(settled)
        await asyncio.sleep(1.0)
        return joined, await phone.eval(CAMERA_DISTANCE)

    joined, reloaded = viewer("laptop", "en", scenario, also=(("phone", "en"),))
    assert joined == pytest.approx(reloaded, rel=0.02)


def test_finishing_a_lesson_brings_the_lesson_complete_box_into_view(viewer):
    """The box sits at the foot of the lesson panel: below the fold, Finish looked as if it did nothing."""

    async def scenario(tab, server):
        await _start(tab, "pid_twins")
        steps = await asyncio.to_thread(qa.play_lesson, server, "pid_twins")
        assert all(s.passed for s in steps), [(s.id, s.failed) for s in steps]
        await asyncio.sleep(1.0)
        await tab.run("document.getElementById('lesson-next').click();")
        await asyncio.sleep(1.5)
        return await tab.eval("""(() => {
            const box = document.getElementById("lesson-finished").getBoundingClientRect();
            const panel = document.getElementById("lesson-panel").getBoundingClientRect();
            return { shown: !document.getElementById("lesson-finished").hidden, top: box.top, bottom: box.bottom,
                     panelTop: panel.top, panelBottom: panel.bottom };
        })()""")

    box = viewer("laptop", "en", scenario)
    assert box["shown"], box
    assert box["panelTop"] <= box["top"] and box["bottom"] <= box["panelBottom"] + 1, box


def test_a_parked_robot_does_not_fill_its_plots_with_wobble(viewer):
    """The sensors lesson drew its parked rover's last 0.4 mm of odometry across the whole plot, between two ticks
    that both read 0, and the lidar's 0.5 mm of settling between 1.506 and 1.506."""

    async def scenario(tab, _server):
        await _start(tab, "sensors")
        await asyncio.sleep(3.0)  # the plots have drawn a few seconds of the rover standing still
        return await tab.eval("window.__robotforge.plotAxes()")

    axes = {a["id"]: a for a in viewer("laptop", "en", scenario)}
    span = {k: a["range"][1] - a["range"][0] for k, a in axes.items()}
    assert span["distance"] >= 0.2 and span["range"] >= 0.2, axes  # metres
    assert span["imu"] >= 1.0, axes  # the lesson's own smallest span


def test_a_joint_sliders_value_sits_on_its_names_baseline(viewer):
    """The small mono value sat 3 px above the joint's name."""

    async def scenario(tab, _server):
        await tab.run("window.__robotforge.select('dog'); document.getElementById('drawer-toggle').click();")
        await asyncio.sleep(0.8)
        return await tab.eval("""(() => {
            const baseline = (el) => {
                const probe = document.createElement("span");
                probe.style.cssText = "display: inline-block; width: 0; height: 0; vertical-align: baseline";
                el.append(probe);
                const y = probe.getBoundingClientRect().top;
                probe.remove();
                return y;
            };
            return [...document.querySelectorAll("#joints .joint")].map((j) =>
                [baseline(j.querySelector(".joint-name")), baseline(j.querySelector(".joint-value"))]);
        })()""")

    rows = viewer("laptop", "en", scenario)
    assert rows, "the drawer showed no joints"
    assert all(abs(name - value) < 0.5 for name, value in rows), rows


def test_plot_tabs_on_a_phone_show_their_whole_labels(viewer):
    """On a phone the sensors lesson's tabs read "Distance trave..." and "Distance ahead..." (no tooltip to tap)."""

    async def scenario(tab, _server):
        await _start(tab, "sensors")
        return await tab.eval("""[...document.querySelectorAll("#lesson-plots .plot-tabs button")].map((b) => ({
            label: b.textContent, needs: b.scrollWidth, has: b.clientWidth,
            cut: getComputedStyle(b).textOverflow === "ellipsis" && b.scrollWidth > b.clientWidth }))""")

    tabs = viewer("phone", "en", scenario)
    assert len(tabs) == 3, tabs
    assert not any(t["cut"] or t["needs"] > t["has"] + 1 for t in tabs), tabs
    assert [t["label"] for t in tabs] == ["Distance travelled", "IMU", "Distance ahead"], tabs
