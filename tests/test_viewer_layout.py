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
    """Run `scenario(tab, server)` against a viewer of a world with a robot dog (or what `world` builds), in Chrome at
    `size` and `lang`; with `also`, more windows ((size, lang) each) watch the same viewer and the scenario gets them
    all."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))

    def run(
        size: str,
        lang: str,
        scenario: Scenario,
        also: tuple[tuple[str, str], ...] = (),
        world: Callable[[Any], None] = lambda pg: pg.add_robot("dog", template="quadruped"),
    ) -> Any:
        world = qa.build_world(tmp_path, world)

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


DOG_ON_SCREEN = """(() => {
    const rf = window.__robotforge;
    const id = rf.scene.bodies.findIndex((b) => b.name.startsWith("dog/") && b.dynamic);
    const [x, y] = rf.screenOf(id);
    const drawer = document.getElementById("drawer").getBoundingClientRect();
    const dock = document.querySelector(".dock").getBoundingClientRect().top;
    return { x, y, free: rf.freeRect(), drawer: drawer.bottom, dock };
})()"""


def test_on_a_phone_the_robots_drawer_leaves_the_robot_in_sight(viewer):
    """The drawer ran from the top bar down to the dock: the robot whose joints you moved was under it."""

    async def scenario(tab, _server):
        await tab.run("window.__robotforge.select('dog'); document.getElementById('drawer-toggle').click();")
        await tab.until("!window.__robotforge.tweening()")
        await asyncio.sleep(1.5)  # the view eases into the room under the drawer
        return await tab.eval(DOG_ON_SCREEN)

    seen = viewer("phone", "en", scenario)
    assert seen["dock"] - seen["drawer"] >= 200, seen  # room for a robot between the drawer and the dock
    assert seen["drawer"] < seen["y"] < seen["dock"] and 0 < seen["x"] < 390, seen


SIGHT = """(async () => {
    const THREE = await import(new URL("static/vendor/three/three.module.min.js", location).href);
    const rf = window.__robotforge;
    const { scene, camera } = rf.three();
    const target = new THREE.Vector3().fromArray(rf.cameraTarget());
    const ray = new THREE.Raycaster(camera.position.clone(), target.clone().sub(camera.position).normalize());
    ray.far = camera.position.distanceTo(target);
    const robot = new Set(rf.scene.bodies.filter((b) => b.name.startsWith("dog/")).map((b) => b.id));
    const meshes = [];
    scene.traverse((o) => { if (o.isMesh && o.visible && o.userData.body !== undefined) meshes.push(o); });
    return ray.intersectObjects(meshes, false).filter((h) => !robot.has(h.object.userData.body))
      .map((h) => rf.scene.bodies[h.object.userData.body].name);
})()"""


def _dog_behind_a_wall(pg: Any) -> None:
    """A wall between the robot and where the camera comes from when it is picked (the viewer looks from +x, -y)."""
    pg.add_robot("dog", template="quadruped", pos=[0.0, 0.0])
    pg.add_object("wall", shape="box", size=[0.05, 1.2, 0.8], pos=[0.6, -0.8], fixed=True, yaw=-52.0)


def test_a_robot_picked_on_a_phone_is_framed_with_nothing_in_between(viewer):
    """On a phone the camera stood twice as far back as on a laptop, and in the park a tree was between it and the
    Go2 just picked from the list: the screen was all foliage."""

    async def scenario(tab, _server):
        await tab.run("document.getElementById('btn-frame').click();")
        await tab.until("!window.__robotforge.tweening()")
        await tab.run("window.__robotforge.select('dog');")
        await asyncio.sleep(0.3)
        await tab.until("!window.__robotforge.tweening()")
        return await tab.eval(SIGHT)

    assert viewer("phone", "en", scenario, world=_dog_behind_a_wall) == []


TITLE_LINES = """(() => {
    const lines = (el) => {  // the text of each line the title takes
        const node = el.firstChild, range = document.createRange(), rows = [];
        for (let i = 0; i < node.data.length; i++) {
            range.setStart(node, i);
            range.setEnd(node, i + 1);
            const top = Math.round(range.getClientRects()[0]?.top ?? -1);
            if (!rows.length || Math.abs(rows[rows.length - 1].top - top) > 4) rows.push({ top, text: "" });
            rows[rows.length - 1].text += node.data[i];
        }
        return rows.map((r) => r.text.trim());
    };
    return [...document.querySelectorAll("#lesson-list .course-title")].map(lines);
})()"""


@pytest.mark.parametrize("lang", ["zh-CN", "en"])
def test_course_titles_on_a_phone_break_between_words(viewer, lang):
    """Balanced, the Chinese title broke "六轴机械臂" as "六轴机 / 械臂", and the English one "six- / joint arms"."""

    async def scenario(tab, _server):
        await tab.run("document.getElementById('learn-toggle').click();")
        await asyncio.sleep(0.8)
        return await tab.eval(TITLE_LINES)

    titles = viewer("phone", lang, scenario)
    ik_arm = next(t for t in titles if "PiPER" in "".join(t) and ("六轴" in "".join(t) or "joint" in "".join(t)))
    assert len(ik_arm) == 2, ik_arm
    words = ("六轴机械臂", "机器人", "扫地机器人") if lang == "zh-CN" else ("six\u2011joint", "cart\u2011pole")
    for title in titles:
        for word in words:
            if word in "".join(title):
                assert any(word in line for line in title), title


def test_a_lesson_opens_at_the_top_of_its_panel(viewer):
    """The panel kept the last lesson's place: after another lesson's Finish had scrolled to its foot, the next lesson
    opened with its title scrolled out of sight."""

    async def scenario(tab, _server):
        await _start(tab, "pid_twins")
        await tab.run("document.getElementById('lesson-panel').scrollTop = 1e6;")
        scrolled = await tab.eval("document.getElementById('lesson-panel').scrollTop")
        await tab.run("document.getElementById('lesson-exit').click();")
        await tab.until("window.__robotforge.lesson() === null", 120)
        await _start(tab, "pd_control")
        return scrolled, await tab.eval("document.getElementById('lesson-panel').scrollTop")

    scrolled, top = viewer("laptop", "en", scenario)
    assert scrolled > 0 and top == 0


TAGS = """(async () => {
    const THREE = await import(new URL("static/vendor/three/three.module.min.js", location).href);
    const rf = window.__robotforge;
    const { scene, camera } = rf.three();
    camera.updateMatrixWorld();
    const tags = [];
    scene.traverse((o) => { if (o.isSprite && o.visible && o.renderOrder === 10) tags.push(o); });
    const [sx, sy] = [camera.projectionMatrix.elements[0], camera.projectionMatrix.elements[5]];
    return tags.map((s) => {
        const p = s.getWorldPosition(new THREE.Vector3()).project(camera);
        const robot = new THREE.Vector3(s.position.x, s.position.y, 0.05).project(camera);  // the robot under it
        const w = s.scale.x * sx, h = s.scale.y * sy;  // the tag's size, -1 to 1 across the view
        const left = p.x - s.center.x * w, bottom = p.y - s.center.y * h;
        return { covers: robot.x > left && robot.x < left + w && robot.y > bottom && robot.y < bottom + h };
    });
})()"""


def test_a_name_tag_seen_from_overhead_does_not_cover_its_robot(viewer):
    """The robot vacuums are watched from above: each tag, centred on a point over its robot, hid the robot."""

    async def scenario(tab, _server):
        await _start(tab, "lidar_autopilot")
        return await tab.eval(TAGS)

    tags = viewer("laptop", "en", scenario)
    assert len(tags) == 2 and not any(t["covers"] for t in tags), tags


def test_the_sensors_camera_on_a_phone_is_a_tab_and_leaves_the_rover_room(viewer):
    """Shown under the plot tabs, the rover's camera and the lesson sheet covered the whole phone, the rover too."""

    async def scenario(tab, server):
        await _start(tab, "sensors")
        steps = await asyncio.to_thread(qa.play_lesson, server, "sensors")
        assert all(s.passed for s in steps), [(s.id, s.failed) for s in steps]
        await asyncio.sleep(1.0)
        return await tab.eval("""(() => ({
            tabs: [...document.querySelectorAll("#lesson-plots .plot-tabs button")].map((b) => b.textContent),
            selected: document.querySelector("#lesson-plots .plot-tabs [aria-selected=true]")?.textContent,
            plots: document.getElementById("lesson-plots").getBoundingClientRect().bottom,
            sheet: document.getElementById("lesson-panel").getBoundingClientRect().top,
        }))()""")

    seen = viewer("phone", "en", scenario)
    assert seen["tabs"][-1] == "Robot camera" and seen["selected"] == "Robot camera", seen
    assert seen["sheet"] - seen["plots"] >= 60, seen


def test_a_slider_value_has_a_space_before_its_unit(viewer):
    """The balance lesson's push read "0.10m/s" beside "0.25 m" in the IK lessons."""

    async def scenario(tab, _server):
        await _start(tab, "balance")
        values = "[...document.querySelectorAll('#lesson-controls .joint-value')].map((v) => v.textContent)"
        return await tab.eval(values)

    assert viewer("laptop", "en", scenario) == ["0.10\u00a0m/s"]


def test_the_docks_focus_ring_on_a_phone_stays_off_its_neighbour(viewer):
    """2 px apart on a phone, the Reset button's ring ran 2 px over the speed buttons beside it."""

    async def scenario(tab, _server):
        await tab.run("document.getElementById('btn-reset').focus();")  # a keyboard's focus: the ring shows
        await asyncio.sleep(0.3)
        return await tab.eval("""(() => {
            const b = document.getElementById("btn-reset"), cs = getComputedStyle(b);
            const ring = b.getBoundingClientRect().right + parseFloat(cs.outlineOffset) + parseFloat(cs.outlineWidth);
            return { ring, next: document.getElementById("speed").getBoundingClientRect().left,
                     shown: cs.outlineStyle !== "none" };
        })()""")

    seen = viewer("phone", "en", scenario)
    assert seen["shown"] and seen["ring"] <= seen["next"], seen
