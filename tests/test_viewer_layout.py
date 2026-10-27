"""The viewer's layout in a real browser, at the sizes where the pixel pass (scripts/viewer_qa.py) found faults.

Each test serves a viewer in this process, opens it in headless Google Chrome over the DevTools protocol at one
window size, and measures the page. A step later in a lesson is reached on the server (`_reach`), not played in real
time, so a busy machine cannot run a step out of its clock. Skipped without Chrome, and kept out of the default run
and CI's slow job: `uv run pytest -m browser tests/test_viewer_layout.py`."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from viewer_camera import settle_camera

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


pytestmark = [pytest.mark.browser, pytest.mark.skipif(_chrome() is None, reason="Google Chrome is not installed")]

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


async def _reach(tab: Any, server: Any, step: int | None = None, passed: bool = False) -> None:
    """Open the running lesson at `step` (the last by default) as if every step before it had been passed, and with
    `passed` that one too; then wait for the page to show it. Playing the steps in real time made these tests depend
    on the machine's load: on a busy one the sensors lesson's IMU step ran out of its clock."""
    runner = server.pg.lessons
    steps = runner.lesson.steps
    index = len(steps) - 1 if step is None else step
    with runner._lock:  # the viewer's loop reads the passed steps 20 times a second
        runner.passed.update(s.id for s in steps[: index + 1 if passed else index])
    runner.handle({"action": "goto", "step": index})
    await tab.until(f"window.__robotforge.lesson()?.step === {index}")
    await asyncio.sleep(1.0)  # the page draws the step and eases its view


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


RIG_ACROSS = """(async () => {
    const THREE = await import(new URL("static/vendor/three/three.module.min.js", location).href);
    const rf = window.__robotforge;
    const { scene, camera } = rf.three();
    camera.updateMatrixWorld();
    const out = {};
    scene.traverse((o) => {
        const name = o.isMesh && o.visible && rf.scene.bodies[o.userData.body]?.name;
        if (!name || !name.includes("/")) return;
        const b = new THREE.Box3().setFromObject(o), robot = name.split("/")[0];
        for (const x of [b.min.x, b.max.x]) for (const y of [b.min.y, b.max.y]) for (const z of [b.min.z, b.max.z]) {
            const px = ((new THREE.Vector3(x, y, z).project(camera).x + 1) / 2) * innerWidth;
            const e = (out[robot] ??= [px, px]);
            out[robot] = [Math.min(e[0], px), Math.max(e[1], px)];
        }
    });
    return { robots: out, width: innerWidth };
})()"""


def test_a_lesson_rig_that_settles_after_its_first_poses_is_framed_as_it_stands(viewer):
    """The PID triplets' arms hang at their first poses and rise level within half a second: framed at the first
    poses, the third arm then reached off a phone's right edge (a reload, which framed the settled rig, kept it in)."""

    async def scenario(tab, _server):
        await _start(tab, "pid_twins")
        # First poses schedule another fit after 800 ms; !tweening alone has a gap between the two fits.
        await tab.until("window.__robotforge.time >= 0.8")
        await settle_camera(tab, quiet_ms=1000)
        return await tab.eval(RIG_ACROSS)

    seen = viewer("phone", "en", scenario)
    assert sorted(seen["robots"]) == ["a", "b", "c"], seen
    assert all(left >= 0 and right <= seen["width"] for left, right in seen["robots"].values()), seen


# Start a lesson that fits its rig and press Fit view as soon as its first poses start the lesson camera's glide, inside
# the time the rig is given to settle before it is fitted again. Returns where Fit view looked once its glide ended,
# and where the camera looks two seconds later.
FIT_VIEW_AS_THE_RIG_SETTLES = """(async () => {
    const rf = window.__robotforge;
    const before = rf.scene;
    %s
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
    const gliding = () => rf.scene !== before && rf.lesson()?.id === "%s" && rf.tweening();
    for (let i = 0; i < 1000 && !gliding(); i++) await wait(10);
    document.getElementById("btn-frame").click();
    for (let i = 0; i < 200 && rf.tweening(); i++) await wait(10);
    const framed = rf.cameraTarget();
    await wait(2000);
    return { framed, kept: rf.cameraTarget() };
})()"""


def test_fit_view_pressed_while_a_lesson_rig_settles_is_kept(viewer):
    """The rig fitted at its first poses is fitted again once it has settled, unless the view was moved by hand: a
    drag cancelled that, but Fit view pressed in those 800 ms was undone by the lesson camera."""

    async def scenario(tab, _server):
        return await tab.eval(FIT_VIEW_AS_THE_RIG_SETTLES % (qa.START % "pid_twins", "pid_twins"))

    seen = viewer("laptop", "en", scenario)
    assert seen["kept"] == pytest.approx(seen["framed"], abs=0.01), seen


def test_finishing_a_lesson_brings_the_lesson_complete_box_into_view(viewer):
    """The box sits at the foot of the lesson panel: below the fold, Finish looked as if it did nothing."""

    async def scenario(tab, server):
        await _start(tab, "pid_twins")
        await _reach(tab, server, passed=True)
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


def test_a_quantity_that_cannot_go_negative_is_not_drawn_below_zero(viewer):
    """Widened evenly around the data to its smallest span, a converged distance drew ticks 2, 0 and -2 mm, the
    distance travelled 0.1, 0 and -0.1 m, and the floor covered 5, 0 and -5 %."""

    async def scenario(tab, _server):
        axes = {}
        for lesson, wait in (("sensors", 3.0), ("ik_arm", 4.0), ("lidar_autopilot", 3.0)):
            await _start(tab, lesson)
            await asyncio.sleep(wait)  # the rover parked, the arm on its target, the vacuums setting off
            axes |= {f"{lesson}:{a['id']}": a["range"] for a in await tab.eval("window.__robotforge.plotAxes()")}
        return axes

    axes = viewer("laptop", "en", scenario)
    for plot in ("sensors:distance", "sensors:range", "ik_arm:err", "lidar_autopilot:covered"):
        lo, hi = axes[plot]
        assert lo >= -0.02 * (hi - lo), (plot, axes[plot])  # at most a settling's hair below zero: no tick there


def test_a_signed_quantity_at_rest_is_drawn_round_zero(viewer):
    """The axis stopped at zero for any plot whose data was not below it, so the gait lesson's body sway (roll and
    pitch, signed) read 0, 2 and 4 with the trace on the floor, and jumped open once pitch dipped below zero."""

    async def scenario(tab, _server):
        await _start(tab, "gait")
        await asyncio.sleep(2.0)  # the dogs stand still
        return {a["id"]: a["range"] for a in await tab.eval("window.__robotforge.plotAxes()")}

    lo, hi = viewer("laptop", "en", scenario)["sway"]
    assert lo <= -0.25 * (hi - lo) and hi >= 0.25 * (hi - lo), (lo, hi)


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
    const robot = new Set(rf.scene.bodies.filter((b) => b.name.startsWith("dog/")).map((b) => b.id));
    const box = new THREE.Box3();
    const meshes = [];
    scene.traverse((o) => {
        if (!o.isMesh || !o.visible || o.userData.body === undefined || o.userData.isPlane) return;
        if (robot.has(o.userData.body)) box.expandByObject(o);
        else meshes.push(o);
    });
    // The robot's middle and the corners of its box drawn in by a sixth, near the robot's own outline.
    const inner = box.clone().expandByVector(box.getSize(new THREE.Vector3()).multiplyScalar(-1 / 6));
    const points = [box.getCenter(new THREE.Vector3())];
    for (const x of [inner.min.x, inner.max.x]) for (const y of [inner.min.y, inner.max.y])
        for (const z of [inner.min.z, inner.max.z]) points.push(new THREE.Vector3(x, y, z));
    return points.flatMap((p) => {
        const way = p.clone().sub(camera.position);
        const ray = new THREE.Raycaster(camera.position.clone(), way.clone().normalize());
        ray.far = way.length();
        return ray.intersectObjects(meshes, false).map((h) => rf.scene.bodies[h.object.userData.body].name);
    });
})()"""


def _dog_behind_a_wall(pg: Any) -> None:
    """A wall between the robot and where the camera comes from when it is picked (the viewer looks from +x, -y)."""
    pg.add_robot("dog", template="quadruped", pos=[0.0, 0.0])
    pg.add_object("wall", shape="box", size=[0.05, 1.2, 0.8], pos=[0.6, -0.8], fixed=True, yaw=-52.0)


def _dog_half_behind_a_wall(pg: Any) -> None:
    """The same wall moved sideways until a ray from the robot's middle just passes its end: it hides half the dog,
    as the warehouse's shelf hid half the Go2 picked from the list."""
    pg.add_robot("dog", template="quadruped", pos=[0.0, 0.0])
    pg.add_object("wall", shape="box", size=[0.05, 1.2, 1.2], pos=[1.1, -0.42], fixed=True, yaw=-52.0)


@pytest.mark.parametrize("world", [_dog_behind_a_wall, _dog_half_behind_a_wall], ids=["wall", "half-wall"])
def test_a_robot_picked_on_a_phone_is_framed_with_nothing_in_between(viewer, world):
    """On a phone the camera stood twice as far back as on a laptop, and in the park a tree was between it and the
    Go2 just picked from the list: the screen was all foliage."""

    async def scenario(tab, _server):
        await tab.run("document.getElementById('btn-frame').click();")
        await tab.until("!window.__robotforge.tweening()")
        await tab.run("window.__robotforge.select('dog');")
        await asyncio.sleep(0.3)
        await tab.until("!window.__robotforge.tweening()")
        return await tab.eval(SIGHT)

    assert viewer("phone", "en", scenario, world=world) == []


CONTENT_EXTENT = """(async () => {
    const THREE = await import(new URL("static/vendor/three/three.module.min.js", location).href);
    const rf = window.__robotforge;
    const { scene, camera } = rf.three();
    camera.updateMatrixWorld();
    const [W, H] = [innerWidth, innerHeight];
    const e = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity };
    scene.traverse((o) => {
        if (!o.isMesh || !o.visible || o.userData.body === undefined || o.userData.isPlane) return;
        const b = new THREE.Box3().setFromObject(o);
        for (const x of [b.min.x, b.max.x]) for (const y of [b.min.y, b.max.y]) for (const z of [b.min.z, b.max.z]) {
            const p = new THREE.Vector3(x, y, z).project(camera);
            const [px, py] = [((p.x + 1) / 2) * W, ((1 - p.y) / 2) * H];
            Object.assign(e, { left: Math.min(e.left, px), right: Math.max(e.right, px),
                               top: Math.min(e.top, py), bottom: Math.max(e.bottom, py) });
        }
    });
    return { content: e, free: rf.freeRect() };
})()"""


def _dog_in_a_yard(pg: Any) -> None:
    """A robot dog with a few posts spread round it: flat, with empty corners, as the park is."""
    pg.add_robot("dog", template="quadruped", pos=[0.0, 0.0])
    for i, (x, y) in enumerate([(-3.0, 2.0), (4.0, -1.0), (1.0, 4.0), (-2.0, -3.0)]):
        pg.add_object(f"post_{i}", shape="box", size=[0.2, 0.2, 1.0], pos=[x, y], fixed=True)


@pytest.mark.parametrize("size, drawer", [("laptop", None), ("wide", None), ("tablet", "drawer"), ("tablet", "world")])
def test_fit_view_fills_the_free_view_with_the_scene(viewer, monkeypatch, size, drawer):
    """Fit view framed the scene's bounding sphere: seen from above a scene is flat and its corners empty, so the
    park filled 45% of a laptop's height and its robots were 16 px tall. Coming in from there, it never went further
    out: on a tablet with the robots drawer or the scene panel open, the free view is narrower than the sphere, and
    60 to 130 px of the scene stayed under the panel."""
    monkeypatch.setitem(qa.SIZES, "tablet", (820, 1180, 2))

    async def scenario(tab, _server):
        if drawer:
            await tab.run(f"document.getElementById('{drawer}-toggle').click();")
            await asyncio.sleep(1.0)
        await tab.run("document.getElementById('btn-frame').click();")
        await tab.until("!window.__robotforge.tweening()")
        await asyncio.sleep(0.5)
        return await tab.eval(CONTENT_EXTENT)

    seen = viewer(size, "zh-CN" if drawer else "en", scenario, world=_dog_in_a_yard)
    c, f = seen["content"], seen["free"]
    assert f["left"] <= c["left"] and c["right"] <= f["right"] and f["top"] <= c["top"] and c["bottom"] <= f["bottom"]
    fill = max((c["right"] - c["left"]) / (f["right"] - f["left"]), (c["bottom"] - c["top"]) / (f["bottom"] - f["top"]))
    assert fill >= 0.75, seen


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


def test_a_new_step_starts_at_the_top_of_the_lesson_panel(viewer):
    """The panel kept the last step's place: on a phone, a step reached with the sheet scrolled down to the last one's
    buttons opened with its title and text scrolled out of sight."""

    async def scenario(tab, server):
        await _start(tab, "pid_twins")
        await tab.run("document.getElementById('lesson-panel').scrollTop = 1e6;")
        scrolled = await tab.eval("document.getElementById('lesson-panel').scrollTop")
        await _reach(tab, server, 1)
        return scrolled, await tab.eval("document.getElementById('lesson-panel').scrollTop")

    scrolled, top = viewer("phone", "en", scenario)  # the lesson sheet is 40% of a phone: it scrolls
    assert scrolled > 0 and top == 0, (scrolled, top)


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
        await _reach(tab, server)
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


RINGS = """(async () => {
    const out = [];
    const tools = [...document.querySelectorAll(".dock > *")].filter((el) => el.getClientRects().length);
    for (const b of document.querySelectorAll(".dock > button, .dock .seg button")) {
        if (!b.getClientRects().length) continue;
        b.focus();  // a keyboard's focus: the ring shows
        await new Promise((r) => setTimeout(r, 50));
        const cs = getComputedStyle(b), r = b.getBoundingClientRect();
        const reach = parseFloat(cs.outlineOffset) + parseFloat(cs.outlineWidth);  // past the button's edge
        // Its neighbours: the buttons beside it in its segment, or the dock's items beside it.
        const row = b.closest(".seg") ? [...b.parentElement.children] : tools;
        const me = row.indexOf(b);
        const prev = row[me - 1]?.getBoundingClientRect().right ?? -1e9;
        const next = row[me + 1]?.getBoundingClientRect().left ?? 1e9;
        out.push({ label: b.textContent.trim() || b.id, shown: cs.outlineStyle !== "none",
                   left: r.left - reach, prev, right: r.right + reach, next });
    }
    return out;
})()"""


@pytest.mark.parametrize("size", ["phone", "laptop"])
def test_the_docks_focus_rings_stay_off_their_neighbours(viewer, size):
    """2 px apart on a phone, the Reset button's ring ran 2 px over the speed buttons beside it; and the speed
    buttons, edge to edge at every size, each ran 4 px over the next."""

    async def scenario(tab, _server):
        return await tab.eval(RINGS)

    rings = viewer(size, "en", scenario)
    assert len(rings) == 8, rings  # play, reset, four speeds, fit, record
    for ring in rings:
        assert ring["shown"] and ring["prev"] <= ring["left"] + 0.01 and ring["right"] <= ring["next"] + 0.01, ring


def test_a_face_off_tables_words_stay_inside_their_columns(viewer):
    """Three columns of "gravity compensation" ran into each other as "compensationcompensationcompensation"."""

    async def scenario(tab, _server):
        await _start(tab, "arm_faceoff")
        return await tab.eval("""[...document.querySelectorAll(".faceoff-table td")]
            .filter((td) => td.scrollWidth > td.clientWidth + 1).map((td) => td.textContent)""")

    assert viewer("laptop", "en", scenario) == []


MARKERS = """(() => {
    const out = [];
    window.__robotforge.three().scene.traverse((o) => {
        if (!o.isMesh || o.geometry.type !== "SphereGeometry" || o.geometry.parameters.radius !== 0.018) return;
        let shown = o.visible;
        for (let p = o.parent; p; p = p.parent) shown = shown && p.visible;
        if (shown) out.push(o.getWorldPosition(o.position.clone()).toArray().map((v) => Math.round(v * 1000) / 1000));
    });
    return out;
})()"""


def test_the_motion_planning_target_shows_where_it_is(viewer):
    """The lesson sends its drawing once; it came before the lesson's scene had loaded in the page and was dropped,
    so step 1's "amber ball on the right" never showed, and the target and the clash marker sat at the world's
    origin as two dots between the arms."""

    async def scenario(tab, _server):
        await _start(tab, "arm_planning")
        return await tab.eval(MARKERS)

    shown = viewer("laptop", "en", scenario)
    assert len(shown) == 1 and shown[0] != [0, 0, 0], shown  # the target alone, at its place


TAG_RECTS = """(async () => {
    const THREE = await import(new URL("static/vendor/three/three.module.min.js", location).href);
    const rf = window.__robotforge;
    const { scene, camera } = rf.three();
    camera.updateMatrixWorld();
    const [W, H] = [innerWidth, innerHeight];
    const [sx, sy] = [camera.projectionMatrix.elements[0], camera.projectionMatrix.elements[5]];
    const px = (p) => [((p.x + 1) / 2) * W, ((1 - p.y) / 2) * H];
    const tags = [];
    scene.traverse((o) => { if (o.isSprite && o.visible && o.renderOrder === 10) tags.push(o); });
    return { free: rf.freeRect(), tags: tags.map((s) => {
        const p = s.getWorldPosition(new THREE.Vector3()).project(camera);
        const [w, h] = [s.scale.x * sx, s.scale.y * sy];
        const [left, top] = px({ x: p.x - s.center.x * w, y: p.y + (1 - s.center.y) * h });
        const [right, bottom] = px({ x: p.x + (1 - s.center.x) * w, y: p.y - s.center.y * h });
        const foot = px(new THREE.Vector3(s.position.x, s.position.y, 0).project(camera));
        const base = px(new THREE.Vector3(s.position.x, s.position.y, 0.05).project(camera));  // the robot's base
        return { left, top, right, bottom, foot, base };
    }), target: (() => {  // the IK lesson's target ball, its middle and its radius on the screen
        const at = rf.lesson()?.live?.target;
        if (!at) return null;
        const ball = new THREE.Vector3().fromArray(at);
        const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).multiplyScalar(0.03);
        const [x, y] = px(ball.clone().project(camera));
        const [x2] = px(ball.clone().add(right).project(camera));
        return { x, y, r: Math.abs(x2 - x) };
    })() };
})()"""


def _inside(rect: dict[str, float], x: float, y: float) -> bool:
    return rect["left"] < x < rect["right"] and rect["top"] < y < rect["bottom"]


@pytest.mark.parametrize(
    "lesson, lang, robots", [("ik_arm", "en", 2), ("arm_planning", "en", 1), ("arm_faceoff", "zh-CN", 3)]
)
def test_a_name_tag_stays_in_the_free_view_while_its_robot_is_in_sight(viewer, lesson, lang, robots):
    """On a phone the PiPER's tag stood under the plots strip, and seen from above in the planning lesson only the
    bottom edge of its tag showed, at the top of the window. Held down below the plots, the planning lesson's tag
    then sat on its arm's base, and the Panda's tag, lifted clear of the PiPER's, went 7 px under the plots again."""

    async def scenario(tab, _server):
        await _start(tab, lesson)
        await tab.until("!window.__robotforge.tweening()")
        await asyncio.sleep(1.0)
        return await tab.eval(TAG_RECTS)

    seen = viewer("phone", lang, scenario)
    free, tags = seen["free"], seen["tags"]
    shown = [t for t in tags if _inside(free, *t["foot"])]
    assert len(shown) == robots, seen  # the planning lesson looks at our arm; the PiPER is out of the picture
    for t in shown:
        assert t["top"] >= free["top"] and t["left"] >= free["left"] - 1 and t["right"] <= free["right"] + 1, seen
        assert t["bottom"] <= free["bottom"] and not _inside(t, *t["base"]), seen


def _clear_of(tag: dict[str, float], x: float, y: float, r: float) -> bool:
    """Is a circle at (x, y) of radius r wholly outside the tag's rectangle?"""
    nearest_x = min(max(x, tag["left"]), tag["right"])
    nearest_y = min(max(y, tag["top"]), tag["bottom"])
    return (x - nearest_x) ** 2 + (y - nearest_y) ** 2 >= r**2


@pytest.mark.parametrize("size", ["phone", "laptop"])
def test_the_ik_target_raised_to_its_highest_is_not_under_a_name_tag(viewer, size):
    """The arms' tags stood at 0.5 and 0.6 m and draw over everything; the target goes up to 0.75 m, so on a phone
    the ball step 1 asks you to drag was hidden under "Our arm" (on a laptop its lower half)."""

    async def scenario(tab, server):
        await _start(tab, "ik_arm")
        runner = server.pg.lessons
        runner.handle({"action": "set", "control": "tx", "value": 0.15})
        seen = []
        for height in (0.7, 0.75):
            runner.handle({"action": "set", "control": "tz", "value": height})
            await asyncio.sleep(4.0)  # the arm reaches up as far as it can
            seen.append(await tab.eval(TAG_RECTS))
        return seen

    for seen in viewer(size, "en", scenario):
        ball = seen["target"]
        assert ball is not None and seen["tags"], seen
        for tag in seen["tags"]:
            assert _clear_of(tag, ball["x"], ball["y"], ball["r"]), seen


# For each robot in a lesson, the share of its mesh vertices (as the viewer's camera projects them) under the name tag
# of another robot. A tag belongs to the robot whose middle is nearest the point it stands on.
UNDER_OTHER_TAGS = """(async () => {
    const THREE = await import(new URL("static/vendor/three/three.module.min.js", location).href);
    const rf = window.__robotforge;
    const { scene, camera } = rf.three();
    camera.updateMatrixWorld();
    scene.updateMatrixWorld(true);
    const [sx, sy] = [camera.projectionMatrix.elements[0], camera.projectionMatrix.elements[5]];
    const tags = [];
    scene.traverse((o) => { if (o.isSprite && o.visible && o.renderOrder === 10) tags.push(o); });
    const rects = tags.map((s) => {
        const p = s.getWorldPosition(new THREE.Vector3()).project(camera);
        const [w, h] = [s.scale.x * sx, s.scale.y * sy];
        const left = p.x - s.center.x * w, bottom = p.y - s.center.y * h;
        return { left, right: left + w, bottom, top: bottom + h, at: s.position };
    });
    const points = {}, middle = {};
    const v = new THREE.Vector3();
    scene.traverse((o) => {
        const body = rf.scene.bodies[o.userData.body];
        if (!o.isMesh || !o.visible || !body?.dynamic || !body.name.includes("/")) return;
        const robot = body.name.split("/")[0];
        const pos = o.geometry.attributes.position;
        const list = (points[robot] ??= []), m = (middle[robot] ??= [0, 0, 0]);
        for (let k = 0; k < pos.count; k++) {
            v.fromBufferAttribute(pos, k).applyMatrix4(o.matrixWorld);
            [m[0], m[1], m[2]] = [m[0] + v.x, m[1] + v.y, m[2] + 1];
            v.project(camera);
            if (v.z < 1) list.push([v.x, v.y]);
        }
    });
    const owner = rects.map((r) => Object.keys(middle).sort((a, b) =>
        Math.hypot(middle[a][0] / middle[a][2] - r.at.x, middle[a][1] / middle[a][2] - r.at.y) -
        Math.hypot(middle[b][0] / middle[b][2] - r.at.x, middle[b][1] / middle[b][2] - r.at.y))[0]);
    return Object.fromEntries(Object.entries(points).map(([robot, list]) => [robot, Math.max(0, ...rects.map((r, i) =>
        owner[i] === robot ? 0 : list.filter(([x, y]) => x > r.left && x < r.right && y > r.bottom && y < r.top).length
            / list.length))]));
})()"""


@pytest.mark.parametrize("size", ["phone", "laptop", "wide"])
@pytest.mark.parametrize("lesson", ["gait", "humanoid_faceoff"])
def test_a_name_tag_does_not_hide_another_robot(viewer, lesson, size):
    """Seen down the row, each tag stood over its own robot's top and on the robot behind it: the template dog's tag
    covered the Go2's front legs in the step that asks which legs step together, and on a phone 'Our humanoid' the
    G1's legs. The test models leave out the visual meshes (the G1 has none), so their collision shapes are shown."""

    async def scenario(tab, _server):
        await tab.run("document.getElementById('show-collision').click();")
        await _start(tab, lesson)
        await tab.until("!window.__robotforge.tweening()")
        await asyncio.sleep(1.0)
        return await tab.eval(UNDER_OTHER_TAGS)

    hidden = viewer(size, "en", scenario)
    assert len(hidden) == 2 and max(hidden.values()) < 0.05, hidden


# Two tags seen one behind the other, and a mark (as the IK lesson's target ball) just where the farther one would be
# lifted to clear the nearer. Returns the free view, the mark and each tag, in CSS px, after declutter has placed them.
LIFTED_ONTO_A_MARK = """(async () => {
    const THREE = await import(new URL("static/vendor/three/three.module.min.js", location).href);
    const { declutter } = await import(new URL("static/lesson.js", location).href);
    const rf = window.__robotforge;
    const { camera } = rf.three();
    camera.updateMatrixWorld();
    const [W, H] = [innerWidth, innerHeight];
    const [sx, sy] = [camera.projectionMatrix.elements[0], camera.projectionMatrix.elements[5]];
    const px = (p) => [((p.x + 1) / 2) * W, ((1 - p.y) / 2) * H];
    const tag = (at) => {
        const s = new THREE.Sprite(new THREE.SpriteMaterial());
        s.scale.set(0.12, 0.034, 1);
        s.position.copy(at);
        s.userData.foot = new THREE.Vector3(at.x, at.y, 0);
        return s;
    };
    const rect = (s) => {
        const p = s.position.clone().project(camera);
        const [w, h] = [s.scale.x * sx, s.scale.y * sy];
        const [left, top] = px({ x: p.x - s.center.x * w, y: p.y + (1 - s.center.y) * h });
        const [right, bottom] = px({ x: p.x + (1 - s.center.x) * w, y: p.y - s.center.y * h });
        return { left, top, right, bottom };
    };
    // The nearer a little under the middle of the view, at the depth of the point the camera looks at; the farther
    // half a metre behind it on the same line of sight.
    const depth = new THREE.Vector3(...rf.cameraTarget()).project(camera).z;
    const near = new THREE.Vector3(0, -0.2, depth).unproject(camera);
    const far = near.clone().addScaledVector(near.clone().sub(camera.position).normalize(), 0.5);
    const tags = [tag(near), tag(far)];
    declutter(tags);  // the farther tag is lifted over the nearer
    const lifted = rect(tags[1]);
    const [cx, cy] = [(lifted.left + lifted.right) / W - 1, 1 - (lifted.top + lifted.bottom) / H];
    const middle = new THREE.Vector3(cx, cy, far.clone().project(camera).z).unproject(camera);
    const r = 0.02;
    const edge = middle.clone().addScaledVector(new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0), r);
    const [mx, my] = px(middle.clone().project(camera)), [ex] = px(edge.project(camera));
    declutter(tags, [{ at: middle, r }]);
    return { free: rf.freeRect(), mark: { x: mx, y: my, r: Math.abs(ex - mx) }, tags: tags.map(rect), lifted };
})()"""


def test_a_tag_moved_off_another_tag_does_not_land_on_what_the_lesson_marks(viewer):
    """Lifted, set beside or dropped below the tag it covered, a tag was placed without the check its first place
    gets: it could land on the IK lesson's target ball, on its own robot, or under the lesson sheet."""

    async def scenario(tab, _server):
        await tab.until("!window.__robotforge.tweening()")
        return await tab.eval(LIFTED_ONTO_A_MARK)

    seen = viewer("laptop", "en", scenario)
    mark, (near, far), free = seen["mark"], seen["tags"], seen["free"]
    assert _clear_of(near, mark["x"], mark["y"], mark["r"]), seen
    assert _clear_of(far, mark["x"], mark["y"], mark["r"]), seen  # it was lifted onto the mark
    apart = far["right"] <= near["left"] or far["left"] >= near["right"]
    assert apart or far["bottom"] <= near["top"] or far["top"] >= near["bottom"], seen
    assert free["left"] <= far["left"] and far["right"] <= free["right"], seen
    assert free["top"] <= far["top"] and far["bottom"] <= free["bottom"], seen
