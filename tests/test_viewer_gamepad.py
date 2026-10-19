"""A gamepad drives robots in the real viewer: simulated Gamepad API events in headless Chrome.

The page polls navigator.getGamepads() once a frame; here a test replaces it with a pad whose sticks and buttons the
test moves, and watches what the simulation does (the server runs in this process). Waits are on what the robots did,
never on how long it took. Skipped without Chrome; kept out of the default run and CI: `uv run pytest -m browser
tests/test_viewer_gamepad.py`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "viewer_qa.py"
FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"
WAIT = 60.0  # s a robot may take to do what the pad asked for: a slow machine, not a slow robot


def _load_qa() -> Any:
    spec = importlib.util.spec_from_file_location("viewer_qa_pad", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["viewer_qa_pad"] = module
    spec.loader.exec_module(module)
    return module


qa = _load_qa()


def _chrome() -> str | None:
    try:
        return qa.screenshots._chrome()
    except SystemExit:
        return None


pytestmark = [pytest.mark.browser, pytest.mark.skipif(_chrome() is None, reason="Google Chrome is not installed")]

PAD = """
window.__pad = {
  id: "Test Pad (STANDARD GAMEPAD)", index: 0, connected: true, mapping: "standard", axes: [0, 0, 0, 0],
  buttons: Array.from({ length: 17 }, () => ({ pressed: false, value: 0 })),
};
navigator.getGamepads = () => [window.__pad];
const e = new Event("gamepadconnected");
e.gamepad = window.__pad;
window.dispatchEvent(e);
"""


def axes(**values: float) -> str:
    """Set sticks: a0..a3 are the left x and y and the right x and y (up is negative, as the browser reports it)."""
    return "".join(f"window.__pad.axes[{k[1:]}] = {v};" for k, v in values.items())


def press(*buttons: int) -> str:
    """Hold exactly these buttons down."""
    held = list(buttons)
    return (
        "for (let i = 0; i < window.__pad.buttons.length; i++) "
        f"window.__pad.buttons[i] = {{ pressed: {held}.includes(i), value: {held}.includes(i) ? 1 : 0 }};"
    )


@pytest.fixture
def viewer(tmp_path, monkeypatch) -> Callable[..., Any]:
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))

    def run(scenario: Callable[[Any, Any], Awaitable[Any]], world: Callable[[Any], None], size: str = "laptop") -> Any:
        built = qa.build_world(tmp_path, world)

        async def main() -> Any:
            port = qa.screenshots._free_port()
            with qa.Server(built, port) as server:
                tab = qa.Tab("en", size, port, tmp_path)
                try:
                    await tab.open()
                    returning = "".join(f"localStorage.setItem({k!r}, {v!r});" for k, v in qa.RETURNING.items())
                    await tab.run(returning)
                    await tab.navigate()
                    return await scenario(tab, server)
                finally:
                    await tab.close()

        return asyncio.run(main())

    return run


def two_robots(pg: Any) -> None:
    pg.add_robot("dog", template="quadruped", pos=[-1.5, 0.0])
    pg.add_robot("arm", template="arm", pos=[1.5, 0.0])


async def until_server(check: Callable[[], bool], what: str) -> None:
    end = time.monotonic() + WAIT
    while not check():
        if time.monotonic() > end:
            raise TimeoutError(f"never {what}")
        await asyncio.sleep(0.2)


def pose(server: Any, robot: str) -> dict[str, Any]:
    return server.pg.behaviors.pose(robot)


def tool(server: Any, arm: str) -> np.ndarray:
    m, d = server.pg.session.model, server.pg.session.data
    return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{arm}/tcp")].copy()


def test_the_left_stick_walks_the_dog_the_right_stick_turns_it_and_letting_go_stops_it(viewer):
    async def scenario(tab, server):
        await tab.run("window.__robotforge.select('dog');")
        await tab.run(PAD)
        await tab.until("window.__robotforge.pad().connected && window.__robotforge.pad().profile === 'drive'")
        start = np.array(pose(server, "dog")["position"][:2])
        await tab.run(axes(a1=-1))  # the left stick all the way forward
        await until_server(
            lambda: np.linalg.norm(np.array(pose(server, "dog")["position"][:2]) - start) > 0.3, "walked"
        )
        assert await tab.eval("window.__robotforge.pad().drive.fwd") == 1
        walking = await tab.eval("document.querySelector('#drive-pad:not([hidden])') !== null")
        await tab.run(axes(a1=0))
        await tab.until("window.__robotforge.pad().drive.fwd === 0")
        await until_server(lambda: _speed(server, "dog") == 0, "stopped")
        stopped_at = np.array(pose(server, "dog")["position"][:2])
        await asyncio.sleep(1.5)
        drift = float(np.linalg.norm(np.array(pose(server, "dog")["position"][:2]) - stopped_at))
        heading = pose(server, "dog")["heading_deg"]
        await tab.run(axes(a2=-1))  # the right stick to the left: turn left
        await until_server(lambda: _turned(pose(server, "dog")["heading_deg"], heading) > 20, "turned")
        await tab.run(axes(a2=0))
        await until_server(lambda: _speed(server, "dog") == 0, "stopped turning")
        return walking, drift, _turned(pose(server, "dog")["heading_deg"], heading)

    walking, drift, turned = viewer(scenario, two_robots)
    assert walking, "the pad chip is not shown"
    assert drift < 0.25, drift  # it stopped: a dog still walking would have gone further
    assert turned > 20


def _speed(server: Any, robot: str) -> float:
    status = next((b for b in server.pg.behaviors.status() if b["robot"] == robot), {})
    return abs(float(status.get("speed", 0.0))) + abs(float(status.get("turn", 0.0)))


def _turned(now: float, before: float) -> float:
    return (now - before + 180) % 360 - 180


def test_the_sticks_jog_an_arm_and_a_and_b_close_and_open_its_gripper(viewer):
    async def scenario(tab, server):
        pg = server.pg
        pg.behaviors.move_arm("arm", [1.65, 0.0, 0.2])
        await until_server(lambda: pg.session.controllers["arm"].done, "reached the start")
        # From behind the arm, looking along +x: up on the left stick is +x, the arm's own reach direction
        await tab.run(
            "window.__robotforge.lookAt([0.3, 0.0, 0.9], [1.65, 0.0, 0.1]); window.__robotforge.select('arm');"
        )
        await tab.run(PAD)
        await tab.until("window.__robotforge.pad().profile === 'arm'")
        assert await tab.eval("!document.getElementById('drive').hidden")  # the HUD appears for an arm too
        before = tool(server, "arm")
        await tab.run(axes(a1=-1))
        await until_server(lambda: tool(server, "arm")[0] - before[0] > 0.04, "jogged away")
        await tab.run(axes(a1=0))
        await asyncio.sleep(1.0)
        held = tool(server, "arm")
        await asyncio.sleep(1.0)
        still = float(np.linalg.norm(tool(server, "arm") - held))
        sideways = abs(held[1] - before[1])
        await tab.run(axes(a3=-1))  # the right stick up raises the tool
        await until_server(lambda: tool(server, "arm")[2] - held[2] > 0.03, "raised")
        await tab.run(axes(a3=0))
        motion = pg.session.controllers["arm"]
        data = pg.session.data
        await tab.run(press(0))  # A
        await until_server(lambda: abs(data.ctrl[motion.gripper] - motion.gripper_closed) < 1e-6, "closed")
        await tab.run(press())
        await tab.run(press(1))  # B
        await until_server(lambda: abs(data.ctrl[motion.gripper] - motion.gripper_open) < 1e-6, "opened")
        await tab.run(press())
        return still, sideways

    still, sideways = viewer(scenario, two_robots)
    assert still < 0.01, still  # a stick let go holds the tool where it is
    assert sideways < 0.03, sideways


def test_a_pad_that_goes_away_stops_what_it_drove_and_the_page_says_so(viewer):
    async def scenario(tab, server):
        await tab.run("window.__robotforge.select('dog');")
        await tab.run(PAD)
        await tab.until("window.__robotforge.pad().connected")
        await tab.run(axes(a1=-1))
        await until_server(lambda: _speed(server, "dog") > 0.05, "set off")
        await tab.run("navigator.getGamepads = () => [null];")  # unplugged with the stick still pushed
        await tab.until("!window.__robotforge.pad().connected")
        await until_server(lambda: _speed(server, "dog") == 0, "stopped")
        return await tab.eval("document.getElementById('toast').textContent")

    assert "disconnected" in viewer(scenario, two_robots)


def test_the_keyboard_still_drives_with_a_pad_plugged_in(viewer):
    async def scenario(tab, server):
        await tab.run("window.__robotforge.select('dog');")
        await tab.run(PAD)
        await tab.until("window.__robotforge.pad().connected")
        start = np.array(pose(server, "dog")["position"][:2])
        await tab.page.call("Input.dispatchKeyEvent", type="keyDown", key="w", code="KeyW", windowsVirtualKeyCode=87)
        await until_server(
            lambda: np.linalg.norm(np.array(pose(server, "dog")["position"][:2]) - start) > 0.2, "walked"
        )
        await tab.page.call("Input.dispatchKeyEvent", type="keyUp", key="w", code="KeyW", windowsVirtualKeyCode=87)
        await until_server(lambda: _speed(server, "dog") == 0, "stopped")
        return True

    assert viewer(scenario, two_robots)


def test_the_page_shows_what_is_being_recorded(viewer):
    async def scenario(tab, server):
        server.pg.demonstrations.start("dog", "shown")
        await tab.until("!document.getElementById('demo-chip').hidden")
        text = await tab.eval("document.getElementById('demo-text').textContent")
        server.pg.demonstrations.stop("shown")
        await tab.until("document.getElementById('demo-chip').hidden")
        return text

    assert "shown" in viewer(scenario, two_robots)


def test_the_hud_and_chip_stay_inside_a_phone_screen(viewer):
    async def scenario(tab, server):
        await tab.run("window.__robotforge.select('dog');")
        await tab.run(PAD)
        await tab.until("window.__robotforge.pad().connected")
        shown = "[...document.querySelectorAll('#drive-pad, #drive')].filter((e) => !e.hidden)"
        return await tab.eval(shown + ".map((e) => e.getBoundingClientRect().right)")

    rights = viewer(scenario, two_robots, size="phone")
    assert all(math.isfinite(r) and r <= 390 for r in rights), rights


MOBILE_ARM = {
    "color": "yellow",
    "motor": "jga25",
    "base": {"shape": "box", "size": [0.34, 0.22, 0.06], "mass": 3.0, "sensors": ["imu"]},
    "parts": [
        {"name": "left_wheel", "at": [0, 0.13, -0.01], "shape": "wheel", "size": [0.1, 0.03], "mirror": True},
        {"name": "front_caster", "at": [0.14, 0, -0.04], "shape": "caster", "size": [0.04]},
        {"name": "rear_caster", "at": [-0.14, 0, -0.04], "shape": "caster", "size": [0.04]},
        {"name": "arm", "at": "top", "shift": [0.06, 0, 0], "robot": {"template": "arm", "params": {"reach": 0.45}}},
    ],
}


def rover_world(pg: Any) -> None:
    pg.add_robot("rover", template="rover", pos=[0.0, 0.0])
    pg.add_robot("mm", build=MOBILE_ARM, pos=[0.0, 2.0])


def test_the_left_stick_drives_a_wheeled_rover_and_turns_it(viewer):
    async def scenario(tab, server):
        await tab.run("window.__robotforge.select('rover');")
        await tab.run(PAD)
        await tab.until("window.__robotforge.pad().profile === 'drive'")
        start = np.array(pose(server, "rover")["position"][:2])
        heading = pose(server, "rover")["heading_deg"]
        await tab.run(axes(a1=-1))
        await until_server(
            lambda: np.linalg.norm(np.array(pose(server, "rover")["position"][:2]) - start) > 0.3, "drove"
        )
        await tab.run(axes(a1=0, a0=-1))  # the left stick to the left: turn left on the spot
        await until_server(lambda: _turned(pose(server, "rover")["heading_deg"], heading) > 20, "turned")
        await tab.run(axes(a0=0))
        await until_server(lambda: _speed(server, "rover") == 0, "stopped")
        return True

    assert viewer(scenario, rover_world)


def test_y_switches_a_mobile_manipulator_between_driving_and_its_arm(viewer):
    async def scenario(tab, server):
        pg = server.pg
        await tab.run("window.__robotforge.select('mm');")
        await tab.run(PAD)
        await tab.until("window.__robotforge.pad().profile === 'both'")
        assert await tab.eval("window.__robotforge.pad().mode") == "drive"
        base = np.array(pose(server, "mm")["position"][:2])
        await tab.run(axes(a1=-1))
        await until_server(lambda: np.linalg.norm(np.array(pose(server, "mm")["position"][:2]) - base) > 0.2, "drove")
        await tab.run(axes(a1=0))
        await until_server(lambda: _speed(server, "mm") == 0, "stopped")
        await tab.run(press(3))  # Y: the sticks now move the arm's tool
        await tab.until("window.__robotforge.pad().mode === 'arm'")
        await tab.run(press())
        parked = np.array(pose(server, "mm")["position"][:2])
        before = tool(server, "mm/arm")
        await tab.run(axes(a3=-1))  # the right stick up raises the tool
        await until_server(lambda: tool(server, "mm/arm")[2] - before[2] > 0.02, "raised the tool")
        await tab.run(axes(a3=0))
        moved = float(np.linalg.norm(np.array(pose(server, "mm")["position"][:2]) - parked))
        await tab.run(press(3))
        await tab.until("window.__robotforge.pad().mode === 'drive'")
        await tab.run(press())
        return moved, pg.behaviors.arms("mm")

    moved, arms = viewer(scenario, rover_world)
    assert arms == ["mm/arm"]
    assert moved < 0.05, moved  # in arm mode the base stays where it is
