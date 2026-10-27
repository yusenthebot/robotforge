"""Real pointer input drives the actual simulation without a keyboard or gamepad."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from test_viewer_creation_browser import click
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_touch_build_browser import touch


async def point(tab, direction):
    return await tab.eval(f"""(()=>{{
      const e=document.querySelector('#screen-drive [data-direction={json.dumps(direction)}]');
      const r=e.getBoundingClientRect();
      if(e.disabled||e.hidden||!r.width)throw Error('unavailable '+e.dataset.direction);
      const x=r.x+r.width/2,y=r.y+r.height/2,hit=document.elementFromPoint(x,y);
      if(hit!==e&&!e.contains(hit))throw Error('covered');return [x,y];}})()""")


async def ready(tab, name="rover"):
    await tab.run(f"window.__robotforge.select({json.dumps(name)})")
    await tab.until(
        "!document.getElementById('screen-drive').hidden && "
        "!document.querySelector('#screen-drive [data-direction=fwd]').disabled"
    )
    await tab.run("""window.__driveMessages=[];window.__drivePointers=[];
      for(const type of ['pointerdown','pointerup','pointercancel','lostpointercapture']){
        document.addEventListener(type,e=>window.__drivePointers.push({type,id:e.pointerId,
          direction:e.target.closest('[data-direction]')?.dataset.direction}),true);
      }
      const native=WebSocket.prototype.send;
      WebSocket.prototype.send=function(s){window.__driveSocket=this;const m=JSON.parse(s);
        if(m.type==='teleop')window.__driveMessages.push(m);return native.call(this,s)};""")


def world(pg):
    pg.add_robot("rover", template="rover", pos=[0, 0])
    pg.add_robot("dog", template="quadruped", pos=[0, 2])
    pg.add_robot("arm", template="arm", pos=[0, -2])


def command(server, robot="rover"):
    pilot = server.pg.session.controllers.get(robot)
    return (float(getattr(pilot, "speed", 0)), float(getattr(pilot, "turn", 0)))


async def until(check):
    for _ in range(150):
        if check():
            return
        await asyncio.sleep(0.04)
    raise AssertionError("simulation command did not reach expected state")


@pytest.mark.parametrize("lang", ["en", "zh-CN"])
def test_phone_two_fingers_drive_and_turn_release_independently(viewer, lang):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab)
        before = np.array(server.pg.behaviors.pose("rover")["position"][:2])
        fwd, left = await point(tab, "fwd"), await point(tab, "left")
        await touch(tab, "touchStart", [(1, *fwd)])
        await until(lambda: command(server)[0] > 0)
        await until(lambda: np.linalg.norm(np.array(server.pg.behaviors.pose("rover")["position"][:2]) - before) > 0.2)
        await touch(tab, "touchStart", [(1, *fwd), (2, *left)])
        await until(lambda: command(server)[0] > 0 and command(server)[1] > 0)
        await touch(tab, "touchEnd", [(1, *fwd)])
        await until(lambda: command(server)[0] == 0 and command(server)[1] > 0)
        await touch(tab, "touchEnd")
        await until(lambda: command(server) == (0, 0))
        sent = await tab.eval("window.__driveMessages")
        assert any(m.get("hold_ms") == 750 and m["speed"] > 0 and m["turn"] > 0 for m in sent)
        assert sent[-1]["speed"] == sent[-1]["turn"] == 0
        count = len(sent)
        await asyncio.sleep(0.85)
        assert len(await tab.eval("window.__driveMessages")) == count, "idle controls must not send heartbeats"
        assert not tab.errors

    viewer("phone", lang, scenario, world=world)


def test_same_button_two_fingers_opposite_directions_and_keyboard_remain_independent(viewer):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab)
        fwd, back = await point(tab, "fwd"), await point(tab, "back")
        await touch(tab, "touchStart", [(1, fwd[0] - 5, fwd[1])])
        await touch(tab, "touchStart", [(1, fwd[0] - 5, fwd[1]), (2, fwd[0] + 5, fwd[1])])
        await touch(tab, "touchEnd", [(1, fwd[0] - 5, fwd[1])])
        assert command(server)[0] > 0
        await touch(tab, "touchStart", [(2, fwd[0] + 5, fwd[1]), (3, *back)])
        await until(lambda: command(server)[0] == 0)
        await touch(tab, "touchEnd", [(3, *back)])
        await until(lambda: command(server)[0] > 0)
        await tab.page.call("Input.dispatchKeyEvent", type="keyDown", key="w", code="KeyW", windowsVirtualKeyCode=87)
        await touch(tab, "touchEnd")
        await until(lambda: command(server)[0] > 0)
        assert not (await tab.eval("window.__driveMessages.at(-1)")).get("hold_ms")
        await tab.page.call("Input.dispatchKeyEvent", type="keyUp", key="w", code="KeyW", windowsVirtualKeyCode=87)
        await until(lambda: command(server) == (0, 0))

    viewer("phone", "en", scenario, world=world)


@pytest.mark.parametrize(
    "boundary",
    ["cancel", "blur", "hidden", "close", "select", "pause", "reset", "external_reset", "disconnect", "creation"],
)
def test_held_direction_does_not_survive_runtime_boundaries(viewer, boundary):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab)
        fwd = await point(tab, "fwd")
        await touch(tab, "touchStart", [(1, *fwd)])
        await until(lambda: command(server)[0] > 0)
        if boundary == "cancel":
            await touch(tab, "touchCancel")
        elif boundary == "blur":
            await tab.run("window.dispatchEvent(new Event('blur'))")
        elif boundary == "hidden":
            await tab.run(
                "Object.defineProperty(document,'hidden',{configurable:true,get:()=>true});"
                "document.dispatchEvent(new Event('visibilitychange'))"
            )
        elif boundary == "close":
            await tab.run("document.getElementById('screen-drive-close').click()")
        elif boundary == "select":
            await tab.run("window.__robotforge.select('dog')")
        elif boundary in {"pause", "reset"}:
            await tab.run(f"document.getElementById('btn-{'play' if boundary == 'pause' else 'reset'}').click()")
        elif boundary == "external_reset":
            server.pg.session.reset()
        elif boundary == "disconnect":
            await tab.run("window.__driveSocket.close()")
        else:
            # Keep the native held finger down while exercising the edit transition.
            await tab.run(
                "document.getElementById('world-toggle').click();document.querySelector('[data-tab=objects]').click();document.querySelector('[data-object=ball]').click()"
            )
        await until(lambda: command(server) == (0, 0))
        if boundary != "cancel":
            await touch(tab, "touchEnd")
        if boundary == "hidden":
            await tab.run("delete document.hidden;document.dispatchEvent(new Event('visibilitychange'))")
        if boundary == "pause":
            await tab.run("document.getElementById('btn-play').click()")
        await asyncio.sleep(1.0)
        assert command(server) == (0, 0)
        assert not await tab.eval("document.querySelector('#screen-drive [aria-pressed=true]') !== null")
        assert not tab.errors

    viewer("phone", "en", scenario, world=world)


def test_desktop_explicit_controls_keyboard_button_and_saved_preference(viewer):  # noqa: F811
    async def scenario(tab, server):
        assert await tab.eval("document.getElementById('screen-drive').hidden")
        await click(tab, "#drawer-toggle")
        await click(tab, "#show-screen-drive")
        await click(tab, "#drawer-toggle")
        await ready(tab)
        await tab.run("document.querySelector('#screen-drive [data-direction=fwd]').focus()")
        await tab.page.call("Input.dispatchKeyEvent", type="keyDown", key=" ", code="Space", windowsVirtualKeyCode=32)
        await until(lambda: command(server)[0] > 0)
        await tab.page.call("Input.dispatchKeyEvent", type="keyUp", key=" ", code="Space", windowsVirtualKeyCode=32)
        await until(lambda: command(server) == (0, 0))
        await tab.navigate()
        await tab.until("!document.getElementById('screen-drive').hidden")
        assert await tab.eval("document.getElementById('show-screen-drive').checked")
        assert command(server) == (0, 0)

    viewer("laptop", "en", scenario, world=world)


@pytest.mark.parametrize("panel", ["world", "drawer"])
@pytest.mark.parametrize("width,height", [(320, 568), (390, 844)])
def test_phone_editor_panels_release_held_drive_and_preserve_preference(viewer, panel, width, height):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call(
            "Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=2, mobile=True
        )
        await ready(tab)
        preference = await tab.eval("localStorage.getItem('robotforge.screen-drive')")
        fwd = await point(tab, "fwd")
        await touch(tab, "touchStart", [(1, *fwd)])
        await until(lambda: command(server)[0] > 0)
        # Keep the native pointer held while a native keyboard activation opens the panel.
        await tab.run(
            f"document.getElementById('{panel}-toggle').focus();window.__panelClick=false;"
            f"document.getElementById('{panel}-toggle').addEventListener('click',e=>window.__panelClick=e.isTrusted,{{once:true}})"
        )
        for event in ["keyDown", "keyUp"]:
            await tab.page.call(
                "Input.dispatchKeyEvent",
                type=event,
                key="Enter",
                code="Enter",
                windowsVirtualKeyCode=13,
                text="\r" if event == "keyDown" else "",
            )
        await until(lambda: command(server) == (0, 0))
        assert await tab.eval("window.__panelClick && document.getElementById('screen-drive').hidden")
        assert not await tab.eval("document.querySelector('#screen-drive [aria-pressed=true]') !== null")
        assert await tab.eval("document.getElementById('show-screen-drive').checked")
        assert await tab.eval("localStorage.getItem('robotforge.screen-drive')") == preference
        await touch(tab, "touchEnd")
        await click(tab, f"#{panel}-toggle")
        await tab.until("!document.getElementById('screen-drive').hidden")
        await asyncio.sleep(0.8)
        assert command(server) == (0, 0)
        assert not tab.errors

    viewer("phone", "zh-CN", scenario, world=world)


def test_open_editor_hides_drive_only_while_window_is_narrow(viewer):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab)
        await click(tab, "#world-toggle")
        assert await tab.eval("document.getElementById('screen-drive').hidden")
        for width, hidden in [(900, False), (700, True), (701, False), (390, True)]:
            await tab.page.call(
                "Emulation.setDeviceMetricsOverride", width=width, height=844, deviceScaleFactor=2, mobile=True
            )
            await tab.until(f"document.getElementById('screen-drive').hidden==={str(hidden).lower()}")
            assert await tab.eval("document.getElementById('show-screen-drive').checked")
            assert command(server) == (0, 0)
        await click(tab, "#world-toggle")
        await tab.until("!document.getElementById('screen-drive').hidden")
        assert not tab.errors

    viewer("phone", "en", scenario, world=world)


@pytest.mark.parametrize("template", ["rover", "quadruped", "boat", "rov", "drone", "arm"])
def test_shared_capability_axes_choose_the_visible_buttons(viewer, template):  # noqa: F811
    def build(pg):
        if template in {"boat", "rov"}:
            pg.set_floor("sand", water=1.0)
        pg.add_robot("test", template=template)

    async def scenario(tab, server):
        await tab.run("window.__robotforge.select('test')")
        if template == "arm":
            assert await tab.eval("document.getElementById('screen-drive').hidden")
            return
        await tab.until("!document.getElementById('screen-drive').hidden")
        shown = await tab.eval(
            "[...document.querySelectorAll('#screen-drive [data-direction]')]"
            ".filter(b=>!b.hidden).map(b=>b.dataset.direction)"
        )
        assert set(shown) == {"fwd", "back", "left", "right"} | (
            {"up", "down"} if template in {"rov", "drone"} else set()
        ) | ({"slide_left", "slide_right"} if template == "drone" else set())
        rects = await tab.eval(
            "[...document.querySelectorAll('#screen-drive [data-direction]')].filter(b=>!b.hidden)"
            ".map(b=>{const r=b.getBoundingClientRect();return [r.x,r.y,r.right,r.bottom,r.width,r.height]})"
        )
        assert all(0 <= r[0] < r[2] <= 390 and 0 <= r[1] < r[3] <= 844 and r[4] >= 44 and r[5] >= 44 for r in rects)
        assert not tab.errors

    viewer("phone", "zh-CN", scenario, world=build)


def test_capture_keeps_one_direction_outside_the_button_and_lost_capture_stops(viewer):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab)
        fwd = await point(tab, "fwd")
        await touch(tab, "touchStart", [(1, *fwd)])
        await until(lambda: command(server)[0] > 0)
        await touch(tab, "touchMove", [(1, 360, 400)])
        assert command(server)[0] > 0
        await tab.run("""const e=document.querySelector('#screen-drive [data-direction=fwd]');
          const id=window.__drivePointers.find(e=>e.type==='pointerdown').id;e.releasePointerCapture(id);""")
        await until(lambda: command(server) == (0, 0))
        await touch(tab, "touchEnd")
        assert command(server) == (0, 0)

    viewer("phone", "en", scenario, world=world)


def test_gamepad_axis_priority_and_touch_release_keep_other_inputs(viewer):  # noqa: F811
    from test_viewer_gamepad import PAD, axes

    async def scenario(tab, server):
        await ready(tab)
        await tab.run(PAD)
        await tab.until("window.__robotforge.pad().connected")
        fwd = await point(tab, "fwd")
        await touch(tab, "touchStart", [(1, *fwd)])
        await until(lambda: command(server)[0] > 0)
        await tab.run(axes(a1=1))
        await until(lambda: command(server)[0] < 0)
        await touch(tab, "touchEnd")
        assert command(server)[0] < 0
        await tab.run(axes(a1=0))
        await until(lambda: command(server) == (0, 0))

    viewer("phone", "en", scenario, world=world)


def test_async_scene_load_disables_old_buttons_until_a_fresh_pointer(viewer):  # noqa: F811
    from test_viewer_startup_browser import READY, reload_instrumented

    async def scenario(tab, server):
        await reload_instrumented(tab)
        await tab.until(READY)
        await ready(tab)
        old = await point(tab, "fwd")
        await touch(tab, "touchStart", [(1, *old)])
        await until(lambda: command(server)[0] > 0)
        await tab.run("window.__startup.holdMesh=true")
        server.pg.update_robot("rover", pos=[1, 0])
        await tab.until("window.__startup.releases.length > 0")
        assert await tab.eval("document.getElementById('screen-drive').hidden")
        sent = len(await tab.eval("window.__driveMessages"))
        await tab.run("""document.querySelector('#screen-drive [data-direction=fwd]').dispatchEvent(
          new PointerEvent('pointerdown',{pointerId:9999,button:0,bubbles:true}));""")
        await asyncio.sleep(0.3)
        assert len(await tab.eval("window.__driveMessages")) == sent
        await touch(tab, "touchEnd")
        await tab.run("window.__startup.releaseMesh()")
        await tab.until(READY)
        await tab.until("!document.querySelector('#screen-drive [data-direction=fwd]').disabled")
        assert command(server) == (0, 0)
        fresh = await point(tab, "fwd")
        await touch(tab, "touchStart", [(2, *fresh)])
        await until(lambda: command(server)[0] > 0)
        await touch(tab, "touchEnd")
        await until(lambda: command(server) == (0, 0))

    viewer("phone", "en", scenario, world=world)
