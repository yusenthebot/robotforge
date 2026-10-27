"""Native touch/mouse camera gestures stay separate from deliberate placement."""

from __future__ import annotations

import asyncio
import json

import pytest
from test_viewer_build_flow_browser import instrument_wire
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_objects_browser import choose_object


async def touch(tab, kind, points=()):
    await tab.page.call(
        "Input.dispatchTouchEvent",
        type=kind,
        touchPoints=[{"id": i, "x": x, "y": y} for i, x, y in points],
    )
    await asyncio.sleep(0.04)


async def eye(tab):
    return await tab.eval("window.__robotforge.three().camera.position.toArray()")


async def distance(tab):
    return await tab.eval("""(()=>{const rf=window.__robotforge,eye=rf.three().camera.position.toArray(),
      target=rf.cameraTarget();return Math.hypot(...eye.map((v,i)=>v-target[i]));})()""")


async def settled_direction(tab):
    previous = None
    for _ in range(30):
        direction = await tab.eval("""(()=>{const c=window.__robotforge.three().camera;
          return [0,0,-1].map((_,i)=>c.matrixWorld.elements[8+i]);})()""")
        if previous is not None and direction == pytest.approx(previous, abs=1e-5):
            return direction
        previous = direction
        await asyncio.sleep(0.1)
    raise AssertionError("Camera damping did not settle")


async def blue_at(tab, point):
    await tab.until("window.__robotforge.renderedVersion === window.__robotforge.scene.version")
    return await tab.eval(
        """(()=>{const rf=window.__robotforge,{w,h,data}=rf.pixels(),p=rf.screenOfPoint(POINT);
          const x=Math.round(p[0]*w/innerWidth),y=h-1-Math.round(p[1]*h/innerHeight);
          let blue=0;for(let dy=-4;dy<=4;dy++)for(let dx=-4;dx<=4;dx++){
            const px=x+dx,py=y+dy;if(px<0||px>=w||py<0||py>=h)continue;
            const i=(py*w+px)*4;if(data[i+2]>data[i]+25&&data[i+2]>data[i+1]+5)blue++;
          }return blue;})()""".replace("POINT", json.dumps(point))
    )


async def tap_at(tab, pos, before):
    x, y = await tab.eval(f"window.__robotforge.screenOfPoint({json.dumps(pos)})")
    await touch(tab, "touchStart", [(1, x, y)])
    assert not await tab.eval("window.__robotforge.busy().busy")
    await touch(tab, "touchEnd")
    await tab.until(f"window.__robotforge.scene.version > {before} && !window.__robotforge.busy().busy")


@pytest.mark.parametrize("lang", ["en", "zh-CN"])
def test_phone_orbit_pinch_and_repeat_taps_do_not_add_during_camera_gestures(viewer, lang):  # noqa: F811
    async def scenario(tab, server):
        await tab.eval("window.__robotforge.lookAt([4,-8,6],[0,0,0])")
        await choose_object(tab, "pin")
        await click(tab, '[data-size="large"]')
        await click(tab, "#place-rotate")
        await click(tab, "#place-repeat")
        await tab.run("""window.__gestureSent=[];const send=WebSocket.prototype.send;
          WebSocket.prototype.send=function(s){window.__gestureSent.push(JSON.parse(s));return send.call(this,s)};""")
        before = await eye(tab)
        await touch(tab, "touchStart", [(1, 170, 320)])
        assert not server.pg.store.load().objects  # old implementation commits on touchStart
        for x, y in [(190, 325), (215, 335), (240, 345)]:
            await touch(tab, "touchMove", [(1, x, y)])
        assert await eye(tab) != pytest.approx(before)
        await touch(tab, "touchMove", [(1, 170, 320)])
        await touch(tab, "touchEnd")
        assert not server.pg.store.load().objects
        assert await tab.eval("document.getElementById('place-here').disabled")
        before = await distance(tab)
        await touch(tab, "touchStart", [(1, 140, 330)])
        await touch(tab, "touchStart", [(1, 140, 330), (2, 250, 330)])
        await touch(tab, "touchMove", [(1, 110, 320), (2, 280, 340)])
        # Lift only finger 2, then continue orbiting with finger 1 before the final release.
        await touch(tab, "touchEnd", [(2, 280, 340)])
        direction = await settled_direction(tab)
        await touch(tab, "touchMove", [(1, 135, 340)])
        changed = await tab.eval("window.__robotforge.three().camera.matrixWorld.elements.slice(8,11)")
        assert changed != pytest.approx(direction, abs=0.002)
        await touch(tab, "touchEnd")
        assert not server.pg.store.load().objects
        after = await distance(tab)
        assert after != pytest.approx(before, rel=0.05)
        assert not await tab.eval("window.__gestureSent.some(m=>m.type==='drag_start')")
        version = await tab.eval("window.__robotforge.scene.version")
        await tap_at(tab, [0, 0, 0], version)
        assert len(server.pg.store.load().objects) == 1
        assert await blue_at(tab, [0, 0, 0.18]) > 5, "The first prop must remain visible at the kept camera distance"
        first_name = server.pg.store.load().objects[0].name
        # The existing click guard prevents another addition, but must not swallow camera input.
        before = await eye(tab)
        await touch(tab, "touchStart", [(1, 170, 320)])
        await touch(tab, "touchMove", [(1, 205, 335)])
        await touch(tab, "touchEnd")
        assert await eye(tab) != pytest.approx(before)
        assert len(server.pg.store.load().objects) == 1
        await asyncio.sleep(0.55)
        version = await tab.eval("window.__robotforge.scene.version")
        await tap_at(tab, [1, 0, 0], version)
        objects = server.pg.store.load().objects
        assert len(objects) == 2
        second_name = objects[1].name
        assert all(o.yaw == 90 and o.size == pytest.approx([0.15, 0.36]) for o in objects)
        await click(tab, "#place-cancel")
        await click(tab, "#undo-add")
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('undo-add').hidden")
        assert [o.name for o in server.pg.store.load().objects] == [first_name]
        assert second_name != first_name
        assert await tab.eval("document.getElementById('empty').hidden")
        assert await blue_at(tab, [0, 0, 0.18]) > 5
        assert not tab.errors

    viewer("phone", lang, scenario, world=lambda pg: None)


@pytest.mark.parametrize("boundary", ["cancel", "reset", "edit"])
def test_held_touch_cannot_place_after_cancel_or_scene_change_and_fresh_tap_recovers(viewer, boundary):  # noqa: F811
    async def scenario(tab, server):
        await choose_object(tab, "ball")
        await touch(tab, "touchStart", [(1, 180, 320)])
        # Explicit keyboard placement must not use a stale center marker while a finger is down.
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        assert not server.pg.store.load().objects
        if boundary == "cancel":
            await touch(tab, "touchCancel")
        else:
            if boundary == "reset":
                server.pg.session.reset()
                server.pg.session.paused = True
            else:
                server.pg.add_object("external", shape="sphere", size=[0.1], pos=[3, 0])
            await tab.until("document.getElementById('placement-bar').hidden")
            await touch(tab, "touchEnd")
            version = server.pg.session.version
            await tab.until(
                f"window.__robotforge.scene.version === {version} && window.__robotforge.renderedVersion === {version}"
            )
            await choose_object(tab, "ball")
        expected = 1 if boundary == "edit" else 0
        assert len(server.pg.store.load().objects) == expected
        before = await tab.eval("window.__robotforge.scene.version")
        await tap_at(tab, [0, 0, 0], before)
        assert len(server.pg.store.load().objects) == expected + 1
        assert not tab.errors

    viewer("phone", "en", scenario, world=lambda pg: None)


def test_touch_begun_while_sending_stays_ineligible_after_reply(viewer):  # noqa: F811
    async def scenario(tab, server):
        await instrument_wire(tab)
        await choose_object(tab, "ball")
        await click(tab, "#place-repeat")
        await tab.run("window.__flow.holdReply=true")
        x, y = await tab.eval("window.__robotforge.screenOfPoint([0,0,0])")
        await touch(tab, "touchStart", [(1, x, y)])
        await touch(tab, "touchEnd")
        await tab.until("window.__flow.replies.length === 1")
        assert len(server.pg.store.load().objects) == 1
        await touch(tab, "touchStart", [(2, 240, 320)])
        await tab.run("window.__flow.releaseReplies()")
        await tab.until("!window.__robotforge.busy().busy")
        await asyncio.sleep(0.55)
        await touch(tab, "touchEnd")
        assert len(server.pg.store.load().objects) == 1
        assert await tab.eval("document.getElementById('place-here').disabled")
        assert not tab.errors

    viewer("phone", "en", scenario, world=lambda pg: None)


def test_desktop_drag_orbits_right_drag_pans_wheel_zooms_and_click_still_places(viewer):  # noqa: F811
    async def scenario(tab, server):
        await choose_object(tab, "block")
        # Pointer Events reports only the first press and final release of a mouse chord.
        for kind, button, buttons in [
            ("mousePressed", "left", 1),
            ("mousePressed", "right", 3),
            ("mouseReleased", "left", 2),
            ("mouseReleased", "right", 0),
        ]:
            await tab.page.call(
                "Input.dispatchMouseEvent", type=kind, x=600, y=320, button=button, buttons=buttons, clickCount=1
            )
        await asyncio.sleep(0.1)
        assert not server.pg.store.load().objects
        before = await eye(tab)
        await tab.page.call("Input.dispatchMouseEvent", type="mousePressed", x=600, y=320, button="left", clickCount=1)
        assert not server.pg.store.load().objects
        await tab.page.call("Input.dispatchMouseEvent", type="mouseMoved", x=680, y=350, button="left", buttons=1)
        await tab.page.call("Input.dispatchMouseEvent", type="mouseReleased", x=680, y=350, button="left", clickCount=1)
        assert not server.pg.store.load().objects
        assert await eye(tab) != pytest.approx(before)
        assert await tab.eval("document.getElementById('place-here').disabled")
        before = await tab.eval("window.__robotforge.cameraTarget()")
        await tab.page.call("Input.dispatchMouseEvent", type="mousePressed", x=600, y=320, button="right", clickCount=1)
        await tab.page.call("Input.dispatchMouseEvent", type="mouseMoved", x=630, y=340, button="right", buttons=2)
        await tab.page.call(
            "Input.dispatchMouseEvent", type="mouseReleased", x=630, y=340, button="right", clickCount=1
        )
        assert await tab.eval("window.__robotforge.cameraTarget()") != pytest.approx(before)
        before = await distance(tab)
        await tab.page.call("Input.dispatchMouseEvent", type="mouseWheel", x=600, y=320, deltaY=-120, deltaX=0)
        await asyncio.sleep(0.15)
        assert await distance(tab) < before * 0.99
        version = await tab.eval("window.__robotforge.scene.version")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([0,0,0])"))
        await tab.until(f"window.__robotforge.scene.version > {version} && !window.__robotforge.busy().busy")
        assert len(server.pg.store.load().objects) == 1
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)
