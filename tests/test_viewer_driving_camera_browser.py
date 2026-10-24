"""Native camera intent remains independent of sustained driving input."""

from __future__ import annotations

import asyncio

import pytest
from test_viewer_creation_browser import click
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_screen_drive_browser import command, point, ready, until
from test_viewer_touch_build_browser import touch


def test_manual_touch_camera_stays_free_while_forward_is_held(viewer):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab)
        fwd = await point(tab, "fwd")
        await touch(tab, "touchStart", [(1, *fwd)])
        await until(lambda: command(server)[0] > 0)
        await tab.until("window.__robotforge.following() !== null")
        assert await tab.eval("document.elementFromPoint(300,140).id") == "view"
        await touch(tab, "touchStart", [(1, *fwd), (2, 300, 140)])
        await touch(tab, "touchMove", [(1, *fwd), (2, 240, 180)])
        after_gesture = len(await tab.eval("window.__driveMessages"))
        await tab.until(f"window.__driveMessages.slice({after_gesture}).some(m=>m.hold_ms===750&&m.speed>0)")
        assert command(server)[0] > 0
        assert await tab.eval("window.__robotforge.following()") is None
        await touch(tab, "touchEnd")
        await until(lambda: command(server) == (0, 0))
        assert not tab.errors

    viewer("phone", "zh-CN", scenario, world=lambda pg: pg.add_robot("rover", template="rover"))


async def camera_ready(tab):
    await tab.until(
        "!document.getElementById('camera-view').hidden && !document.getElementById('camera-view').disabled"
    )


async def header_controls(tab):
    return await tab.eval("""(()=>{const buttons=[...document.querySelectorAll('.top button')]
      .filter(e=>!e.hidden&&e.getClientRects().length);return buttons.map(e=>{
        const r=e.getBoundingClientRect(),hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
        return {id:e.id,left:r.left,right:r.right,top:r.top,bottom:r.bottom,hit:e===hit||e.contains(hit)};});})()""")


async def assert_header_accessible(tab):
    controls = await header_controls(tab)
    width = await tab.eval("innerWidth")
    height = await tab.eval("innerHeight")
    assert all(
        c["hit"] and c["left"] >= 0 and c["right"] <= width and c["top"] >= 0 and c["bottom"] <= height
        for c in controls
    ), controls
    for i, a in enumerate(controls):
        for b in controls[i + 1 :]:
            assert min(a["right"], b["right"]) <= max(a["left"], b["left"]) or min(a["bottom"], b["bottom"]) <= max(
                a["top"], b["top"]
            ), controls


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("laptop", "zh-CN"), ("phone", "en"), ("phone", "zh-CN")])
def test_camera_button_finds_offscreen_robot_while_paused_and_stops_its_glide(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await camera_ready(tab)
        server.pg.session.paused = True
        await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
        await tab.run(
            "window.__robotforge.lookAt([20,-3,2],[20,0,0]);window.__cameraWire=[];"
            "const send=WebSocket.prototype.send;WebSocket.prototype.send=function(s){"
            "const m=JSON.parse(s);if(m.type!=='seen')window.__cameraWire.push(m);return send.call(this,s)}"
        )
        await click(tab, "#camera-view")
        await tab.until("!window.__robotforge.tweening()")
        assert await tab.eval("document.getElementById('camera-view').getAttribute('aria-pressed')") == "true"
        assert await tab.eval("window.__robotforge.following()") is not None
        framed = await tab.eval("""(()=>{const rf=window.__robotforge,p=rf.screenOf(rf.following()),r=rf.freeRect();
          return p[0]>r.left+20&&p[0]<r.right-20&&p[1]>r.top+20&&p[1]<r.bottom-20})()""")
        assert framed
        assert server.pg.session.paused
        assert await tab.eval("window.__cameraWire") == []
        await assert_header_accessible(tab)
        await click(tab, "#camera-view")
        assert await tab.eval("window.__robotforge.following()") is None
        await click(tab, "#camera-view")
        await click(tab, "#camera-view")
        assert not await tab.eval("window.__robotforge.tweening()")
        assert await tab.eval("window.__robotforge.following()") is None
        assert await tab.eval("window.__cameraWire") == []
        await tab.capture(tab.home / "camera-controls.png")
        assert not tab.errors

    viewer(size, lang, scenario, world=lambda pg: pg.add_robot("rover", template="rover"))


@pytest.mark.parametrize("lang", ["en", "zh-CN"])
def test_camera_header_stays_accessible_at_320_pixels(viewer, lang):  # noqa: F811
    async def scenario(tab, _server):
        await tab.page.call(
            "Emulation.setDeviceMetricsOverride", width=320, height=720, deviceScaleFactor=2, mobile=True
        )
        await camera_ready(tab)
        await assert_header_accessible(tab)
        await click(tab, "#camera-view")
        assert await tab.eval("window.__robotforge.following()") is not None
        await tab.capture(tab.home / "camera-320.png")
        assert not tab.errors

    viewer("phone", lang, scenario, world=lambda pg: pg.add_robot("rover", template="rover"))


@pytest.mark.parametrize("source", ["keyboard", "gamepad"])
def test_explicit_free_view_survives_repeated_desktop_input(viewer, source):  # noqa: F811
    from test_viewer_gamepad import PAD, axes

    async def scenario(tab, server):
        await camera_ready(tab)
        await click(tab, "#camera-view")
        assert await tab.eval("window.__robotforge.following()") is not None
        if source == "keyboard":
            await tab.page.call("Input.dispatchKeyEvent", type="keyDown", key="f", code="KeyF")
            await tab.page.call("Input.dispatchKeyEvent", type="keyUp", key="f", code="KeyF")
            for repeat in [False, True, True]:
                await tab.page.call(
                    "Input.dispatchKeyEvent",
                    type="keyDown",
                    key="w",
                    code="KeyW",
                    windowsVirtualKeyCode=87,
                    autoRepeat=repeat,
                )
                await asyncio.sleep(0.05)
        else:
            await click(tab, "#camera-view")
            await tab.run(PAD)
            await tab.until("window.__robotforge.pad().connected")
            await tab.run(axes(a1=-1))
            await until(lambda: command(server)[0] > 0)
            await tab.run(axes(a0=0.5))
            await until(lambda: abs(command(server)[1]) > 0)
        await until(lambda: command(server)[0] > 0)
        assert await tab.eval("window.__robotforge.following()") is None
        assert await tab.eval("document.getElementById('camera-view').getAttribute('aria-pressed')") == "false"
        if source == "keyboard":
            await tab.page.call("Input.dispatchKeyEvent", type="keyUp", key="w", code="KeyW", windowsVirtualKeyCode=87)
        else:
            await tab.run(axes(a0=0, a1=0))
        await until(lambda: command(server) == (0, 0))
        await click(tab, "#camera-view")
        assert await tab.eval("window.__robotforge.following()") is not None
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.add_robot("rover", template="rover"))


def test_fixed_arm_has_focus_action_without_follow_or_control_commands(viewer):  # noqa: F811
    async def scenario(tab, server):
        await camera_ready(tab)
        server.pg.session.paused = True
        await tab.run("window.__robotforge.lookAt([20,-3,2],[20,0,0])")
        await tab.run("""window.__armCameraWire=[];const send=WebSocket.prototype.send;
          WebSocket.prototype.send=function(s){const m=JSON.parse(s);
            if(m.type!=='seen')window.__armCameraWire.push(m);return send.call(this,s)};""")
        before = server.pg.session.data.ctrl.copy()
        assert await tab.eval("document.getElementById('camera-view').getAttribute('aria-pressed')") is None
        await click(tab, "#camera-view")
        await tab.until("!window.__robotforge.tweening()")
        assert await tab.eval("window.__robotforge.following()") is None
        assert await tab.eval("window.__robotforge.cameraTarget()[0]") == pytest.approx(0, abs=0.3)
        assert (before == server.pg.session.data.ctrl).all()
        assert await tab.eval("window.__armCameraWire") == []
        assert server.pg.session.paused
        assert not tab.errors

    viewer("phone", "zh-CN", scenario, world=lambda pg: pg.add_robot("arm", template="arm"))


def test_creation_cancel_keeps_building_camera_free_until_explicit_resume(viewer):  # noqa: F811
    from test_viewer_creation_browser import key
    from test_viewer_objects_browser import choose_object

    async def scenario(tab, server):
        await camera_ready(tab)
        await click(tab, "#camera-view")
        await choose_object(tab, "ball")
        assert await tab.eval("document.getElementById('camera-view').hidden")
        assert await tab.eval("window.__robotforge.following()") is None
        await key(tab, "Escape")
        await camera_ready(tab)
        await key(tab, "w", "KeyW")
        await until(lambda: command(server)[0] > 0)
        assert await tab.eval("window.__robotforge.following()") is None
        await key(tab, "w", "KeyW", "keyUp")
        await until(lambda: command(server) == (0, 0))
        await click(tab, "#camera-view")
        assert await tab.eval("window.__robotforge.following()") is not None
        assert not server.pg.store.load().objects
        assert not tab.errors

    viewer("phone", "zh-CN", scenario, world=lambda pg: pg.add_robot("rover", template="rover"))


def test_delayed_same_size_rebuild_selection_and_reconnect_drop_old_camera_target(viewer):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab)
        await camera_ready(tab)
        await click(tab, "#camera-view")
        old = await tab.eval("window.__robotforge.scene.version")
        await tab.run("""const native=window.fetch;window.fetch=async(...args)=>{
          const response=await native(...args);
          if(String(args[0]).startsWith('mesh/'))await new Promise(resolve=>{
            window.__releaseCameraMesh=()=>{window.fetch=native;resolve()};});
          return response;};""")
        server.pg.update_robot("rover", pos=[2, 0])
        await tab.until("!!window.__releaseCameraMesh")
        assert await tab.eval("document.getElementById('camera-view').disabled")
        assert await tab.eval("window.__robotforge.following()") is None
        await click(tab, "#camera-view")
        assert await tab.eval("window.__robotforge.following()") is None
        await tab.run("window.__releaseCameraMesh()")
        await tab.until(f"window.__robotforge.scene.version > {old}")
        await camera_ready(tab)
        assert await tab.eval("window.__robotforge.following()") is None
        await click(tab, "#camera-view")
        assert await tab.eval("window.__robotforge.following()") is not None
        await tab.run("window.__robotforge.select('other')")
        assert await tab.eval("window.__robotforge.following()") is None
        await tab.page.call("Input.dispatchKeyEvent", type="keyDown", key="w", code="KeyW", windowsVirtualKeyCode=87)
        await until(lambda: command(server, "other")[0] > 0)
        assert await tab.eval("""(()=>{const rf=window.__robotforge;
          return rf.scene.bodies.find(b=>b.id===rf.following()).name.startsWith('other/');})()""")
        await tab.page.call("Input.dispatchKeyEvent", type="keyUp", key="w", code="KeyW", windowsVirtualKeyCode=87)
        await until(lambda: command(server, "other") == (0, 0))
        await tab.run("window.__driveSocket.close()")
        await tab.until("document.getElementById('camera-view').disabled")
        assert await tab.eval("window.__robotforge.following()") is None
        await camera_ready(tab)
        assert await tab.eval("window.__robotforge.following()") is None
        assert not tab.errors

    def world(pg):
        pg.add_robot("rover", template="rover")
        pg.add_robot("other", template="rover", pos=[0, 2])

    viewer("phone", "en", scenario, world=world)


def test_lesson_follow_is_independent_and_exiting_restores_free_world_view(viewer):  # noqa: F811
    from test_viewer_creation_browser import key
    from test_viewer_layout import _start

    async def scenario(tab, _server):
        await camera_ready(tab)
        await click(tab, "#camera-view")
        await click(tab, "#camera-view")
        await _start(tab, "maze_escape")
        assert await tab.eval("document.getElementById('camera-view').hidden")
        await key(tab, "w", "KeyW")
        await tab.until("window.__robotforge.following() !== null")
        await key(tab, "w", "KeyW", "keyUp")
        await click(tab, "#lesson-exit")
        await tab.until("window.__robotforge.lesson() === null")
        await camera_ready(tab)
        assert await tab.eval("window.__robotforge.following()") is None
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.add_robot("rover", template="rover"))


@pytest.mark.parametrize("width,lang", [(320, "en"), (320, "zh-CN"), (390, "en"), (390, "zh-CN")])
def test_native_build_undo_does_not_cover_driving(viewer, width, lang):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call(
            "Emulation.setDeviceMetricsOverride", width=width, height=844, deviceScaleFactor=2, mobile=True
        )
        await build_drive_route(tab, server)
        if await tab.eval("document.getElementById('btn-play').classList.contains('is-paused')"):
            await click(tab, "#btn-play")
        await ready(tab)
        for direction in ("fwd", "left", "back", "right"):
            await point(tab, direction)
        rects = await tab.eval("""(()=>{const items=[document.getElementById('undo-add'),
          ...document.querySelectorAll('#screen-drive [data-direction]:not([hidden])')];return items.map(e=>{
            const r=e.getBoundingClientRect(),hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
            return {left:r.left,right:r.right,top:r.top,bottom:r.bottom,hit:e===hit||e.contains(hit)};});})()""")
        assert len(rects) == 5
        assert all(
            r["hit"] and 0 <= r["left"] < r["right"] <= width and 0 <= r["top"] < r["bottom"] <= 844 for r in rects
        )
        for other in rects[1:]:
            undo = rects[0]
            assert undo["bottom"] <= other["top"] or undo["right"] <= other["left"] or undo["left"] >= other["right"]
        left = await point(tab, "left")
        await touch(tab, "touchStart", [(1, *left)])
        await until(lambda: command(server)[1] > 0)
        await touch(tab, "touchEnd")
        await until(lambda: command(server) == (0, 0))
        await tab.capture(tab.home / "undo-and-drive.png")
        await click(tab, "#undo-add")
        await until(lambda: not server.pg.store.load().objects)
        assert not tab.errors

    viewer("phone", lang, scenario, world=lambda pg: pg.add_robot("rover", template="rover", pos=[-1.2, 0]))


async def build_drive_route(tab, server):
    from test_viewer_objects_browser import choose_object

    await tab.run("window.__robotforge.lookAt([2.6,-4.8,3.2],[-.4,0,0])")
    await choose_object(tab, "barrier")
    await tab.eval("""(async()=>{let last=null,stable=0;const start=performance.now();
      while(stable<8){if(performance.now()-start>5000)throw Error('camera did not settle');
        const state=await new Promise(resolve=>requestAnimationFrame(()=>{
          const rf=window.__robotforge,c=rf.three().camera;
          resolve(JSON.stringify([c.projectionMatrix.toArray(),c.position.toArray(),rf.cameraTarget()]));}));
        stable=state===last?stable+1:0;last=state;}})()""")
    version = server.pg.session.version
    await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([0,0,0])"))
    await tab.until(f"window.__robotforge.renderedVersion>{version} && !document.getElementById('undo-add').hidden")
    assert len(server.pg.store.load().objects) == 1


@pytest.mark.parametrize("lang", ["en", "zh-CN"])
def test_native_build_undo_and_keyboard_hint_share_narrow_hud(viewer, lang):  # noqa: F811
    from test_viewer_creation_browser import key

    async def scenario(tab, server):
        await tab.page.call(
            "Emulation.setDeviceMetricsOverride", width=720, height=844, deviceScaleFactor=1, mobile=False
        )
        await build_drive_route(tab, server)
        assert await tab.eval("document.getElementById('screen-drive').hidden")
        rects = await tab.eval("""(()=>{return ['drive','undo-add','btn-play'].map(id=>{
          const e=document.getElementById(id),r=e.getBoundingClientRect();
          const hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
          return {left:r.left,right:r.right,top:r.top,bottom:r.bottom,hit:e===hit||e.contains(hit)};});})()""")
        await tab.capture(tab.home / "narrow-hud-before-check.png")
        assert all(
            r["hit"] and 0 <= r["left"] < r["right"] <= 720 and 0 <= r["top"] < r["bottom"] <= 844 for r in rects
        ), rects
        assert rects[0]["bottom"] <= rects[1]["top"] < rects[1]["bottom"] <= rects[2]["top"]
        if await tab.eval("document.getElementById('btn-play').classList.contains('is-paused')"):
            await click(tab, "#btn-play")
        await key(tab, "a", "KeyA")
        await until(lambda: command(server)[1] > 0)
        await key(tab, "a", "KeyA", "keyUp")
        await until(lambda: command(server) == (0, 0))
        await tab.capture(tab.home / "undo-and-keyboard.png")
        await click(tab, "#undo-add")
        await until(lambda: not server.pg.store.load().objects)
        assert not tab.errors

    viewer("laptop", lang, scenario, world=lambda pg: pg.add_robot("rover", template="rover", pos=[-1.2, 0]))
