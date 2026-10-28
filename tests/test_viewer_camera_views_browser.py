"""Native world-axis inspection, return and editing without transient raycast submissions."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_startup_browser import READY
from viewer_camera import settle_camera


def world(pg):
    pg.add_robot("rover", template="rover", pos=[-2, -1])
    pg.add_object("block", shape="box", size=[0.4, 0.4, 0.2], pos=[0, 0], fixed=True)


async def pose(tab):
    return await tab.run("""
      const c=window.__robotforge.three().camera,t=window.__robotforge.cameraTarget(),e=c.position.toArray();
      const d=Math.hypot(...e.map((v,i)=>v-t[i]));
      return {eye:e,target:t,distance:d,direction:e.map((v,i)=>(v-t[i])/d),
        perspective:c.isPerspectiveCamera,up:c.up.toArray()};
    """)


async def view(tab, name):
    await click(tab, "#view-toggle")
    await click(tab, "#view-back" if name == "back" else f'[data-camera-view="{name}"]')
    await settle_camera(tab)
    return await pose(tab)


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_axis_views_keep_current_aim_distance_physics_and_original_return(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        await tab.run("window.__robotforge.lookAt([4,-6,4],[.1,.2,.25])")
        await settle_camera(tab)
        original = await pose(tab)
        saved = server.pg.store.file.read_bytes()
        state = {k: getattr(server.pg.session.data, k).copy() for k in ["qpos", "qvel", "ctrl"]}
        resources = await tab.eval("window.__robotforge.three().renderer.info.memory")
        for name, direction in [("top", [0, -1e-6, 1]), ("front", [0, -1, 0]), ("side", [1, 0, 0])]:
            result = await view(tab, name)
            assert result["direction"] == pytest.approx(direction, abs=1e-8)
            assert result["target"] == pytest.approx(original["target"], abs=1e-10)
            assert result["distance"] == pytest.approx(original["distance"], abs=1e-10)
            assert result["perspective"] and result["up"] == [0, 0, 1]
            assert (
                await tab.eval(f'document.querySelector("[data-camera-view={name}]").getAttribute("aria-pressed")')
                == "true"
            )
            assert await tab.eval("document.getElementById('view-panel').hidden")
        returned = await view(tab, "back")
        assert returned["eye"] == pytest.approx(original["eye"], abs=1e-9)
        assert returned["target"] == pytest.approx(original["target"], abs=1e-9)
        assert await tab.eval("document.getElementById('view-back').disabled")
        assert await tab.eval("window.__robotforge.following()") is None
        assert await tab.eval("window.__robotforge.three().renderer.info.memory") == resources
        assert server.pg.store.file.read_bytes() == saved and server.pg.session.paused
        for k, expected in state.items():
            np.testing.assert_array_equal(getattr(server.pg.session.data, k), expected)
        assert not tab.errors

    viewer(size, lang, scenario, world=world)


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_views_cancel_stale_points_arc_taps_and_dismissal_without_cancelling_editor(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        await tab.run("window.__robotforge.lookAt([4,-6,5],[0,0,0])")
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="objects"]')
        await click(tab, '[data-object="ball"]')
        saved = server.pg.store.file.read_bytes()
        await click(tab, "#view-toggle")
        await key(tab, "Escape")
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('view-panel').hidden")
        await click(tab, "#view-toggle")
        await click(tab, '[data-camera-view="top"]')
        assert await tab.eval("document.getElementById('place-here').disabled")
        assert await tab.eval("window.__robotforge.tweening()")
        # Trusted input during the arc may stop the camera, but cannot become a placement.
        await click(tab, point=[100, 200])
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        await settle_camera(tab)
        assert server.pg.store.file.read_bytes() == saved
        await click(tab, "#view-toggle")
        await click(tab, point=[100, 200])
        assert await tab.eval("document.getElementById('view-panel').hidden")
        assert server.pg.store.file.read_bytes() == saved
        await view(tab, "top")
        await asyncio.sleep(0.1)  # outside the documented post-arc pointer cooldown
        point = await tab.eval("window.__robotforge.screenOfPoint([1,0,0])")
        version = server.pg.session.version
        await click(tab, point=point)
        await tab.until(f"{READY} && window.__robotforge.scene.version>{version}")
        assert len(server.pg.store.load().objects) == 2
        assert await tab.eval("document.getElementById('view-back').disabled")
        await click(tab, "#undo-add")
        await tab.until(f"{READY} && document.getElementById('undo-add').hidden")
        assert server.pg.store.file.read_bytes() == saved
        assert not tab.errors

    viewer(size, lang, scenario, world=world)


@pytest.mark.parametrize("width,height,lang", [(320, 568, "en"), (320, 568, "zh-CN"), (640, 360, "en")])
def test_view_picker_and_all_dock_controls_are_reachable_on_small_screens(viewer, width, height, lang):  # noqa: F811
    async def scenario(tab, _server):
        await tab.page.call(
            "Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=2, mobile=True
        )
        await tab.until(READY)
        await settle_camera(tab)
        result = await tab.run("""
          return [...document.querySelectorAll('.dock button')].filter(e=>e.getClientRects().length).map(e=>{
            const r=e.getBoundingClientRect(),hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
            return {id:e.id,left:r.left,right:r.right,top:r.top,bottom:r.bottom,hit:e===hit||e.contains(hit)};});
        """)
        assert all(
            r["hit"] and 0 <= r["left"] < r["right"] <= width and 0 <= r["top"] < r["bottom"] <= height for r in result
        ), result
        await click(tab, "#view-toggle")
        for name in ["top", "front", "side"]:
            assert await tab.eval(f"""(()=>{{const e=document.querySelector('[data-camera-view="{name}"]');
              const r=e.getBoundingClientRect();
              return r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=innerHeight
                && e.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2));}})()""")
        await key(tab, "Escape")
        assert await tab.eval("document.activeElement.id") == "view-toggle"
        assert not tab.errors

    viewer("phone", lang, scenario, world=world)


def test_camera_arc_keeps_radius_even_between_opposite_views_and_top_projection_is_stable(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await tab.until(READY)
        samples = await tab.run("""
          const {cameraArc,arcEye}=await import('./static/camera-views.js');
          const arc=cameraArc([-3,0,0],[0,0,0],[3,0,0],[0,0,0]);
          return Array.from({length:21},(_,i)=>arcEye(arc,[0,0,0],i/20));
        """)
        assert all(np.linalg.norm(point) == pytest.approx(3, abs=1e-10) for point in samples)
        await view(tab, "top")
        origin, x, y = await tab.eval("[[0,0,0],[1,0,0],[0,1,0]].map(p=>window.__robotforge.screenOfPoint(p))")
        assert x[0] > origin[0] and abs(x[1] - origin[1]) < 0.01
        assert y[1] < origin[1] and abs(y[0] - origin[0]) < 0.01
        assert not tab.errors

    viewer("laptop", "en", scenario, world=world)


def test_view_history_clears_on_new_scene_and_presets_do_not_enable_auto_follow(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        await view(tab, "top")
        await key(tab, "w", "KeyW")
        assert await tab.eval("window.__robotforge.following()") is None
        await key(tab, "w", "KeyW", kind="keyUp")
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="maps"]')
        version = server.pg.session.version
        await click(tab, "#map-list .choice:first-child")
        await tab.until(f"{READY} && window.__robotforge.scene.version>{version}")
        assert await tab.eval("document.getElementById('view-back').disabled")
        assert await tab.eval("document.getElementById('view-panel').hidden")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=world)


@pytest.mark.parametrize("size", ["laptop", "phone"])
def test_manual_orbit_after_top_keeps_original_return_and_picker_owns_driving_keys(viewer, size):  # noqa: F811
    async def scenario(tab, server):
        from test_viewer_touch_build_browser import touch

        server.pg.session.paused = True
        await tab.until(READY)
        await tab.run("window.__robotforge.lookAt([4,-6,4],[0,0,0])")
        await settle_camera(tab)
        original = await pose(tab)
        top = await view(tab, "top")
        await click(tab, "#view-toggle")
        before = len(await tab.eval("window.__qaErrors"))
        await key(tab, "w", "KeyW")
        await key(tab, "w", "KeyW", kind="keyUp")
        assert await tab.eval("window.__robotforge.following()") is None
        await key(tab, "Escape")
        if size == "phone":
            await touch(tab, "touchStart", [(1, 100, 210)])
            await touch(tab, "touchMove", [(1, 150, 175)])
            await touch(tab, "touchEnd")
        else:
            for kind, x, y in [("mousePressed", 100, 210), ("mouseMoved", 150, 175), ("mouseReleased", 150, 175)]:
                await tab.page.call(
                    "Input.dispatchMouseEvent",
                    type=kind,
                    x=x,
                    y=y,
                    button="left",
                    buttons=0 if kind == "mouseReleased" else 1,
                    clickCount=1,
                )
        await settle_camera(tab)
        moved = await pose(tab)
        assert moved["eye"] != pytest.approx(top["eye"], abs=1e-4)
        assert await tab.eval("document.querySelectorAll('[data-camera-view][aria-pressed=true]').length") == 0
        restored = await view(tab, "back")
        assert restored["eye"] == pytest.approx(original["eye"], abs=1e-9)
        assert restored["target"] == pytest.approx(original["target"], abs=1e-9)
        assert len(await tab.eval("window.__qaErrors")) == before == 0

    viewer(size, "en", scenario, world=world)


def test_reduced_motion_snaps_views_without_an_animation(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await tab.page.call(
            "Emulation.setEmulatedMedia", features=[{"name": "prefers-reduced-motion", "value": "reduce"}]
        )
        await tab.navigate()
        await tab.until(READY)
        await click(tab, "#view-toggle")
        await click(tab, '[data-camera-view="top"]')
        assert not await tab.eval("window.__robotforge.tweening()")
        result = await pose(tab)
        assert result["direction"] == pytest.approx([0, -1e-6, 1], abs=1e-8)
        assert not tab.errors

    viewer("laptop", "en", scenario, world=world)


def test_opening_picker_invalidates_a_held_canvas_pointer_before_its_release(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        await tab.run("window.__robotforge.lookAt([4,-6,5],[0,0,0])")
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="objects"]')
        await click(tab, '[data-object="ball"]')
        await settle_camera(tab)
        saved, version = server.pg.store.file.read_bytes(), server.pg.session.version
        x, y = await tab.eval("window.__robotforge.screenOfPoint([1.2,.5,0])")
        await tab.page.call(
            "Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", buttons=1, clickCount=1
        )
        # A keyboard user can open the picker while a pointer remains pressed on the canvas.
        for _ in range(40):
            if await tab.eval("document.activeElement.id") == "view-toggle":
                break
            await key(tab, "Tab")
            await key(tab, "Tab", kind="keyUp")
        assert await tab.eval("document.activeElement.id") == "view-toggle"
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        assert not await tab.eval("document.getElementById('view-panel').hidden")
        await tab.page.call(
            "Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", buttons=0, clickCount=1
        )
        await asyncio.sleep(0.5)
        assert server.pg.session.version == version
        assert server.pg.store.file.read_bytes() == saved
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=world)
