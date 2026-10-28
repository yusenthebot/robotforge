"""Shared native fine headings, fractional authors, busy ownership and held-pointer safety."""

from __future__ import annotations

import asyncio
import math

import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_object_edit_browser import choose as choose_prop
from test_viewer_objects_browser import choose_object
from test_viewer_placement_browser import spacing
from test_viewer_robot_edit_browser import begin as choose_robot
from test_viewer_startup_browser import READY as DRAWN
from test_viewer_terrain_create_browser import choose as choose_terrain
from test_viewer_terrain_edit_browser import choose as choose_terrain_edit
from viewer_camera import settle_camera

READY = f"({DRAWN}) && !window.__robotforge.busy().busy"


def world(pg):
    pg.add_robot("rover", template="rover", pos=[-4, -3], yaw=13.5)
    pg.add_terrain(
        "ramp",
        name="dock",
        pos=[-3, 1],
        yaw=7213.5,
        params={"length": 0.5, "landing": 0.3, "width": 0.6, "friction": 0.72},
    )


async def prepare(tab, server):
    server.pg.session.paused = True
    server.pg.session.reset()
    await tab.until(READY + " && window.__robotforge.time===0")
    if tab.size == "phone":
        await tab.page.call(
            "Emulation.setDeviceMetricsOverride",
            width=320,
            height=812,
            deviceScaleFactor=2.75,
            mobile=True,
        )
    await tab.run("window.__robotforge.lookAt([6,-8,8],[0,0,0])")
    await settle_camera(tab)


async def place(tab, position, *, repeating=False):
    await tab.run(f"window.__robotforge.lookAt({[position[0] + 6, position[1] - 8, position[2] + 8]},{position})")
    await settle_camera(tab)
    point = await tab.eval(f"window.__robotforge.screenOfPoint({position})")
    assert await tab.eval(f"document.elementFromPoint({point[0]},{point[1]})?.id") == "view"
    version = await tab.eval("window.__robotforge.scene.version")
    await click(tab, point=point)
    ending = (
        "!document.getElementById('placement-bar').hidden && document.getElementById('place-here').disabled"
        if repeating
        else "document.getElementById('placement-bar').hidden"
    )
    await tab.until(f"{READY} && window.__robotforge.scene.version>{version} && {ending}")


async def heading(tab, expected):
    assert (await tab.eval("document.getElementById('placement-heading').textContent")).endswith(f"{expected}°")


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_native_fine_prop_create_copy_repeat_move_and_inverse(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        await choose_object(tab, "barrier")
        await click(tab, "#place-rotate-plus")
        await click(tab, "#place-rotate-plus")
        await heading(tab, 30)
        preview = await tab.eval(
            "window.__robotforge.three().scene.getObjectByName('placement-size-preview').rotation.z"
        )
        assert preview == pytest.approx(math.radians(30))
        await spacing(tab, "0.1")
        await place(tab, [1.03, 0.03, 0])
        original = server.pg.current_world().objects[0]
        assert original.yaw == 30 and original.pos == [1, 0]
        await choose_prop(tab, original.name)
        await click(tab, "#object-copy")
        await click(tab, "#place-rotate-minus")
        await heading(tab, 15)
        await click(tab, "#place-repeat")
        for point in ([1, 1, 0], [1, 2, 0]):
            await asyncio.sleep(0.55)
            await place(tab, point, repeating=True)
            await heading(tab, 15)
        assert [o.yaw for o in server.pg.current_world().objects] == [30, 15, 15]
        await key(tab, "Escape")
        await click(tab, "#undo-edit")
        await tab.until(READY + " && document.getElementById('undo-edit').hidden")
        assert [o.yaw for o in server.pg.current_world().objects] == [30, 15]
        await choose_prop(tab, original.name)
        await click(tab, "#object-move")
        await click(tab, "#place-rotate-plus")
        await heading(tab, 45)
        await place(tab, [2, -1, 0])
        assert server.pg.current_world().objects[0].yaw == 45
        await click(tab, "#undo-edit")
        await tab.until(READY + " && document.getElementById('undo-edit').hidden")
        assert server.pg.current_world().objects[0] == original and not tab.errors

    viewer(size, lang, scenario, world=world)


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_native_fine_terrain_create_move_authored_turns_undo_and_reload(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        original = server.pg.current_world()
        await choose_terrain(tab, "ramp")
        await click(tab, "#place-rotate-plus")
        await heading(tab, 15)
        await place(tab, [3, 3, 0])
        assert server.pg.current_world().terrain[-1].yaw == 15
        await click(tab, "#undo-add")
        await tab.until(READY + " && document.getElementById('undo-add').hidden")
        assert server.pg.current_world() == original
        await choose_terrain_edit(tab)
        await heading(tab, 13.5)
        await click(tab, "#place-rotate-plus")
        await heading(tab, 28.5)
        await place(tab, [2, -3, 0])
        assert server.pg.current_world().terrain[0].yaw == 7228.5
        await click(tab, "#undo-edit")
        await tab.until(READY + " && document.getElementById('undo-edit').hidden")
        assert server.pg.current_world() == original
        await choose_terrain(tab, "stairs")
        await click(tab, "#place-rotate-minus")
        await heading(tab, 345)
        await place(tab, [3, 3, 0])
        saved = server.pg.current_world()
        assert saved.terrain[-1].yaw == 345
        origin = await tab.eval("performance.timeOrigin")
        await tab.page.call("Page.reload")
        await tab.until(f"performance.timeOrigin!=={origin} && {READY}")
        assert server.pg.current_world() == saved and not tab.errors

    viewer(size, lang, scenario, world=world)


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_native_fractional_robot_heading_and_busy_controls_cannot_mutate_submission(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        original = server.pg.current_world().robots[0]
        await choose_robot(tab)
        await click(tab, "#place-rotate-plus")
        await heading(tab, 28.5)
        await tab.run("""
          window.__headingSend=WebSocket.prototype.send;
          WebSocket.prototype.send=function(s){
            if(JSON.parse(s).action==='edit_robot_start'){window.__headingHeld={socket:this,value:s};return;}
            return window.__headingSend.call(this,s);
          };
        """)
        await tab.run("window.__robotforge.lookAt([8,-10,8],[2,-2,0])")
        await settle_camera(tab)
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([2,-2,0])"))
        await tab.until("!!window.__headingHeld && window.__robotforge.busy().busy")
        assert await tab.eval(
            "['place-rotate-minus','place-rotate','place-rotate-plus'].every(id=>document.getElementById(id).disabled)"
        )
        await tab.run("""
          for(const id of ['place-rotate-minus','place-rotate','place-rotate-plus'])
            document.getElementById(id).dispatchEvent(new MouseEvent('click',{bubbles:true}));
        """)
        await heading(tab, 28.5)
        assert await tab.eval("JSON.parse(window.__headingHeld.value).yaw") == 28.5
        await tab.run("""
          WebSocket.prototype.send=window.__headingSend;
          window.__headingSend.call(window.__headingHeld.socket,window.__headingHeld.value);
        """)
        await tab.until(READY + " && document.getElementById('placement-bar').hidden")
        assert server.pg.current_world().robots[0].yaw == 28.5
        await click(tab, "#undo-edit")
        await tab.until(READY + " && document.getElementById('undo-edit').hidden")
        assert server.pg.current_world().robots[0] == original and not tab.errors

    viewer(size, lang, scenario, world=world)


def test_keyboard_fine_rotation_invalidates_held_canvas_pointer(viewer):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        await choose_object(tab, "barrier")
        await settle_camera(tab)
        saved, version = server.pg.store.file.read_bytes(), server.pg.session.version
        x, y = await tab.eval("window.__robotforge.screenOfPoint([1,.5,0])")
        await tab.page.call(
            "Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", buttons=1, clickCount=1
        )
        for _ in range(40):
            if await tab.eval("document.activeElement.id") == "place-rotate-plus":
                break
            await key(tab, "Tab")
            await key(tab, "Tab", kind="keyUp")
        assert await tab.eval("document.activeElement.id") == "place-rotate-plus"
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        await heading(tab, 15)
        await tab.page.call(
            "Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", buttons=0, clickCount=1
        )
        await asyncio.sleep(0.5)
        assert server.pg.session.version == version and server.pg.store.file.read_bytes() == saved
        assert not await tab.eval("document.getElementById('placement-bar').hidden") and not tab.errors

    viewer("laptop", "en", scenario, world=world)
