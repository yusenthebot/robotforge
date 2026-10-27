"""Native reuse of configured props, repeated guards, saved variants and delayed completion."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_object_edit_browser import choose

READY = "window.__robotforge.scene?.version === window.__robotforge.renderedVersion && !window.__robotforge.busy().busy"


def course(pg):
    pg.add_robot("rover", template="rover", pos=[-1.5, -1.5])
    pg.add_object(
        "custom_wall",
        shape="box",
        size=[0.65, 0.13, 0.24],
        color="#547e6b",
        material="wood",
        mass=2.8,
        fixed=True,
        yaw=15,
        pos=[-1.5, 0],
    )


async def place(tab, position):
    version = await tab.eval("window.__robotforge.scene.version")
    await click(tab, point=await tab.eval(f"window.__robotforge.screenOfPoint({position})"))
    await tab.until(f"{READY} && window.__robotforge.scene.version > {version}")


async def journey(tab, server):
    await tab.eval("window.__robotforge.lookAt([3.5,-5,3.2],[-.2,0,.1])")
    original = server.pg.store.load().objects[0]
    await choose(tab, "custom_wall")
    assert await tab.eval("document.getElementById('object-copy').textContent") == (
        "Duplicate" if tab.lang == "en" else "复制"
    )
    await click(tab, "#object-copy")
    assert not await tab.eval("document.querySelector('.repeat-placement').hidden")
    assert await tab.eval("document.getElementById('prop-material-control').hidden")
    assert await tab.eval("[...document.querySelectorAll('.prop-size-control')].every(e=>e.hidden)")
    assert await tab.eval("""(()=>{const r=document.getElementById('placement-bar').getBoundingClientRect(),
      dock=document.querySelector('.dock').getBoundingClientRect();
      return r.left>=0&&r.right<=innerWidth&&r.top>0&&r.bottom<=dock.top;})()""")
    assert (await tab.eval("document.getElementById('place-rotate').textContent")).endswith("15°")
    await click(tab, "#place-repeat")
    await click(tab, "#place-rotate")
    for i, point in enumerate([[0.5, -0.9, 0], [0.5, 1, 0]], 2):
        await asyncio.sleep(0.55)
        await place(tab, point)
        assert len(server.pg.store.load().objects) == i
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('place-here').disabled")
        await tab.page.call("Input.dispatchKeyEvent", type="keyDown", key="Enter", code="Enter", autoRepeat=True)
        await key(tab, "Enter", kind="keyUp")
        assert len(server.pg.store.load().objects) == i
        if i == 2:
            await click(tab, "#place-rotate")
    await key(tab, "Escape")
    assert await tab.eval("document.getElementById('placement-bar').hidden")
    world = server.pg.store.load()
    assert [o.name for o in world.objects] == ["custom_wall", "custom_wall_2", "custom_wall_3"]
    assert [o.yaw for o in world.objects] == [15, 105, 195]
    assert world.objects[0] == original
    assert all(
        o.model_dump(exclude={"name", "pos", "yaw"}) == original.model_dump(exclude={"name", "pos", "yaw"})
        for o in world.objects
    )
    await click(tab, "#undo-edit")
    await tab.until(f"{READY} && document.getElementById('undo-edit').hidden")
    assert [o.name for o in server.pg.store.load().objects] == ["custom_wall", "custom_wall_2"]
    assert "undefined" not in await tab.eval("document.getElementById('toast').textContent")
    # Keep two copies in the saved variation after demonstrating the one-use inverse.
    await choose(tab, "custom_wall")
    await click(tab, "#object-copy")
    await place(tab, [0.5, 1, 0])
    variant = server.pg.store.load().model_dump(mode="json")
    await click(tab, "#world-toggle")
    await click(tab, '[data-tab="maps"]')
    await tab.until("!document.getElementById('layout-save').disabled")
    await click(tab, "#layout-name")
    await tab.page.call("Input.insertText", text="Custom course" if tab.lang == "en" else "自定义路线")
    await click(tab, "#layout-save")
    await tab.until("document.querySelector('.layout-row') && !document.getElementById('layout-save').disabled")
    await click(tab, "#map-list .choice:first-child")
    await tab.until("!window.__robotforge.busy().busy && !document.getElementById('layout-save').disabled")
    assert not server.pg.store.load().objects
    await click(tab, '[data-layout-action="restore"]')
    await tab.until(
        "!document.getElementById('layout-confirm').hidden && !document.getElementById('layout-apply').disabled"
    )
    await click(tab, "#layout-apply")
    await tab.until("!window.__robotforge.busy().busy && !document.getElementById('layout-save').disabled")
    assert server.pg.store.load().model_dump(mode="json") == variant
    origin = await tab.eval("performance.timeOrigin")
    await tab.page.call("Page.reload")
    await tab.until(f"performance.timeOrigin !== {origin} && !!window.__robotforge?.scene")
    await tab.until("!document.getElementById('layout-save').disabled")
    assert server.pg.store.load().model_dump(mode="json") == variant
    await click(tab, "#world-toggle")
    await tab.eval("window.__robotforge.select('rover');document.getElementById('view').focus()")
    start = server.pg.session.data.qpos[:2].copy()
    await click(tab, "#btn-play")
    await key(tab, "w", "KeyW")
    await asyncio.sleep(1.3)
    await key(tab, "w", "KeyW", kind="keyUp")
    distance = float(np.linalg.norm(server.pg.session.data.qpos[:2] - start))
    assert distance > 0.15
    assert not tab.errors
    return {"variant": variant, "drive_distance_m": distance}


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("laptop", "zh-CN"), ("phone", "en"), ("phone", "zh-CN")])
def test_configured_copies_repeat_rotate_undo_save_restore_and_drive(viewer, size, lang):  # noqa: F811
    viewer(size, lang, journey, world=course)


def test_native_copy_stacks_on_its_source_without_excluding_original_geoms(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.eval("window.__robotforge.lookAt([.8,-1,1],[0,0,.15])")
        await choose(tab, "source")
        await click(tab, "#object-copy")
        await place(tab, [0, 0, 0.2])
        objects = server.pg.store.load().objects
        assert len(objects) == 2 and objects[0].pos == [0, 0]
        assert objects[1].pos == pytest.approx([0, 0, 0.3], abs=0.003)
        assert not tab.errors

    viewer(
        "laptop",
        "en",
        scenario,
        world=lambda pg: pg.add_object("source", shape="box", size=[0.3, 0.3, 0.2], fixed=True, pos=[0, 0]),
    )


def test_authored_full_turns_copy_the_same_heading_within_request_bounds(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.eval("window.__robotforge.lookAt([4,-6,5],[0,0,0])")
        await choose(tab, "source")
        await click(tab, "#object-copy")
        assert (await tab.eval("document.getElementById('place-rotate').textContent")).endswith("40°")
        await place(tab, [2, 0, 0])
        assert [o.yaw for o in server.pg.store.load().objects] == [4000, 40]
        assert not tab.errors

    viewer(
        "laptop",
        "en",
        scenario,
        world=lambda pg: pg.add_object("source", shape="box", fixed=True, yaw=4000, pos=[0, 0]),
    )


def test_occupied_copy_can_retry_and_cancel_only_discards_preview(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.eval("window.__robotforge.lookAt([4,-6,5],[0,0,0])")
        await choose(tab, "source")
        await click(tab, "#object-copy")
        before = server.pg.store.file.read_bytes()
        # Ground next to the source is visible, but the new copy's footprint intersects it.
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([.25,.25,0])"))
        await tab.until(
            "!window.__robotforge.busy().busy && document.getElementById('toast').textContent.includes('occupied')"
        )
        assert server.pg.store.file.read_bytes() == before
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        await place(tab, [2, 0, 0])
        await choose(tab, "source")
        await click(tab, "#object-copy")
        await key(tab, "Escape")
        assert [o.name for o in server.pg.store.load().objects] == ["source", "source_2"]
        assert not tab.errors

    viewer(
        "laptop",
        "en",
        scenario,
        world=lambda pg: pg.add_object("source", shape="box", size=[0.4, 0.4, 0.2], fixed=True, pos=[0, 0]),
    )


COPY_DELAY = """
window.__copyIds=new Set();const send=WebSocket.prototype.send;
WebSocket.prototype.send=function(s){const m=JSON.parse(s);
 if(m.action==='duplicate_object')window.__copyIds.add(m.id);return send.call(this,s)};
const descriptor=Object.getOwnPropertyDescriptor(WebSocket.prototype,'onmessage');
Object.defineProperty(WebSocket.prototype,'onmessage',{get:descriptor.get,set(fn){descriptor.set.call(this,ev=>{
 const m=typeof ev.data==='string'?JSON.parse(ev.data):null;
 if(window.__delayCopy && m?.type==='reply' && window.__copyIds.has(m.id)){
  window.__delayCopy=false;window.__releaseCopy=()=>fn.call(this,ev);return;
 }return fn.call(this,ev);
});}});
"""


def test_external_reset_before_delayed_copy_reply_cancels_repeat_without_resubmission(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=COPY_DELAY)
        await tab.navigate()
        await tab.eval("window.__robotforge.lookAt([4,-6,5],[0,0,0])")
        await choose(tab, "custom_wall")
        await click(tab, "#object-copy")
        await click(tab, "#place-repeat")
        await tab.run("window.__delayCopy=true")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([1,0,0])"))
        await tab.until("!!window.__releaseCopy")
        server.pg.session.reset()
        await asyncio.sleep(0.4)
        await tab.run("window.__releaseCopy()")
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden")
        assert len(server.pg.store.load().objects) == 2
        await asyncio.sleep(0.6)
        assert len(server.pg.store.load().objects) == 2
        assert await tab.eval("document.getElementById('undo-edit').hidden")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=course)
