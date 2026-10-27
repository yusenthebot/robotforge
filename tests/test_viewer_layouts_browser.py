"""Named creation loops in native Chrome, plus delayed metadata/pose regression cases."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401

READY = "!document.getElementById('layout-save').disabled"
RESTORE = '[data-layout-action="restore"]'
REMOVE = '[data-layout-action="remove"]'


def course(pg):
    pg.add_robot("rover", template="rover")
    pg.add_object("barrier", shape="box", pos=[2, 0], size=[0.8, 0.12, 0.22], fixed=True)
    pg.add_terrain("ramp", pos=[3, 2])


async def open_layouts(tab):
    if await tab.eval("document.getElementById('world-panel').hidden"):
        await click(tab, "#world-toggle")
    await click(tab, '[data-tab="maps"]')
    await tab.until(READY)


async def journey(tab, server):
    """Shared with real screencast evidence: native input, actual server state and physics checks."""
    await open_layouts(tab)
    original = server.pg.current_world().model_dump(mode="json")
    name = "我的坡道路线" if tab.lang == "zh-CN" else "My ramp course"
    await click(tab, "#layout-name")
    await tab.page.call("Input.insertText", text=name)
    await key(tab, "Enter")
    await key(tab, "Enter", kind="keyUp")
    await tab.until("document.querySelectorAll('.layout-row').length === 1 && " + READY)
    assert await tab.eval("document.querySelector('.layout-row span').textContent") == name
    files = list((server.pg.store.dir / ".layouts").glob("*.json"))
    assert len(files) == 1
    frozen = files[0].read_bytes()
    await click(tab, "#map-list .choice:first-child")
    await tab.until("!window.__robotforge.busy().busy && " + READY)
    assert not server.pg.current_world().objects and not server.pg.current_world().terrain
    await click(tab, RESTORE)
    await tab.until(
        "!document.getElementById('layout-confirm').hidden && document.activeElement.id === 'layout-cancel'"
    )
    assert await tab.eval("document.activeElement.id") == "layout-cancel"
    await key(tab, "Escape")
    await key(tab, "Escape", kind="keyUp")
    assert await tab.eval("document.getElementById('layout-confirm').hidden")
    assert not server.pg.current_world().objects
    await click(tab, RESTORE)
    await click(tab, "#layout-apply")
    await tab.until("!window.__robotforge.busy().busy && " + READY)
    assert server.pg.current_world().model_dump(mode="json") == original
    assert server.pg.session.paused and server.pg.session.data.time == 0
    # Reload reads the same durable record; it does not depend on browser local storage.
    origin = await tab.eval("performance.timeOrigin")
    await tab.page.call("Page.reload")
    await tab.until(f"performance.timeOrigin !== {origin} && !!window.__robotforge?.scene")
    await open_layouts(tab)
    assert await tab.eval("document.querySelector('.layout-row span').textContent") == name
    assert files[0].read_bytes() == frozen
    await click(tab, REMOVE)
    await click(tab, "#layout-cancel")
    assert files[0].exists()
    await click(tab, "#world-toggle")
    await tab.eval("window.__robotforge.select('rover');document.getElementById('view').focus()")
    start = server.pg.session.data.qpos[:3].copy()
    await key(tab, " ", "Space")
    await key(tab, " ", "Space", kind="keyUp")
    await key(tab, "w", "KeyW")
    await asyncio.sleep(1.3)
    await key(tab, "w", "KeyW", kind="keyUp")
    distance = float(np.linalg.norm(server.pg.session.data.qpos[:2] - start[:2]))
    assert distance > 0.15 and server.pg.session.data.time > 0.5
    assert np.isfinite(server.pg.session.data.qpos).all()
    await open_layouts(tab)
    await click(tab, REMOVE)
    await click(tab, "#layout-apply")
    await tab.until("document.querySelectorAll('.layout-row').length === 0 && " + READY)
    assert not files[0].exists() and server.pg.current_world().model_dump(mode="json") == original
    assert not tab.errors
    return {"name": name, "drive_distance_m": distance, "saved_bytes": len(frozen)}


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("laptop", "zh-CN"), ("phone", "en"), ("phone", "zh-CN")])
def test_save_map_restore_reload_and_drive(viewer, size, lang):  # noqa: F811
    viewer(size, lang, journey, world=course)


DELAY = """
const onmessage=Object.getOwnPropertyDescriptor(WebSocket.prototype,'onmessage');
Object.defineProperty(WebSocket.prototype,'onmessage',{
 get:onmessage.get, set(fn){onmessage.set.call(this,ev=>{
  const msg=typeof ev.data==='string'?JSON.parse(ev.data):null;
  if(window.__holdList && msg?.type==='reply' && msg.data?.entries){
   window.__holdList=false;window.__releaseList=()=>fn.call(this,ev);return;
  }
  if(window.__holdMutation && msg?.type==='reply' && msg.data?.name && msg.data?.entries){
   window.__holdMutation=false;window.__releaseMutation=()=>fn.call(this,ev);return;
  }
  if(window.__holdPoses && !msg){window.__lastPose=ev;this.send(JSON.stringify({type:'seen'}));return;}
  return fn.call(this,ev);
 });}
});
"""


def test_first_matching_poses_enable_layout_controls_without_another_refresh(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=DELAY)
        await tab.navigate()
        await open_layouts(tab)
        await tab.run("window.__holdPoses=true")
        old = server.pg.session.version
        await click(tab, "#map-list .choice:first-child")
        await tab.until(f"window.__robotforge.scene.version > {old} && !window.__robotforge.busy().busy")
        assert await tab.eval("document.getElementById('layout-save').disabled")
        await tab.run("window.__holdPoses=false")
        await tab.until(READY)
        assert not tab.errors

    viewer("laptop", "en", scenario, world=course)


def test_reset_before_delayed_save_reply_reconciles_after_mutation(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=DELAY)
        await tab.navigate()
        await open_layouts(tab)
        await tab.run("window.__holdMutation=true")
        await click(tab, "#layout-name")
        await tab.page.call("Input.insertText", text="Delayed save")
        await click(tab, "#layout-save")
        await tab.until("!!window.__releaseMutation")
        server.pg.session.reset()
        await asyncio.sleep(0.4)
        await tab.run("window.__releaseMutation()")
        await tab.until(READY)
        assert await tab.eval("document.querySelector('.layout-row span').textContent") == "Delayed save"
        assert not tab.errors

    viewer("laptop", "en", scenario, world=course)


def test_reset_during_first_pending_list_discards_its_old_generation_reply(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=DELAY)
        await tab.navigate()
        await tab.run("window.__holdList=true")
        await click(tab, "#world-toggle")
        await tab.until("!!window.__releaseList")
        server.pg.session.reset()
        await tab.until(READY)
        await tab.run("window.__releaseList()")
        await asyncio.sleep(0.4)
        assert await tab.eval(READY) and not tab.errors

    viewer("laptop", "en", scenario, world=course)
