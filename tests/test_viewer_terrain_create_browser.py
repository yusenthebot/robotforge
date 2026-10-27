"""Native local-route building, safe retry, durable layouts and real display failure recovery."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_startup_browser import READY
from viewer_camera import settle_camera


def course(pg):
    pg.set_floor("wood")
    pg.add_robot("rover", template="rover", pos=[-2, -2])
    pg.add_object("marker", shape="box", size=[0.3, 0.3, 0.3], pos=[-3, 2], fixed=True)


async def prepare(tab):
    await tab.until(READY + " && window.__robotforge.time > 0.1")
    await click(tab, "#btn-play")
    await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
    await tab.run("window.__robotforge.lookAt([9,-15,14],[0,0,.2])")


async def choose(tab, name):
    if await tab.eval("document.getElementById('world-panel').hidden"):
        await click(tab, "#world-toggle")
    await click(tab, '[data-tab="terrain"]')
    await click(tab, f'#terrain-create-list [data-terrain="{name}"]')
    await tab.until("!document.getElementById('placement-bar').hidden")
    assert await tab.eval("document.querySelector('.repeat-placement').hidden")
    assert await tab.eval("[...document.querySelectorAll('.prop-size-control')].every(e=>e.hidden)")


async def tap(tab, position):
    await settle_camera(tab)
    point = await tab.eval(f"window.__robotforge.screenOfPoint({position})")
    assert await tab.eval(f"document.elementFromPoint({point[0]},{point[1]})?.id") == "view"
    await click(tab, point=point)


async def place(tab, position):
    version = await tab.eval("window.__robotforge.scene.version")
    await tap(tab, position)
    await tab.until(
        f"{READY} && window.__robotforge.scene.version>{version} && "
        "document.getElementById('placement-bar').hidden && !window.__robotforge.busy().busy"
    )


async def journey(tab, server):
    await prepare(tab)
    saved = server.pg.store.file.read_bytes()
    data = server.pg.session.data
    state = [a.copy() for a in (data.qpos, data.qvel, data.ctrl)]
    stamp = data.time
    await choose(tab, "ramp")
    await tab.until("!!window.__robotforge.three().scene.getObjectByName('placement-size-preview')")
    preview = await tab.run("""
      const rf=window.__robotforge,root=rf.three().scene.getObjectByName('placement-size-preview');
      return {parts:root.children.map(p=>({center:p.position.toArray(),pitch:p.rotation.y,
        wire:p.material.wireframe,shadow:p.castShadow})),parent:root.parent===rf.three().scene};
    """)
    boxes = next(
        p["boxes"] for p in await tab.eval("window.__robotforge.scene.catalog.terrain_presets") if p["name"] == "ramp"
    )
    assert len(preview["parts"]) == 3 and preview["parent"]
    for part, box in zip(preview["parts"], boxes, strict=True):
        assert part["center"] == box["center"] and part["pitch"] == box["pitch"]
        assert part["wire"] and not part["shadow"]
    await key(tab, "Escape")
    assert server.pg.store.file.read_bytes() == saved and server.pg.session.data is data
    for actual, before in zip((data.qpos, data.qvel, data.ctrl), state, strict=True):
        np.testing.assert_array_equal(actual, before)
    assert data.time == stamp and server.pg.session.paused
    await choose(tab, "ramp")
    await place(tab, [2, -2, 0])
    assert len(server.pg.current_world().terrain) == 1 and server.pg.session.paused
    assert server.pg.session.data.time == 0
    await choose(tab, "stairs")
    version = server.pg.session.version
    await tap(tab, [4.5, -2, 0])  # Center is clear, but the whole staircase crosses the existing ramp edge.
    await tab.until(
        "!window.__robotforge.busy().busy && document.getElementById('toast').textContent.includes('occupied')"
    )
    assert server.pg.session.version == version and len(server.pg.current_world().terrain) == 1
    assert not await tab.eval("document.getElementById('placement-bar').hidden")
    await click(tab, "#place-rotate")
    await place(tab, [1, 2, 0])
    world = server.pg.current_world()
    assert [(p.type, p.yaw) for p in world.terrain] == [("ramp", 0), ("stairs", 90)]
    assert all(p.style == "wood" for p in world.terrain) and len(world.objects) == 1
    await click(tab, "#undo-add")
    await tab.until(READY + " && !window.__robotforge.busy().busy")
    assert [p.type for p in server.pg.current_world().terrain] == ["ramp"]
    await choose(tab, "stairs")
    await click(tab, "#place-rotate")
    await place(tab, [1, 2, 0])
    original = server.pg.current_world().model_dump(mode="json")
    await click(tab, "#world-toggle")
    await click(tab, '[data-tab="maps"]')
    await tab.until("!document.getElementById('layout-save').disabled")
    await click(tab, "#layout-name")
    await tab.page.call("Input.insertText", text="Local route")
    await click(tab, "#layout-save")
    await tab.until(
        "document.querySelectorAll('.layout-row').length===1 && !document.getElementById('layout-save').disabled"
    )
    await click(tab, "#map-list .choice:first-child")
    await tab.until(READY + " && !document.getElementById('layout-save').disabled")
    assert not server.pg.current_world().terrain
    await click(tab, '[data-layout-action="restore"]')
    await tab.until("!document.getElementById('layout-confirm').hidden")
    await click(tab, "#layout-apply")
    await tab.until(READY + " && !document.getElementById('layout-save').disabled")
    assert server.pg.current_world().model_dump(mode="json") == original
    origin = await tab.eval("performance.timeOrigin")
    await tab.page.call("Page.reload")
    await tab.until(f"performance.timeOrigin!=={origin} && {READY}")
    assert server.pg.current_world().model_dump(mode="json") == original
    if not await tab.eval("document.getElementById('world-panel').hidden"):
        await click(tab, "#world-toggle")
    start = server.pg.session.data.qpos[:2].copy()
    await click(tab, "#btn-play")
    await key(tab, "w", "KeyW")
    try:
        await tab.until("window.__robotforge.time >= 2.5", timeout=10)
    finally:
        await key(tab, "w", "KeyW", "keyUp")
    await asyncio.sleep(0.25)
    distance = float(np.linalg.norm(server.pg.session.data.qpos[:2] - start))
    assert distance > 0.25 and np.isfinite(server.pg.session.data.qpos).all()
    np.testing.assert_array_equal(server.pg.session.data.ctrl, np.zeros(server.pg.session.model.nu))
    await click(tab, "#btn-play")
    await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
    await click(tab, "#btn-reset")
    await tab.until(
        "window.__robotforge.time===0 && document.getElementById('btn-play').classList.contains('is-paused')"
    )
    from robotforge.sim.session import fresh_data

    home = fresh_data(server.pg.session.model)
    np.testing.assert_array_equal(server.pg.session.data.qpos, home.qpos)
    np.testing.assert_array_equal(server.pg.session.data.ctrl, home.ctrl)
    assert not tab.errors
    return {"world": original, "preview": preview, "drive_distance_m": distance}


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_native_local_route_cancel_retry_undo_save_restore_reload(viewer, size, lang):  # noqa: F811
    viewer(size, lang, journey, world=course)


HOLD = """
const Native=window.WebSocket;
window.__terrainWire={poses:[],reply:null,hold:false};
window.WebSocket=class extends Native{
 set onmessage(handler){super.onmessage=event=>{
  const msg=typeof event.data==='string'?JSON.parse(event.data):null,wire=window.__terrainWire;
  if(msg?.type==='scene'&&wire.afterVersion&&msg.version>wire.afterVersion)wire.hold=true;
  if(msg?.type==='reply'&&typeof msg.data?.terrain==='string')wire.reply=msg.data;
  if(!msg&&wire.hold){wire.poses.push(()=>handler(event));return}
  handler(event);
 }}
};
"""


def test_committed_terrain_keeps_undo_after_reply_before_poses_and_real_context_loss(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=HOLD)
        await tab.navigate()
        await prepare(tab)
        await choose(tab, "ramp")
        await tab.run("window.__terrainWire.afterVersion=window.__robotforge.scene.version")
        await tap(tab, [2, -2, 0])
        await tab.until("!!window.__terrainWire.reply && window.__terrainWire.poses.length>0")
        supported = await tab.run("""window.__terrainContext=window.__robotforge.three().renderer.getContext()
          .getExtension('WEBGL_lose_context');if(window.__terrainContext)window.__terrainContext.loseContext();
          return !!window.__terrainContext;""")
        if not supported:
            pytest.skip("driver does not expose WEBGL_lose_context")
        await tab.until("window.__robotforge.three().renderer.getContext().isContextLost()")
        await tab.run(
            "const w=window.__terrainWire;w.afterVersion=null;w.hold=false;"
            "for(const release of w.poses.splice(0))release()"
        )
        await tab.until(
            "window.__robotforge.time===0 && document.getElementById('btn-play').classList.contains('is-paused')"
        )
        await tab.until(
            "!window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden", timeout=20
        )
        assert len(server.pg.current_world().terrain) == 1 and server.pg._play_addition is not None
        assert not await tab.eval("document.getElementById('undo-add').hidden")
        (tab.home / "terrain-timeout.json").write_text(json.dumps(await tab.eval("window.__terrainWire.reply")))
        await tab.run("window.__terrainContext.restoreContext()")
        await tab.until(READY)
        await click(tab, "#undo-add")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert not server.pg.current_world().terrain and not tab.errors

    viewer("laptop", "en", scenario, world=course)
