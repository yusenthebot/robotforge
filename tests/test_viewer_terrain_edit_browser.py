"""Native route reuse: instance previews, safe retry, inverse edits and durable layouts."""

from __future__ import annotations

import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_startup_browser import READY
from test_viewer_terrain_create_browser import HOLD, place, prepare, tap


def course(pg):
    pg.set_floor("wood")
    pg.add_robot("rover", template="rover", pos=[-3, -2])
    pg.add_terrain(
        "ramp",
        name="dock",
        pos=[0, 0],
        yaw=13.5,
        params={
            "length": 0.6,
            "width": 0.9,
            "angle": 8,
            "landing": 0.4,
            "direction": "up_down",
            "style": "concrete",
            "friction": 0.72,
        },
    )
    pg.add_terrain(
        "stairs",
        name="steps",
        pos=[-2, 3],
        params={
            "steps": 3,
            "step_height": 0.05,
            "step_depth": 0.2,
            "width": 0.8,
            "landing": 0.4,
        },
    )
    pg.add_object("marker", shape="box", size=[0.4, 0.4, 0.2], pos=[2, 2], fixed=True)


async def choose(tab, name="dock"):
    if await tab.eval("document.getElementById('world-panel').hidden"):
        await click(tab, "#world-toggle")
    await click(tab, '[data-tab="terrain"]')
    selector = f'[data-edit-terrain="{name}"]'
    await tab.until(f"!document.querySelector('{selector}').disabled")
    await click(tab, selector)
    await tab.until("!document.getElementById('placement-bar').hidden")


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_native_custom_terrain_cancel_retry_rotate_undo_and_layout_replay(viewer, size, lang):  # noqa: F811
    async def journey(tab, server):
        await prepare(tab)
        original = server.pg.current_world()
        saved, data = server.pg.store.file.read_bytes(), server.pg.session.data
        await choose(tab)
        preview = await tab.run("""
          const s=window.__robotforge.three().scene,p=s.getObjectByName('placement-size-preview');
          return {parts:p.children.map(o=>({half:o.geometry.parameters,
            center:o.position.toArray(),pitch:o.rotation.y})),
            yaw:p.rotation.z,highlight:!!s.getObjectByName('terrain-edit-highlight')};
        """)
        boxes = await tab.eval("window.__robotforge.scene.terrain_edit.terrains.find(e=>e.name==='dock').boxes")
        assert preview["highlight"] and len(boxes) == len(preview["parts"]) == 3
        for part, box in zip(preview["parts"], boxes, strict=True):
            assert part["center"] == box["center"] and part["pitch"] == box["pitch"]
            assert [part["half"][axis] / 2 for axis in ("width", "height", "depth")] == box["half"]
        await key(tab, "Escape")
        assert server.pg.store.file.read_bytes() == saved and server.pg.session.data is data
        assert not await tab.eval("!!window.__robotforge.three().scene.getObjectByName('terrain-edit-highlight')")
        await choose(tab)
        await click(tab, "#place-rotate")
        await tap(tab, [2.5, 2, 0])
        await tab.until(
            "!window.__robotforge.busy().busy && document.getElementById('toast').textContent.includes('occupied')"
        )
        assert server.pg.store.file.read_bytes() == saved and server.pg.session.data is data
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        await place(tab, [3, -2, 0])
        moved = server.pg.current_world()
        assert moved.terrain[0].pos == pytest.approx([3, -2], abs=0.015) and moved.terrain[0].yaw == 103.5
        assert moved.terrain[0].model_dump(exclude={"pos", "yaw"}) == original.terrain[0].model_dump(
            exclude={"pos", "yaw"}
        )
        assert moved.terrain[1:] == original.terrain[1:] and moved.objects == original.objects
        assert server.pg.session.paused and server.pg.session.data.time == 0
        await click(tab, "#undo-edit")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert server.pg.current_world() == original and await tab.eval("document.getElementById('undo-edit').hidden")
        await choose(tab)
        await click(tab, "#place-rotate")
        await place(tab, [3, -2, 0])
        moved = server.pg.current_world()
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="maps"]')
        await tab.until("!document.getElementById('layout-save').disabled")
        await click(tab, "#layout-name")
        await tab.page.call("Input.insertText", text="Moved custom terrain")
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
        assert server.pg.current_world() == moved
        origin = await tab.eval("performance.timeOrigin")
        await tab.page.call("Page.reload")
        await tab.until(f"performance.timeOrigin!=={origin} && {READY}")
        assert server.pg.current_world() == moved
        if not await tab.eval("document.getElementById('world-panel').hidden"):
            await click(tab, "#world-toggle")
        await click(tab, "#btn-reset")
        await tab.until(
            "window.__robotforge.time===0 && document.getElementById('btn-play').classList.contains('is-paused')"
        )
        assert server.pg.current_world() == moved and not tab.errors

    viewer(size, lang, journey, world=course)


def test_terrain_move_receipt_survives_real_context_loss_and_display_timeout(viewer):  # noqa: F811
    async def journey(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=HOLD)
        await tab.navigate()
        await prepare(tab)
        before = server.pg.current_world()
        await choose(tab)
        await tab.run("window.__terrainWire.afterVersion=window.__robotforge.scene.version")
        await tap(tab, [3, -2, 0])
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
            "!window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden", timeout=20
        )
        assert server.pg._play_terrain_edit is not None and not await tab.eval(
            "document.getElementById('undo-edit').hidden"
        )
        await tab.run("window.__terrainContext.restoreContext()")
        await tab.until(READY)
        await click(tab, "#undo-edit")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert server.pg.current_world() == before and not tab.errors

    viewer("laptop", "en", journey, world=course)


NOOP_WIRE = """
const Native=window.WebSocket;
window.__noopWire={hold:false,waiting:null};
window.WebSocket=class extends Native {
 send(value){const m=JSON.parse(value);
  // Controlled boundary case: submit the exact authored coordinates, without pixel rounding.
  if(window.__noopWire.hold&&m.action==='edit_terrain'){
   m.pos=window.__robotforge.scene.terrain_edit.terrains.find(e=>e.name===m.name).pos;
   m.yaw_delta=0;value=JSON.stringify(m);
  }
  super.send(value);
 }
 set onmessage(handler){super.onmessage=event=>{
  const m=typeof event.data==='string'?JSON.parse(event.data):null;
  if(m?.type==='reply'&&m.data?.unchanged&&window.__noopWire.hold){
   window.__noopWire.waiting={data:m.data,release:()=>handler(event)};return;
  }
  handler(event);
  if(m?.type==='state')window.__noopWire.generation=m.generation;
 }}
};
"""


@pytest.mark.parametrize("reset", [False, True])
def test_delayed_actual_noop_reply_keeps_only_current_generation_undo(viewer, reset):  # noqa: F811
    async def journey(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=NOOP_WIRE)
        await tab.navigate()
        await prepare(tab)
        await choose(tab)
        await place(tab, [3, -2, 0])
        assert not await tab.eval("document.getElementById('undo-edit').hidden")
        await choose(tab)
        await tab.run("window.__noopWire.hold=true")
        await tap(tab, [3, -2, 0])
        await tab.until("!!window.__noopWire.waiting")
        assert await tab.eval("window.__noopWire.waiting.data.unchanged")
        if reset:
            generation = server.pg.session.generation
            server.pg.session.reset()
            await tab.until(f"window.__noopWire.generation>{generation}")
        await tab.run("window.__noopWire.hold=false;window.__noopWire.waiting.release()")
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('undo-edit').hidden") is reset
        assert not tab.errors

    viewer("laptop", "en", journey, world=course)
