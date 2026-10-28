"""Native surface editing, physical-material separation, reopen and stale reply boundaries."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from test_viewer_creation_browser import click
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_object_edit_browser import choose
from test_viewer_startup_browser import READY
from viewer_camera import settle_camera


def world(pg):
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    pg.add_object("block", shape="box", size=[0.5, 0.3, 0.2], pos=[0, 0], material="steel", color="#a98a65")
    pg.add_object("desk", shape="table", size=[1, 0.6, 0.7], pos=[0, 1.5])


async def select_surface(tab, value):
    # Closed native select typeahead/keys produce trusted input without assigning the value.
    await tab.run("""const e=document.getElementById('object-surface-material');e.focus();
      window.__surfaceChanges=[];window.__surfaceListener=event=>
        window.__surfaceChanges.push({trusted:event.isTrusted,value:e.value});
      e.addEventListener('change',window.__surfaceListener);""")
    if tab.size == "phone":
        steps = ["", "plastic", "wood", "steel"].index(value)
        for name, code in [("Home", 36), *[("ArrowDown", 40)] * steps]:
            for event in ["keyDown", "keyUp"]:
                await tab.page.call("Input.dispatchKeyEvent", type=event, key=name, windowsVirtualKeyCode=code)
    else:
        await asyncio.sleep(1.05)
        letter = await tab.eval(
            f"[...document.getElementById('object-surface-material').options].find(o=>o.value==={json.dumps(value)}).textContent[0]"
        )
        for event in ["keyDown", "keyUp"]:
            await tab.page.call(
                "Input.dispatchKeyEvent", type=event, key=letter, text=letter if event == "keyDown" else ""
            )
    assert await tab.eval("document.getElementById('object-surface-material').value") == value
    changes = await tab.eval("window.__surfaceChanges")
    await tab.run(
        "document.getElementById('object-surface-material').removeEventListener('change',window.__surfaceListener)"
    )
    assert changes and all(x["trusted"] for x in changes)


async def apply_surface(tab, value):
    await select_surface(tab, value)
    await click(tab, "#object-surface-apply")
    await tab.until(
        READY + " && !window.__robotforge.busy().busy && !document.getElementById('object-surface-controls').hidden"
    )
    # Reselection publishes its controls before the highlight's first GPU upload.
    # Compare resources in the same settled view, after the selected scene has actually drawn.
    await settle_camera(tab)
    await tab.run("await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame)")


INSPECT = """(()=>{const rf=window.__robotforge,entry=rf.scene.object_edit.objects.find(e=>e.name==='block'),rows=[];
  rf.three().scene.traverse(o=>{if(o.isMesh&&o.visible&&entry.geoms.includes(o.userData.geom)){
    const g=rf.scene.geoms.find(g=>g.id===o.userData.geom);
    rows.push({name:g.material,metalness:o.material.metalness,wood:!!o.material.userData.woodGrainAxis});}});
  return {rows,memory:rf.three().renderer.info.memory,gl:rf.three().renderer.getContext().getError()};})()"""


@pytest.mark.parametrize("size,lang,width", [("laptop", "en", 1280), ("laptop", "en", 821), ("phone", "zh-CN", 390)])
def test_native_three_finishes_noop_undo_and_reopen(viewer, size, lang, width):  # noqa: F811
    async def journey(tab, server):
        if width == 821:
            await tab.page.call(
                "Emulation.setDeviceMetricsOverride", width=821, height=800, deviceScaleFactor=1, mobile=False
            )
        server.pg.session.paused = True
        await choose(tab, "block")
        await click(tab, "#object-surface")
        assert await tab.eval("document.getElementById('object-surface-apply').textContent") == (
            "应用并重置" if lang == "zh-CN" else "Apply and reset"
        )
        initial = server.pg.store.load()
        samples = []
        for finish, preset in [
            ("wood", "rf_wood"),
            ("steel", "rf_metal"),
            ("plastic", "rf_shell"),
            ("wood", "rf_wood"),
        ]:
            await apply_surface(tab, finish)
            entry = server.pg.store.load().objects[0]
            assert entry.surface_material == finish and entry.material == "steel"
            assert len(server.pg.store.load().objects) == len(initial.objects)
            assert server.pg.session.paused and server.pg.session.data.time == 0
            actual = await tab.eval(INSPECT)
            assert actual["rows"] and all(r["name"] == preset for r in actual["rows"])
            assert all(r["wood"] == (finish == "wood") for r in actual["rows"])
            assert actual["gl"] == 0
            samples.append(actual["memory"])
            assert not await tab.eval("document.getElementById('object-surface-apply').disabled")
        assert samples[0] == samples[-1]
        receipt, data = server.pg._play_edit, server.pg.session.data
        await click(tab, "#object-surface-apply")
        await tab.until("!window.__robotforge.busy().busy")
        assert server.pg._play_edit is receipt and server.pg.session.data is data
        await click(tab, "#object-edit-done")
        await click(tab, "#undo-edit")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert server.pg.store.load().objects[0].surface_material == "plastic"
        await tab.run("window.__surfaceOldDocument=true")
        await tab.page.call("Page.reload")
        await tab.until("!window.__surfaceOldDocument && " + READY + " && document.getElementById('loading').hidden")
        await choose(tab, "block")
        await click(tab, "#object-surface")
        assert await tab.eval("document.getElementById('object-surface-material').value") == "plastic"
        await apply_surface(tab, "")
        assert server.pg.store.load().objects[0].surface_material is None
        assert all(r["name"] == "rf_metal" for r in (await tab.eval(INSPECT))["rows"])
        await click(tab, "#object-edit-done")
        try:
            await choose(tab, "desk")
        except Exception:
            shot = await tab.page.call("Page.captureScreenshot", format="png")
            (tab.home / "surface-failure.png").write_bytes(base64.b64decode(shot["data"]))
            (tab.home / "surface-failure.json").write_text(
                json.dumps(
                    await tab.run("""
              const e=document.querySelector('[data-edit-object="desk"]'),r=e.getBoundingClientRect();
              return {rect:r.toJSON(),disabled:e.disabled,
                hit:document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)?.outerHTML,
                panel:document.getElementById('world-panel').getBoundingClientRect().toJSON(),busy:window.__robotforge.busy()};
            """)
                ),
                encoding="utf-8",
            )
            raise
        if width >= 821:
            assert await tab.run("""const p=document.getElementById('world-panel').getBoundingClientRect(),
              h=document.querySelector('.drive-hud').getBoundingClientRect();
              return h.left>=p.right && h.right<=innerWidth;""")
        assert await tab.eval("document.getElementById('object-surface').hidden")
        assert not tab.errors and await tab.eval("window.__qaErrors") == []

    viewer(size, lang, journey, world=world)


NOOP_WIRE = """
const Native=window.WebSocket;window.__surfaceWire={hold:false,waiting:null};
window.WebSocket=class extends Native{
 set onmessage(handler){super.onmessage=event=>{
  const m=typeof event.data==='string'?JSON.parse(event.data):null,w=window.__surfaceWire;
  if(m?.type==='reply'&&m.data?.unchanged&&w.hold){w.waiting={data:m.data,release:()=>handler(event)};return}
  handler(event);if(m?.type==='state')w.generation=m.generation;
 }}
};
"""


@pytest.mark.parametrize("reset", [False, True])
def test_actual_late_noop_ack_preserves_only_current_undo(viewer, reset):  # noqa: F811
    async def journey(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=NOOP_WIRE)
        await tab.navigate()
        await choose(tab, "block")
        await click(tab, "#object-surface")
        await apply_surface(tab, "wood")
        await tab.run("window.__surfaceWire.hold=true")
        await click(tab, "#object-surface-apply")
        await tab.until("!!window.__surfaceWire.waiting")
        if reset:
            generation = server.pg.session.generation
            server.pg.session.reset()
            await tab.until(f"window.__surfaceWire.generation>{generation}")
        await tab.run("window.__surfaceWire.hold=false;window.__surfaceWire.waiting.release()")
        await tab.until("!window.__robotforge.busy().busy")
        if not reset:
            await click(tab, "#object-edit-done")
        assert await tab.eval("document.getElementById('object-edit-bar').hidden")
        assert await tab.eval("document.getElementById('undo-edit').hidden") is reset
        assert not tab.errors

    viewer("laptop", "en", journey, world=world)


POSE_WIRE = """
const Native=window.WebSocket;window.__surfacePoses={hold:false,poses:[],reply:null};
window.WebSocket=class extends Native{
 set onmessage(handler){super.onmessage=event=>{
  const m=typeof event.data==='string'?JSON.parse(event.data):null,w=window.__surfacePoses;
  if(m?.type==='scene'&&w.afterVersion&&m.version>w.afterVersion)w.hold=true;
  if(m?.type==='reply'&&m.data?.object)w.reply=m.data;
  if(!m&&w.hold){w.poses.push(()=>handler(event));return}
  handler(event);if(m?.type==='state')w.generation=m.generation;
 }}
};
"""


@pytest.mark.parametrize("reset", [False, True])
def test_display_wait_context_loss_keeps_only_current_committed_undo(viewer, reset):  # noqa: F811
    async def journey(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=POSE_WIRE)
        await tab.navigate()
        await choose(tab, "block")
        await click(tab, "#object-surface")
        await select_surface(tab, "wood")
        await tab.run("window.__surfacePoses.afterVersion=window.__robotforge.scene.version")
        await click(tab, "#object-surface-apply")
        await tab.until("!!window.__surfacePoses.reply && window.__surfacePoses.poses.length>0")
        extension = await tab.run("""window.__surfaceContext=window.__robotforge.three().renderer.getContext()
          .getExtension('WEBGL_lose_context');if(window.__surfaceContext)window.__surfaceContext.loseContext();
          return !!window.__surfaceContext;""")
        if not extension:
            pytest.skip("driver does not expose WEBGL_lose_context")
        await tab.until("window.__robotforge.three().renderer.getContext().isContextLost()")
        if reset:
            generation = server.pg.session.generation
            server.pg.session.reset()
            await tab.until(f"window.__surfacePoses.generation>{generation}")
        await tab.run(
            "const w=window.__surfacePoses;w.afterVersion=null;w.hold=false;for(const f of w.poses.splice(0))f()"
        )
        await tab.until(
            "!window.__robotforge.busy().busy && document.getElementById('object-edit-bar').hidden", timeout=20
        )
        assert await tab.eval("document.getElementById('undo-edit').hidden") is reset
        await tab.run("window.__surfaceContext.restoreContext()")
        await tab.until(READY)
        assert server.pg.current_world().objects[0].surface_material == "wood"
        if not reset:
            await click(tab, "#undo-edit")
            await tab.until(READY + " && !window.__robotforge.busy().busy")
            assert server.pg.current_world().objects[0].surface_material is None
        assert not tab.errors

    viewer("laptop", "en", journey, world=world)
