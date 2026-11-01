"""Trusted group surface choices, heterogeneous Undo and scene ownership boundaries."""

from __future__ import annotations

import pytest
from test_viewer_creation_browser import click
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_object_edit_browser import choose
from test_viewer_object_surface_browser import select_surface
from test_viewer_startup_browser import READY
from viewer_camera import settle_camera

NAMES = ("box", "ball", "column")


def world(pg):
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    pg.add_object("box", size=[0.25, 0.25, 0.25], material="steel")
    pg.add_object("ball", shape="sphere", size=[0.12], pos=[0.6, 0], surface_material="plastic")
    pg.add_object("column", shape="cylinder", size=[0.12, 0.3], pos=[1.2, 0], surface_material="steel")
    pg.add_object("desk", shape="table", size=[1, 0.6, 0.7], pos=[0, 1.5])


async def select_group(tab, names=NAMES):
    """Select exact members through the existing trusted object-list controls."""
    if await tab.eval("document.getElementById('world-panel').hidden"):
        await click(tab, "#world-toggle")
    await click(tab, '[data-tab="objects"]')
    await click(tab, "#select-object-group")
    await click(tab, "#world-toggle")
    for name in names:
        await click(tab, f'[data-edit-object="{name}"]')
    await click(tab, "#world-toggle")
    await settle_camera(tab)


async def pause(tab, server):
    with server.pg.session.lock:
        server.pg.session.paused = True
    await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")


async def apply_surface(tab, finish):
    """Choose a native finish and wait for all freshly selected members to render."""
    await select_surface(tab, finish)
    await click(tab, "#object-surface-apply")
    await tab.until(
        READY + " && !window.__robotforge.busy().busy && !document.getElementById('object-surface-controls').hidden"
    )
    await settle_camera(tab)
    await tab.run("await new Promise(requestAnimationFrame);await new Promise(requestAnimationFrame)")


WIRE = """
const Native=window.WebSocket;window.__batchWire={ids:new Set(),sent:[],holdReply:false,holdPoses:false,poses:[]};
window.WebSocket=class extends Native{
 send(raw){const m=typeof raw==='string'?JSON.parse(raw):null,w=window.__batchWire;
  if(['set_object_surface','set_object_surfaces'].includes(m?.action)){
   w.socket=this;w.ids.add(m.id);w.sent.push(m);w.sourceVersion=m.version;}
  return super.send(raw);
 }
 set onmessage(handler){super.onmessage=event=>{
  const m=typeof event.data==='string'?JSON.parse(event.data):null,w=window.__batchWire;
  if(m?.type==='reply'&&w.ids.has(m.id)){
   w.reply=m.data;if(w.holdReply){w.releaseReply=()=>handler(event);return;}
  }
  if(!m&&w.holdPoses&&w.sourceVersion&&w.newScene){w.poses.push(()=>handler(event));return;}
  if(m?.type==='scene'&&m.version>w.sourceVersion)w.newScene=true;
  handler(event);if(m?.type==='state')w.generation=m.generation;
 }}
};
"""


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_native_mixed_apply_noop_original_done_undo_reload_and_bounded_highlights(viewer, size, lang):  # noqa: F811
    async def journey(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=WIRE)
        await tab.navigate()
        await pause(tab, server)
        initial = server.pg.store.load()
        await select_group(tab)
        await click(tab, "#object-surface")
        assert await tab.eval("document.getElementById('object-surface-material').value") == "mixed"
        assert await tab.eval("document.getElementById('object-surface-material').selectedOptions[0].disabled")
        assert await tab.eval("document.getElementById('object-surface-apply').disabled")
        version = server.pg.session.version
        resources = []
        for _ in range(3):
            await apply_surface(tab, "wood")
            assert [o.surface_material for o in server.pg.store.load().objects[:3]] == ["wood"] * 3
            assert await tab.eval("document.getElementById('undo-edit').hidden")
            assert await tab.eval("document.querySelectorAll('#edit-object-list [aria-pressed=true]').length") == 3
            assert await tab.eval("""(()=>{const rf=window.__robotforge;
              const entries=rf.scene.object_edit.objects.filter(e=>['box','ball','column'].includes(e.name));
              return entries.length===3&&rf.scene.version===rf.renderedVersion
                &&entries.every(e=>e.surface_material==='wood');})()""")
            resources.append(await tab.eval("window.__robotforge.three().renderer.info.memory"))
            receipt, data = server.pg._play_edit, server.pg.session.data
            await click(tab, "#object-surface-apply")
            await tab.until("!window.__robotforge.busy().busy")
            assert server.pg._play_edit is receipt and server.pg.session.data is data
            await click(tab, "#object-edit-done")
            assert await tab.eval("document.getElementById('undo-edit').textContent") == (
                "Undo group surfaces" if lang == "en" else "撤销整组外观"
            )
            await click(tab, "#undo-edit")
            await tab.until(READY + " && !window.__robotforge.busy().busy")
            assert server.pg.store.load() == initial
            await select_group(tab)
            await click(tab, "#object-surface")
        assert resources[0] == resources[-1]
        assert server.pg.session.version == version + 6
        assert await tab.eval("window.__batchWire.sent.length") == 6
        assert (
            await tab.eval("""window.__robotforge.three().scene.children.filter(
          n=>n.name==='object-group-member-highlight').length""")
            == 3
        )
        await apply_surface(tab, "")
        assert [o.surface_material for o in server.pg.store.load().objects[:3]] == [None] * 3
        await click(tab, "#object-edit-done")
        await click(tab, "#undo-edit")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert server.pg.store.load() == initial
        await tab.run("window.__batchOldDocument=true")
        await tab.page.call("Page.reload")
        await tab.until("!window.__batchOldDocument && " + READY)
        await select_group(tab)
        await click(tab, "#object-surface")
        assert await tab.eval("document.getElementById('object-surface-material').value") == "mixed"
        # A draft alone does not save, and normal copy/move remains available after closing.
        await select_surface(tab, "wood")
        await click(tab, "#object-surface")
        assert server.pg.store.load() == initial
        assert not await tab.eval("document.getElementById('object-group-copy').disabled")
        assert not await tab.eval("document.getElementById('object-group-move').disabled")
        assert not tab.errors and await tab.eval("window.__qaErrors") == []

    viewer(size, lang, journey, world=world)


def test_furniture_and_selection_change_never_submit_partial_surface_set(viewer):  # noqa: F811
    async def journey(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=WIRE)
        await tab.navigate()
        await pause(tab, server)
        await select_group(tab, ("box", "desk"))
        assert not await tab.eval("document.getElementById('object-surface').hidden")
        assert await tab.eval("window.__batchWire.sent.length") == 0
        await click(tab, "#world-toggle")
        await click(tab, '[data-edit-object="desk"]')
        await click(tab, '[data-edit-object="ball"]')
        await click(tab, "#world-toggle")
        await click(tab, "#object-surface")
        await select_surface(tab, "wood")
        await click(tab, "#world-toggle")
        await click(tab, '[data-edit-object="ball"]')
        await click(tab, '[data-edit-object="column"]')
        await click(tab, "#world-toggle")
        assert await tab.eval("document.getElementById('object-surface-material').value") == "mixed"
        assert await tab.eval("document.getElementById('object-surface-apply').disabled")
        assert await tab.eval("window.__batchWire.sent.length") == 0
        await apply_surface(tab, "wood")
        assert await tab.eval("window.__batchWire.sent[0].names") == ["box", "column"]
        assert server.pg.store.load().objects[1].surface_material == "plastic"
        assert not tab.errors and await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", journey, world=world)


@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("unchanged", [False, True])
def test_late_apply_or_noop_reply_cannot_reopen_reset_selection(viewer, reset, unchanged):  # noqa: F811
    async def journey(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=WIRE)
        await tab.navigate()
        await pause(tab, server)
        await select_group(tab)
        await click(tab, "#object-surface")
        if unchanged:
            await apply_surface(tab, "wood")
        else:
            await select_surface(tab, "wood")
        await tab.run("window.__batchWire.holdReply=true")
        await click(tab, "#object-surface-apply")
        await tab.until("!!window.__batchWire.releaseReply")
        if reset:
            generation = server.pg.session.generation
            server.pg.session.reset()
            await tab.until(f"window.__batchWire.generation>{generation}")
        await tab.run("window.__batchWire.holdReply=false;window.__batchWire.releaseReply()")
        await tab.until("!window.__robotforge.busy().busy")
        assert await tab.eval("document.getElementById('object-edit-bar').hidden") is reset
        if not reset:
            await click(tab, "#object-edit-done")
        assert await tab.eval("document.getElementById('undo-edit').hidden") is reset
        assert not tab.errors and await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", journey, world=world)


@pytest.mark.parametrize("interruption", ["context", "reset", "disconnect"])
def test_render_wait_requires_all_members_and_retains_only_current_committed_receipt(viewer, interruption):  # noqa: F811
    async def journey(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=WIRE)
        await tab.navigate()
        await pause(tab, server)
        await select_group(tab)
        await click(tab, "#object-surface")
        await select_surface(tab, "wood")
        await tab.run("window.__batchWire.holdPoses=true")
        await click(tab, "#object-surface-apply")
        await tab.until("!!window.__batchWire.reply && window.__batchWire.poses.length>0")
        assert await tab.eval("window.__robotforge.busy().busy")
        assert await tab.eval("document.getElementById('undo-edit').hidden")
        if interruption == "disconnect":
            # Close the actual viewer connection; automatic reconnect must not revive its old edit.
            await tab.run("window.__batchWire.socket.close()")
        else:
            extension = await tab.run("""window.__batchContext=window.__robotforge.three().renderer.getContext()
              .getExtension('WEBGL_lose_context');if(window.__batchContext)window.__batchContext.loseContext();
              return !!window.__batchContext;""")
            if not extension:
                pytest.skip("driver does not expose WEBGL_lose_context")
            await tab.until("window.__robotforge.three().renderer.getContext().isContextLost()")
            if interruption == "reset":
                generation = server.pg.session.generation
                server.pg.session.reset()
                await tab.until(f"window.__batchWire.generation>{generation}")
        await tab.run("const w=window.__batchWire;w.holdPoses=false;for(const f of w.poses.splice(0))f()")
        await tab.until(
            "!window.__robotforge.busy().busy && document.getElementById('object-edit-bar').hidden", timeout=20
        )
        assert await tab.eval("document.getElementById('undo-edit').hidden") is (interruption != "context")
        if interruption != "disconnect":
            await tab.run("window.__batchContext.restoreContext()")
            await tab.until(READY)
        assert [o.surface_material for o in server.pg.store.load().objects[:3]] == ["wood"] * 3
        assert not tab.errors and await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", journey, world=world)


@pytest.mark.parametrize("group", [False, True])
def test_noop_context_loss_preserves_current_single_or_group_undo(viewer, group):  # noqa: F811
    async def journey(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=WIRE)
        await tab.navigate()
        await pause(tab, server)
        initial = server.pg.store.load()
        if group:
            await select_group(tab)
        else:
            await choose(tab, "box")
        await click(tab, "#object-surface")
        await apply_surface(tab, "wood")
        receipt, data = server.pg._play_edit, server.pg.session.data
        await tab.run("window.__batchWire.holdReply=true")
        await click(tab, "#object-surface-apply")
        await tab.until("!!window.__batchWire.releaseReply")
        extension = await tab.run("""window.__batchContext=window.__robotforge.three().renderer.getContext()
          .getExtension('WEBGL_lose_context');if(window.__batchContext)window.__batchContext.loseContext();
          return !!window.__batchContext;""")
        if not extension:
            pytest.skip("driver does not expose WEBGL_lose_context")
        await tab.until("window.__robotforge.three().renderer.getContext().isContextLost()")
        await tab.run("window.__batchWire.holdReply=false;window.__batchWire.releaseReply()")
        await tab.until("!window.__robotforge.busy().busy")
        assert server.pg._play_edit is receipt and server.pg.session.data is data
        assert await tab.eval("document.getElementById('object-edit-bar').hidden")
        assert not await tab.eval("document.getElementById('undo-edit').hidden")
        await tab.run("window.__batchContext.restoreContext()")
        await tab.until(READY)
        version = server.pg.session.version
        await click(tab, "#undo-edit")
        await tab.until(READY + f" && window.__robotforge.scene.version>{version}")
        assert server.pg.store.load() == initial
        assert server.pg._play_edit is None
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", journey, world=world)
