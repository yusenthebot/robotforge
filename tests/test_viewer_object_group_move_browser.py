"""Native group moves preserve identity, support picking, resources and inverse edits."""

from __future__ import annotations

import asyncio
import json

import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_object_group_browser import PREPARE_DELAY, READY, select_group, table_layout
from test_viewer_placement_browser import spacing
from viewer_camera import settle_camera


async def prepare(tab, server):
    with server.pg.session.lock:
        server.pg.session.paused = True
    await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
    await tab.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
    await settle_camera(tab)


async def begin_move(tab):
    await select_group(tab)
    assert not await tab.eval("document.getElementById('object-group-move').disabled")
    await click(tab, "#object-group-move")
    await tab.until("!document.getElementById('placement-bar').hidden && !window.__robotforge.busy().busy")


async def move_to(tab, point):
    await settle_camera(tab)
    await click(tab, point=await tab.eval(f"window.__robotforge.screenOfPoint({json.dumps(point)})"))
    await tab.until(
        f"{READY} && document.getElementById('placement-bar').hidden && !document.getElementById('undo-edit').hidden"
    )


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_group_move_stack_cancel_rotate_identity_and_undo_preservation(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        before = server.pg.store.load().model_dump(mode="json")
        top_height = float(server.pg.session.data.body("top").xpos[2])
        await begin_move(tab)
        assert await tab.eval("document.querySelector('.repeat-placement').hidden")
        assert await tab.eval("document.getElementById('prop-material-control').hidden")
        assert await tab.eval("[...document.querySelectorAll('.prop-size-control')].every(node=>node.hidden)")
        ghost = await tab.run("""
          const scene=window.__robotforge.three().scene;
          const ghost=scene.getObjectByName('placement-size-preview');
          const owned=new Set(window.__robotforge.scene.object_edit.objects.flatMap(entry=>entry.geoms));
          const sources=[];scene.traverse(node=>{
            if(node.isMesh&&node.visible&&owned.has(node.userData.geom))sources.push(node);
          });
          window.__moveDisposals=0;
          for(const source of sources)source.geometry.addEventListener('dispose',()=>window.__moveDisposals++);
          return {visible:ghost?.visible,count:ghost?.children.length,sources:sources.length,
            borrowed:ghost?.children.every(part=>sources.some(source=>source.geometry===part.geometry))};
        """)
        assert ghost["visible"] and ghost["borrowed"]
        assert ghost["count"] == ghost["sources"] and ghost["count"] > 3
        await key(tab, "Escape")
        assert server.pg.store.load().model_dump(mode="json") == before
        assert await tab.eval("window.__moveDisposals") == 0

        await begin_move(tab)
        for selector in ("#place-rotate", "#place-rotate-plus", "#place-rotate-minus"):
            await click(tab, selector)
        assert (await tab.eval("document.getElementById('placement-heading').textContent")).endswith("90°")
        version = server.pg.session.version
        await move_to(tab, [2, 0, 0])
        assert server.pg.session.version == version + 1
        world = server.pg.store.load()
        assert [obj.name for obj in world.objects] == ["base", "top", "bin"]
        assert [obj.yaw for obj in world.objects] == pytest.approx([90, 90, 90])
        assert [obj.pos[2] for obj in world.objects] == pytest.approx([0, top_height, 0.5], abs=0.002)
        for obj, xy, original in zip(world.objects, ([2, 0], [2, -0.25], [2, 0.25]), before["objects"], strict=True):
            assert obj.pos[:2] == pytest.approx(xy, abs=0.002)
            actual = obj.model_dump(mode="json")
            assert {k: v for k, v in actual.items() if k not in {"pos", "yaw"}} == {
                k: v for k, v in original.items() if k not in {"pos", "yaw"}
            }
        undo_label = "Undo group move" if lang == "en" else "撤销整组移动"
        assert await tab.eval("document.getElementById('undo-edit').textContent") == undo_label
        moved = world.model_dump(mode="json")

        # Cancel and an exact no-op must both retain the last committed group inverse.
        await begin_move(tab)
        await click(tab, "#place-cancel")
        assert await tab.eval("document.getElementById('undo-edit').textContent") == undo_label
        assert not await tab.eval("document.getElementById('undo-edit').hidden")
        await begin_move(tab)
        await move_to(tab, [2, 0, 0])
        assert server.pg.session.version == version + 1
        assert server.pg.store.load().model_dump(mode="json") == moved
        assert await tab.eval("document.getElementById('undo-edit').textContent") == undo_label
        await click(tab, "#undo-edit")
        await tab.until(f"{READY} && document.getElementById('undo-edit').hidden")
        assert server.pg.store.load().model_dump(mode="json") == before
        assert server.pg.session.version == version + 2
        assert not tab.errors

    viewer(size, lang, scenario, world=table_layout)


def test_group_move_picks_floor_through_all_old_member_geoms_and_snap_recheck(viewer):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        before = server.pg.store.load().model_dump(mode="json")
        await begin_move(tab)
        await spacing(tab, "0.25")
        x, y = await tab.eval("window.__robotforge.screenOfPoint([-1.75,0,0])")
        await tab.page.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        await tab.until("!document.getElementById('place-here').disabled")
        point = await tab.eval(
            "window.__robotforge.three().scene.getObjectByName('placement-point-marker').position.toArray()"
        )
        assert point == pytest.approx([-1.75, 0, 0.008], abs=0.0001)
        # Changing the spacing rechecks the support ray at the old source footprint.
        await spacing(tab, "0.1")
        assert not await tab.eval("document.getElementById('place-here').disabled")
        await spacing(tab, "0.25")
        await click(tab, "#place-here")
        await tab.until(f"{READY} && document.getElementById('placement-bar').hidden")
        world = server.pg.store.load()
        assert [obj.name for obj in world.objects] == ["base", "top", "bin"]
        assert world.objects[0].pos == pytest.approx([-1.75, 0, 0], abs=0.002)
        await click(tab, "#undo-edit")
        await tab.until(f"{READY} && document.getElementById('undo-edit').hidden")
        assert server.pg.store.load().model_dump(mode="json") == before
        assert not tab.errors

    viewer("laptop", "en", scenario, world=table_layout)


@pytest.mark.parametrize("interruption", ["reset", "pointer"])
def test_delayed_move_prepare_cannot_claim_reset_or_later_pointer(viewer, interruption):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=PREPARE_DELAY)
        await tab.navigate()
        await prepare(tab, server)
        await select_group(tab)
        saved = server.pg.store.file.read_bytes()
        await tab.run("window.__delayGroupPrepare=true")
        await click(tab, "#object-group-move")
        await tab.until("!!window.__releaseGroupPrepare")
        if interruption == "reset":
            server.pg.session.reset()
            await asyncio.sleep(0.4)
        else:
            await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([1,0,0])"))
        await tab.run("window.__releaseGroupPrepare()")
        await tab.until("!window.__robotforge.busy().busy")
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('undo-edit').hidden")
        assert server.pg.store.file.read_bytes() == saved and not tab.errors

    viewer("laptop", "en", scenario, world=table_layout)


MOVE_COMPLETION_GUARD = """
window.__moveIds=new Set();const send=WebSocket.prototype.send;
WebSocket.prototype.send=function(s){const m=JSON.parse(s);
 if(m.action==='move_object_group'){window.__moveIds.add(m.id);window.__guardNextMoveScene=true;}
 return send.call(this,s)};
const descriptor=Object.getOwnPropertyDescriptor(WebSocket.prototype,'onmessage');
Object.defineProperty(WebSocket.prototype,'onmessage',{get:descriptor.get,set(fn){descriptor.set.call(this,ev=>{
 const m=typeof ev.data==='string'?JSON.parse(ev.data):null;
 if(window.__guardNextMoveScene&&m?.type==='scene'){
  window.__guardNextMoveScene=false;window.__guardedMoveVersion=m.version;
  if(window.__moveGuard==='delay-scene'){
   window.__releaseMoveScene=()=>fn.call(this,ev);return;
  }
  if(window.__moveGuard==='missing-member'){
   for(const node of [...m.bodies,...m.geoms])
    if(node.name==='bin'||node.name?.startsWith('bin/'))node.name='withheld_move_member';
   return fn.call(this,new MessageEvent('message',{data:JSON.stringify(m)}));
  }
 }
 if(m?.type==='reply'&&window.__moveIds.has(m.id)&&window.__moveGuard==='unknown-result'){
  // Corrupt a returned identity without adding a body. Existing source names must not
  // let completion accept a result for a different group.
  m.data.objects[m.data.objects.length-1]='unselected_member';
  return fn.call(this,new MessageEvent('message',{data:JSON.stringify(m)}));
 }
 return fn.call(this,ev);
});}});
"""


@pytest.mark.parametrize("fault", ["delay-scene", "missing-member", "unknown-result"])
def test_group_move_waits_for_new_render_and_every_same_named_member(viewer, fault):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=MOVE_COMPLETION_GUARD)
        await tab.navigate()
        await prepare(tab, server)
        await begin_move(tab)
        version = server.pg.session.version
        await tab.run(f"window.__moveGuard={json.dumps(fault)}")
        await settle_camera(tab)
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([2,0,0])"))
        await tab.until("!!window.__guardedMoveVersion")
        assert server.pg.session.version == version + 1
        assert [obj.name for obj in server.pg.store.load().objects] == ["base", "top", "bin"]
        if fault == "delay-scene":
            # All names still exist in the old displayed scene. That is insufficient.
            assert await tab.eval("window.__robotforge.scene.version") == version
            assert await tab.eval("window.__robotforge.busy().busy")
            assert await tab.eval("document.getElementById('undo-edit').hidden")
            await tab.run("window.__releaseMoveScene()")
            await tab.until(f"{READY} && !document.getElementById('undo-edit').hidden")
            await click(tab, "#undo-edit")
            await tab.until(f"{READY} && document.getElementById('undo-edit').hidden")
        else:
            await tab.until("!window.__robotforge.busy().busy")
            assert await tab.eval("document.getElementById('placement-bar').hidden")
            assert await tab.eval("document.getElementById('undo-edit').hidden")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=table_layout)


def test_group_move_three_undo_cycles_release_borrowed_ghosts_and_bound_resources(viewer):  # noqa: F811
    async def resources(tab):
        await settle_camera(tab)
        return await tab.run("""
          for(let frame=0;frame<4;frame++)await new Promise(requestAnimationFrame);
          const memory=window.__robotforge.three().renderer.info.memory;
          return {geometries:memory.geometries,textures:memory.textures};
        """)

    async def scenario(tab, server):
        await prepare(tab, server)
        before = server.pg.store.load().model_dump(mode="json")
        await begin_move(tab)
        await resources(tab)
        await key(tab, "Escape")
        baseline = await resources(tab)
        moved, restored, lifetimes = [], [], []
        for _ in range(3):
            await begin_move(tab)
            await resources(tab)
            await tab.run("""
              window.__moveLifetime={disposals:0,whileGhostAttached:0};
              const ghost=window.__robotforge.three().scene.getObjectByName('placement-size-preview');
              for(const geometry of new Set(ghost.children.map(part=>part.geometry)))
                geometry.addEventListener('dispose',()=>{
                  window.__moveLifetime.disposals++;
                  if(window.__robotforge.three().scene.getObjectByName('placement-size-preview'))
                    window.__moveLifetime.whileGhostAttached++;
                });
            """)
            await move_to(tab, [2, 0, 0])
            assert len(server.pg.store.load().objects) == 3
            moved.append(await resources(tab))
            lifetimes.append(await tab.eval("({...window.__moveLifetime})"))
            await click(tab, "#undo-edit")
            await tab.until(f"{READY} && document.getElementById('undo-edit').hidden")
            assert server.pg.store.load().model_dump(mode="json") == before
            restored.append(await resources(tab))
        assert moved == [baseline] * 3, {"baseline": baseline, "moved": moved}
        assert restored == [baseline] * 3, {"baseline": baseline, "restored": restored}
        assert all(lifetime["disposals"] > 0 for lifetime in lifetimes)
        assert all(lifetime["whileGhostAttached"] == 0 for lifetime in lifetimes)
        assert not tab.errors

    viewer("laptop", "en", scenario, world=table_layout)


def test_group_move_keyboard_rotation_invalidates_held_canvas_pointer(viewer):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        await begin_move(tab)
        await settle_camera(tab)
        saved, version = server.pg.store.file.read_bytes(), server.pg.session.version
        x, y = await tab.eval("window.__robotforge.screenOfPoint([2,0,0])")
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
        assert (await tab.eval("document.getElementById('placement-heading').textContent")).endswith("15°")
        await tab.page.call(
            "Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", buttons=0, clickCount=1
        )
        await asyncio.sleep(0.4)
        assert server.pg.session.version == version and server.pg.store.file.read_bytes() == saved
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        await key(tab, "Escape")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=table_layout)
