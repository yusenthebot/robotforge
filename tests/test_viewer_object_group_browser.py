"""Native selection, complete group ghosts, one-shot copying and grouped inverse."""

from __future__ import annotations

import asyncio

import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from viewer_camera import settle_camera

READY = "window.__robotforge.scene?.version === window.__robotforge.renderedVersion && !window.__robotforge.busy().busy"


def table_layout(pg):
    pg.add_object("base", shape="table", size=[1.2, 0.8, 0.5], pos=[-2.0, 0.0], fixed=True)
    pg.add_object("top", shape="box", size=[0.2, 0.2, 0.2], pos=[-2.25, 0.0, 0.6], material="steel")
    pg.add_object("bin", shape="bin", size=[0.3, 0.3, 0.2], pos=[-1.75, 0.0, 0.5], fixed=True)


async def select_group(tab):
    if await tab.eval("document.getElementById('world-panel').hidden"):
        await click(tab, "#world-toggle")
    await click(tab, '[data-tab="objects"]')
    await click(tab, "#select-object-group")
    await click(tab, "#world-toggle")
    for index, name in enumerate(("base", "top", "bin")):
        await click(tab, f'[data-edit-object="{name}"]')
        assert await tab.eval("document.getElementById('object-group-copy').disabled") is (index == 0)
    assert await tab.eval("document.querySelectorAll('#edit-object-list [aria-pressed=true]').length") == 3
    await click(tab, "#world-toggle")
    await settle_camera(tab)


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_tabletop_group_ghost_cancel_rotate_copy_and_all_member_undo(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        with server.pg.session.lock:
            server.pg.session.paused = True
        await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
        await tab.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
        before = server.pg.store.load().model_dump(mode="json")
        top_height = float(server.pg.session.data.body("top").xpos[2])
        await select_group(tab)
        top = await tab.eval("window.__robotforge.screenOfPoint([-2.25,0,.7])")
        await click(tab, point=top)
        assert await tab.eval("document.getElementById('object-edit-title').textContent") == (
            "Group · 2 / 16 props" if lang == "en" else "已选 2 / 16 个道具"
        )
        await click(tab, point=top)
        assert (
            await tab.eval("""window.__robotforge.three().scene.children.filter(
          node=>node.name==='object-group-member-highlight'&&node.visible).length""")
            == 3
        )
        assert await tab.eval("""['object-move','object-copy','object-remove'].every(
          id=>document.getElementById(id).hidden)""")
        assert await tab.eval("""!document.getElementById('object-surface').hidden &&
          !document.getElementById('object-surface').disabled""")
        await click(tab, "#object-group-copy")
        await tab.until("!document.getElementById('placement-bar').hidden && !window.__robotforge.busy().busy")
        assert await tab.eval("document.querySelector('.repeat-placement').hidden")
        assert await tab.eval("document.getElementById('prop-material-control').hidden")
        assert await tab.eval("[...document.querySelectorAll('.prop-size-control')].every(node=>node.hidden)")
        assert (await tab.eval("document.getElementById('placement-heading').textContent")).endswith("0°")
        ghost = await tab.run("""
          const scene=window.__robotforge.three().scene, ghost=scene.getObjectByName('placement-size-preview');
          if(!ghost?.visible)throw Error('no complete visible group ghost');
          const description=window.__robotforge.scene;
          const owned=new Set(description.object_edit.objects.flatMap(entry=>entry.geoms));
          const geoms=new Set(description.geoms.filter(geom=>owned.has(geom.id)&&
            geom.group<description.visibleGroupLimit&&geom.rgba[3]>0).map(geom=>geom.id));
          const sources=[];scene.traverse(node=>{
            if(node.isMesh&&node.visible&&geoms.has(node.userData.geom))sources.push(node);
          });
          window.__groupSourceDisposals=0;
          for(const source of sources)source.geometry.addEventListener('dispose',()=>window.__groupSourceDisposals++);
          return {count:ghost.children.length,sources:sources.length,hidden:owned.size-geoms.size,
            shared:ghost.children.every(part=>sources.some(source=>source.geometry===part.geometry)),
            transforms:ghost.children.every(part=>sources.some(source=>source.geometry===part.geometry&&
              Math.abs(part.matrix.elements[12]-(source.matrixWorld.elements[12]+2))<1e-6&&
              Math.abs(part.matrix.elements[13]-source.matrixWorld.elements[13])<1e-6&&
              Math.abs(part.matrix.elements[14]-source.matrixWorld.elements[14])<1e-6))};
        """)
        assert ghost["count"] == ghost["sources"] and ghost["count"] >= 3
        assert ghost["hidden"] > 0
        assert ghost["shared"] and ghost["transforms"]
        await key(tab, "Escape")
        assert await tab.eval("window.__groupSourceDisposals") == 0
        assert server.pg.store.load().model_dump(mode="json") == before
        await select_group(tab)
        await click(tab, "#object-group-copy")
        await tab.until("!document.getElementById('placement-bar').hidden && !window.__robotforge.busy().busy")
        for selector in ("#place-rotate", "#place-rotate-plus", "#place-rotate-minus"):
            await click(tab, selector)
        assert (await tab.eval("document.getElementById('placement-heading').textContent")).endswith("90°")
        await settle_camera(tab)
        version = server.pg.session.version
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([2,0,0])"))
        await tab.until(
            f"{READY} && document.getElementById('placement-bar').hidden"
            " && !document.getElementById('undo-edit').hidden"
        )
        assert server.pg.session.version == version + 1
        world = server.pg.store.load()
        assert len(world.objects) == 6
        assert world.model_dump(mode="json")["objects"][:3] == before["objects"]
        clones = world.objects[3:]
        assert [obj.yaw for obj in clones] == pytest.approx([90, 90, 90])
        assert [obj.pos[2] for obj in clones] == pytest.approx([0, top_height, 0.5], abs=0.002)
        for clone, xy in zip(clones, ([2, 0], [2, -0.25], [2, 0.25]), strict=True):
            assert clone.pos[:2] == pytest.approx(xy, abs=0.002)
        await click(tab, "#undo-edit")
        await tab.until(f"{READY} && document.getElementById('undo-edit').hidden")
        assert server.pg.session.version == version + 2
        assert server.pg.store.load().model_dump(mode="json") == before
        assert await tab.eval("""!window.__robotforge.scene.geoms.some(geom=>
          ['base_2','top_2','bin_2'].some(name=>geom.name===name||geom.name?.startsWith(name+'/')))""")
        assert not tab.errors

    viewer(size, lang, scenario, world=table_layout)


PREPARE_DELAY = """
window.__groupPrepareIds=new Set();const send=WebSocket.prototype.send;
WebSocket.prototype.send=function(s){const m=JSON.parse(s);
 if(m.action==='prepare_object_group')window.__groupPrepareIds.add(m.id);return send.call(this,s)};
const descriptor=Object.getOwnPropertyDescriptor(WebSocket.prototype,'onmessage');
Object.defineProperty(WebSocket.prototype,'onmessage',{get:descriptor.get,set(fn){descriptor.set.call(this,ev=>{
 const m=typeof ev.data==='string'?JSON.parse(ev.data):null;
 if(window.__delayGroupPrepare&&m?.type==='reply'&&window.__groupPrepareIds.has(m.id)){
  window.__delayGroupPrepare=false;window.__releaseGroupPrepare=()=>fn.call(this,ev);return;
 }return fn.call(this,ev);
});}});
"""


@pytest.mark.parametrize("interruption", ["reset", "pointer"])
def test_delayed_prepare_after_reset_or_pointer_epoch_cannot_enter_group_placement(viewer, interruption):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=PREPARE_DELAY)
        await tab.navigate()
        with server.pg.session.lock:
            server.pg.session.paused = True
        await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
        await tab.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
        await select_group(tab)
        saved = server.pg.store.file.read_bytes()
        await tab.run("window.__delayGroupPrepare=true")
        await click(tab, "#object-group-copy")
        await tab.until("!!window.__releaseGroupPrepare")
        if interruption == "reset":
            server.pg.session.reset()
            await asyncio.sleep(0.4)
        else:
            # Finish a real canvas tap while the reply is held: no active pointer remains,
            # but that later gesture must prevent preparation from taking ownership.
            await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([1,0,0])"))
        await tab.run("window.__releaseGroupPrepare()")
        await tab.until("!window.__robotforge.busy().busy")
        assert await tab.eval("document.getElementById('object-edit-bar').hidden") is (interruption == "reset")
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('undo-edit').hidden")
        assert server.pg.store.file.read_bytes() == saved
        assert not tab.errors

    viewer("laptop", "en", scenario, world=table_layout)


MISSING_GROUP_MEMBER = """
const send=WebSocket.prototype.send;
WebSocket.prototype.send=function(s){const m=JSON.parse(s);
 if(m.action==='duplicate_object_group')window.__omitLastGroupMember=true;
 return send.call(this,s)};
const descriptor=Object.getOwnPropertyDescriptor(WebSocket.prototype,'onmessage');
Object.defineProperty(WebSocket.prototype,'onmessage',{get:descriptor.get,set(fn){descriptor.set.call(this,ev=>{
 const m=typeof ev.data==='string'?JSON.parse(ev.data):null;
 if(window.__omitLastGroupMember&&m?.type==='scene'){
  // Keep legal geometry and poses, while withholding the last copied member's identity.
  // The first two copies are available, so waiting for only the first name would succeed.
  for(const node of [...m.bodies,...m.geoms]){
   if(node.name==='bin_2'||node.name?.startsWith('bin_2/'))node.name='withheld_group_member';
  }
  window.__omittedGroupVersion=m.version;window.__omitLastGroupMember=false;
  return fn.call(this,new MessageEvent('message',{data:JSON.stringify(m)}));
 }
 return fn.call(this,ev);
});}});
"""


def test_group_completion_requires_last_member_before_exposing_undo(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=MISSING_GROUP_MEMBER)
        await tab.navigate()
        with server.pg.session.lock:
            server.pg.session.paused = True
        await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
        await tab.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
        await select_group(tab)
        await click(tab, "#object-group-copy")
        await tab.until("!document.getElementById('placement-bar').hidden && !window.__robotforge.busy().busy")
        await settle_camera(tab)
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([2,0,0])"))
        await tab.until("!!window.__omittedGroupVersion && !window.__robotforge.busy().busy")
        assert len(server.pg.store.load().objects) == 6
        assert await tab.eval("""['base_2','top_2'].every(name=>window.__robotforge.scene.geoms.some(
          geom=>geom.name===name||geom.name?.startsWith(name+'/')))""")
        assert await tab.eval("""!window.__robotforge.scene.geoms.some(geom=>
          geom.name==='bin_2'||geom.name?.startsWith('bin_2/'))""")
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('undo-edit').hidden")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=table_layout)


def test_repeated_group_copy_undo_releases_ghost_before_source_buffers_and_bounds_resources(viewer):  # noqa: F811
    async def resources(tab):
        await settle_camera(tab)
        return await tab.run("""
          for(let frame=0;frame<4;frame++)await new Promise(requestAnimationFrame);
          const memory=window.__robotforge.three().renderer.info.memory;
          return {geometries:memory.geometries,textures:memory.textures};
        """)

    async def begin_copy(tab):
        await select_group(tab)
        await click(tab, "#object-group-copy")
        await tab.until("""!window.__robotforge.busy().busy &&
          !document.getElementById('placement-bar').hidden &&
          window.__robotforge.three().scene.getObjectByName('placement-size-preview')?.visible""")

    async def scenario(tab, server):
        with server.pg.session.lock:
            server.pg.session.paused = True
        await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
        await tab.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
        before = server.pg.store.load().model_dump(mode="json")
        # Warm the persistent placement marker and its heading once, so later samples
        # compare identical scenes rather than counting their first GPU allocation as a leak.
        await begin_copy(tab)
        await resources(tab)
        await tab.run("""
          window.__groupLifetime={disposals:0,whileGhostAttached:0};
          const ghost=window.__robotforge.three().scene.getObjectByName('placement-size-preview');
          for(const geometry of new Set(ghost.children.map(part=>part.geometry)))
            geometry.addEventListener('dispose',()=>window.__groupLifetime.disposals++);
        """)
        await key(tab, "Escape")
        baseline = await resources(tab)
        assert await tab.eval("window.__groupLifetime.disposals") == 0
        copies, inverses, lifetimes = [], [], []
        for _ in range(3):
            await begin_copy(tab)
            await resources(tab)
            await tab.run("""
              window.__groupLifetime={disposals:0,whileGhostAttached:0};
              const ghost=window.__robotforge.three().scene.getObjectByName('placement-size-preview');
              for(const geometry of new Set(ghost.children.map(part=>part.geometry)))
                geometry.addEventListener('dispose',()=>{
                  window.__groupLifetime.disposals++;
                  // A scene rebuild may dispose the owner. The borrowed ghost must already
                  // be detached so the next render cannot upload those freed buffers again.
                  if(window.__robotforge.three().scene.getObjectByName('placement-size-preview'))
                    window.__groupLifetime.whileGhostAttached++;
                });
            """)
            await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([2,0,0])"))
            await tab.until(
                f"{READY} && document.getElementById('placement-bar').hidden"
                " && !document.getElementById('undo-edit').hidden"
            )
            assert len(server.pg.store.load().objects) == 6
            copies.append(await resources(tab))
            lifetimes.append(await tab.eval("({...window.__groupLifetime})"))
            await click(tab, "#undo-edit")
            await tab.until(f"{READY} && document.getElementById('undo-edit').hidden")
            assert server.pg.store.load().model_dump(mode="json") == before
            inverses.append(await resources(tab))
        assert inverses == [baseline] * 3, {"baseline": baseline, "copies": copies, "undo": inverses}
        assert copies == [copies[0]] * 3, {"copies": copies, "undo": inverses}
        assert all(lifetime["disposals"] > 0 for lifetime in lifetimes)
        assert all(lifetime["whileGhostAttached"] == 0 for lifetime in lifetimes), lifetimes
        assert not tab.errors

    viewer("laptop", "en", scenario, world=table_layout)
