"""Explicit support selection is read-only and joins existing native group actions."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from pathlib import Path
from uuid import uuid4

import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, qa, viewer  # noqa: F401
from test_viewer_object_group_browser import READY
from test_viewer_object_group_move_browser import prepare
from viewer_camera import settle_camera

NAMES = ["base", "top", "bin", "stacked"]
HIGHLIGHTS = (
    "window.__robotforge.three().scene.children.filter("
    "node=>node.name==='object-group-member-highlight'&&node.visible).length"
)
SINGLE = (
    "window.__robotforge.three().scene.children.filter(node=>node.name==='object-edit-highlight'&&node.visible).length"
)
CAMERA = "[...window.__robotforge.three().camera.position.toArray(),...window.__robotforge.cameraTarget()]"


def carried_world(pg):
    pg.add_object("base", shape="table", size=[1.2, 0.8, 0.5], pos=[-2, 0], fixed=True)
    pg.add_object("top", shape="box", size=[0.2, 0.2, 0.2], pos=[-2.25, 0, 0.6], material="steel")
    pg.add_object("bin", shape="bin", size=[0.3, 0.3, 0.2], pos=[-1.75, 0, 0.5], fixed=True)
    pg.add_object("stacked", shape="box", size=[0.1, 0.1, 0.1], pos=[-2.25, 0, 0.75], material="wood")


WIRE = """
window.__carriedRequests=[];window.__carriedReplies=[];window.__carriedIds=new Set();
const send=WebSocket.prototype.send;
WebSocket.prototype.send=function(s){const m=JSON.parse(s);window.__carriedSocket=this;
 if(m.action==='select_object_group')window.__carriedIds.add(m.id);
 if(m.type==='world')window.__carriedRequests.push(m);
 return send.call(this,s);
};
const descriptor=Object.getOwnPropertyDescriptor(WebSocket.prototype,'onmessage');
Object.defineProperty(WebSocket.prototype,'onmessage',{get:descriptor.get,set(fn){descriptor.set.call(this,ev=>{
 const m=typeof ev.data==='string'?JSON.parse(ev.data):null;
 if(m?.type==='reply'&&window.__carriedIds.has(m.id)){
  window.__carriedReplies.push(structuredClone(m));
  if(m.ok){const d=m.data,f=window.__carriedFault;
   if(f==='duplicates')d.names[1]=d.names[0];
   if(f==='unknown')d.names[1]='not_an_editable_prop';
   if(f==='too-many')d.names=Array.from({length:17},(_,i)=>'prop'+i);
   if(f==='missing-root')d.names=d.names.filter(name=>name!==d.name);
   if(f==='wrong-root')d.name='top';
   if(f==='wrong-order')d.names.reverse();
   if(f==='wrong-generation')d.generation++;
   if(f==='wrong-version')d.version++;
   if(f==='wrong-world')d.world_sha256='0'.repeat(64);
   if(f==='null')m.data=null;
   if(f==='error'){m.ok=false;m.message='Selection rejected';}
  }
  const delivered=new MessageEvent('message',{data:JSON.stringify(m)});
  if(window.__holdCarried){window.__holdCarried=false;window.__releaseCarried=()=>fn.call(this,delivered);return;}
  return fn.call(this,delivered);
 }
 return fn.call(this,ev);
});}});
"""


async def ready(tab, server):
    await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=WIRE)
    await tab.navigate()
    await prepare(tab, server)


async def select_single(tab, name="base"):
    if await tab.eval("document.getElementById('world-panel').hidden"):
        await click(tab, "#world-toggle")
    await click(tab, '[data-tab="objects"]')
    await click(tab, f'[data-edit-object="{name}"]')
    await tab.until(
        "!document.getElementById('object-edit-bar').hidden"
        f" && document.getElementById('object-edit-title').textContent==={json.dumps(name)}"
    )
    await settle_camera(tab)


async def select_carried(tab):
    await select_single(tab)
    assert not await tab.eval("document.getElementById('object-select-carried').disabled")
    await click(tab, "#object-select-carried")
    await tab.until(f"{HIGHLIGHTS}===4 && !window.__robotforge.busy().busy")


def state(server):
    session = server.pg.session
    with session.lock:
        return (
            server.pg.store.file.read_bytes(),
            session.version,
            session.generation,
            session.data.time,
            session.data.qpos.tobytes(),
            session.data.qvel.tobytes(),
            session.data.act.tobytes(),
            session.data.ctrl.tobytes(),
            server.pg._play_edit,
            server.pg._play_addition,
        )


async def inverse(tab):
    await click(tab, "#undo-edit")
    await tab.until(f"{READY} && document.getElementById('undo-edit').hidden")


async def capture_evidence(tab, phase):
    evidence = os.environ.get("ROBOTFORGE_QA_EVIDENCE")
    if not evidence:
        return None
    directory = Path(evidence)
    directory.mkdir(parents=True, exist_ok=True)
    case = re.sub(r"[^A-Za-z0-9_.-]+", "-", os.environ.get("PYTEST_CURRENT_TEST", "supported-selection"))
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", phase)
    path = directory / f"{case[:120]}-{label[:40]}-{uuid4().hex[:12]}.png"
    shot = await tab.page.call("Page.captureScreenshot", format="png", fromSurface=True, captureBeyondViewport=False)
    path.write_bytes(base64.b64decode(shot["data"]))
    return str(path)


async def move_to(tab, point, *, phase):
    await settle_camera(tab)
    diagnostic = await tab.run(
        """
      const point=window.__robotforge.screenOfPoint(TARGET),canvas=document.getElementById('view');
      const finite=Array.isArray(point)&&point.length===2&&point.every(Number.isFinite);
      const inside=finite&&point[0]>=0&&point[1]>=0&&point[0]<innerWidth&&point[1]<innerHeight;
      const hit=inside?document.elementFromPoint(...point):null;
      const rects=Object.fromEntries(['#view','#placement-bar','#object-edit-bar','#world-panel','#drawer','.dock'].map(
        selector=>{const node=document.querySelector(selector),r=node?.getBoundingClientRect();
          return [selector,{hidden:node?.hidden,rect:r?[r.x,r.y,r.width,r.height]:null}];}));
      return {point,finite,inside,canvasHit:hit===canvas,viewport:[innerWidth,innerHeight],
        hit:hit?{tag:hit.tagName,id:hit.id,class:hit.className}:null,rects};
    """.replace("TARGET", json.dumps(point))
    )
    diagnostic["evidence"] = await capture_evidence(tab, phase)
    assert diagnostic["finite"] and diagnostic["inside"] and diagnostic["canvasHit"], json.dumps(diagnostic)
    await click(tab, point=diagnostic["point"])
    await tab.until(
        f"{READY} && document.getElementById('placement-bar').hidden && !document.getElementById('undo-edit').hidden"
    )


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_supported_selection_native_move_copy_save_and_undo(viewer, monkeypatch, size, lang):  # noqa: F811
    if size == "phone":
        monkeypatch.setitem(qa.SIZES, "phone", (320, 740, 2))

    async def scenario(tab, server):
        await ready(tab, server)
        await select_single(tab)
        destination = [0, 1, 0] if size == "phone" else [2, 0, 0]
        expected_xs = [0.0625, -0.1875, 0.3125, -0.1875] if size == "phone" else [2.0625, 1.8125, 2.3125, 1.8125]
        button = await tab.run("""
          const button=document.getElementById('object-select-carried'),r=button.getBoundingClientRect();
          const point=[r.x+r.width/2,r.y+r.height/2],hit=document.elementFromPoint(...point);
          return {rect:[r.x,r.y,r.width,r.height],point,viewport:[innerWidth,innerHeight],
            inside:r.width>0&&r.height>0&&r.left>=0&&r.top>=0&&r.right<=innerWidth&&r.bottom<=innerHeight,
            hit:button.contains(hit),disabled:button.disabled};
        """)
        button["evidence"] = await capture_evidence(tab, "support-selected")
        assert button["inside"] and button["hit"] and not button["disabled"], json.dumps(button)
        assert await tab.eval("document.getElementById('object-select-carried').textContent") == (
            "Include items on top" if lang == "en" else "连同上面的物品选择"
        )
        before = state(server)
        world = server.pg.store.load().model_dump(mode="json")
        # Fresh paused fixture origins match authored positions, including world-body-only fixed bins.
        source_positions = [
            (*entry["pos"][:2], entry["pos"][2] if len(entry["pos"]) == 3 else 0) for entry in world["objects"]
        ]
        camera = await tab.eval(CAMERA)
        await click(tab, "#object-select-carried")
        await tab.until(f"{HIGHLIGHTS}===4 && !window.__robotforge.busy().busy")
        assert state(server) == before
        await settle_camera(tab)
        assert await tab.eval(CAMERA) == pytest.approx(camera, abs=1e-9)
        assert await tab.eval(SINGLE) == 0
        assert await tab.eval("document.getElementById('object-select-carried').hidden")
        wire = await tab.eval("window.__carriedRequests.filter(message=>message.action==='select_object_group')")
        assert len(wire) == 1
        assert {k: v for k, v in wire[0].items() if k != "id"} == {
            "type": "world",
            "action": "select_object_group",
            "name": "base",
            "version": before[1],
            "generation": before[2],
            "world_sha256": await tab.eval("window.__robotforge.scene.object_edit.world_sha256"),
        }
        assert await tab.eval("window.__carriedReplies[0].data.names") == NAMES
        await click(tab, "#world-toggle")
        assert (
            await tab.eval(
                "[...document.querySelectorAll('#edit-object-list [aria-pressed=true]')].map(n=>n.dataset.editObject)"
            )
            == NAMES
        )
        await click(tab, "#world-toggle")

        await click(tab, "#object-group-move")
        await tab.until(f"{READY} && !document.getElementById('placement-bar').hidden")
        await move_to(tab, destination, phase="move")
        moved = server.pg.store.load()
        assert [obj.name for obj in moved.objects] == NAMES
        # The established group anchor averages all four live member origins, including the upper box.
        assert [obj.pos[0] for obj in moved.objects] == pytest.approx(expected_xs, abs=0.002)
        assert sum(obj.pos[0] for obj in moved.objects) / len(moved.objects) == pytest.approx(destination[0], abs=0.002)
        assert [obj.pos[1] for obj in moved.objects] == pytest.approx([destination[1]] * 4, abs=0.002)
        assert [obj.pos[2] for obj in moved.objects] == pytest.approx([0, 0.6, 0.5, 0.75], abs=0.002)
        for obj, source in zip(moved.objects, source_positions, strict=True):
            assert [obj.pos[axis] - moved.objects[0].pos[axis] for axis in range(3)] == pytest.approx(
                [source[axis] - source_positions[0][axis] for axis in range(3)], abs=0.002
            )
        await inverse(tab)
        assert server.pg.store.load().model_dump(mode="json") == world

        await select_carried(tab)
        await click(tab, "#object-group-copy")
        await tab.until(f"{READY} && !document.getElementById('placement-bar').hidden")
        await move_to(tab, destination, phase="copy")
        assert len(server.pg.store.load().objects) == 8
        assert server.pg.store.load().model_dump(mode="json")["objects"][:4] == world["objects"]
        copies = server.pg.store.load().objects[4:]
        assert [obj.pos[0] for obj in copies] == pytest.approx(expected_xs, abs=0.002)
        assert [obj.pos[1] for obj in copies] == pytest.approx([destination[1]] * 4, abs=0.002)
        for obj, source in zip(copies, source_positions, strict=True):
            assert [obj.pos[axis] - copies[0].pos[axis] for axis in range(3)] == pytest.approx(
                [source[axis] - source_positions[0][axis] for axis in range(3)], abs=0.002
            )
        receipt = server.pg._play_edit
        await select_carried(tab)
        await click(tab, "#arrangement-save-open")
        await tab.page.call("Input.insertText", text="Supported workbench")
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        await tab.until(
            "document.getElementById('arrangement-save-form').hidden"
            " && document.querySelectorAll('.arrangement-row').length===1"
        )
        assert server.pg._play_edit is receipt
        await key(tab, "Escape")
        await inverse(tab)
        assert server.pg.store.load().model_dump(mode="json") == world
        assert not tab.errors

    viewer(size, lang, scenario, world=carried_world)


def test_supported_leaf_keeps_single_selection_and_running_action_is_disabled(viewer):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab, server)
        await select_single(tab, "stacked")
        before, camera = state(server), await tab.eval(CAMERA)
        await click(tab, "#object-select-carried")
        await tab.until(
            "!window.__robotforge.busy().busy && document.getElementById('toast').classList.contains('show')"
        )
        assert await tab.eval("document.getElementById('toast').textContent") == "There are no other items on top."
        assert await tab.eval("document.getElementById('object-edit-title').textContent") == "stacked"
        assert await tab.eval(SINGLE) == 1 and await tab.eval(HIGHLIGHTS) == 0
        assert state(server) == before
        assert await tab.eval(CAMERA) == pytest.approx(camera, abs=1e-9)
        await click(tab, "#btn-play")
        await tab.until("document.getElementById('object-select-carried').disabled")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=carried_world)


def test_busy_supported_query_blocks_native_selection_changes(viewer):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab, server)
        await select_single(tab)
        before = state(server)
        await tab.run("window.__holdCarried=true")
        await click(tab, "#object-select-carried")
        await tab.until("!!window.__releaseCarried")
        await click(tab, "#world-toggle")
        for name in ("top", "base"):
            selector = f'[data-edit-object="{name}"]'
            assert await tab.eval(f"document.querySelector({json.dumps(selector)}).disabled")
            await click(tab, selector)
            assert await tab.eval("document.getElementById('object-edit-title').textContent") == "base"
            assert await tab.eval(SINGLE) == 1 and await tab.eval(HIGHLIGHTS) == 0
        await click(tab, "#world-toggle")
        await tab.run("window.__releaseCarried()")
        await tab.until(f"{HIGHLIGHTS}===4 && !window.__robotforge.busy().busy")
        assert state(server) == before
        assert not tab.errors

    viewer("laptop", "en", scenario, world=carried_world)


@pytest.mark.parametrize(
    "fault",
    [
        "duplicates",
        "unknown",
        "too-many",
        "missing-root",
        "wrong-root",
        "wrong-order",
        "wrong-generation",
        "wrong-version",
        "wrong-world",
        "null",
        "error",
    ],
)
def test_invalid_supported_reply_preserves_selection_and_prior_undo(viewer, fault):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab, server)
        # Make an actual edit first so a failed selection must retain a usable inverse.
        await select_carried(tab)
        await click(tab, "#object-group-copy")
        await tab.until(f"{READY} && !document.getElementById('placement-bar').hidden")
        await move_to(tab, [2, 0, 0], phase="invalid-reply-copy")
        await select_single(tab)
        before, camera = state(server), await tab.eval(CAMERA)
        await tab.run(f"window.__carriedFault={json.dumps(fault)}")
        await click(tab, "#object-select-carried")
        await tab.until("!window.__robotforge.busy().busy")
        assert await tab.eval("document.getElementById('object-edit-title').textContent") == "base"
        assert await tab.eval(SINGLE) == 1 and await tab.eval(HIGHLIGHTS) == 0
        assert await tab.eval("document.getElementById('undo-edit').hidden")
        assert state(server) == before
        assert await tab.eval(CAMERA) == pytest.approx(camera, abs=1e-9)
        await key(tab, "Escape")
        await tab.until(
            "!document.getElementById('undo-edit').hidden && !document.getElementById('undo-edit').disabled"
        )
        await inverse(tab)
        assert len(server.pg.store.load().objects) == 4
        assert not tab.errors

    viewer("laptop", "en", scenario, world=carried_world)


@pytest.mark.parametrize("interruption", ["reset", "play", "escape", "pointer", "disconnect"])
def test_held_supported_reply_loses_ownership_after_interruption(viewer, interruption):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab, server)
        await select_single(tab)
        saved = server.pg.store.file.read_bytes()
        await tab.run("window.__holdCarried=true")
        await click(tab, "#object-select-carried")
        await tab.until("!!window.__releaseCarried")
        assert await tab.eval("window.__robotforge.busy().busy")
        if interruption == "reset":
            await click(tab, "#btn-reset")
        elif interruption == "play":
            await click(tab, "#btn-play")
            await tab.until("!document.getElementById('btn-play').classList.contains('is-paused')")
            # Returning to pause must not revive a query made before Play.
            await click(tab, "#btn-play")
            await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
        elif interruption == "escape":
            await key(tab, "Escape")
        elif interruption == "pointer":
            await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([1,0,0])"))
        else:
            await tab.run("window.__carriedSocket.close()")
            await tab.until("!window.__robotforge.busy().busy")
            await asyncio.sleep(0.7)
            await tab.until(READY)
        await tab.run("window.__releaseCarried()")
        await tab.until("!window.__robotforge.busy().busy")
        assert await tab.eval(HIGHLIGHTS) == 0
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert server.pg.store.file.read_bytes() == saved
        assert not tab.errors

    viewer("laptop", "en", scenario, world=carried_world)
