"""Native prop inspection frames current geometry without changing the world or simulation."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, qa, viewer  # noqa: F401
from test_viewer_object_edit_browser import choose
from test_viewer_object_group_select_browser import (
    NAMES,
    carried_world,
    select_carried,
    select_single,
)
from test_viewer_object_group_select_browser import (
    ready as ready_carried,
)
from test_viewer_object_group_select_browser import (
    state as selection_state,
)
from test_viewer_object_surface_browser import apply_surface
from test_viewer_startup_browser import READY
from viewer_camera import settle_camera


def world(pg):
    pg.add_object("route", shape="box", size=[0.8, 0.12, 0.22], pos=[0, 0], fixed=True)
    pg.add_object("falling", shape="sphere", size=[0.18], pos=[0, 1, 1.5])
    pg.add_object("table", shape="table", size=[1.1, 0.6, 0.7], pos=[2, 0], fixed=True)
    pg.add_object("tiny", shape="box", size=[0.015, 0.02, 0.01], pos=[-2, 0], fixed=True)


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_native_focus_frames_fixed_free_compound_and_tiny_props(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await tab.until(READY)
        await asyncio.sleep(1.0)  # A free object falls from its saved spawn in the real simulation.
        await click(tab, "#btn-play")
        await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
        saved = server.pg.store.file.read_bytes()
        with server.pg.session.lock:
            data = server.pg.session.data
            before = [v.copy() for v in (data.qpos, data.qvel, data.ctrl)]
            stamp = data.time
        await tab.eval("window.__robotforge.lookAt([5,-8,4],[0,0,.2])")
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="objects"]')
        await click(tab, "#edit-objects")
        assert await tab.eval("document.getElementById('object-focus').disabled")
        await key(tab, "Escape")
        for name in ("route", "falling", "table", "tiny"):
            await tab.eval("window.__robotforge.lookAt([12,-16,10],[0,0,.2])")
            await choose(tab, name)
            initial = await tab.run("""
              const rf=window.__robotforge,{camera}=rf.three(),T=await import('three');
              const away=camera.position.clone().sub(new T.Vector3().fromArray(rf.cameraTarget()));
              return {distance:away.length(),direction:away.normalize().toArray()};
            """)
            await click(tab, "#object-focus")
            await tab.until("!window.__robotforge.tweening()")
            await asyncio.sleep(0.3)  # The panel's free-view projection also eases.
            framing = await tab.run("""
              const rf=window.__robotforge,{scene,camera}=rf.three(),T=await import('three');
              const highlight=scene.getObjectByName('object-edit-highlight'),box=highlight.box;
              const rect=rf.freeRect(),points=[];camera.updateMatrixWorld();
              for(const x of [box.min.x,box.max.x])for(const y of [box.min.y,box.max.y])
                for(const z of [box.min.z,box.max.z]){
                const p=new T.Vector3(x,y,z).project(camera);
                points.push([(p.x+1)*innerWidth/2,(1-p.y)*innerHeight/2,p.z]);
              }
              const direction=camera.position.clone().sub(new T.Vector3().fromArray(rf.cameraTarget())).normalize();
              return {points,rect,center:box.getCenter(new T.Vector3()).toArray(),target:rf.cameraTarget(),
                direction:direction.toArray(),
                distance:camera.position.distanceTo(new T.Vector3().fromArray(rf.cameraTarget())),
                gl:rf.three().renderer.getContext().getError()};
            """)
            assert framing["target"] == pytest.approx(framing["center"], abs=1e-6)
            assert framing["direction"] == pytest.approx(initial["direction"], abs=1e-5)
            assert framing["distance"] < initial["distance"] * 0.8
            r = framing["rect"]
            for x, y, z in framing["points"]:
                assert r["left"] <= x <= r["right"] and r["top"] <= y <= r["bottom"]
                assert -1 < z < 1
            if name == "falling":
                assert framing["center"][2] < 0.2
            assert framing["gl"] == 0
            await key(tab, "Escape")
        assert server.pg.store.file.read_bytes() == saved
        with server.pg.session.lock:
            current = server.pg.session.data
            assert current is data and server.pg.session.paused
            for actual, expected in zip((current.qpos, current.qvel, current.ctrl), before, strict=True):
                np.testing.assert_array_equal(actual, expected)
            assert current.time == stamp
        assert await tab.eval("window.__qaErrors") == []

    viewer(size, lang, scenario, world=world)


async def manual_group(tab, names):
    if await tab.eval("document.getElementById('world-panel').hidden"):
        await click(tab, "#world-toggle")
    await click(tab, '[data-tab="objects"]')
    await click(tab, "#select-object-group")
    assert await tab.eval("document.getElementById('object-focus').disabled")
    await click(tab, "#world-toggle")
    for name in names:
        await click(tab, f'[data-edit-object="{name}"]')
    await click(tab, "#world-toggle")
    await settle_camera(tab)


async def group_framing(tab):
    return await tab.run("""
      const rf=window.__robotforge,{scene,camera}=rf.three(),T=await import('three');
      const names=[...document.querySelectorAll('#edit-object-list [aria-pressed=true]')]
        .map(node=>node.dataset.editObject);
      const union=new T.Box3(),members=[];
      for(const name of names){
        const entry=rf.scene.object_edit.objects.find(entry=>entry.name===name),box=new T.Box3();
        scene.traverse(node=>{if(node.isMesh&&node.visible&&entry.geoms.includes(node.userData.geom))
          box.expandByObject(node)});
        if(!box.isEmpty()){union.union(box);members.push({name,min:box.min.toArray(),max:box.max.toArray()})}
      }
      const helpers=scene.children.filter(node=>node.name==='object-group-member-highlight'&&node.visible)
        .map(node=>({min:node.box.min.toArray(),max:node.box.max.toArray()}));
      const points=[];camera.updateMatrixWorld();
      if(!union.isEmpty())for(const x of [union.min.x,union.max.x])for(const y of [union.min.y,union.max.y])
        for(const z of [union.min.z,union.max.z]){
          const p=new T.Vector3(x,y,z).project(camera);
          points.push([(p.x+1)*innerWidth/2,(1-p.y)*innerHeight/2,p.z]);
        }
      const target=rf.cameraTarget(),offset=camera.position.clone().sub(new T.Vector3().fromArray(target));
      return {names,members,helpers,points,rect:rf.freeRect(),
        center:union.isEmpty()?null:union.getCenter(new T.Vector3()).toArray(),
        target,direction:offset.clone().normalize().toArray(),distance:offset.length(),
        eye:camera.position.toArray(),near:camera.near,far:camera.far,follow:rf.following(),
        gl:rf.three().renderer.getContext().getError()};
    """)


def assert_group_framed(framing, direction):
    assert framing["target"] == pytest.approx(framing["center"], abs=1e-6)
    assert framing["direction"] == pytest.approx(direction, abs=1e-5)
    assert len(framing["helpers"]) == len(framing["members"]) > 0
    for member in framing["members"]:
        assert {key: member[key] for key in ("min", "max")} in framing["helpers"]
    r = framing["rect"]
    for x, y, z in framing["points"]:
        assert r["left"] <= x <= r["right"] and r["top"] <= y <= r["bottom"]
        assert -1 < z < 1
    assert framing["follow"] is None and framing["gl"] == 0


async def focus_group(tab):
    button = await tab.run("""
      const button=document.getElementById('object-focus'),r=button.getBoundingClientRect();
      const hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
      return {hidden:button.hidden,disabled:button.disabled,text:button.textContent,
        inside:r.width>0&&r.height>0&&r.left>=0&&r.top>=0&&r.right<=innerWidth&&r.bottom<=innerHeight,
        hit:button===hit||button.contains(hit)};
    """)
    assert not button["hidden"] and not button["disabled"] and button["inside"] and button["hit"], button
    assert button["text"] == ("查看" if tab.lang == "zh-CN" else "Focus")
    await click(tab, "#object-focus")
    await settle_camera(tab)
    return await group_framing(tab)


@pytest.mark.parametrize("size,lang,carried", [("laptop", "en", False), ("phone", "zh-CN", True)])
def test_group_focus_frames_live_bounds_and_preserves_selection_and_undo(viewer, monkeypatch, size, lang, carried):  # noqa: F811
    if size == "phone":
        monkeypatch.setitem(qa.SIZES, "phone", (320, 740, 2))

    async def scenario(tab, server):
        if carried:
            await ready_carried(tab, server)
            original = server.pg.store.load().model_dump(mode="json")
            await select_single(tab)
            await click(tab, "#object-surface")
            await apply_surface(tab, "steel")
            await key(tab, "Escape")
            await select_carried(tab)
            names = NAMES
            assert await tab.eval("document.getElementById('undo-edit').hidden")
        else:
            await tab.until(READY)
            await asyncio.sleep(1.0)
            await click(tab, "#btn-play")
            await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
            names = ["route", "falling", "table", "tiny"]
            await manual_group(tab, names)
        await tab.run("window.__robotforge.lookAt([12,-16,10],[0,0,.2])")
        await settle_camera(tab)
        if size == "laptop":
            composition = await tab.run("""
              const focus=document.getElementById('object-focus');
              const measure=()=>{
                const bar=document.getElementById('object-edit-bar').getBoundingClientRect();
                const hint=document.getElementById('object-edit-hint').getBoundingClientRect();
                return {bar:[bar.top,bar.height],hint:[hint.width,hint.height],free:window.__robotforge.freeRect()};
              };
              const current=measure(),hidden=focus.hidden;
              // Removing only the new action reconstructs the existing group's flex composition.
              // Restore it before yielding, so no frame or native click uses this comparison layout.
              try{focus.hidden=true;return {current,legacy:measure()}}finally{focus.hidden=hidden}
            """)
            assert composition["current"] == composition["legacy"]
        initial = await group_framing(tab)
        before = selection_state(server)
        model, data = server.pg.session.model, server.pg.session.data
        receipt = server.pg._play_edit
        undo = await tab.eval(
            "[document.getElementById('undo-edit').hidden,document.getElementById('undo-edit').textContent]"
        )
        await tab.run("""
          window.__focusWorldRequests=[];const send=WebSocket.prototype.send;
          WebSocket.prototype.send=function(raw){const msg=JSON.parse(raw);
            if(msg.type==='world')window.__focusWorldRequests.push(msg);return send.call(this,raw)};
        """)
        framing = await focus_group(tab)
        assert_group_framed(framing, initial["direction"])
        assert framing["names"] == names and framing["members"] == initial["members"]
        assert framing["distance"] < initial["distance"]
        if not carried:
            falling = next(member for member in framing["members"] if member["name"] == "falling")
            assert (falling["min"][2] + falling["max"][2]) / 2 < 0.2
        assert selection_state(server) == before
        assert server.pg.session.model is model and server.pg.session.data is data
        assert server.pg._play_edit is receipt
        assert (
            await tab.eval(
                "[document.getElementById('undo-edit').hidden,document.getElementById('undo-edit').textContent]"
            )
            == undo
        )
        assert await tab.eval("window.__focusWorldRequests") == []
        await key(tab, "Escape")
        if carried:
            await tab.until(
                "!document.getElementById('undo-edit').hidden && !document.getElementById('undo-edit').disabled"
            )
            await click(tab, "#undo-edit")
            await tab.until(READY + " && document.getElementById('undo-edit').hidden")
            assert server.pg.store.load().model_dump(mode="json") == original
        assert not tab.errors

    viewer(size, lang, scenario, world=carried_world if carried else world)


def sixteen_props(pg):
    for index in range(16):
        pg.add_object(
            f"prop_{index}", size=[0.2, 0.3, 0.4], pos=[index % 4 * 3 - 4.5, index // 4 * 3 - 4.5], fixed=True
        )


def test_group_focus_16_members_partial_empty_and_single_member_bounds(viewer):  # noqa: F811
    async def scenario(tab, server):
        await ready_carried(tab, server)
        names = [f"prop_{index}" for index in range(16)]
        await manual_group(tab, names)
        before = selection_state(server)
        await tab.run("window.__robotforge.lookAt([24,-24,18],[0,0,.2])")
        await settle_camera(tab)
        initial = await group_framing(tab)
        framing = await focus_group(tab)
        assert framing["names"] == names and len(framing["members"]) == 16
        assert_group_framed(framing, initial["direction"])
        await tab.run("""
          const rf=window.__robotforge;
          const geoms=new Set(rf.scene.object_edit.objects.filter(entry=>!['prop_0','prop_1'].includes(entry.name))
            .flatMap(entry=>entry.geoms));window.__focusHidden=[];
          rf.three().scene.traverse(node=>{if(node.isMesh&&node.visible&&geoms.has(node.userData.geom)){
            window.__focusHidden.push(node);node.visible=false}});
        """)
        try:
            partial = await focus_group(tab)
            assert partial["names"] == names
            assert [member["name"] for member in partial["members"]] == names[:2]
            assert_group_framed(partial, initial["direction"])
            await tab.run("""
              const rf=window.__robotforge,geoms=new Set(rf.scene.object_edit.objects.flatMap(entry=>entry.geoms));
              rf.three().scene.traverse(node=>{if(node.isMesh&&node.visible&&geoms.has(node.userData.geom)){
                window.__focusHidden.push(node);node.visible=false}});
            """)
            empty_before = await group_framing(tab)
            empty_after = await focus_group(tab)
            assert not empty_after["members"] and not empty_after["helpers"]
            for name in ("eye", "target", "near", "far"):
                assert empty_after[name] == pytest.approx(empty_before[name], abs=1e-9)
            assert empty_after["follow"] == empty_before["follow"]
            assert not await tab.eval("window.__robotforge.tweening()")
        finally:
            await tab.run("for(const node of window.__focusHidden)node.visible=true")
        assert selection_state(server) == before
        await click(tab, "#world-toggle")
        for name in names[1:]:
            await click(tab, f'[data-edit-object="{name}"]')
        await click(tab, "#world-toggle")
        single = await focus_group(tab)
        assert single["names"] == names[:1]
        assert_group_framed(single, initial["direction"])
        await click(tab, "#world-toggle")
        await click(tab, '[data-edit-object="prop_0"]')
        await click(tab, "#world-toggle")
        assert await tab.eval("document.getElementById('object-focus').disabled")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=sixteen_props)
