"""Real native navigation and repeated building, with current-scene/reset boundaries."""

from __future__ import annotations

import asyncio
import json

import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_objects_browser import choose_object

READY = "window.__robotforge.scene?.version === window.__robotforge.renderedVersion"


async def select_material(tab, value):
    # macOS desktop Chrome uses a native popup for arrow keys; closed-select typeahead
    # produces trusted input/change without depending on that OS popup's key handling.
    await tab.run("""const e=document.getElementById('place-material');e.focus();
      window.__materialChanges=[];window.__materialListener=event=>
        window.__materialChanges.push({trusted:event.isTrusted,value:e.value});
      e.addEventListener('change',window.__materialListener);""")
    if tab.size == "phone":
        steps = ["", "plastic", "wood", "steel"].index(value)
        for key_name, code in [("Home", 36), *[("ArrowDown", 40)] * steps]:
            for event in ("keyDown", "keyUp"):
                await tab.page.call("Input.dispatchKeyEvent", type=event, key=key_name, windowsVirtualKeyCode=code)
    else:
        # Chromium accumulates printable keys for one second; start a new choice after that window.
        await asyncio.sleep(1.05)
        letter = await tab.eval(f"document.querySelector('#place-material option[value={value}]').textContent[0]")
        for event in ("keyDown", "keyUp"):
            await tab.page.call(
                "Input.dispatchKeyEvent", type=event, key=letter, text=letter if event == "keyDown" else ""
            )
    assert await tab.eval("document.getElementById('place-material').value") == value
    changes = await tab.eval("window.__materialChanges")
    await tab.run("document.getElementById('place-material').removeEventListener('change',window.__materialListener)")
    assert changes and all(change["trusted"] for change in changes)
    assert changes[-1]["value"] == value


async def place_at(tab, point):
    before = await tab.eval("window.__robotforge.scene.version")
    projected = await tab.eval(f"window.__robotforge.screenOfPoint({json.dumps(point)})")
    assert await tab.eval(f"document.elementFromPoint({projected[0]},{projected[1]})?.id") == "view"
    await click(tab, point=projected)
    await tab.until(f"{READY} && window.__robotforge.scene.version > {before} && !window.__robotforge.busy().busy")


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("laptop", "zh-CN"), ("phone", "en"), ("phone", "zh-CN")])
def test_repeat_props_preserves_parameters_requires_new_input_and_undoes_only_latest(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await tab.eval("window.__robotforge.lookAt([4,-8,6],[0,0,0])")
        await tab.run("""window.__sent=[];const old=WebSocket.prototype.send;WebSocket.prototype.send=function(s){
          window.__sent.push(JSON.parse(s));return old.call(this,s)};""")
        await choose_object(tab, "pin")
        assert not await tab.eval("document.getElementById('place-repeat').checked")
        assert not await tab.eval("document.getElementById('prop-material-control').hidden")
        assert await tab.eval("[...document.getElementById('place-material').options].map(o=>o.textContent)") == (
            ["Default", "Plastic", "Wood", "Steel"] if lang == "en" else ["默认", "塑料", "木材", "钢"]
        )
        for material in ("plastic", "wood", "steel"):
            await select_material(tab, material)
        await click(tab, '[data-size="large"]')
        await click(tab, "#place-rotate")
        await click(tab, "#place-repeat")
        assert await tab.eval("document.getElementById('place-cancel').textContent") == (
            "Done" if lang == "en" else "完成"
        )
        assert await tab.eval("""(()=>{const r=document.getElementById('placement-bar').getBoundingClientRect(),
          d=document.querySelector('.dock').getBoundingClientRect();
          return r.left>=0&&r.right<=innerWidth&&r.top>0&&r.bottom<=d.top;})()""")
        for i, x in enumerate([-1.2, 0, 1.2], 1):
            await asyncio.sleep(0.55)  # fresh intentional clicks, outside the double-click guard
            await place_at(tab, [x, 0, 0])
            assert len(server.pg.store.load().objects) == i
            assert not await tab.eval("document.getElementById('placement-bar').hidden")
            assert await tab.eval("document.getElementById('place-here').disabled")
            assert not await tab.eval("!!window.__robotforge.three().scene.getObjectByName('placement-size-preview')")
            assert await tab.eval("document.querySelector('[data-size=large]').getAttribute('aria-pressed')") == "true"
            assert await tab.eval("document.getElementById('place-material').value") == "steel"
            # Held Enter and Enter without a new raycast cannot reuse the previous position.
            await tab.page.call("Input.dispatchKeyEvent", type="keyDown", key="Enter", code="Enter", autoRepeat=True)
            await key(tab, "Enter", kind="keyUp")
            await key(tab, "Enter")
            await key(tab, "Enter", kind="keyUp")
            assert len(server.pg.store.load().objects) == i
        objects = server.pg.store.load().objects
        assert all(o.material == "steel" and o.yaw == 90 and o.size == pytest.approx([0.15, 0.36]) for o in objects)
        assert await tab.eval("window.__sent.filter(m=>m.action==='add_object').every(m=>m.material==='steel')")
        # Escape works even when the new native checkbox owns keyboard focus.
        await tab.eval("document.getElementById('place-repeat').focus()")
        await key(tab, "Escape")
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        await click(tab, "#undo-add")
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('undo-add').hidden")
        assert len(server.pg.store.load().objects) == 2
        await choose_object(tab, "table")
        assert await tab.eval("document.getElementById('prop-material-control').hidden")
        await key(tab, "Escape")
        await choose_object(tab, "pin")
        assert await tab.eval("document.getElementById('place-material').value") == ""
        await place_at(tab, [-1.2, 1.2, 0])
        assert server.pg.store.load().objects[-1].material == "plastic"
        assert await tab.eval("!('material' in window.__sent.filter(m=>m.action==='add_object').at(-1))")
        assert not tab.errors

    viewer(size, lang, scenario, world=lambda pg: None)


@pytest.mark.parametrize("size", ["laptop", "phone"])
def test_scene_tabs_share_one_row_native_keyboard_does_not_drive_and_build_opens_robot_picker(viewer, size):  # noqa: F811
    async def scenario(tab, _server):
        await tab.run("localStorage.setItem('robotforge.tab','removed-tab')")
        await tab.navigate()
        await tab.until(READY)
        await click(tab, "#world-toggle")
        tabs = await tab.eval("""[...document.querySelectorAll('#world-panel [role=tab]')].map(e=>{
          const r=e.getBoundingClientRect();
          return {top:r.top,right:r.right,left:r.left,controls:e.getAttribute('aria-controls')};})""")
        assert max(t["top"] for t in tabs) - min(t["top"] for t in tabs) < 1
        width = await tab.eval("innerWidth")
        assert all(0 <= t["left"] < t["right"] <= width for t in tabs)
        await click(tab, '[data-tab="maps"]')
        await tab.run("""window.__sent=[];const old=WebSocket.prototype.send;WebSocket.prototype.send=function(s){
          window.__sent.push(JSON.parse(s));return old.call(this,s)};""")
        for value, expected in [
            ("ArrowRight", "terrain"),
            ("End", "modes"),
            ("ArrowRight", "maps"),
            ("ArrowLeft", "modes"),
            ("Home", "maps"),
        ]:
            await key(tab, value)
            await key(tab, value, kind="keyUp")
            assert await tab.eval("document.activeElement.dataset.tab") == expected
            assert await tab.eval("document.querySelector('#world-panel [aria-selected=true]').dataset.tab") == expected
            assert await tab.eval("document.querySelectorAll('#world-panel [role=tab][tabindex=\"0\"]').length") == 1
        assert not await tab.eval("window.__sent.some(m=>m.type==='teleop')")
        await click(tab, '[data-tab="objects"]')
        await click(tab, "#build-add-robot")
        assert await tab.eval("document.getElementById('world-panel').hidden")
        assert not await tab.eval("document.getElementById('create-options').hidden")
        assert await tab.eval("document.querySelectorAll('#create-list .create-card').length") == 4
        await click(tab, '.create-card[data-template="rover"]')
        assert await tab.eval("document.getElementById('object-placement-controls').hidden")
        assert await tab.eval("document.getElementById('prop-material-control').hidden")
        await key(tab, "Escape")
        await click(tab, "#world-toggle")
        await key(tab, "Escape")
        assert await tab.eval("document.activeElement.id") == "world-toggle"
        assert not tab.errors

    viewer(size, "en", scenario, world=lambda pg: pg.add_robot("rover", template="rover", pos=[-3, 0]))


def test_repeat_idle_external_zero_time_reset_and_edit_cancel_without_resubmitting(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.reset()
        server.pg.session.paused = True
        await tab.until(
            "window.__robotforge.time === 0 && document.getElementById('btn-play').classList.contains('is-paused')"
        )
        await choose_object(tab, "ball")
        await click(tab, "#place-repeat")
        assert server.pg.session.data.time == 0
        before = await tab.eval("window.__robotforge.scene.version")
        server.pg.session.reset()
        await tab.until("document.getElementById('placement-bar').hidden")
        assert await tab.eval("window.__robotforge.scene.version") == before
        assert not server.pg.store.load().objects
        await choose_object(tab, "ball")
        await click(tab, "#place-repeat")
        server.pg.add_object("external", shape="sphere", size=[0.1], pos=[2, 0])
        await tab.until("document.getElementById('placement-bar').hidden")
        assert [o.name for o in server.pg.store.load().objects] == ["external"]
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)


WIRE = """
window.__flow={holdState:false,holdReply:false,replies:[],heldMessages:[],generation:null};
const flow=window.__flow,Native=window.WebSocket;
window.WebSocket=class extends Native{
  constructor(...args){super(...args);flow.socket=this}
  set onmessage(handler){super.onmessage=event=>{
    if(typeof event.data==='string'){
      const msg=JSON.parse(event.data);
      if(msg.type==='state'){
        flow.generation=msg.generation;
        if(flow.holdState){flow.releaseState=()=>handler(event);return}
      }
      if(msg.type==='reply'&&flow.holdReply){flow.heldMessages.push(msg);flow.replies.push(()=>handler(event));return}
    }
    handler(event);
  }}
};
flow.releaseReplies=()=>{flow.holdReply=false;for(const send of flow.replies.splice(0))send()};
"""


async def instrument_wire(tab, flags=""):
    await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=WIRE + flags)
    await tab.page.call("Page.reload")
    await tab.until(f"!!window.__flow && !!window.__robotforge && {READY}")
    await tab.eval("window.__robotforge.lookAt([4,-8,6],[0,0,0])")


def test_repeat_waits_for_initial_generation_and_does_not_assume_legacy_reply_is_current(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await instrument_wire(tab, "flow.holdState=true;")
        await choose_object(tab, "ball")
        await click(tab, "#place-repeat")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([1,0,0])"))
        assert not server.pg.store.load().objects
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        await tab.run("window.__flow.holdState=false;window.__flow.releaseState()")
        await place_at(tab, [1, 0, 0])
        assert len(server.pg.store.load().objects) == 1
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)


def test_reset_after_commit_before_delayed_reply_cannot_rearm_repeat(viewer):  # noqa: F811
    async def scenario(tab, server):
        await instrument_wire(tab)
        await choose_object(tab, "ball")
        await select_material(tab, "wood")
        await click(tab, "#place-repeat")
        await tab.run("window.__flow.holdReply=true")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([1,0,0])"))
        await tab.until("window.__flow.replies.length === 1")
        assert len(server.pg.store.load().objects) == 1
        assert await tab.eval("window.__robotforge.busy().busy")
        assert await tab.eval("document.getElementById('place-material').disabled")
        version = server.pg.session.version
        server.pg.session.reset()
        server.pg.session.paused = True
        generation = server.pg.session.generation
        await tab.until(f"window.__flow.generation === {generation}")
        assert server.pg.session.version == version
        await tab.run("window.__flow.releaseReplies()")
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('undo-add').hidden")
        assert len(server.pg.store.load().objects) == 1
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)


@pytest.mark.parametrize("action", ["local_reset", "disconnect", "lesson"])
def test_repeat_exits_on_reset_disconnect_or_lesson_without_adding(viewer, action):  # noqa: F811
    async def scenario(tab, server):
        await instrument_wire(tab)
        await choose_object(tab, "block")
        await click(tab, "#place-repeat")
        store = server.pg.store
        if action == "local_reset":
            await click(tab, "#btn-reset")
        elif action == "disconnect":
            await tab.run("window.__flow.socket.close()")
        else:
            server.pg.lessons.start("pid_twins")
        await tab.until("document.getElementById('placement-bar').hidden")
        assert not store.load().objects
        assert not tab.errors

    viewer("phone", "zh-CN", scenario, world=lambda pg: None)


def test_repeat_occupied_spot_keeps_parameters_and_accepts_new_explicit_location(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.eval("window.__robotforge.lookAt([4,-8,6],[0,0,0])")
        await choose_object(tab, "barrier")
        await select_material(tab, "plastic")
        await click(tab, '[data-size="large"]')
        await click(tab, "#place-rotate")
        await click(tab, "#place-repeat")
        await place_at(tab, [-2, 0, 0])
        await asyncio.sleep(0.55)
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([-1.9,0,0])"))
        await tab.until(
            "!window.__robotforge.busy().busy && document.getElementById('toast').textContent.includes('occupied')"
        )
        assert len(server.pg.store.load().objects) == 1
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('place-material').value") == "plastic"
        assert not await tab.eval("document.getElementById('place-material').disabled")
        await place_at(tab, [1, 0, 0])
        assert len(server.pg.store.load().objects) == 2
        assert all(o.material == "plastic" for o in server.pg.store.load().objects)
        assert all(o.yaw == 90 and o.size == pytest.approx([1.2, 0.18, 0.33]) for o in server.pg.store.load().objects)
        await click(tab, "#place-cancel")
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)


def test_plain_model_build_links_to_maps_and_enables_creation_after_real_map_load(viewer):  # noqa: F811
    async def scenario(tab, server):
        before = await tab.eval("window.__robotforge.scene.version")
        model = server.pg.store.dir / "plain-model.xml"
        model.write_text('<mujoco><worldbody><geom type="plane" size="3 3 .1"/></worldbody></mujoco>', encoding="utf-8")
        server.pg.open_model(str(model))
        await tab.until(f"window.__robotforge.scene.version > {before} && !window.__robotforge.scene.world && {READY}")
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="objects"]')
        assert not await tab.eval("document.getElementById('build-pick-map').hidden")
        assert await tab.eval("document.getElementById('build-add-robot').hidden")
        await click(tab, "#build-pick-map")
        assert await tab.eval("document.activeElement.dataset.tab") == "maps"
        await click(tab, "#map-list button:first-child")
        await tab.until(f"!!window.__robotforge.scene.world && {READY} && !window.__robotforge.busy().busy")
        await click(tab, '[data-tab="objects"]')
        assert await tab.eval("document.getElementById('build-pick-map').hidden")
        assert not await tab.eval("document.getElementById('build-add-robot').hidden")
        assert await tab.eval("document.querySelectorAll('.object-card').length") == 5
        assert not tab.errors

    viewer("phone", "zh-CN", scenario, world=lambda pg: None)
