"""One picker session places independent robots without moving the composition."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from test_viewer_creation_browser import choose, click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_placement_browser import spacing
from test_viewer_robot_surface_browser import RESET_WIRE
from test_viewer_startup_browser import READY
from test_viewer_terrain_create_browser import tap
from viewer_camera import settle_camera

from robotforge.design import combine
from robotforge.design.world import RobotEntry
from robotforge.playground import PlaygroundError
from robotforge.viewer.creation import undo_addition

CAMERA = """(()=>{const r=window.__robotforge,c=r.three().camera;
return [...c.position.toArray(),...r.cameraTarget(),...c.projectionMatrix.elements]})()"""


def world(pg):
    pg.add_robot("existing", template="rover", pos=[-3, -3])
    pg.add_object("obstacle", shape="box", size=[0.6, 0.6, 0.7], pos=[0, 1], fixed=True)
    combine.save(RobotEntry(name="source", template="arm", params={"color": "red"}), "local_arm", pg.store.dir)


async def begin(tab, kind):
    if kind == "template":
        await choose(tab, "rover")
    else:
        if await tab.eval("document.getElementById('drawer').hidden"):
            await click(tab, "#drawer-toggle")
        if await tab.eval("document.getElementById('create-options').hidden"):
            await click(tab, "#create-toggle")
        name, source = ("go2", "library") if kind == "library" else ("local_arm", "design")
        await click(tab, f'#model-list [data-model="{name}"][data-source="{source}"]')
    await tab.until("!document.getElementById('placement-bar').hidden")
    assert await tab.eval("document.getElementById('place-repeat').getClientRects().length>0")
    assert not await tab.eval("document.getElementById('place-repeat').checked")
    assert await tab.eval("document.getElementById('object-placement-controls').hidden")
    await click(tab, "#place-repeat")
    await tab.run("window.__robotforge.lookAt([9,-15,14],[0,1,.2])")
    await settle_camera(tab)


async def prepare(tab, server):
    server.pg.session.paused = True
    await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
    await tab.run("""window.__repeatRobots=[];const send=WebSocket.prototype.send;
      WebSocket.prototype.send=function(s){const m=JSON.parse(s);
        if(['add_robot','add_model'].includes(m.action))window.__repeatRobots.push(m);return send.call(this,s)};""")


async def place(tab, server, position):
    version = server.pg.session.version
    await asyncio.sleep(0.55)  # existing post-create click suppression
    await tap(tab, position)
    await tab.until(READY + f"&&window.__robotforge.scene.version>{version}&&!window.__robotforge.busy().busy")
    assert not await tab.eval("document.getElementById('placement-bar').hidden")
    assert await tab.eval("document.getElementById('place-here').disabled")
    assert await tab.eval("document.getElementById('place-repeat').checked")


@pytest.mark.parametrize("kind", ["template", "library", "design"])
@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_continuous_robots_keep_composition_fresh_guards_latest_undo_and_reload(viewer, kind, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        await begin(tab, kind)
        await spacing(tab, "0.25")
        await settle_camera(tab)
        camera = await tab.eval(CAMERA)
        original = server.pg.current_world().robots[0]
        for position in [[2, -1, 0], [2, 2, 0]]:
            version, generation = server.pg.session.version, server.pg.session.generation
            await place(tab, server, position)
            np.testing.assert_allclose(await tab.eval(CAMERA), camera, rtol=0, atol=1e-9)
            sent = await tab.eval("window.__repeatRobots.at(-1)")
            assert (sent["version"], sent["generation"]) == (version, generation)
            assert await tab.eval("document.getElementById('place-spacing').value") == "0.25"
            assert await tab.eval("window.__robotforge.selected()") == server.pg.current_world().robots[-1].name
            before = len(await tab.eval("window.__repeatRobots"))
            await key(tab, "Enter")
            await key(tab, "Enter", kind="keyUp")
            assert len(await tab.eval("window.__repeatRobots")) == before
        entries = server.pg.current_world().robots
        assert entries[0] == original
        assert [r.pos for r in entries[1:]] == [[2, -1], [2, 2]]
        assert len({r.name for r in entries}) == 3
        if kind == "design":
            assert all(r.params == {"color": "red"} for r in entries[1:])
        await click(tab, "#place-cancel")
        await click(tab, "#undo-add")
        await tab.until(READY + "&&!window.__robotforge.busy().busy")
        assert server.pg.current_world().robots == entries[:2]
        saved = server.pg.current_world().model_dump(mode="json")
        assert not await tab.eval("window.__qaErrors")
        assert await tab.eval("window.__robotforge.three().renderer.getContext().getError()") == 0
        origin = await tab.eval("performance.timeOrigin")
        await tab.page.call("Page.reload")
        await tab.until(f"performance.timeOrigin!=={origin}&&({READY})")
        assert server.pg.current_world().model_dump(mode="json") == saved
        await begin(tab, kind)  # a new session starts opt-in off, then helper explicitly enables it
        await key(tab, "Escape")
        assert not await tab.eval("window.__qaErrors")
        assert await tab.eval("window.__robotforge.three().renderer.getContext().getError()") == 0

    viewer(size, lang, scenario, world=world)


@pytest.mark.parametrize("kind", ["template", "library", "design"])
def test_occupied_repeat_retries_new_point_without_losing_last_success(viewer, kind):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        await begin(tab, kind)
        await place(tab, server, [2, -1, 0])
        saved, receipt = server.pg.store.file.read_bytes(), server.pg._play_addition
        state = server.pg.session.version, server.pg.session.generation
        await asyncio.sleep(0.55)
        await tap(tab, [0.35, 1, 0])  # clear ground center; robot footprint intersects the fixed box
        await tab.until(
            "!window.__robotforge.busy().busy&&document.getElementById('toast').textContent.includes('occupied')"
        )
        assert server.pg.store.file.read_bytes() == saved and server.pg._play_addition is receipt
        assert (server.pg.session.version, server.pg.session.generation) == state
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        await place(tab, server, [2, 2, 0])
        assert len(server.pg.current_world().robots) == 3
        assert len(await tab.eval("window.__repeatRobots")) == 3
        assert not await tab.eval("window.__qaErrors")

    viewer("laptop", "en", scenario, world=world)


@pytest.mark.parametrize("state_first", [True, False], ids=["state-first", "display-first"])
def test_reset_during_robot_repeat_handles_both_real_state_orders(viewer, state_first):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=RESET_WIRE)
        await tab.navigate()
        await prepare(tab, server)
        await begin(tab, "template")
        await tab.run("""const old=window.fetch;window.__heldRepeat=false;
          window.fetch=async(...args)=>{const r=await old(...args);
            if(String(args[0]).includes('mesh/')){window.__heldRepeat=true;
              await new Promise(resolve=>window.__releaseRepeat=resolve)}return r};""")
        await tap(tab, [2, -1, 0])
        await tab.until("window.__heldRepeat")
        receipt = server.pg._play_addition
        await tab.until(f"window.__partReset.generation==={receipt.generation}")
        generation = receipt.generation + 1
        await tab.run(f"window.__partReset.holdGeneration={generation}")
        server.pg.session.reset()
        await tab.until("window.__partReset.states.length>0")
        if state_first:
            await tab.run("window.__releasePartReset()")
            await tab.until(f"window.__partReset.generation==={generation}")
        await tab.run("window.__releaseRepeat()")
        await tab.until(READY + "&&!window.__robotforge.busy().busy")
        if not state_first:
            assert not await tab.eval("document.getElementById('placement-bar').hidden")
            await tab.run("window.__releasePartReset()")
            await tab.until(f"window.__partReset.generation==={generation}")
        await tab.until("document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('undo-add').hidden")
        assert len(await tab.eval("window.__repeatRobots")) == 1
        assert len(server.pg.current_world().robots) == 2
        with pytest.raises(PlaygroundError, match="simulation changed"):
            undo_addition(server.pg, {"token": receipt.token, "version": receipt.version})
        assert not await tab.eval("window.__qaErrors")

    viewer("laptop", "en", scenario, world=world)


MISSING_GENERATION = """
const Native=window.WebSocket;window.__dropRepeatGeneration=false;
window.WebSocket=class extends Native{
 set onmessage(handler){super.onmessage=event=>{
  if(typeof event.data==='string'){
   const m=JSON.parse(event.data);
   if(window.__dropRepeatGeneration&&m.type==='reply'&&m.ok&&m.data?.robot){
    const data={...m.data};delete data.generation;
    handler(new MessageEvent('message',{data:JSON.stringify({...m,data})}));return;
   }
  }
  handler(event);
 }}
};
"""


@pytest.mark.parametrize("kind", ["template", "design"])
def test_missing_repeat_generation_cannot_reuse_previous_undo(viewer, kind):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=MISSING_GENERATION)
        await tab.navigate()
        await prepare(tab, server)
        await begin(tab, kind)
        await place(tab, server, [2, -1, 0])
        old_receipt = server.pg._play_addition
        await click(tab, "#place-cancel")
        assert not await tab.eval("document.getElementById('undo-add').hidden")
        await begin(tab, kind)
        await tab.run("window.__dropRepeatGeneration=true")
        version = server.pg.session.version
        await tap(tab, [2, 2, 0])
        await tab.until(
            READY + f"&&window.__robotforge.scene.version>{version}"
            "&&document.getElementById('placement-bar').hidden&&!window.__robotforge.busy().busy"
        )
        assert len(server.pg.current_world().robots) == 3
        assert server.pg._play_addition.token != old_receipt.token
        assert await tab.eval("document.getElementById('undo-add').hidden")
        assert len(await tab.eval("window.__repeatRobots")) == 2
        assert not await tab.eval("window.__qaErrors")

    viewer("laptop", "en", scenario, world=world)


@pytest.mark.parametrize("kind", ["template", "design"])
def test_committed_repeat_display_timeout_keeps_valid_undo_without_rearming(viewer, kind):  # noqa: F811
    async def scenario(tab, server):
        await tab.until("window.__robotforge.time>0.2")
        await prepare(tab, server)
        await begin(tab, kind)
        saved = server.pg.current_world().model_dump(mode="json")
        await tab.run("""const old=window.fetch;window.__heldRepeat=false;
          window.fetch=async(...args)=>{const r=await old(...args);
            if(String(args[0]).includes('mesh/')){window.fetch=old;window.__heldRepeat=true;
              await new Promise(resolve=>window.__releaseRepeat=resolve)}return r};""")
        await tap(tab, [2, -1, 0])
        await tab.until("window.__heldRepeat")
        receipt = server.pg._play_addition
        await tab.until("!window.__robotforge.busy().busy&&document.getElementById('placement-bar').hidden", timeout=25)
        assert len(server.pg.current_world().robots) == 2
        assert receipt.generation == server.pg.session.generation
        assert not await tab.eval("document.getElementById('undo-add').hidden")
        assert len(await tab.eval("window.__repeatRobots")) == 1
        await tab.run("window.__releaseRepeat()")
        await tab.until(READY)
        await click(tab, "#undo-add")
        await tab.until(READY + "&&!window.__robotforge.busy().busy")
        assert server.pg.current_world().model_dump(mode="json") == saved
        assert not await tab.eval("window.__qaErrors")

    viewer("laptop", "en", scenario, world=world)


@pytest.mark.parametrize("event", ["map", "disconnect"])
def test_repeat_stops_when_world_or_connection_changes(viewer, event):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        await tab.run("""const send=WebSocket.prototype.send;
          WebSocket.prototype.send=function(s){window.__repeatSocket=this;return send.call(this,s)};""")
        await begin(tab, "template")
        await place(tab, server, [2, -1, 0])
        if event == "map":
            server.pg.new_world("another_world")
            version = server.pg.session.version
            await tab.until(READY + f"&&window.__robotforge.scene.version==={version}")
            assert not server.pg.current_world().robots
        else:
            await tab.run("window.__repeatSocket.close()")
        await tab.until("document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('undo-add').hidden")
        assert len(await tab.eval("window.__repeatRobots")) == 1
        assert not await tab.eval("window.__qaErrors")

    viewer("laptop", "en", scenario, world=world)


def test_same_version_time_rewind_still_invalidates_undo_before_reset_state(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=RESET_WIRE)
        await tab.navigate()
        await prepare(tab, server)
        await begin(tab, "template")
        await place(tab, server, [2, -1, 0])
        await click(tab, "#place-cancel")
        assert not await tab.eval("document.getElementById('undo-add').hidden")
        await click(tab, "#btn-play")
        await tab.until("window.__robotforge.time>0.2")
        receipt = server.pg._play_addition
        await tab.until(f"window.__partReset.generation==={receipt.generation}")
        generation = receipt.generation + 1
        await tab.run(f"window.__partReset.holdGeneration={generation}")
        version = server.pg.session.version
        server.pg.session.paused = True
        server.pg.session.reset()
        await tab.until("window.__partReset.states.length>0")
        await tab.until("window.__robotforge.time===0&&document.getElementById('undo-add').hidden")
        assert await tab.eval("window.__partReset.generation") == receipt.generation
        assert server.pg.session.version == version
        await tab.run("window.__releasePartReset()")
        with pytest.raises(PlaygroundError, match="simulation changed"):
            undo_addition(server.pg, {"token": receipt.token, "version": receipt.version})
        assert not await tab.eval("window.__qaErrors")

    viewer("laptop", "en", scenario, world=world)
