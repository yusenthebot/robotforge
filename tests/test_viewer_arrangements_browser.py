"""Native persistent prop shelf, owned outlines and grouped insertion completion."""

from __future__ import annotations

import asyncio

import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_object_group_browser import select_group, table_layout
from viewer_camera import settle_camera

READY = "window.__robotforge.scene?.version === window.__robotforge.renderedVersion && !window.__robotforge.busy().busy"
INSERT = '[data-arrangement-action="insert"]'
REMOVE = '[data-arrangement-action="remove"]'
OUTLINE = "window.__robotforge.three().scene.getObjectByName('placement-size-preview')"


async def trace_arrangement_requests(tab):
    await tab.run("""
      window.__arrangementWire={requests:[],replies:[]};
      const ids=new Map(), sockets=new WeakSet(), send=WebSocket.prototype.send;
      WebSocket.prototype.send=function(raw){
        const msg=JSON.parse(raw);
        if(msg.action?.startsWith('arrangement_')){
          ids.set(msg.id,msg.action);window.__arrangementWire.requests.push(msg);
          window.__arrangementWire.requests=window.__arrangementWire.requests.slice(-8);
        }
        if(!sockets.has(this)){
          sockets.add(this);this.addEventListener('message',event=>{
            if(typeof event.data!=='string')return;
            const reply=JSON.parse(event.data);
            if(reply.type==='reply'&&ids.has(reply.id)){
              window.__arrangementWire.replies.push({action:ids.get(reply.id),...reply});
              window.__arrangementWire.replies=window.__arrangementWire.replies.slice(-8);
            }
          });
        }
        return send.call(this,raw);
      };
    """)


async def placement_diagnostics(tab):
    return await tab.run("""
      const rf=window.__robotforge, by=id=>document.getElementById(id);
      return {wire:window.__arrangementWire,toast:by('toast').textContent,busy:rf.busy(),
        version:rf.scene?.version,renderedVersion:rf.renderedVersion,
        placementHidden:by('placement-bar').hidden,placeDisabled:by('place-here').disabled,
        point:by('placement-position').textContent,status:by('placement-coordinate-status').textContent,
        heading:by('placement-heading').textContent,outline:!!rf.three().scene.getObjectByName('placement-size-preview')};
    """)


async def pause(tab, server):
    with server.pg.session.lock:
        server.pg.session.paused = True
    await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
    await tab.until(READY)


async def open_shelf(tab, *, wait_ready=True):
    if await tab.eval("document.getElementById('world-panel').hidden"):
        await click(tab, "#world-toggle")
    await click(tab, '[data-tab="objects"]')
    if wait_ready:
        await tab.until("!document.getElementById('arrangement-refresh').disabled")


async def reload(tab):
    origin = await tab.eval("performance.timeOrigin")
    await tab.page.call("Page.reload")
    await tab.until(f"performance.timeOrigin !== {origin} && !!window.__robotforge?.scene")
    await tab.until(READY)


async def save_group(tab, name="Workbench <shared>"):
    await select_group(tab)
    await click(tab, "#arrangement-save-open")
    assert await tab.eval("document.activeElement.id") == "arrangement-name"
    await tab.page.call("Input.insertText", text=name)
    await key(tab, "Enter")
    await key(tab, "Enter", kind="keyUp")
    await tab.until(
        "document.getElementById('arrangement-save-form').hidden"
        " && document.querySelectorAll('.arrangement-row').length===1"
    )
    assert await tab.eval("document.querySelector('.arrangement-row strong').textContent") == name
    assert await tab.eval("document.querySelector('.arrangement-row strong').children.length") == 0
    return await tab.eval("document.querySelector('.arrangement-row').dataset.arrangement")


async def prepare(tab):
    await open_shelf(tab)
    await tab.until(
        "!!document.querySelector('[data-arrangement-action=insert]')"
        " && !document.querySelector('[data-arrangement-action=insert]').disabled"
    )
    await click(tab, INSERT)
    await tab.until(f"!document.getElementById('placement-bar').hidden && {OUTLINE}?.visible && {READY}")
    await settle_camera(tab)


async def place(tab, point):
    await settle_camera(tab)
    screen = await tab.eval(f"window.__robotforge.screenOfPoint({point})")
    width, height = await tab.eval("[innerWidth, innerHeight]")
    assert 0 <= screen[0] < width and 0 <= screen[1] < height, (point, screen, width, height)
    await click(tab, point=screen)
    try:
        await tab.until(
            f"{READY} && document.getElementById('placement-bar').hidden"
            " && !document.getElementById('undo-edit').hidden",
            timeout=20,
        )
    except TimeoutError as exc:
        raise AssertionError(await placement_diagnostics(tab)) from exc


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_native_save_reload_other_world_occupied_retry_latest_group_undo(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        if size == "phone":
            await tab.page.call(
                "Emulation.setDeviceMetricsOverride", width=320, height=740, deviceScaleFactor=2, mobile=True
            )
        await pause(tab, server)
        await tab.eval("window.__robotforge.lookAt([6,-8,6],[0,0,.2])")
        source = server.pg.current_world().model_dump(mode="json")
        version = server.pg.session.version
        name = "工作台 <共享>" if lang == "zh-CN" else "Workbench <shared>"
        identity = await save_group(tab, name)
        assert server.pg.session.version == version
        assert server.pg.current_world().model_dump(mode="json") == source
        await reload(tab)
        await open_shelf(tab)
        await tab.until("document.querySelectorAll('.arrangement-row').length===1")
        assert await tab.eval("document.querySelector('.arrangement-row').dataset.arrangement") == identity
        # A native map change removes the sources; the shelf remains global and visible with zero props.
        await click(tab, '[data-tab="maps"]')
        await click(tab, "#map-list .choice:first-child")
        await tab.until(READY)
        await open_shelf(tab)
        await tab.until("document.querySelectorAll('.arrangement-row').length===1")
        assert not server.pg.current_world().objects
        assert await tab.eval("!document.getElementById('arrangement-shelf').hidden")
        server.pg.add_robot("keep_rover", template="rover", pos=[0, -3])
        server.pg.add_object(
            "keep_steel", shape="box", size=[0.2, 0.2, 0.45], pos=[2.55, 0.3], fixed=True, material="steel"
        )
        server.pg.add_terrain("ramp", pos=[-3, 3])
        await pause(tab, server)
        await tab.until(f"window.__robotforge.scene.version==={server.pg.session.version} && {READY}")
        destination = server.pg.current_world().model_dump(mode="json")
        await tab.eval("window.__robotforge.lookAt([6,-8,6],[0,0,.2])")
        await prepare(tab)
        assert await tab.eval("document.querySelector('.repeat-placement').hidden")
        assert await tab.eval(
            "document.getElementById('placement-hint').textContent.includes('轮廓')"
            if lang == "zh-CN"
            else "document.getElementById('placement-hint').textContent.includes('Outline')"
        )
        outline = await tab.run(f"""
          const ghost={OUTLINE};
          return {{count:ghost.children.length,owned:ghost.children.every(part=>part.userData.previewOwnGeometry),
            shapes:ghost.children.map(part=>part.geometry.type)}};
        """)
        assert outline["count"] == 3 and outline["owned"]
        await click(tab, "#place-rotate")
        assert (await tab.eval("document.getElementById('placement-heading').textContent")).endswith("90°")
        # The floor anchor is clear; the blocker intersects the actual leg at [2.55, .35].
        # Rotate back to zero so this exercises a real collision rather than the empty underside.
        for _ in range(3):
            await click(tab, "#place-rotate")
        await trace_arrangement_requests(tab)
        failed_version = server.pg.session.version
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([2,0,0])"))
        try:
            await tab.until(
                f"{READY} && !document.getElementById('place-here').disabled && "
                "window.__arrangementWire.replies.some(reply=>reply.action==='arrangement_insert'&&reply.code==='arrangement_occupied')",
                timeout=20,
            )
        except TimeoutError as cause:
            diagnostics = await placement_diagnostics(tab)
            raise AssertionError(
                f"Occupied retry did not settle: {diagnostics!r}; server_version={server.pg.session.version}"
            ) from cause
        assert server.pg.session.version == failed_version
        assert await tab.eval(f"!document.getElementById('placement-bar').hidden && {OUTLINE}?.visible")
        assert server.pg.current_world().model_dump(mode="json") == destination
        await click(tab, "#place-rotate")
        await settle_camera(tab)
        camera = await tab.eval(
            "[window.__robotforge.three().camera.position.toArray(),window.__robotforge.cameraTarget()]"
        )
        await place(tab, [0, 1.5, 0])
        first = server.pg.current_world().model_dump(mode="json")
        assert len(first["objects"]) == len(destination["objects"]) + 3
        assert first["objects"][:1] == destination["objects"]
        assert first["robots"] == destination["robots"] and first["terrain"] == destination["terrain"]
        assert all(obj["yaw"] == pytest.approx(90) for obj in first["objects"][1:])
        assert (
            await tab.eval("[window.__robotforge.three().camera.position.toArray(),window.__robotforge.cameraTarget()]")
            == camera
        )
        # A second deliberate selection inserts another copy; Undo removes only that latest group.
        await prepare(tab)
        # This clear point stays inside the fixed camera on the 320px touch viewport too.
        await place(tab, [1.5, -2, 0])
        assert len(server.pg.current_world().objects) == len(destination["objects"]) + 6
        await click(tab, "#undo-edit")
        await tab.until(f"{READY} && document.getElementById('undo-edit').hidden")
        assert server.pg.current_world().model_dump(mode="json") == first
        await reload(tab)
        assert server.pg.current_world().model_dump(mode="json") == first
        await open_shelf(tab)
        await tab.until("document.querySelectorAll('.arrangement-row').length===1")
        # Remove confirmation supports native Escape and preserves placed copies.
        await click(tab, REMOVE)
        assert await tab.eval("document.activeElement.id") == "arrangement-remove-cancel"
        await key(tab, "Escape")
        await key(tab, "Escape", kind="keyUp")
        assert await tab.eval("document.getElementById('arrangement-confirm').hidden")
        await click(tab, REMOVE)
        await click(tab, "#arrangement-remove-apply")
        await tab.until("document.querySelectorAll('.arrangement-row').length===0")
        assert server.pg.current_world().model_dump(mode="json") == first
        assert await tab.eval("window.__qaErrors") == []

    viewer(size, lang, scenario, world=table_layout)


DELAY = """
window.__arrangementSent=[];window.__arrangementActions=new Map();
const send=WebSocket.prototype.send;
WebSocket.prototype.send=function(s){const m=JSON.parse(s);
 if(m.action?.startsWith('arrangement_')){window.__arrangementSent.push(m);window.__arrangementActions.set(m.id,m.action)}
 return send.call(this,s)};
const descriptor=Object.getOwnPropertyDescriptor(WebSocket.prototype,'onmessage');
Object.defineProperty(WebSocket.prototype,'onmessage',{get:descriptor.get,set(fn){descriptor.set.call(this,ev=>{
 const m=typeof ev.data==='string'?JSON.parse(ev.data):null;
 if(m?.type==='reply'&&window.__holdArrangement===window.__arrangementActions.get(m.id)){
  window.__holdArrangement=null;window.__releaseArrangement=()=>fn.call(this,ev);return;
 }
 return fn.call(this,ev);
});}});
"""


@pytest.mark.parametrize("action", ["list", "save", "prepare"])
@pytest.mark.parametrize("interruption", ["reset", "map"])
def test_delayed_shelf_reply_cannot_restore_old_context(viewer, action, interruption):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=DELAY)
        await tab.navigate()
        assert await tab.eval("window.__arrangementSent.length") == 0
        await pause(tab, server)
        await tab.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
        if action == "save":
            await select_group(tab)
            await click(tab, "#arrangement-save-open")
            await tab.page.call("Input.insertText", text="Delayed save")
            await tab.run("window.__holdArrangement='arrangement_save'")
            await click(tab, "#arrangement-save")
        elif action == "prepare":
            await save_group(tab)
            await open_shelf(tab)
            await tab.until("!document.querySelector('[data-arrangement-action=insert]').disabled")
            await tab.run("window.__holdArrangement='arrangement_prepare'")
            await click(tab, INSERT)
        else:
            await tab.run("window.__holdArrangement='arrangement_list'")
            await open_shelf(tab, wait_ready=False)
        await tab.until("!!window.__releaseArrangement")
        if interruption == "reset":
            old_generation = server.pg.session.generation
            server.pg.session.reset()
            assert server.pg.session.generation > old_generation
            await asyncio.sleep(0.4)
            await tab.until(READY)
        else:
            if await tab.eval("document.getElementById('world-panel').hidden"):
                await click(tab, "#world-toggle")
            await click(tab, '[data-tab="maps"]')
            await click(tab, "#map-list .choice:first-child")
            await tab.until(READY)
        await tab.run("window.__releaseArrangement()")
        await asyncio.sleep(0.3)
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('arrangement-save-form').hidden")
        assert await tab.eval("document.getElementById('undo-edit').hidden")
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=table_layout)


MISSING_MEMBER = """
const send=WebSocket.prototype.send;
WebSocket.prototype.send=function(s){const m=JSON.parse(s);
 if(m.action==='arrangement_insert')window.__omitArrangementMember=true;return send.call(this,s)};
const descriptor=Object.getOwnPropertyDescriptor(WebSocket.prototype,'onmessage');
Object.defineProperty(WebSocket.prototype,'onmessage',{get:descriptor.get,set(fn){descriptor.set.call(this,ev=>{
 const m=typeof ev.data==='string'?JSON.parse(ev.data):null;
 if(window.__omitArrangementMember&&m?.type==='scene'){
  for(const node of [...m.bodies,...m.geoms])
    if(node.name==='bin_2'||node.name?.startsWith('bin_2/'))node.name='withheld_arrangement_member';
  window.__omittedArrangementVersion=m.version;window.__omitArrangementMember=false;
  return fn.call(this,new MessageEvent('message',{data:JSON.stringify(m)}));
 }
 return fn.call(this,ev);
});}});
"""


def test_all_members_must_be_posed_and_rendered_before_arrangement_success_or_undo(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=MISSING_MEMBER)
        await tab.navigate()
        await pause(tab, server)
        await tab.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
        await save_group(tab)
        await prepare(tab)
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([2,0,0])"))
        await tab.until("!!window.__omittedArrangementVersion && !window.__robotforge.busy().busy", timeout=22)
        assert len(server.pg.current_world().objects) == 6
        assert await tab.eval(
            """['base_2','top_2'].every(name=>window.__robotforge.scene.bodies.some(body=>body.name===name))"""
        )
        assert await tab.eval("!window.__robotforge.scene.bodies.some(body=>body.name==='bin_2')")
        assert await tab.eval("document.getElementById('undo-edit').hidden")
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval("!document.getElementById('toast').textContent.includes('Inserted')")
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=table_layout)


def test_three_outline_cancel_insert_undo_cycles_dispose_owned_buffers_once(viewer):  # noqa: F811
    async def memory(tab):
        await settle_camera(tab)
        return await tab.eval("({...window.__robotforge.three().renderer.info.memory})")

    async def scenario(tab, server):
        await pause(tab, server)
        await tab.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
        original = server.pg.current_world().model_dump(mode="json")
        await save_group(tab)
        await prepare(tab)
        await key(tab, "Escape")
        baseline = await memory(tab)
        samples = []
        for _ in range(3):
            await prepare(tab)
            await tab.run(f"""
              window.__arrangementDisposals={{owned:0,source:0}};
              const ghost={OUTLINE}, owned=new Set(ghost.children.map(part=>part.geometry));
              for(const geometry of owned)
                geometry.addEventListener('dispose',()=>window.__arrangementDisposals.owned++);
              const source=new Set();window.__robotforge.three().scene.traverse(node=>{{
                if(node.isMesh&&node.userData.geom!==undefined)source.add(node.geometry);
              }});
              if([...owned].some(geometry=>source.has(geometry)))throw Error('outline borrowed a source geometry');
              for(const geometry of source)
                geometry.addEventListener('dispose',()=>window.__arrangementDisposals.source++);
            """)
            await key(tab, "Escape")
            assert await tab.eval("({...window.__arrangementDisposals})") == {"owned": 3, "source": 0}
            assert await memory(tab) == baseline
            await prepare(tab)
            await place(tab, [2, 0, 0])
            await click(tab, "#undo-edit")
            await tab.until(f"{READY} && document.getElementById('undo-edit').hidden")
            assert server.pg.current_world().model_dump(mode="json") == original
            samples.append(await memory(tab))
        assert samples == [baseline] * 3
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=table_layout)


@pytest.mark.parametrize("interruption", ["play", "help", "drawer"])
def test_pending_prepare_cannot_open_after_play_or_native_panel_close(viewer, interruption):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=DELAY)
        await tab.navigate()
        if interruption == "drawer":
            await tab.page.call(
                "Emulation.setDeviceMetricsOverride", width=320, height=740, deviceScaleFactor=2, mobile=True
            )
        await pause(tab, server)
        await tab.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
        await save_group(tab)
        await open_shelf(tab)
        await tab.until("!document.querySelector('[data-arrangement-action=insert]').disabled")
        await tab.run("window.__holdArrangement='arrangement_prepare'")
        await click(tab, INSERT)
        await tab.until("!!window.__releaseArrangement")
        before = server.pg.session.version
        await click(tab, {"play": "#btn-play", "help": "#help-toggle", "drawer": "#drawer-toggle"}[interruption])
        if interruption == "play":
            await tab.until("!document.getElementById('btn-play').classList.contains('is-paused')")
            assert server.pg.session.version == before
        else:
            assert await tab.eval("document.getElementById('world-panel').hidden")
            # Reopening before the old reply arrives must not make that reply current again.
            await open_shelf(tab)
        await tab.run("window.__releaseArrangement()")
        await asyncio.sleep(0.3)
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval(f"!{OUTLINE}")
        assert await tab.eval("window.__qaErrors") == []

    viewer("phone" if interruption == "drawer" else "laptop", "en", scenario, world=table_layout)


def test_save_form_resets_after_selection_change_and_can_save_the_new_pair(viewer):  # noqa: F811
    async def scenario(tab, server):
        await pause(tab, server)
        await tab.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
        await select_group(tab)
        await click(tab, "#arrangement-save-open")
        await tab.page.call("Input.insertText", text="Stale three props")
        await click(tab, "#world-toggle")
        await click(tab, '[data-edit-object="top"]')
        assert await tab.eval("document.querySelectorAll('#edit-object-list [aria-pressed=true]').length") == 2
        assert await tab.eval("document.getElementById('arrangement-save-form').hidden")
        assert await tab.eval("document.getElementById('arrangement-name').value") == ""
        await click(tab, "#world-toggle")
        await click(tab, "#arrangement-save-open")
        await tab.page.call("Input.insertText", text="Current pair")
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        await tab.until("document.querySelectorAll('.arrangement-row').length===1")
        assert await tab.eval("document.querySelector('.arrangement-row strong').textContent") == "Current pair"
        assert await tab.eval("document.querySelector('.arrangement-row small').textContent") == "2 props"
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=table_layout)


@pytest.mark.parametrize("interruption", ["reset", "reconnect"])
def test_late_insert_ack_after_reset_or_reconnect_cannot_restore_creator_or_undo(viewer, interruption):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call(
            "Page.addScriptToEvaluateOnNewDocument",
            source=DELAY.replace(
                "window.__arrangementSent.push(m);", "window.__arrangementSocket=this;window.__arrangementSent.push(m);"
            ),
        )
        await tab.navigate()
        await pause(tab, server)
        await tab.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
        await save_group(tab)
        await prepare(tab)
        await tab.run("window.__holdArrangement='arrangement_insert'")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([2,0,0])"))
        await tab.until("!!window.__releaseArrangement")
        assert len(server.pg.current_world().objects) == 6
        assert await tab.eval("window.__robotforge.busy().busy")
        if interruption == "reset":
            await click(tab, "#btn-reset")
        else:
            await tab.run("window.__arrangementSocket.close()")
        await tab.until("document.getElementById('placement-bar').hidden")
        await tab.run("window.__releaseArrangement()")
        await tab.until(READY)
        assert await tab.eval("document.getElementById('undo-edit').hidden")
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval(f"!{OUTLINE}")
        assert len(server.pg.current_world().objects) == 6
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=table_layout)


def test_missing_third_member_mesh_keeps_all_descriptor_identities_but_blocks_success(viewer):  # noqa: F811
    async def scenario(tab, server):
        await pause(tab, server)
        await tab.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
        await save_group(tab)
        await prepare(tab)
        await click(tab, "#place-rotate")
        await tab.run("""
          const rf=window.__robotforge, renderer=rf.three().renderer, original=renderer.render.bind(renderer);
          window.__hiddenArrangementMeshes=[];
          renderer.render=function(scene,camera){
            const member=rf.scene?.object_edit?.objects.find(entry=>entry.name==='bin_2');
            if(member){
              const ids=new Set(member.geoms);
              scene.traverse(node=>{
                if(node.isMesh&&ids.has(node.userData.geom)){
                  node.visible=false;
                  if(!window.__hiddenArrangementMeshes.includes(node.userData.geom))window.__hiddenArrangementMeshes.push(node.userData.geom);
                }
              });
            }
            return original(scene,camera);
          };
        """)
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([2,0,0])"))
        await tab.until("window.__hiddenArrangementMeshes.length>0 && !window.__robotforge.busy().busy")
        assert len(server.pg.current_world().objects) == 6
        displayed = await tab.run("""
          const rf=window.__robotforge, desc=rf.scene, drawn=new Set();
          rf.three().scene.traverse(node=>{if(node.isMesh&&node.visible)drawn.add(node.userData.geom)});
          return ['base_2','top_2','bin_2'].map(name=>{
            const member=desc.object_edit.objects.find(entry=>entry.name===name);
            return {name,body:desc.bodies.some(body=>body.name===name),entry:!!member,
              geoms:member?.geoms.every(id=>desc.geoms.some(geom=>geom.id===id)),
              displayed:member?.geoms.some(id=>drawn.has(id))};
          });
        """)
        assert all(member["entry"] and member["geoms"] for member in displayed), displayed
        assert all(member["body"] for member in displayed), displayed
        assert [member["displayed"] for member in displayed] == [True, True, False]
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('undo-edit').hidden")
        assert await tab.eval("!document.getElementById('toast').textContent.includes('Inserted')")
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=table_layout)


HOLD_INSERT_POSES = DELAY.replace(
    "return fn.call(this,ev);",
    """if(window.__holdInsertPoses&&typeof ev.data!=='string'){
       window.__heldInsertPoseCount=(window.__heldInsertPoseCount||0)+1;
       this.send(JSON.stringify({type:'seen'}));return;
     }
     return fn.call(this,ev);""",
)


@pytest.mark.parametrize("held_stage", ["poses", "render"])
def test_real_creation_deadline_without_first_display_clears_owned_outline_and_no_undo(viewer, held_stage):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=HOLD_INSERT_POSES)
        await tab.navigate()
        await pause(tab, server)
        await tab.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
        await save_group(tab)
        await prepare(tab)
        await tab.run(f"""
          window.__timedOutlineDisposals=0;
          for(const geometry of new Set({OUTLINE}.children.map(node=>node.geometry)))
            geometry.addEventListener('dispose',()=>window.__timedOutlineDisposals++);
          window.__insertDeadlineStart=performance.now();
        """)
        if held_stage == "poses":
            await tab.run("window.__holdInsertPoses=true")
        else:
            await tab.run("""
              const rf=window.__robotforge, gl=rf.three().renderer.getContext(), sourceVersion=rf.scene.version;
              const original=gl.isContextLost.bind(gl);
              // Withhold the main renderer's completion signal while retaining real poses and meshes.
              gl.isContextLost=function(){
                if(rf.scene.version>sourceVersion){
                  window.__heldArrangementRenderChecks=(window.__heldArrangementRenderChecks||0)+1;return true;
                }
                return original();
              };
            """)
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([2,0,0])"))
        stage_oracle = (
            "window.__heldInsertPoseCount>0" if held_stage == "poses" else "window.__heldArrangementRenderChecks>0"
        )
        await tab.until(stage_oracle + " && window.__robotforge.busy().busy")
        assert len(server.pg.current_world().objects) == 6
        assert await tab.eval("window.__robotforge.renderedVersion < window.__robotforge.scene.version")
        assert await tab.eval("document.getElementById('undo-edit').hidden")
        await tab.until("!window.__robotforge.busy().busy", timeout=23)
        assert await tab.eval("performance.now()-window.__insertDeadlineStart") >= 14500
        assert await tab.eval("window.__timedOutlineDisposals") == 3
        assert await tab.eval(f"!{OUTLINE}")
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('undo-edit').hidden")
        assert await tab.eval("!document.getElementById('toast').textContent.includes('Inserted')")
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=table_layout)


def test_late_real_changed_record_error_after_native_reset_cannot_reopen_or_refresh_shelf(viewer):  # noqa: F811
    async def scenario(tabs, server):
        primary, other = tabs
        await primary.page.call(
            "Page.addScriptToEvaluateOnNewDocument",
            source=DELAY.replace(
                "window.__holdArrangement=null;", "window.__heldArrangementReply=m;window.__holdArrangement=null;"
            ),
        )
        await primary.navigate()
        await pause(primary, server)
        await primary.eval("window.__robotforge.lookAt([5,-7,5],[0,0,.15])")
        await save_group(primary)
        await prepare(primary)
        original = server.pg.current_world().model_dump(mode="json")
        version = server.pg.session.version
        # Remove the real shared record through another native client, leaving the
        # primary client's already-prepared insertion and current world untouched.
        await pause(other, server)
        await open_shelf(other)
        await other.until(
            "document.querySelectorAll('.arrangement-row').length===1"
            " && !document.querySelector('[data-arrangement-action=remove]').disabled"
        )
        await click(other, REMOVE)
        await click(other, "#arrangement-remove-apply")
        await other.until("document.querySelectorAll('.arrangement-row').length===0")
        assert server.pg.session.version == version
        assert server.pg.current_world().model_dump(mode="json") == original
        assert await primary.eval("!document.getElementById('placement-bar').hidden")
        await primary.run("window.__holdArrangement='arrangement_insert'")
        await click(primary, point=await primary.eval("window.__robotforge.screenOfPoint([2,0,0])"))
        await primary.until("!!window.__releaseArrangement")
        failure = await primary.eval("window.__heldArrangementReply")
        assert failure["ok"] is False
        assert failure["code"] in {"arrangement_changed", "arrangement_damaged"}, failure
        assert server.pg.session.version == version
        assert server.pg.current_world().model_dump(mode="json") == original
        await click(primary, "#btn-reset")
        await primary.until("document.getElementById('placement-bar').hidden")
        assert await primary.eval("document.getElementById('world-panel').hidden")
        requests_before = await primary.eval(
            "window.__arrangementSent.filter(message=>message.action==='arrangement_list').length"
        )
        await primary.run("window.__releaseArrangement()")
        await primary.until(READY)
        await asyncio.sleep(0.3)
        assert (
            await primary.eval("window.__arrangementSent.filter(message=>message.action==='arrangement_list').length")
            == requests_before
        )
        assert await primary.eval("document.getElementById('world-panel').hidden")
        assert await primary.eval("document.getElementById('placement-bar').hidden")
        assert await primary.eval("document.getElementById('undo-edit').hidden")
        assert await primary.eval(f"!{OUTLINE}")
        assert await primary.eval("!document.getElementById('toast').textContent.includes('Inserted')")
        assert server.pg.current_world().model_dump(mode="json") == original
        assert await primary.eval("window.__qaErrors") == []
        assert await other.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, also=(("laptop", "zh-CN"),), world=table_layout)
