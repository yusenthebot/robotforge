"""Explicit row placement uses ordered server targets and the native group move transaction."""

from __future__ import annotations

import json

import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_object_group_browser import READY
from test_viewer_object_group_move_browser import prepare
from test_viewer_placement_browser import spacing
from viewer_camera import settle_camera

ORDER = ["bin", "narrow", "wide"]


def row_world(pg):
    pg.add_object("wide", shape="table", size=[1.2, 0.7, 0.45], pos=[-2, -1.4], fixed=True)
    pg.add_object("bin", shape="bin", size=[0.6, 0.4, 0.3], pos=[-2, 1.2], fixed=True, yaw=15)
    pg.add_object("narrow", shape="box", size=[0.2, 0.3, 0.2], pos=[-0.8, 0], material="wood")
    pg.add_object("obstacle", shape="box", size=[0.3, 0.3, 0.6], pos=[2, -1], fixed=True)


# Installed before opening the app: inspect the actual wire contract and selectively
# withhold/corrupt a real server preparation, without fabricating successful placement.
ROW_WIRE = """
window.__rowRequests=[];window.__rowReplies=[];window.__rowIds=new Set();
const send=WebSocket.prototype.send;
WebSocket.prototype.send=function(s){const m=JSON.parse(s);
 if(m.action==='prepare_object_group'&&m.layout==='row')window.__rowIds.add(m.id);
 if(['prepare_object_group','move_object_group'].includes(m.action))window.__rowRequests.push(m);
 return send.call(this,s);
};
const descriptor=Object.getOwnPropertyDescriptor(WebSocket.prototype,'onmessage');
Object.defineProperty(WebSocket.prototype,'onmessage',{get:descriptor.get,set(fn){descriptor.set.call(this,ev=>{
 const m=typeof ev.data==='string'?JSON.parse(ev.data):null;
 if(m?.type==='reply'&&window.__rowIds.has(m.id)&&m.data?.members){
  window.__rowReplies.push(structuredClone(m.data));
  const d=m.data,f=window.__rowFault;
  if(f==='missing-offset')delete d.members[0].target_offset;
  if(f==='wrong-order')d.members.reverse();
  if(f==='wrong-gap')d.gap+=.01;
  if(f==='wrong-layout')delete d.layout;
  if(f==='invalid-vector')d.members[0].target_offset[0]=null;
  if(f==='duplicate-geom')d.members[1].geoms.push(d.members[0].geoms[0]);
  if(f==='missing-geom')d.members[0].geoms.pop();
  const delivered=new MessageEvent('message',{data:JSON.stringify(m)});
  if(window.__holdRow){window.__holdRow=false;window.__releaseRow=()=>fn.call(this,delivered);return;}
  return fn.call(this,delivered);
 }
 return fn.call(this,ev);
});}});
"""


async def ready(tab, server):
    await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=ROW_WIRE)
    await tab.navigate()
    await prepare(tab, server)


async def select_row(tab):
    if await tab.eval("document.getElementById('world-panel').hidden"):
        await click(tab, "#world-toggle")
    await click(tab, '[data-tab="objects"]')
    await click(tab, "#select-object-group")
    await click(tab, "#world-toggle")
    for name in ORDER:
        await click(tab, f'[data-edit-object="{name}"]')
    await click(tab, "#world-toggle")
    await settle_camera(tab)


async def gap(tab, value):
    await click(tab, "#object-row-gap")
    await tab.run("""window.__rowGapInput=null;
      document.getElementById('object-row-gap').addEventListener('input',event=>{
        window.__rowGapInput={trusted:event.isTrusted,inputType:event.inputType};
      },{once:true});""")
    # CDP does not infer the platform's editing command from Meta+A consistently,
    # especially for the native number editor. Dispatch the browser edit command.
    await tab.page.call(
        "Input.dispatchKeyEvent", type="keyDown", key="a", code="KeyA", modifiers=4, commands=["selectAll"]
    )
    await tab.page.call("Input.dispatchKeyEvent", type="keyUp", key="a", code="KeyA", modifiers=0)
    await tab.page.call("Input.insertText", text=value)
    witness = await tab.eval("""({value:document.getElementById('object-row-gap').value,
      active:document.activeElement.id,input:window.__rowGapInput})""")
    assert witness["value"] == value and witness["active"] == "object-row-gap", witness
    assert witness["input"] and witness["input"]["trusted"], witness


async def begin_row(tab):
    await click(tab, "#object-group-row")
    await tab.until("!document.getElementById('placement-bar').hidden && !window.__robotforge.busy().busy")


GHOST = """
const scene=window.__robotforge.three().scene,ghost=scene.getObjectByName('placement-size-preview');
const data=window.__rowReplies.at(-1),desc=window.__robotforge.scene;
const owned=new Map(data.members.flatMap(member=>member.geoms.map(id=>[id,member])));
const visible=new Set(desc.geoms.filter(g=>owned.has(g.id)&&g.group<desc.visibleGroupLimit&&g.rgba[3]>0).map(g=>g.id));
const sources=[];scene.traverse(n=>{if(n.isMesh&&n.visible&&visible.has(n.userData.geom))sources.push(n)});
window.__rowDisposed=0;
for(const geometry of new Set(sources.map(n=>n.geometry)))
 geometry.addEventListener('dispose',()=>window.__rowDisposed++);
return {count:ghost?.children.length,sources:sources.length,borrowed:ghost?.children.every(part=>sources.some(source=>{
 const member=owned.get(source.userData.geom),expected=source.matrixWorld.elements.slice();
 for(let axis=0;axis<3;axis++)expected[12+axis]+=member.target_offset[axis]-member.pos[axis];
 return part.geometry===source.geometry&&expected.every((v,index)=>Math.abs(v-part.matrix.elements[index])<1e-6);
})),order:data.members.map(m=>m.name),gap:data.gap};
"""


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_row_choice_compound_ghost_rotate_snap_atomic_move_and_undo(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab, server)
        before = server.pg.store.load().model_dump(mode="json")
        await select_row(tab)
        assert await tab.eval("document.getElementById('object-row-gap').value") == "0.25"
        assert await tab.eval("document.getElementById('object-group-row').textContent") == (
            "Arrange in a row" if lang == "en" else "排成一行"
        )
        # The narrow selection bar stays inside the viewport and above the dock;
        # native clicks below also verify the actual controls are unobstructed.
        assert await tab.eval("""(()=>{const r=document.getElementById('object-edit-bar').getBoundingClientRect();
          const dock=document.querySelector('.dock').getBoundingClientRect();
          return r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=dock.top;})()""")
        await gap(tab, "0.3")
        await begin_row(tab)
        ghost = await tab.run(GHOST)
        assert ghost["borrowed"] and ghost["count"] == ghost["sources"] and ghost["count"] > len(ORDER)
        assert ghost["order"] == ORDER and ghost["gap"] == 0.3
        wire = await tab.eval("window.__rowRequests.at(-1)")
        assert {name: wire[name] for name in ("mode", "layout", "gap", "names")} == {
            "mode": "move",
            "layout": "row",
            "gap": 0.3,
            "names": ORDER,
        }
        await key(tab, "Escape")
        assert await tab.eval("window.__rowDisposed") == 0
        assert server.pg.store.load().model_dump(mode="json") == before
        await select_row(tab)
        await begin_row(tab)
        prepared = await tab.eval("window.__rowReplies.at(-1)")
        await click(tab, "#place-rotate")
        await spacing(tab, "0.25")
        await tab.eval("window.__robotforge.lookAt([7,-7,5],[2,0,.15])")
        await settle_camera(tab)
        version = server.pg.session.version
        # At this location one end of the proposed row intersects the obstacle;
        # picking the floor at its centre still succeeds, so the server rejects it.
        occupied_point = await tab.eval("window.__robotforge.screenOfPoint([2,0,0])")
        await click(tab, point=occupied_point)
        await tab.until("!window.__robotforge.busy().busy")
        assert await tab.eval("window.__rowRequests.filter(m=>m.action==='move_object_group').length") == 1
        assert server.pg.session.version == version
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        assert server.pg.store.load().model_dump(mode="json") == before
        clear_point = await tab.eval("window.__robotforge.screenOfPoint([3.5,0,0])")
        hit = await tab.eval(f"""(()=>{{const p={json.dumps(clear_point)},e=document.elementFromPoint(...p);
          return {{point:p,viewport:[innerWidth,innerHeight],tag:e?.tagName,id:e?.id}};}})()""")
        assert hit["tag"] == "CANVAS", hit
        await click(tab, point=clear_point)
        await tab.until(
            f"{READY} && document.getElementById('placement-bar').hidden"
            " && !document.getElementById('undo-edit').hidden"
        )
        assert server.pg.session.version == version + 1
        moved = server.pg.store.load().model_dump(mode="json")
        assert [obj["name"] for obj in moved["objects"]] == [obj["name"] for obj in before["objects"]]
        by_name = {obj["name"]: obj for obj in moved["objects"]}
        for member in prepared["members"]:
            target = member["target_offset"]
            obj = by_name[member["name"]]
            assert obj["pos"] == pytest.approx([3.5 - target[1], target[0], target[2]], abs=0.002)
            assert obj["yaw"] == pytest.approx(member["yaw"] + 90)
            original = next(item for item in before["objects"] if item["name"] == member["name"])
            assert {k: v for k, v in obj.items() if k not in {"pos", "yaw"}} == {
                k: v for k, v in original.items() if k not in {"pos", "yaw"}
            }
        submits = await tab.eval("window.__rowRequests.filter(m=>m.action==='move_object_group')")
        assert len(submits) == 2
        assert all(not {"layout", "gap", "target_offset"}.intersection(request) for request in submits)
        await click(tab, "#undo-edit")
        await tab.until(f"{READY} && document.getElementById('undo-edit').hidden")
        assert server.pg.store.load().model_dump(mode="json") == before
        assert server.pg.session.version == version + 2 and not tab.errors

    viewer(size, lang, scenario, world=row_world)


@pytest.mark.parametrize(
    "fault",
    ["missing-offset", "wrong-order", "wrong-gap", "wrong-layout", "invalid-vector", "duplicate-geom", "missing-geom"],
)
def test_row_prepare_malformed_response_never_falls_back_to_group_move(viewer, fault):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab, server)
        await select_row(tab)
        saved = server.pg.store.file.read_bytes()
        await tab.run(f"window.__rowFault={json.dumps(fault)}")
        await click(tab, "#object-group-row")
        await tab.until("window.__rowReplies.length===1&&!window.__robotforge.busy().busy")
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval("window.__rowRequests.every(m=>m.action!=='move_object_group')")
        assert server.pg.store.file.read_bytes() == saved and not tab.errors

    viewer("laptop", "en", scenario, world=row_world)


@pytest.mark.parametrize("interruption", ["gap", "selection", "pointer"])
def test_row_prepare_stale_parameter_or_selection_cannot_claim_preview(viewer, interruption):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab, server)
        await select_row(tab)
        saved = server.pg.store.file.read_bytes()
        await tab.run("window.__holdRow=true")
        await click(tab, "#object-group-row")
        await tab.until("!!window.__releaseRow")
        if interruption == "gap":
            await gap(tab, "0.4")
        else:
            point = [-2, 1.2, 0.15] if interruption == "selection" else [0, 0, 0]
            await click(tab, point=await tab.eval(f"window.__robotforge.screenOfPoint({point})"))
        await tab.run("window.__releaseRow()")
        await tab.until("!window.__robotforge.busy().busy")
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert server.pg.store.file.read_bytes() == saved and not tab.errors

    viewer("laptop", "en", scenario, world=row_world)


def test_row_invalid_gap_does_not_send_and_hidden_parameter_change_revokes_prepared_ghost(viewer):  # noqa: F811
    async def scenario(tab, server):
        await ready(tab, server)
        await select_row(tab)
        for value in ("0.01", "2.01", ""):
            # Input events also cover empty and out-of-range intermediate edits.
            await tab.run(f"""const e=document.getElementById('object-row-gap');e.value={json.dumps(value)};
              e.dispatchEvent(new Event('input',{{bubbles:true}}));""")
            assert await tab.eval("document.getElementById('object-group-row').disabled")
        assert await tab.eval("window.__rowRequests.length") == 0
        await gap(tab, "0.25")
        await begin_row(tab)
        await tab.run("""const e=document.getElementById('object-row-gap');e.value='0.5';
          e.dispatchEvent(new Event('input',{bubbles:true}));""")
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        assert not await tab.eval(
            "window.__robotforge.three().scene.getObjectByName('placement-size-preview')?.visible"
        )
        assert await tab.eval("window.__rowRequests.every(m=>m.action!=='move_object_group')")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=row_world)
