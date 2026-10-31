"""A retained terrain inverse takes priority over a stale group move receipt."""

from __future__ import annotations

from test_viewer_creation_browser import click
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_object_group_browser import READY, table_layout
from test_viewer_object_group_move_browser import begin_move, move_to, prepare
from test_viewer_terrain_create_browser import tap
from test_viewer_terrain_edit_browser import choose as choose_terrain
from viewer_camera import settle_camera


def course(pg):
    table_layout(pg)
    pg.add_terrain(
        "ramp",
        name="dock",
        pos=[0, 3],
        params={"length": 0.6, "width": 0.6, "angle": 8, "landing": 0.4},
    )


HOLD_TERRAIN_EDIT_SCENE = """
const Native=window.WebSocket;
window.__inversePriority={heldScene:null,edit:null,undo:null,lastState:null,timeoutFired:false};
window.WebSocket=class extends Native {
 send(value){
  const m=JSON.parse(value),wire=window.__inversePriority;
  if(m.action==='edit_terrain')wire.edit={id:m.id,sourceVersion:m.version};
  if(m.action==='undo_terrain_edit')wire.undo={id:m.id,request:m};
  super.send(value);
 }
 set onmessage(handler){super.onmessage=event=>{
  const m=typeof event.data==='string'?JSON.parse(event.data):null,wire=window.__inversePriority;
  if(m?.type==='scene'&&wire.edit&&m.version===wire.edit.sourceVersion+1){
   // Keep the previous group-move scene installed throughout the terrain timeout.
   // Delivering this scene would run onScene and erase the stale object receipt.
   wire.heldScene=m;return;
  }
  if(m?.type==='reply'&&m.id===wire.edit?.id)wire.edit.reply=m;
  if(m?.type==='reply'&&m.id===wire.undo?.id)wire.undo.reply=m;
  handler(event);
  if(m?.type==='state')wire.lastState={version:m.version,generation:m.generation};
 }}
};
const nativeTimeout=window.setTimeout.bind(window);
window.setTimeout=function(callback,delay,...args){
 const wire=window.__inversePriority;
 if(delay===15000&&wire.edit?.reply?.ok&&!wire.timeoutFired){
  // Exercise the application's real display-timeout callback with a shorter deadline.
  // Only this edit's creation wait is affected; no receipt or private app state is injected.
  return nativeTimeout(()=>{wire.timeoutFired=true;callback(...args);},2000);
 }
 return nativeTimeout(callback,delay,...args);
};
"""


def test_terrain_display_timeout_undo_uses_terrain_kind_after_group_move(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=HOLD_TERRAIN_EDIT_SCENE)
        await tab.navigate()
        await prepare(tab, server)
        await begin_move(tab)
        await move_to(tab, [2, 0, 0])
        assert await tab.eval("document.getElementById('undo-edit').textContent") == "Undo group move"
        after_group = server.pg.store.load().model_dump(mode="json")
        group_version = server.pg.session.version

        await tab.eval("window.__robotforge.lookAt([9,-15,14],[0,0,.2])")
        await settle_camera(tab)
        await choose_terrain(tab, "dock")
        await tap(tab, [3, 2, 0])
        await tab.until("""(()=>{
          const w=window.__inversePriority,d=w.edit?.reply?.data;
          return w.heldScene&&w.edit.reply.ok&&w.lastState?.version===d?.version&&
            w.lastState?.generation===d.generation;
        })()""")
        edit = await tab.eval("window.__inversePriority.edit.reply.data")
        assert server.pg.session.version == group_version + 1 == edit["version"]
        assert edit["undo_terrain_edit"]["version"] == edit["version"]
        assert await tab.eval("window.__robotforge.scene.version") == group_version
        assert await tab.eval("window.__robotforge.renderedVersion") == group_version
        assert server.pg.store.load().model_dump(mode="json") != after_group

        await tab.until(
            "window.__inversePriority.timeoutFired && !window.__robotforge.busy().busy"
            " && document.getElementById('placement-bar').hidden"
            " && !document.getElementById('undo-edit').hidden",
            timeout=6,
        )
        assert await tab.eval("document.getElementById('undo-edit').textContent") == "Undo terrain move"
        # The terrain scene must still be absent at the exact instant Undo chooses its
        # receipt. Its later inverse scene is delivered normally, preserving the race.
        assert await tab.eval("window.__robotforge.scene.version") == group_version
        await click(tab, "#undo-edit")
        await tab.until(
            f"{READY} && document.getElementById('undo-edit').hidden"
            " && document.getElementById('toast').textContent==='Terrain move undone. Paused at saved starts.'"
        )
        undo = await tab.eval("window.__inversePriority.undo")
        assert undo["request"]["token"] == edit["undo_terrain_edit"]["token"]
        assert undo["request"]["version"] == edit["version"]
        assert undo["reply"]["ok"]
        assert server.pg.session.version == group_version + 2
        assert server.pg.store.load().model_dump(mode="json") == after_group
        assert await tab.eval("window.__inversePriority.heldScene.version") == edit["version"]
        assert not tab.errors

    viewer("laptop", "en", scenario, world=course)
