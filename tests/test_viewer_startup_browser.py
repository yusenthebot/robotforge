"""Real viewer readiness under delayed data, failed loads and context restoration."""

from __future__ import annotations

import asyncio

import pytest
from test_viewer_creation_browser import click
from test_viewer_layout import pytestmark, viewer  # noqa: F401

INSTRUMENT = """
window.__startup = {holdPoses:false, holdMesh:false, failMesh:null, meshAttempts:0,
  seen:0, frames:0, draws:[], outbound:[], releases:[]};
const p=window.__startup;
function frame(){p.frames++;requestAnimationFrame(frame)}requestAnimationFrame(frame);
const nativeFetch=window.fetch;
window.fetch=async (...args)=>{
  if(!String(args[0]).includes('mesh/'))return nativeFetch(...args);
  p.meshAttempts++;
  if(p.failMesh==='network')throw new TypeError('injected network failure');
  if(p.failMesh==='http')return new Response('unavailable',{status:503});
  const response=await nativeFetch(...args);
  if(p.holdMesh)await new Promise(resolve=>p.releases.push(resolve));
  return response;
};
p.releaseMesh=()=>{p.holdMesh=false;for(const resolve of p.releases.splice(0))resolve()};
const Socket=window.WebSocket;
window.WebSocket=class extends Socket{
  set onmessage(handler){super.onmessage=ev=>{
    if(typeof ev.data!=='string'&&p.holdPoses){
      p.deliverPose=()=>handler(ev);this.send(JSON.stringify({type:'seen'}));return;
    }
    handler(ev);
  }}
  send(data){const msg=JSON.parse(data);if(msg.type==='seen')p.seen++;else p.outbound.push(msg);
    return super.send(data)}
};
p.releasePoses=()=>{p.holdPoses=false;p.deliverPose?.();p.deliverPose=null};
let api;
Object.defineProperty(window,'__robotforge',{configurable:true,get:()=>api,set:value=>{
  api=value;const {renderer,camera}=api.three(),render=renderer.render;
  renderer.render=function(scene,cam){
    if(api.scene)p.draws.push({version:api.scene.version,main:cam===camera,
      loading:!document.getElementById('loading').hidden,time:api.time});
    return render.call(this,scene,cam);
  };
}});
"""
READY = "!!window.__robotforge?.scene && window.__robotforge.renderedVersion === window.__robotforge.scene.version"


async def reload_instrumented(tab, flags=""):
    await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=INSTRUMENT + flags)
    await tab.page.call("Page.reload")
    await tab.until("!!window.__startup && !!window.__robotforge")


def test_first_pose_precedes_all_world_draws_and_ring_measurement_even_at_time_zero(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.reset()
        server.pg.session.paused = True
        await reload_instrumented(tab, "p.holdPoses=true;")
        await tab.until("!!window.__robotforge.scene && window.__startup.seen > 8 && window.__startup.frames > 12")
        assert await tab.eval("window.__startup.draws.length") == 0
        assert not await tab.eval("document.getElementById('loading').hidden")
        await tab.run("window.__startup.releasePoses()")
        await tab.until(READY)
        assert await tab.eval("window.__robotforge.time") == 0
        assert await tab.eval("window.__startup.draws[0].loading")
        assert await tab.eval("document.getElementById('loading').hidden")
        ring = await tab.run("""
          const THREE=await import('three'),rf=window.__robotforge,{scene}=rf.three();
          const ring=scene.children.find(o=>o.children.some(m=>m.geometry?.type==='RingGeometry'));
          const ids=new Set(rf.scene.bodies.filter(b=>b.name.startsWith('r/')).map(b=>b.id));
          const box=new THREE.Box3();scene.updateMatrixWorld(true);
          scene.traverse(o=>{if(o.isMesh&&o.visible&&ids.has(o.userData.body))box.expandByObject(o)});
          const size=box.getSize(new THREE.Vector3());
          return {actual:ring.scale.x,expected:Math.max(.12,Math.hypot(size.x,size.y)*.525),x:ring.position.x};
        """)
        assert ring["actual"] == pytest.approx(ring["expected"], abs=1e-6)
        assert ring["x"] == pytest.approx(2, abs=0.01)
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.add_robot("r", template="rover", pos=[2, 0]))


def test_delayed_mesh_drops_poses_acks_and_only_latest_same_size_scene_draws(viewer):  # noqa: F811
    async def scenario(tab, server):
        await reload_instrumented(tab)
        await tab.until(READY)
        await tab.run("window.__startup.holdMesh=true;window.__startup.outbound=[]")
        original = await tab.eval("window.__robotforge.scene.version")
        count = await tab.eval("window.__robotforge.scene.nbody")
        server.pg.update_robot("r", pos=[1, 0])
        await tab.until("window.__startup.releases.length > 0")
        # Old-scene frames before this held fetch are legitimate. Observe only the active loading gate.
        await tab.run("window.__startup.draws=[]")
        before = await tab.eval("window.__robotforge.bodyPosition(1)")
        seen = await tab.eval("window.__startup.seen")
        await tab.until(f"window.__startup.seen > {seen + 5}")
        assert await tab.eval("window.__startup.draws.length") == 0
        assert await tab.eval("window.__robotforge.renderedVersion") == 0
        assert await tab.eval("window.__robotforge.bodyPosition(1)") == before
        # An obsolete ground image must not send a navigation request.
        await tab.run("""document.getElementById('view').dispatchEvent(new MouseEvent('dblclick',
          {clientX:300,clientY:650,bubbles:true}))""")
        assert not await tab.eval("window.__startup.outbound.some(m=>m.type==='go_to')")
        server.pg.update_robot("r", pos=[2, 0])
        await tab.until("window.__startup.releases.length > 1")
        await tab.run("window.__startup.releaseMesh()")
        await tab.until(READY)
        assert await tab.eval("window.__robotforge.scene.version") >= original + 2
        assert await tab.eval("window.__robotforge.scene.nbody") == count
        assert (await tab.eval("window.__robotforge.bodyPosition(1)"))[0] == pytest.approx(2, abs=0.01)
        versions = await tab.eval("[...new Set(window.__startup.draws.map(d=>d.version))]")
        assert versions == [await tab.eval("window.__robotforge.scene.version")]
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.add_robot("r", template="rover"))


@pytest.mark.parametrize("size,lang,fault", [("laptop", "en", "http"), ("phone", "zh-CN", "network")])
def test_failed_mesh_offers_working_native_retry_without_an_automatic_loop(viewer, size, lang, fault):  # noqa: F811
    async def scenario(tab, _server):
        await reload_instrumented(tab, f"p.failMesh={fault!r};")
        await tab.until("!document.getElementById('scene-retry').hidden")
        await tab.until("window.__startup.frames > 15")
        assert await tab.eval("window.__startup.meshAttempts") == 1
        assert await tab.eval("window.__robotforge.renderedVersion") == 0
        assert not await tab.eval("document.getElementById('loading').hidden")
        assert await tab.eval("document.getElementById('scene-retry').textContent") == (
            "Retry" if lang == "en" else "重试"
        )
        await tab.run("window.__startup.failMesh=null")
        await click(tab, "#scene-retry")
        await tab.until(READY)
        assert await tab.eval("document.getElementById('loading').hidden")
        assert await tab.eval("document.getElementById('scene-retry').hidden")
        assert await tab.eval("window.__startup.meshAttempts") == 2
        assert not tab.errors

    viewer(size, lang, scenario, world=lambda pg: pg.add_robot("r", template="rover"))


def test_context_loss_invalidates_submission_marker_until_restored_main_draw(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await tab.until(READY)
        supported = await tab.run("""window.__lose=window.__robotforge.three().renderer.getContext()
          .getExtension('WEBGL_lose_context');if(window.__lose)window.__lose.loseContext();return !!window.__lose;""")
        if not supported:
            pytest.skip("driver does not expose WEBGL_lose_context")
        await tab.until("window.__robotforge.renderedVersion === 0")
        await asyncio.sleep(0.1)
        assert not await tab.eval("document.getElementById('loading').hidden")
        assert await tab.eval("window.__robotforge.renderedVersion") == 0
        await tab.run("window.__lose.restoreContext()")
        await tab.until(READY)
        assert await tab.eval("document.getElementById('loading').hidden")
        assert not await tab.eval("window.__robotforge.three().renderer.getContext().isContextLost()")
        assert not tab.errors

    viewer("laptop", "en", scenario)
