"""GPU depth caching preserves caster changes and the sensor render boundary."""

from __future__ import annotations

import asyncio

import pytest
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_lighting_browser import SETUP
from test_viewer_startup_browser import READY

PROBE = """
let shadowDraws=0;cube.onBeforeShadow=()=>{shadowDraws++};
function pixels(){const previous=renderer.getRenderTarget();draw();
  const data=new Uint8Array(320*320*4);renderer.readRenderTargetPixels(target,0,0,320,320,data);
  renderer.setRenderTarget(previous);let hash=2166136261;
  for(const v of data)hash=Math.imul(hash^v,16777619);return hash>>>0;}
function fresh(){rig.sun.shadow.autoUpdate=true;try{return pixels()}finally{rig.sun.shadow.autoUpdate=false}}
cubeNode.position.z=.05;camera.position.set(.5,.6,.5);camera.lookAt(0,0,0);
"""


@pytest.mark.parametrize("dynamic", [False, True])
def test_static_depth_reuses_gpu_work_and_small_caster_motion_refreshes(viewer, dynamic):  # noqa: F811
    async def scenario(tab, _server):
        result = await tab.run(
            SETUP
            + PROBE
            + f"desc.bodies[1].dynamic={str(dynamic).lower()};"
            + """
          try {
            rig.bind(world,[floorNode,cubeNode],desc);rig.update();
            const initial=pixels(),first=shadowDraws;
            for(let i=0;i<10;i++)check(pixels()===initial,'static pixels changed');
            check(first>0&&shadowDraws===first,'static depth redrawn');
            const fit=rig.state();cubeNode.position.x+=.001;rig.update();
            check(rig.state()===fit&&rig.sun.shadow.needsUpdate,'small move did not dirty cached depth');
            const moved=pixels();check(shadowDraws>first,'GPU depth not refreshed');
            check(moved===fresh(),'small-move depth differs from uncached rendering');
            const beforeBind=shadowDraws;cubeNode.position.x+=.02;
            rig.bind(world,[floorNode,cubeNode],desc);rig.update();const rebound=pixels();
            check(shadowDraws>beforeBind,'rebind kept old depth');
            check(rebound===fresh(),'rebound depth differs from uncached rendering');
            return {gl:renderer.getContext().getError()};
          } finally {cleanup()}
        """
        )
        assert result["gl"] == 0
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)


@pytest.mark.parametrize("animated", ["rotor", "flex"])
def test_animated_worlds_conservatively_keep_original_depth_updates(viewer, animated):  # noqa: F811
    async def scenario(tab, _server):
        setup = "desc.rotors=[{body:1,center:[0,0,0],radius:.1}];" if animated == "rotor" else "desc.flexes=[{}];"
        result = await tab.run(
            SETUP
            + PROBE
            + setup
            + """
          try {
            rig.bind(world,[floorNode,cubeNode],desc);rig.update();pixels();
            const first=shadowDraws;cube.rotation.z=.5;pixels();
            check(rig.sun.shadow.autoUpdate&&shadowDraws>first,'animated depth was cached');
            rig.bind(world,[floorNode,cubeNode],{...desc,rotors:[],flexes:[]});rig.update();pixels();
            const frozen=shadowDraws;pixels();
            check(!rig.sun.shadow.autoUpdate&&shadowDraws===frozen,'rigid rebind not cached');
            return {gl:renderer.getContext().getError()};
          } finally {cleanup()}
        """
        )
        assert result["gl"] == 0
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)


@pytest.mark.parametrize("mode", ["rgb", "depth", "segmentation", "stereo"])
def test_actual_sensor_picture_does_not_consume_or_pollute_dirty_main_depth(viewer, mode):  # noqa: F811
    async def scenario(tab, _server):
        result = await tab.run(
            SETUP
            + PROBE
            + f"const mode={mode!r};"
            + """
          const {picture}=await import('./static/sensorpicture.js');
          const pip={...picture,cam:new THREE.PerspectiveCamera(),size:[0,0],flat:new Map(),lumas:new Map(),
            light:null,target:null,meshes:null,tone:null,depthMaterial:null,busy:false,last:0,renders:0};
          pip.api={renderer,scene,world:()=>world,bodyNode:()=>floorNode,desc:()=>desc};
          const canvas=document.createElement('canvas');
          const entry={body:0,mode,size:[64,64],fov_deg:45,range_m:10,baseline_m:.05,
            local_pos:camera.position.toArray(),local_quat:[camera.quaternion.w,...camera.quaternion.toArray().slice(0,3)]};
          try {
            rig.bind(world,[floorNode,cubeNode],desc);rig.update();pixels();
            const original=cube.material,first=shadowDraws;cubeNode.position.x+=.001;rig.update();
            const previous=renderer.getRenderTarget();
            check(rig.sun.shadow.needsUpdate,'probe is not dirty');pip.draw(entry,canvas);
            check(rig.sun.shadow.needsUpdate&&shadowDraws===first,'sensor consumed main depth');
            check(cube.material===original&&renderer.shadowMap.autoUpdate,'sensor state not restored');
            const end=performance.now()+5000;
            while(pip.busy&&performance.now()<end)await new Promise(r=>setTimeout(r,20));
            check(!pip.busy&&pip.renders===1,'sensor picture incomplete');
            check(rig.sun.shadow.needsUpdate&&shadowDraws===first,'completed sensor consumed main depth');
            check(cube.material===original&&renderer.shadowMap.autoUpdate,'completed sensor state not restored');
            check(renderer.getRenderTarget()===previous,'completed sensor target not restored');
            const actual=pixels();check(shadowDraws>first&&actual===fresh(),'main depth polluted by sensor material');
            return {gl:renderer.getContext().getError()};
          } finally {
            pip.light?.dispose();pip.target?.dispose();pip.depthMaterial?.dispose();
            for(const m of pip.flat.values())m.dispose();
            pip.tone?.material.dispose();pip.tone?.scene.traverse(o=>o.geometry?.dispose());cleanup();
          }
        """
        )
        assert result["gl"] == 0
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)


def test_main_context_restore_rebuilds_cached_shadow_and_preserves_rgb(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        await tab.run("""
          window.__shadowRGB=async()=>{
            const rf=window.__robotforge,{renderer,scene,camera}=rf.three(),T=await import('three');
            const target=new T.WebGLRenderTarget(160,100),previous=renderer.getRenderTarget();
            try{renderer.setRenderTarget(target);renderer.render(scene,camera);
              const data=new Uint8Array(160*100*4);renderer.readRenderTargetPixels(target,0,0,160,100,data);
              let hash=2166136261;for(const v of data)hash=Math.imul(hash^v,16777619);
              return {hash:hash>>>0,gl:renderer.getContext().getError()};
            }finally{renderer.setRenderTarget(previous);target.dispose()}
          };
          window.__robotforge.select(null);window.__robotforge.lookAt([1,-2,1],[0,0,.1]);
        """)
        await asyncio.sleep(0.6)
        before = await tab.run("return await window.__shadowRGB()")
        for _ in range(2):
            available = await tab.run("""
              window.__shadowContext=window.__robotforge.three().renderer.getContext().getExtension('WEBGL_lose_context');
              if(window.__shadowContext)window.__shadowContext.loseContext();return !!window.__shadowContext;
            """)
            if not available:
                pytest.skip("driver does not expose WEBGL_lose_context")
            await tab.until("window.__robotforge.renderedVersion===0")
            await tab.run("window.__shadowContext.restoreContext()")
            await tab.until(READY)
            assert await tab.run("return await window.__shadowRGB()") == before
        assert before["gl"] == 0
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.add_robot("rover", template="rover"))
