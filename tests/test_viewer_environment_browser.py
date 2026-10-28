"""Authored environment multipliers compose with shared scene illumination and surface shaders."""

from __future__ import annotations

import json

import pytest
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_startup_browser import READY


@pytest.mark.parametrize(
    "surface",
    [
        {},
        {"clearcoat": 0.4, "clearcoatRoughness": 0.2},
        {"texture": "brushed"},
        {"texture": "weave"},
        {"texture": "brushed", "textureStrength": 0},
    ],
    ids=["standard", "physical", "grain", "weave", "zero-grain"],
)
def test_authored_environment_matches_explicit_reference_without_cross_material_leaks(viewer, surface):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        result = await tab.run(
            """
          const T=await import('three'),{surfaceMaterial}=await import('./static/look.js');
          const {renderer,scene:main}=window.__robotforge.three(),gl=renderer.getContext();
          const extra=SURFACE,env=main.environment,materials=[],geometry=new T.SphereGeometry(.3,32,24);
          const scene=new T.Scene();scene.background=new T.Color(.12,.13,.14);
          scene.environment=env;scene.environmentIntensity=.85;
          const key=new T.DirectionalLight(0xffffff,1.1);key.position.set(-3,-4,5);scene.add(key);
          const camera=new T.PerspectiveCamera(35,1280/800,.01,100);
          camera.up.set(0,0,1);camera.position.set(0,-4,2);camera.lookAt(0,0,0);
          function material(scale){
            const m=surfaceMaterial('rf_probe',new T.Color(.3,.45,.6),1,
              {rf_probe:{roughness:.35,metalness:.7,...extra,env:scale}});
            materials.push(m);return m;
          }
          const scales=[0,1.45,.5],meshes=scales.map((scale,i)=>{
            const mesh=new T.Mesh(geometry,material(scale));mesh.position.x=(i-1)*.85;scene.add(mesh);return mesh;
          });
          function pixels(){
            const pack=gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING);
            try{
              gl.bindBuffer(gl.PIXEL_PACK_BUFFER,null);renderer.render(scene,camera);gl.finish();
              const a=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);
              gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,a);
              if(gl.getError()!==0)throw Error('environment reference readback failed');return a;
            }finally{gl.bindBuffer(gl.PIXEL_PACK_BUFFER,pack);}
          }
          function difference(a,b){
            let maximum=0,total=0,changed=0;
            for(let i=0;i<a.length;i++){
              const d=Math.abs(a[i]-b[i]);maximum=Math.max(maximum,d);total+=d;if(d)changed++;
            }return {maximum,mean:total/a.length,changed};
          }
          try{
            const implicit=pixels(),keys=meshes.map(o=>o.material.customProgramCacheKey());
            const repeated=difference(implicit,pixels());
            meshes.forEach((o,i)=>{o.material.envMap=env;o.material.envMapIntensity=.85*scales[i];});
            const explicit=difference(implicit,pixels());
            // Default-scale reference materials use Three's explicit map intensity directly.
            meshes.forEach((o,i)=>{
              const m=material(1);m.envMap=env;m.envMapIntensity=.85*scales[i];o.material=m;
            });
            const reference=difference(implicit,pixels());
            meshes.forEach(o=>{o.material=material(1);});
            const ignored=difference(implicit,pixels());
            scene.environment=null;
            const direct=pixels();meshes.forEach((o,i)=>{o.material=material(scales[i]);});
            const withoutEnvironment=difference(direct,pixels());
            return {reference,explicit,repeated,ignored,withoutEnvironment,keys};
          }finally{
            geometry.dispose();for(const m of materials)m.dispose();
            renderer.render(main,window.__robotforge.three().camera);
          }
        """.replace("SURFACE", json.dumps(surface))
        )
        assert result["reference"]["maximum"] <= 1 and result["reference"]["mean"] < 0.03, result
        assert result["explicit"]["maximum"] <= 1 and result["explicit"]["mean"] < 0.03, result
        assert result["repeated"]["changed"] == 0 and result["withoutEnvironment"]["changed"] == 0, result
        assert result["ignored"]["changed"] > 100, result
        assert len(set(result["keys"])) == 1, result
        assert not tab.errors

    viewer("laptop", "en", scenario)


def test_authored_surfaces_survive_lighting_rgb_sensor_and_context_restoration(viewer):  # noqa: F811
    import numpy as np
    from viewer_camera import settle_camera

    def world(pg):
        pg.add_robot("alloy", template="arm", params={"material": "aluminum", "reach": 0.7}, pos=[-0.6, 0])
        pg.add_robot("lab", template="arm", params={"style": "lab", "reach": 0.7}, pos=[0.6, 0])

    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        await tab.until("window.__robotforge.sensors().sensors.includes('alloy/wrist_cam')", timeout=10)
        await tab.run("window.__robotforge.showSensor('alloy/wrist_cam')")
        await tab.until(
            "window.__robotforge.sensors().picture==='alloy/wrist_cam' && window.__robotforge.sensors().pictures>0",
            timeout=10,
        )
        await tab.run("window.__robotforge.lookAt([2,-3,2],[0,0,.35])")
        await settle_camera(tab)
        saved = server.pg.store.file.read_bytes()
        state = {k: getattr(server.pg.session.data, k).copy() for k in ["qpos", "qvel", "ctrl"]}
        inventory = await tab.run("""
          const rf=window.__robotforge,found=[];
          rf.three().scene.traverse(o=>{
            if(o.isMesh&&o.userData.geom!==undefined&&o.material.envMapIntensity>1)
              found.push({geom:o.userData.geom,scale:o.material.envMapIntensity});
          });return {found,surfaces:rf.scene.surfaces};
        """)
        assert inventory["found"] and {x["scale"] for x in inventory["found"]} >= {1.25, 1.45}
        probe = """
          const rf=window.__robotforge,{renderer}=rf.three(),gl=renderer.getContext();
          const pack=gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING);let p;
          try{gl.bindBuffer(gl.PIXEL_PACK_BUFFER,null);p=rf.pixels();
            if(gl.getError()!==0)throw Error('surface lifecycle readback failed');}
          finally{gl.bindBuffer(gl.PIXEL_PACK_BUFFER,pack);}
          return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',p.data)))
            .map(x=>x.toString(16).padStart(2,'0')).join('');
        """
        await tab.run("document.getElementById('lighting-studio').click()")
        await settle_camera(tab)
        studio = await tab.run(probe)
        await tab.run("document.getElementById('lighting-daylight').click()")
        await settle_camera(tab)
        daylight = await tab.run(probe)
        assert daylight != studio
        pictures = await tab.eval("window.__robotforge.sensors().pictures")
        await tab.until(f"window.__robotforge.sensors().pictures>{pictures}", timeout=10)
        assert await tab.eval("window.__robotforge.three().lighting.preset()") == "daylight"
        assert await tab.run(probe) == daylight
        # Exercise the same temporary Studio boundary used by the actual RGB pass above.
        await tab.run("window.__restoreSurfaceLighting=window.__robotforge.three().lighting.sensorLighting()")
        assert await tab.run(probe) == studio
        await tab.run("window.__restoreSurfaceLighting()")
        assert await tab.run(probe) == daylight
        extension = await tab.run("""
          window.__environmentContext=window.__robotforge.three().renderer.getContext()
            .getExtension('WEBGL_lose_context');return !!window.__environmentContext;
        """)
        if not extension:
            pytest.skip("driver does not expose WEBGL_lose_context")
        await tab.run("window.__environmentContext.loseContext()")
        await tab.until("window.__robotforge.three().renderer.getContext().isContextLost()")
        await tab.run("window.__environmentContext.restoreContext()")
        await tab.until(f"{READY} && !window.__robotforge.three().renderer.getContext().isContextLost()")
        await settle_camera(tab)
        assert await tab.run(probe) == daylight
        await tab.run("document.getElementById('lighting-studio').click()")
        await settle_camera(tab)
        assert await tab.run(probe) == studio
        assert server.pg.store.file.read_bytes() == saved
        for k, before in state.items():
            np.testing.assert_array_equal(getattr(server.pg.session.data, k), before)
        assert not tab.errors

    viewer("laptop", "en", scenario, world=world)
