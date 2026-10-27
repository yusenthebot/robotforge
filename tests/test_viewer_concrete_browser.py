"""Concrete colour can evolve without changing relief, other styles or cached texture ownership."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_startup_browser import READY
from viewer_camera import settle_camera

BASELINE = json.loads((Path(__file__).parent / "fixtures" / "ground-tile-pixels-v1.json").read_text(encoding="utf-8"))


def concrete_world(pg):
    pg.set_floor("concrete")
    pg.add_robot("rover", template="rover")
    pg.add_terrain("ramp", pos=[3, 0], style="concrete")


def test_concrete_preserves_relief_other_tiles_mean_and_shared_preview(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await tab.until(READY)
        await settle_camera(tab)
        result = await tab.run("""
          const T=await import('three'),g=await import('./static/ground-look.js'),rows={};
          const materials=[];
          async function pixels(canvas){
            const data=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
            const hash=await crypto.subtle.digest('SHA-256',data);
            let linearMean=0;
            for(let i=0;i<data.length;i+=4){const v=data[i]/255;
              linearMean+=v<=.04045?v/12.92:Math.pow((v+.055)/1.055,2.4);}
            return {rgba_sha256:Array.from(new Uint8Array(hash)).map(x=>x.toString(16).padStart(2,'0')).join(''),
              linearMean:linearMean/(data.length/4),size:[canvas.width,canvas.height]};
          }
          try {
            for(const name of ['concrete','grass','wood','dirt','sand','rock','snow','tile','checker']){
              const m=g.groundMaterial(name,new T.Color(1,1,1),4);materials.push(m);
              rows[name]={color:await pixels(m.map.image),detail:await pixels(m.bumpMap.image),
                sharedDetail:m.bumpMap===m.roughnessMap,bumpScale:m.bumpScale};
            }
            const a=g.groundMaterial('concrete',new T.Color(1,1,1),4),
              b=g.groundMaterial('unknown',new T.Color(1,1,1),4);materials.push(a,b);
            a.map.repeat.set(2,3);a.bumpMap.offset.set(.2,.4);
            const preview=g.groundPreview('concrete',[1,1,1]);
            const expected=document.createElement('canvas');expected.width=96;expected.height=64;
            expected.getContext('2d').drawImage(a.map.image,0,0,96,64);
            return {rows,shared:a.map.source===b.map.source&&a.bumpMap.source===b.bumpMap.source,
              ownUV:b.map.repeat.toArray(),ownOffset:b.bumpMap.offset.toArray(),
              previewExact:preview.toDataURL()===expected.toDataURL(),
              fallback:g.groundStyle('external',{pattern:'speckle'}),
              woodRequests:performance.getEntriesByType('resource').filter(x=>x.name.includes('/ground-wood.js')).length};
          } finally {
            for(const m of materials){
              for(const tex of new Set([m.map,m.bumpMap,m.roughnessMap]))tex.dispose();m.dispose();
            }
          }
        """)
        for style, expected in BASELINE["styles"].items():
            row = result["rows"][style]
            assert row["color"]["size"] == row["detail"]["size"] == [256, 256]
            assert row["detail"]["rgba_sha256"] == expected["detail_rgba_sha256"]
            assert row["sharedDetail"]
            if style == "concrete":
                assert row["color"]["rgba_sha256"] != expected["color_rgba_sha256"]
                assert row["color"]["linearMean"] == pytest.approx(BASELINE["concrete_linear_mean"], rel=0.01)
                assert row["bumpScale"] == 0.0012
            else:
                assert row["color"]["rgba_sha256"] == expected["color_rgba_sha256"]
        assert result["shared"] and result["previewExact"]
        assert result["ownUV"] == [1, 1] and result["ownOffset"] == [0, 0]
        assert result["fallback"] == "concrete" and result["woodRequests"] == 0
        assert not tab.errors

    viewer("laptop", "en", scenario, world=concrete_world)


def test_concrete_colour_upload_survives_context_restoration(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        await settle_camera(tab)
        before = await tab.run("""
          window.__probeConcrete=async()=>{
            const T=await import('three'),{renderer,scene}=window.__robotforge.three();let source;
            scene.traverse(o=>{if(o.userData.isPlane&&o.material?.userData.groundSource)source=o.material.map});
            if(!source)throw Error('concrete plane is absent');
            const map=source.clone();map.repeat.set(1,1);map.offset.set(0,0);map.needsUpdate=true;
            const probe=new T.Scene(),material=new T.MeshBasicMaterial({map,toneMapped:false}),
              geometry=new T.PlaneGeometry(1,1),camera=new T.OrthographicCamera(-.5,.5,.5,-.5,.01,10);
            probe.add(new T.Mesh(geometry,material));camera.position.set(0,0,1);camera.lookAt(0,0,0);
            const target=new T.WebGLRenderTarget(64,64),previous=renderer.getRenderTarget();
            try {
              renderer.setRenderTarget(target);renderer.render(probe,camera);
              const data=new Uint8Array(64*64*4);renderer.readRenderTargetPixels(target,0,0,64,64,data);
              let hash=2166136261,min=255,max=0;
              for(let i=0;i<data.length;i++)if(i%4!==3){hash=Math.imul(hash^data[i],16777619)>>>0;
                min=Math.min(min,data[i]);max=Math.max(max,data[i]);}
              return {hash,min,max,gl:renderer.getContext().getError()};
            } finally {
              renderer.setRenderTarget(previous);target.dispose();geometry.dispose();material.dispose();map.dispose();
            }
          };
          return await window.__probeConcrete();
        """)
        assert before["max"] - before["min"] > 2 and before["gl"] == 0
        available = await tab.run("""
          window.__concreteContext=window.__robotforge.three().renderer.getContext().getExtension('WEBGL_lose_context');
          if(window.__concreteContext)window.__concreteContext.loseContext();return !!window.__concreteContext;
        """)
        if not available:
            pytest.skip("driver does not expose WEBGL_lose_context")
        await tab.until("window.__robotforge.renderedVersion===0")
        await tab.run("window.__concreteContext.restoreContext()")
        await tab.until(READY)
        assert await tab.run("return await window.__probeConcrete()") == before
        assert not tab.errors

    viewer("laptop", "en", scenario, world=concrete_world)
