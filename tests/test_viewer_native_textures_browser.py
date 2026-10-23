"""Native mesh images retain interior graphics and survive failed/superseded scene loads."""

from __future__ import annotations

import asyncio

import pytest
from PIL import Image, ImageDraw
from test_viewer_creation_browser import click
from test_viewer_layout import pytestmark, viewer  # noqa: F401

READY = "!!window.__robotforge?.scene && window.__robotforge.renderedVersion === window.__robotforge.scene.version"


@pytest.fixture
def image_robot(tmp_path):
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, 45, 45), fill="red")  # every corner is white; vertex sampling cannot see this.
    draw.rectangle((0, 0, 12, 12), fill="blue")
    draw.rectangle((51, 51, 63, 63), fill="green")
    image.save(tmp_path / "label.png")
    xml = tmp_path / "labelled.xml"
    xml.write_text("""<mujoco><asset>
      <texture name="label" type="2d" file="label.png"/>
      <material name="printed" texture="label" rgba="1 1 1 1"/>
      <mesh name="slab" vertex="-.5 -.5 0 .5 -.5 0 .5 .5 0 -.5 .5 0 -.5 -.5 -.02 .5 -.5 -.02 .5 .5 -.02 -.5 .5 -.02"
        face="0 1 2 0 2 3 4 6 5 4 7 6 0 4 5 0 5 1 1 5 6 1 6 2 2 6 7 2 7 3 3 7 4 3 4 0"
        texcoord="0 0 1 0 1 1 0 1 0 0 1 0 1 1 0 1"/>
      </asset><worldbody><body name="panel">
      <geom name="panel" type="mesh" mesh="slab" material="printed"/></body></worldbody>
      </mujoco>""")
    return xml


def test_imported_mesh_draws_its_interior_image_and_authored_uv_orientation(viewer, image_robot):  # noqa: F811
    async def scenario(tab, _server):
        await tab.until(READY)
        result = await tab.run("""
          const T=await import('three'), rf=window.__robotforge, {renderer,scene}=rf.three();
          const entry=rf.scene.geoms.find(g=>g.name==='labelled/panel');
          let actual;scene.updateMatrixWorld(true);
          scene.traverse(o=>{if(o.userData.geom===entry.id&&o.isMesh)actual=o});
          if(!actual?.material.map)throw Error('native model image is missing');
          const basic=new T.MeshBasicMaterial({map:actual.material.map,color:actual.material.color});
          const copy=new T.Mesh(actual.geometry,basic);
          copy.applyMatrix4(actual.matrixWorld);
          const probe=new T.Scene();probe.add(copy);
          const camera=new T.OrthographicCamera(-.5,.5,.5,-.5,.01,10);camera.position.set(0,0,2);camera.lookAt(0,0,0);
          const target=new T.WebGLRenderTarget(64,64,{type:T.FloatType}), previous=renderer.getRenderTarget();
          try {
            renderer.setRenderTarget(target);renderer.render(probe,camera);
            const data=new Float32Array(64*64*4);renderer.readRenderTargetPixels(target,0,0,64,64,data);
            const at=(x,y)=>Array.from(data.subarray((y*64+x)*4,(y*64+x)*4+3));
            return {center:at(32,32),lower:at(6,6),upper:at(57,57),
              vertexColors:actual.material.vertexColors,uv:actual.geometry.attributes.uv.count,
              imageSize:[actual.material.map.image.width,actual.material.map.image.height],gl:renderer.getContext().getError()};
          } finally {renderer.setRenderTarget(previous);target.dispose();copy.material.dispose()}
        """)
        assert result["center"][0] > 0.8 and max(result["center"][1:]) < 0.05, result
        assert result["lower"][2] > 0.8 and result["lower"][0] < 0.05, result
        assert result["upper"][1] > 0.1 and result["upper"][0] < 0.05, result
        assert result["imageSize"] == [64, 64] and result["uv"] > 0
        assert result["vertexColors"] is False and result["gl"] == 0
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.add_robot("labelled", file=str(image_robot)))


def test_shared_mesh_materials_keep_distinct_images_plain_color_and_alpha(viewer, image_robot):  # noqa: F811
    Image.new("RGBA", (4, 4), (0, 255, 0, 96)).save(image_robot.parent / "glass.png")
    xml = (
        image_robot.read_text(encoding="utf-8")
        .replace(
            '<texture name="label"',
            '<texture name="glass" type="2d" nchannel="4" file="glass.png"/>'
            '<material name="glass" texture="glass" texrepeat="2 3"/>'
            '<texture name="label"',
        )
        .replace(
            '<geom name="panel" type="mesh" mesh="slab" material="printed"/>',
            '<geom name="panel" type="mesh" mesh="slab" material="printed"/>'
            '<geom name="glass" type="mesh" mesh="slab" material="glass" pos="2 0 0"/>'
            '<geom name="plain" type="mesh" mesh="slab" rgba="1 1 0 1" pos="4 0 0"/>',
        )
    )
    image_robot.write_text(xml, encoding="utf-8")

    async def scenario(tab, _server):
        await tab.until(READY)
        result = await tab.run("""
          const rf=window.__robotforge,{scene,renderer}=rf.three(),objects={};
          scene.traverse(o=>{if(!o.isMesh||!Number.isInteger(o.userData.geom))return;
            const name=rf.scene.geoms.find(g=>g.id===o.userData.geom)?.name;
            if(name?.startsWith('labelled/'))objects[name.split('/').pop()]=o});
          const {panel,glass,plain}=objects;
          return {shared:panel.geometry===glass.geometry&&glass.geometry===plain.geometry,
            distinct:panel.material.map!==glass.material.map,
            image:Array.from(glass.material.map.image.data.slice(0,4)),repeat:glass.material.map.repeat.toArray(),
            transparent:glass.material.transparent,depthWrite:glass.material.depthWrite,
            plainMap:plain.material.map,plainVertexColors:plain.material.vertexColors,
            plainColor:plain.material.color.toArray(),gl:renderer.getContext().getError()};
        """)
        assert result == {
            "shared": True,
            "distinct": True,
            "image": [0, 255, 0, 96],
            "repeat": [2, 3],
            "transparent": True,
            "depthWrite": False,
            "plainMap": None,
            "plainVertexColors": False,
            "plainColor": [1, 1, 0],
            "gl": 0,
        }
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.add_robot("labelled", file=str(image_robot)))


def test_native_texture_transforms_share_pixels_and_reject_bad_bundles(viewer):  # noqa: F811
    async def scenario(tab, _server):
        result = await tab.run("""
          const T=await import('three'), {modelTextures}=await import('./static/model-textures.js');
          const entries=[{id:0,width:2,height:1,offset:0,length:8,alpha:false}];
          const get=modelTextures(entries,new Uint8Array([255,0,0,255,0,0,255,255]).buffer,16);
          const a=get({id:0,repeat:[1,1]}),b=get({id:0,repeat:[2,3]});
          const invalid=[];
          for(const [e,buffer] of [[entries,new ArrayBuffer(4)],[entries,new ArrayBuffer(4*512*512+1)],
            [[{...entries[0],width:513}],new ArrayBuffer(8)]]){
            try{modelTextures(e,buffer,16);invalid.push(false)}catch{invalid.push(true)}
          }
          let badRepeat=false;try{get({id:0,repeat:[NaN,1]})}catch{badRepeat=true}
          const result={sameSource:a.source===b.source,sameMap:a===get({id:0,repeat:[1,1]}),
            distinctMap:a!==b,repeat:b.repeat.toArray(),srgb:a.colorSpace===T.SRGBColorSpace,
            flip:a.flipY,mips:a.generateMipmaps,anisotropy:a.anisotropy,invalid,badRepeat};
          a.dispose();b.dispose();return result;
        """)
        assert result == {
            "sameSource": True,
            "sameMap": True,
            "distinctMap": True,
            "repeat": [2, 3],
            "srgb": True,
            "flip": False,
            "mips": True,
            "anisotropy": 4,
            "invalid": [True, True, True],
            "badRepeat": True,
        }
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)


def test_native_image_decode_matches_legacy_srgb_values_on_the_gpu(viewer):  # noqa: F811
    async def scenario(tab, _server):
        result = await tab.run("""
          const T=await import('three'), {modelTextures}=await import('./static/model-textures.js');
          const {renderer}=window.__robotforge.three(), previous=renderer.getRenderTarget();
          const entries=[0,1].map(id=>({id,
            width:1,height:1,offset:id*4,length:4,alpha:false}));
          const get=modelTextures(entries,new Uint8Array([128,128,128,255,128,64,192,255]).buffer,4);
          const geometry=new T.PlaneGeometry(2,2),camera=new T.OrthographicCamera(-1,1,1,-1,.1,10);
          camera.position.z=2;
          const target=new T.WebGLRenderTarget(4,4,{type:T.FloatType});
          const values=[];
          try {
            renderer.setRenderTarget(target);
            for(let id=0;id<2;id++){
              const map=get({id,repeat:[1,1]}),material=new T.MeshBasicMaterial({map});
              const scene=new T.Scene();scene.add(new T.Mesh(geometry,material));renderer.render(scene,camera);
              const pixel=new Float32Array(4);renderer.readRenderTargetPixels(target,2,2,1,1,pixel);
              values.push(Array.from(pixel.subarray(0,3)));material.dispose();map.dispose();
            }
            return {values,gl:renderer.getContext().getError()};
          } finally {renderer.setRenderTarget(previous);target.dispose();geometry.dispose()}
        """)

        def expected(v):
            return ((v / 255 + 0.055) / 1.055) ** 2.4

        assert result["values"][0] == pytest.approx([expected(128)] * 3, abs=0.002)
        assert result["values"][1] == pytest.approx([expected(v) for v in (128, 64, 192)], abs=0.002)
        assert result["gl"] == 0 and not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)


TEXTURE_FETCH = """
window.__nativeLoad={hold:false,fail:false,releases:[],requests:[]};
const originalFetch=window.fetch;
window.fetch=async(...args)=>{
 if(!String(args[0]).includes('texture/'))return originalFetch(...args);
 const p=window.__nativeLoad;p.requests.push(String(args[0]));
 if(p.fail)return new Response('injected texture failure',{status:503});
 const result=await originalFetch(...args);
 if(p.hold)await new Promise(resolve=>p.releases.push(resolve));
 return result;
};
"""


def test_failed_texture_fetch_preserves_old_world_and_native_retry_recovers(viewer, image_robot):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=TEXTURE_FETCH)
        await tab.page.call("Page.reload")
        await tab.until(READY + " && !!window.__nativeLoad")
        previous = await tab.eval("window.__robotforge.scene.version")
        textures = await tab.eval("window.__robotforge.three().renderer.info.memory.textures")
        await tab.run("window.__nativeLoad.fail=true")
        server.pg.update_robot("labelled", pos=[1, 0])
        version = server.pg.session.version
        await tab.until("!document.getElementById('scene-retry').hidden")
        assert await tab.eval("window.__robotforge.scene.version") == previous
        assert await tab.eval("window.__robotforge.three().renderer.info.memory.textures") == textures
        await tab.run("window.__nativeLoad.fail=false")
        await click(tab, "#scene-retry")
        await tab.until(READY + f" && window.__robotforge.scene.version==={version}")
        assert await tab.eval("window.__robotforge.three().renderer.info.memory.textures") == textures
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.add_robot("labelled", file=str(image_robot)))


def test_superseded_texture_fetch_and_switches_do_not_keep_old_images(viewer, image_robot):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=TEXTURE_FETCH)
        await tab.page.call("Page.reload")
        await tab.until(READY + " && !!window.__nativeLoad")
        textures = await tab.eval("window.__robotforge.three().renderer.info.memory.textures")
        await tab.run("window.__nativeLoad.hold=true")
        server.pg.update_robot("labelled", pos=[1, 0])
        await tab.until("window.__nativeLoad.releases.length===1")
        server.pg.update_robot("labelled", pos=[2, 0])
        version = server.pg.session.version
        await tab.until("window.__nativeLoad.releases.length===2")
        await tab.run("window.__nativeLoad.hold=false;window.__nativeLoad.releases.pop()()")
        await tab.until(READY + f" && window.__robotforge.scene.version==={version}")
        await tab.run("window.__nativeLoad.releases.pop()()")
        await asyncio.sleep(0.15)
        assert await tab.eval("window.__robotforge.scene.version") == version
        assert await tab.eval("window.__robotforge.three().renderer.info.memory.textures") == textures
        for _ in range(3):
            server.pg.new_world("empty_native")
            version = server.pg.session.version
            await tab.until(READY + f" && window.__robotforge.scene.version==={version}")
            assert await tab.eval("window.__robotforge.three().renderer.info.memory.textures") == textures - 1
            server.pg.open_world("qa")
            version = server.pg.session.version
            await tab.until(READY + f" && window.__robotforge.scene.version==={version}")
            assert await tab.eval("window.__robotforge.three().renderer.info.memory.textures") == textures
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.add_robot("labelled", file=str(image_robot)))
