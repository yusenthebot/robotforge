"""Optional natural grass images are ready atomically and never replace a newer world."""

from __future__ import annotations

import asyncio

import pytest
from starlette.responses import Response
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_startup_browser import READY, reload_instrumented

from robotforge.viewer import server as viewer_server

IMAGE_INSTRUMENT = """
p.decodes=0;p.held=0;p.failGrass=null;p.holdGrass=false;p.grassWarnings=[];
const warn=console.warn;
console.warn=(...args)=>{if(String(args[0]).includes('optional grass'))p.grassWarnings.push(args.map(String).join(' '));
  return warn(...args)};
const decode=HTMLImageElement.prototype.decode;
HTMLImageElement.prototype.decode=async function(){
  if(!this.src.startsWith('data:image/'))return decode.call(this);
  const channel=this.src.startsWith('data:image/jpeg;')?'color':'detail';
  p.decodes++;
  await decode.call(this);
  if(p.holdGrass&&channel==='color'){
    p.held++;await new Promise(resolve=>p.releaseGrass=()=>{p.holdGrass=false;resolve()});
  }
  if(p.failGrass===channel)throw new DOMException('injected decode failure','EncodingError');
  if(p.failGrass==='dimensions'&&channel==='detail')Object.defineProperty(this,'naturalWidth',{value:2});
};
p.groundRows=()=>{const rows=[];window.__robotforge.three().scene.traverse(o=>{
  if(o.isMesh&&o.material?.userData?.groundSource)rows.push({source:o.material.userData.groundSource,
    color:[o.material.map.image.width,o.material.map.image.height],
    detail:[o.material.bumpMap.image.width,o.material.bumpMap.image.height],
    sharedDetail:o.material.bumpMap===o.material.roughnessMap});
});return rows};
"""


def grass_world(pg):
    pg.set_floor("grass")
    pg.add_robot("rover", template="rover")


def test_natural_grass_preserves_sync_api_shares_images_and_matches_preview(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT)
        await tab.until(READY)
        rows = await tab.eval("window.__startup.groundRows()")
        assert rows and all(x["source"] == "natural-leafy-grass" for x in rows)
        assert all(x["color"] == [256, 256] and x["detail"] == [256, 256] and x["sharedDetail"] for x in rows)
        result = await tab.run("""
          const T=await import('three'),g=await import('./static/ground-look.js');
          const tile=await g.prepareGrassTextures(window.__robotforge.scene);
          const a=g.groundMaterial('grass',new T.Color(1,1,1),4,null,tile);
          const b=g.groundMaterial('grass',new T.Color(1,1,1),4,null,tile);
          const legacy=g.groundMaterial('grass',new T.Color(1,1,1),4);
          a.map.repeat.set(2,3);
          const preview=g.groundPreview('grass',[1,1,1]);
          const expected=document.createElement('canvas');expected.width=96;expected.height=64;
          expected.getContext('2d').drawImage(tile.color,0,0,96,64);
          const out={sync:a.isMaterial&&legacy.isMaterial,
            shared:a.map.source===b.map.source&&a.bumpMap.source===b.bumpMap.source,
            ownUV:b.map.repeat.toArray(),legacy:legacy.userData.groundSource,legacyWidth:legacy.map.image.width,
            preview:preview.toDataURL()===expected.toDataURL(),moduleRequests:performance.getEntriesByType('resource')
              .filter(x=>x.name.includes('/ground-grass.js')).length};
          for(const m of [a,b,legacy]){
            for(const tex of new Set([m.map,m.bumpMap,m.roughnessMap]))tex.dispose();m.dispose();
          }
          return out;
        """)
        assert result == {
            "sync": True,
            "shared": True,
            "ownUV": [1, 1],
            "legacy": "procedural",
            "legacyWidth": 256,
            "preview": True,
            "moduleRequests": 1,
        }
        assert await tab.eval("window.__startup.decodes") == 2
        assert not tab.errors

    viewer("laptop", "en", scenario, world=grass_world)


def test_nongrass_and_unused_catalog_or_checker_override_do_not_load_images(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await tab.until(READY)
        result = await tab.run("""
          const g=await import('./static/ground-look.js');
          g.groundPreview('grass',[1,1,1],false);
          const catalog=await g.prepareGrassTextures({geoms:[{material:'ground_concrete',type:'plane'}],
            grounds:{ground_grass:{pattern:'noise'},ground_concrete:{pattern:'noise'}}});
          const checker=await g.prepareGrassTextures({geoms:[{material:'ground_grass',type:'plane'}],
            grounds:{ground_grass:{pattern:'checker'}}});
          const requests=performance.getEntriesByType('resource').filter(x=>x.name.includes('/ground-grass.js')).length;
          return {catalog,checker,requests};
        """)
        assert result == {"catalog": None, "checker": None, "requests": 0}
        assert not tab.errors

    viewer("laptop", "en", scenario)


def test_first_world_draw_waits_for_both_grass_images(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT + "p.holdGrass=true;")
        await tab.until("window.__startup.held===1 && window.__startup.decodes===2")
        await asyncio.sleep(0.12)
        assert await tab.eval("window.__startup.draws.length") == 0
        assert not await tab.eval("document.getElementById('loading').hidden")
        await tab.run("window.__startup.releaseGrass()")
        await tab.until(READY)
        assert await tab.eval("window.__startup.groundRows().every(x=>x.source==='natural-leafy-grass')")
        assert await tab.eval("window.__startup.draws[0].loading")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=grass_world)


MIXED_INSTRUMENT = """
p.heldGrass=0;p.heldWood=0;p.releases=[];p.failStyle=null;p.holdBoth=false;p.firstDecode=null;
const decode=HTMLImageElement.prototype.decode;
HTMLImageElement.prototype.decode=async function(){
  if(!this.src.startsWith('data:image/'))return decode.call(this);
  p.firstDecode??=performance.now();
  await decode.call(this);
  if(!this.src.startsWith('data:image/jpeg;'))return;
  const style=this.naturalWidth===512?'wood':'grass';
  if(p.holdBoth){if(style==='wood')p.heldWood++;else p.heldGrass++;
    await new Promise(resolve=>p.releases.push(resolve));}
  if(p.failStyle===style)throw new DOMException('injected mixed color failure','EncodingError');
};
p.releaseBoth=()=>{p.holdBoth=false;for(const resolve of p.releases.splice(0))resolve()};
p.mixedRows=()=>{const rows=[];window.__robotforge.three().scene.traverse(o=>{
  if(o.isMesh&&o.material?.userData?.groundSource)rows.push(o.material.userData.groundSource);
});return [...new Set(rows)].sort()};
"""


def mixed_world(pg):
    pg.set_floor("wood")
    pg.add_terrain("hills", style="grass", pos=[2, 0], params={"size": [2, 2]})
    pg.add_robot("rover", template="rover")


@pytest.mark.parametrize("failure", [None, "grass", "wood"])
def test_mixed_world_prepares_independent_pairs_and_isolates_failures(viewer, failure):  # noqa: F811
    async def scenario(tab, _server):
        await reload_instrumented(tab, MIXED_INSTRUMENT + f"p.failStyle={failure!r};".replace("None", "null"))
        await tab.until(READY)
        expected = {
            None: ["natural-leafy-grass", "natural-oak"],
            "grass": ["natural-oak", "procedural"],
            "wood": ["natural-leafy-grass", "procedural"],
        }[failure]
        assert await tab.eval("window.__startup.mixedRows()") == expected
        assert await tab.eval("window.__robotforge.three().renderer.getContext().getError()") == 0
        assert not tab.errors

    viewer("laptop", "en", scenario, world=mixed_world)


def test_mixed_deadlines_run_concurrently_and_late_cache_does_not_replace_scene(viewer):  # noqa: F811
    async def scenario(tab, server):
        await reload_instrumented(tab, MIXED_INSTRUMENT + "p.holdBoth=true;")
        await tab.until("window.__startup.heldGrass===1&&window.__startup.heldWood===1")
        assert await tab.eval("window.__startup.draws.length") == 0
        await tab.until(READY)
        elapsed = await tab.eval("performance.now()-window.__startup.firstDecode")
        assert 800 < elapsed < 1700  # two serial1000ms deadlines would exceed this bound
        assert await tab.eval("window.__startup.mixedRows()") == ["procedural"]
        await tab.run("window.__startup.releaseBoth()")
        await asyncio.sleep(0.15)
        assert await tab.eval("window.__startup.mixedRows()") == ["procedural"]
        server.pg.update_robot("rover", pos=[1, 0])
        version = server.pg.session.version
        await tab.until(f"({READY})&&window.__robotforge.scene.version==={version}")
        assert await tab.eval("window.__startup.mixedRows()") == ["natural-leafy-grass", "natural-oak"]
        assert not tab.errors

    viewer("laptop", "en", scenario, world=mixed_world)


@pytest.mark.parametrize("failure", ["color", "detail", "dimensions"])
def test_either_invalid_image_uses_atomic_fallback_and_a_later_scene_can_retry(viewer, failure):  # noqa: F811
    async def scenario(tab, server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT + f"p.failGrass={failure!r};")
        await tab.until(READY)
        rows = await tab.eval("window.__startup.groundRows()")
        assert rows and all(x["source"] == "procedural" and x["color"] == [256, 256] for x in rows)
        assert len(await tab.eval("window.__startup.grassWarnings")) == 1
        await tab.run("window.__startup.failGrass=null")
        server.pg.new_world("retry", floor="grass")
        version = server.pg.session.version
        await tab.until(f"({READY}) && window.__robotforge.scene.version==={version}")
        assert await tab.eval("window.__startup.groundRows().every(x=>x.source==='natural-leafy-grass')")
        assert await tab.eval("window.__startup.decodes") == 4
        assert not tab.errors

    viewer("laptop", "en", scenario, world=grass_world)


def test_missing_optional_module_keeps_scene_available_and_warns_once(viewer, monkeypatch):  # noqa: F811
    original = viewer_server.RevalidatingStaticFiles.get_response

    async def response(self, path, scope):
        if path == "ground-grass.js":
            return Response("missing optional image module", status_code=404)
        return await original(self, path, scope)

    monkeypatch.setattr(viewer_server.RevalidatingStaticFiles, "get_response", response)

    async def scenario(tab, server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT)
        await tab.until(READY)
        assert await tab.eval("window.__startup.groundRows().every(x=>x.source==='procedural')")
        server.pg.new_world("fallback_again", floor="grass")
        version = server.pg.session.version
        await tab.until(f"({READY}) && window.__robotforge.scene.version==={version}")
        assert len(await tab.eval("window.__startup.grassWarnings")) == 1
        assert await tab.eval("window.__startup.decodes") == 0
        assert not tab.errors

    viewer("laptop", "en", scenario, world=grass_world)


def test_timeout_does_not_replace_installed_fallback_when_images_arrive_late(viewer):  # noqa: F811
    async def scenario(tab, server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT + "p.holdGrass=true;")
        await tab.until("window.__startup.held===1")
        await tab.until(READY)  # real1000ms optional-resource deadline, no fake clock
        assert await tab.eval("window.__startup.groundRows().every(x=>x.source==='procedural')")
        await tab.run("window.__startup.releaseGrass()")
        await asyncio.sleep(0.15)
        assert await tab.eval("window.__startup.groundRows().every(x=>x.source==='procedural')")
        assert len(await tab.eval("window.__startup.grassWarnings")) == 1
        server.pg.new_world("cached_later", floor="grass")
        version = server.pg.session.version
        await tab.until(f"({READY}) && window.__robotforge.scene.version==={version}")
        assert await tab.eval("window.__startup.groundRows().every(x=>x.source==='natural-leafy-grass')")
        assert await tab.eval("window.__startup.decodes") == 2
        assert not tab.errors

    viewer("laptop", "en", scenario, world=grass_world)


def test_new_nongrass_scene_wins_over_old_delayed_grass_scene(viewer):  # noqa: F811
    async def scenario(tab, server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT + "p.holdGrass=true;")
        await tab.until("window.__startup.held===1")
        server.pg.new_world("new_plain", floor="concrete")
        version = server.pg.session.version
        await tab.until(f"({READY}) && window.__robotforge.scene.version==={version}")
        await tab.run("window.__startup.releaseGrass()")
        await asyncio.sleep(0.15)
        assert await tab.eval("window.__robotforge.scene.version") == version
        assert await tab.eval("window.__startup.groundRows().every(x=>x.source==='procedural')")
        assert await tab.eval(f"window.__startup.draws.every(x=>x.version==={version})")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=grass_world)


def test_preview_before_scene_load_does_not_poison_natural_tile_cache(viewer):  # noqa: F811
    async def scenario(tab, server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT + "p.holdGrass=true;")
        await tab.until(READY)  # initial scene is nongrass
        await tab.run("""
          const g=await import('./static/ground-look.js');
          window.__grassPreview=g.groundPreview('grass',[1,1,1]);
          document.body.append(window.__grassPreview);
        """)
        await tab.until("window.__startup.held===1")
        await tab.run("window.__startup.releaseGrass()")
        server.pg.new_world("grass_after_preview", floor="grass")
        version = server.pg.session.version
        await tab.until(f"({READY}) && window.__robotforge.scene.version==={version}")
        assert await tab.eval("window.__startup.groundRows().every(x=>x.source==='natural-leafy-grass')")
        assert await tab.eval("window.__startup.decodes") == 2
        assert not tab.errors

    viewer("laptop", "en", scenario)


def test_natural_images_survive_webgl_context_restoration(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await reload_instrumented(tab, IMAGE_INSTRUMENT)
        await tab.until(READY)
        await asyncio.sleep(0.3)
        before = await tab.eval("window.__startup.groundRows()")
        before_pixels = await tab.run("""
          window.__startup.probeGrass=async()=>{
            const T=await import('three'),{renderer,scene}=window.__robotforge.three();let source;
            scene.traverse(o=>{if(o.material?.userData?.groundSource==='natural-leafy-grass')source=o.material.map});
            if(!source)throw Error('natural grass is absent');
            const map=source.clone();map.repeat.set(1,1);map.offset.set(0,0);map.needsUpdate=true;
            const probe=new T.Scene(),material=new T.MeshBasicMaterial({map,toneMapped:false});
            const geometry=new T.PlaneGeometry(1,1);probe.add(new T.Mesh(geometry,material));
            const camera=new T.OrthographicCamera(-.5,.5,.5,-.5,.01,10);
            camera.position.set(0,0,1);camera.lookAt(0,0,0);
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
          return await window.__startup.probeGrass();
        """)
        assert before_pixels["max"] - before_pixels["min"] > 10 and before_pixels["gl"] == 0
        available = await tab.run("""
          window.__groundContext=window.__robotforge.three().renderer.getContext().getExtension('WEBGL_lose_context');
          if(window.__groundContext)window.__groundContext.loseContext();return !!window.__groundContext;
        """)
        if not available:
            pytest.skip("driver does not expose WEBGL_lose_context")
        await tab.until("window.__robotforge.renderedVersion===0")
        await tab.run("window.__groundContext.restoreContext()")
        await tab.until(READY)
        assert await tab.eval("window.__startup.groundRows()") == before
        assert await tab.eval("window.__startup.decodes") == 2
        assert await tab.run("return await window.__startup.probeGrass()") == before_pixels
        assert not tab.errors

    viewer("laptop", "en", scenario, world=grass_world)
