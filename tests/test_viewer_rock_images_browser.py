"""Natural rock installs atomically; stochastic sampling preserves the metric shader."""

from __future__ import annotations

import asyncio
from threading import Event

import pytest
from starlette.responses import Response
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_startup_browser import READY, reload_instrumented

from robotforge.viewer import server as viewer_server

IMAGE_INSTRUMENT = """
p.decodes=0;p.held=0;p.failRock=null;p.holdRock=false;p.rockWarnings=[];
const warn=console.warn;
console.warn=(...args)=>{if(String(args[0]).includes('optional rock'))p.rockWarnings.push(args.map(String).join(' '));
  return warn(...args)};
const decode=HTMLImageElement.prototype.decode;
HTMLImageElement.prototype.decode=async function(){
  if(!this.src.startsWith('data:image/'))return decode.call(this);
  const channel=this.src.startsWith('data:image/jpeg;')?'color':'detail';
  p.decodes++;await decode.call(this);
  if(p.holdRock&&channel==='color'){
    p.held++;await new Promise(resolve=>p.releaseRock=()=>{p.holdRock=false;resolve()});
  }
  if(p.failRock===channel)throw new DOMException('injected decode failure','EncodingError');
  if(p.failRock==='dimensions'&&channel==='detail')Object.defineProperty(this,'naturalWidth',{value:2});
};
p.groundRows=()=>{const rows=[];window.__robotforge.three().scene.traverse(o=>{
  if(o.isMesh&&o.material?.userData?.groundSource)rows.push({source:o.material.userData.groundSource,
    color:[o.material.map.image.width,o.material.map.image.height],
    detail:[o.material.bumpMap.image.width,o.material.bumpMap.image.height],
    sharedDetail:o.material.bumpMap===o.material.roughnessMap,key:o.material.customProgramCacheKey()});
});return rows};
p.hasGroundSource=source=>{const rows=p.groundRows();return rows.length>0&&rows.every(x=>x.source===source)};
"""


def rock_world(pg):
    pg.set_floor("rock")
    pg.add_robot("rover", template="rover")


def test_natural_rock_shares_sources_keeps_legacy_api_preview_and_compiles_metric_shader(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT)
        await tab.until(READY)
        rows = await tab.eval("window.__startup.groundRows()")
        assert rows and all(row["source"] == "natural-rock-face-03" for row in rows)
        assert all(row["sharedDetail"] and row["color"] == row["detail"] == [256, 256] for row in rows)
        assert all(row["key"] == "rf-ground-metric-v1:square-quintic-rock-v1" for row in rows)
        result = await tab.run(r"""
          const T=await import('three'),g=await import('./static/ground-look.js');
          const {stochasticRock}=await import('./static/ground-stochastic.js');
          const tile=await g.prepareRockTextures(window.__robotforge.scene);
          const a=g.groundMaterial('rock',new T.Color(1,1,1),4,null,null,tile);
          const b=g.groundMaterial('rock',new T.Color(1,1,1),4,null,null,tile);
          const legacy=g.groundMaterial('rock',new T.Color(1,1,1),4);
          const hook=a.onBeforeCompile,key=a.customProgramCacheKey;
          stochasticRock(a);const once=a.onBeforeCompile===hook&&a.customProgramCacheKey===key;
          let shader=null;a.onBeforeCompile=function(s,r){hook.call(this,s,r);shader=s.fragmentShader};
          a.map.repeat.set(2,3);a.bumpMap.repeat.copy(a.map.repeat);
          const preview=g.groundPreview('rock',[1,1,1]);
          const expected=document.createElement('canvas');expected.width=96;expected.height=64;
          expected.getContext('2d').drawImage(tile.color,0,0,96,64);
          const {renderer}=window.__robotforge.three(),scene=new T.Scene();
          const geometry=new T.PlaneGeometry(1,1);scene.add(new T.Mesh(geometry,a));
          scene.add(new T.AmbientLight(0xffffff,2));
          const camera=new T.OrthographicCamera(-.5,.5,.5,-.5,.01,10);
          camera.position.set(0,0,1);camera.lookAt(0,0,0);
          const target=new T.WebGLRenderTarget(32,32),previous=renderer.getRenderTarget();
          try {
            renderer.setRenderTarget(target);renderer.render(scene,camera);
            const pixels=new Uint8Array(32*32*4);renderer.readRenderTargetPixels(target,0,0,32,32,pixels);
            const values=Array.from(pixels).filter((_,i)=>i%4!==3);
            return {sync:a.isMaterial&&legacy.isMaterial,once,
              shared:a.map.source===b.map.source&&a.bumpMap.source===b.bumpMap.source,
              ownUV:b.map.repeat.toArray(),legacy:legacy.userData.groundSource,
              legacyKey:legacy.customProgramCacheKey(),bump:a.bumpScale,
              preview:preview.toDataURL()===expected.toDataURL(),
              samples:(shader.match(/rfRockSample\(/g)||[]).length,
              gradients:(shader.match(/textureGrad\(/g)||[]).length,metric:shader.includes('rfMetricBump('),
              range:Math.max(...values)-Math.min(...values),gl:renderer.getContext().getError(),
              requests:performance.getEntriesByType('resource').filter(x=>x.name.includes('/ground-rock.js')).length,
              shaderRequests:performance.getEntriesByType('resource')
                .filter(x=>x.name.includes('/ground-stochastic.js')).length};
          } finally {
            renderer.setRenderTarget(previous);target.dispose();geometry.dispose();
            for(const material of [a,b,legacy]){
              for(const tex of new Set([material.map,material.bumpMap,material.roughnessMap]))tex.dispose();
              material.dispose();
            }
          }
        """)
        assert result.pop("range") > 2, result
        assert result == {
            "sync": True,
            "once": True,
            "shared": True,
            "ownUV": [1, 1],
            "legacy": "procedural",
            "legacyKey": "rf-ground-metric-v1",
            "bump": 0.003,
            "preview": True,
            "samples": 7,
            "gradients": 4,
            "metric": True,
            "gl": 0,
            "requests": 1,
            "shaderRequests": 1,
        }
        assert await tab.eval("window.__startup.decodes") == 2
        assert await tab.eval("window.__startup.rockWarnings") == []
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=rock_world)


def test_unused_rock_catalog_checker_and_flat_preview_do_not_load_images(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await tab.until(READY)
        result = await tab.run("""
          const g=await import('./static/ground-look.js');g.groundPreview('rock',[1,1,1],false);
          const catalog=await g.prepareRockTextures({geoms:[{material:'ground_concrete',type:'plane'}],
            grounds:{ground_rock:{pattern:'noise'},ground_concrete:{pattern:'noise'}}});
          const checker=await g.prepareRockTextures({geoms:[{material:'ground_rock',type:'plane'}],
            grounds:{ground_rock:{pattern:'checker'}}});
          return {catalog,checker,requests:performance.getEntriesByType('resource')
            .filter(x=>x.name.includes('/ground-rock.js')).length,
            shaderRequests:performance.getEntriesByType('resource')
              .filter(x=>x.name.includes('/ground-stochastic.js')).length};
        """)
        assert result == {"catalog": None, "checker": None, "requests": 0, "shaderRequests": 0}
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario)


def test_first_world_waits_for_the_complete_rock_pair(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT + "p.holdRock=true;")
        await tab.until("window.__startup.held===1&&window.__startup.decodes===2")
        await asyncio.sleep(0.12)
        assert await tab.eval("window.__startup.draws.length") == 0
        await tab.run("window.__startup.releaseRock()")
        await tab.until(READY)
        assert await tab.eval("window.__startup.hasGroundSource('natural-rock-face-03')")
        assert await tab.eval("window.__startup.draws[0].loading")
        assert await tab.eval("window.__startup.rockWarnings") == []
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=rock_world)


@pytest.mark.parametrize("failure", ["color", "detail", "dimensions"])
def test_invalid_pair_falls_back_atomically_and_next_scene_retries(viewer, failure):  # noqa: F811
    async def scenario(tab, server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT + f"p.failRock={failure!r};")
        await tab.until(READY)
        rows = await tab.eval("window.__startup.groundRows()")
        assert rows and all(row["source"] == "procedural" and row["key"] == "rf-ground-metric-v1" for row in rows)
        assert len(await tab.eval("window.__startup.rockWarnings")) == 1
        await tab.run("window.__startup.failRock=null")
        server.pg.new_world("retry_rock", floor="rock")
        version = server.pg.session.version
        await tab.until(f"({READY})&&window.__robotforge.scene.version==={version}")
        assert await tab.eval("window.__startup.hasGroundSource('natural-rock-face-03')")
        assert await tab.eval("window.__startup.decodes") == 4
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=rock_world)


def test_deadline_installs_fallback_then_only_caches_late_images(viewer):  # noqa: F811
    async def scenario(tab, server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT + "p.holdRock=true;")
        await tab.until("window.__startup.held===1&&window.__startup.decodes===2")
        assert await tab.eval("window.__startup.draws.length") == 0
        await tab.until(READY)
        assert await tab.eval("window.__startup.hasGroundSource('procedural')")
        await tab.run("window.__startup.releaseRock()")
        await asyncio.sleep(0.15)
        assert await tab.eval("window.__startup.hasGroundSource('procedural')")
        server.pg.new_world("cached_rock", floor="rock")
        version = server.pg.session.version
        await tab.until(f"({READY})&&window.__robotforge.scene.version==={version}")
        assert await tab.eval("window.__startup.hasGroundSource('natural-rock-face-03')")
        assert await tab.eval("window.__startup.decodes") == 2
        assert len(await tab.eval("window.__startup.rockWarnings")) == 1
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=rock_world)


def test_new_world_wins_over_old_delayed_rock_pair(viewer):  # noqa: F811
    async def scenario(tab, server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT + "p.holdRock=true;")
        await tab.until("window.__startup.held===1")
        server.pg.new_world("new_plain", floor="concrete")
        version = server.pg.session.version
        await tab.until(f"({READY})&&window.__robotforge.scene.version==={version}")
        await tab.run("window.__startup.releaseRock()")
        await asyncio.sleep(0.15)
        assert await tab.eval("window.__robotforge.scene.version") == version
        assert await tab.eval("window.__startup.hasGroundSource('procedural')")
        assert await tab.eval(
            f"window.__startup.draws.length>0&&window.__startup.draws.every(x=>x.version==={version})"
        )
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=rock_world)


@pytest.mark.parametrize("missing", ["ground-rock.js", "ground-stochastic.js"])
def test_missing_optional_rock_module_warns_once_and_keeps_fallback(viewer, monkeypatch, missing):  # noqa: F811
    original = viewer_server.RevalidatingStaticFiles.get_response

    async def response(self, path, scope):
        if path == missing:
            return Response("missing optional rock module", status_code=404)
        return await original(self, path, scope)

    monkeypatch.setattr(viewer_server.RevalidatingStaticFiles, "get_response", response)

    async def scenario(tab, server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT)
        await tab.until(READY)
        assert await tab.eval("window.__startup.hasGroundSource('procedural')")
        server.pg.new_world("fallback_again", floor="rock")
        version = server.pg.session.version
        await tab.until(f"({READY})&&window.__robotforge.scene.version==={version}")
        assert await tab.eval("window.__startup.hasGroundSource('procedural')")
        assert await tab.eval("window.__startup.groundRows().every(x=>x.key==='rf-ground-metric-v1')")
        assert len(await tab.eval("window.__startup.rockWarnings")) == 1
        assert await tab.eval("window.__startup.decodes") == 0
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=rock_world)


@pytest.mark.parametrize("held", ["ground-rock.js", "ground-stochastic.js"])
def test_either_optional_import_uses_the_rock_deadline_and_only_warms_late_cache(  # noqa: F811
    viewer,  # noqa: F811
    monkeypatch,
    held,
):
    original = viewer_server.RevalidatingStaticFiles.get_response
    armed, release = Event(), Event()

    async def response(self, path, scope):
        if path == held and armed.is_set():
            while not release.is_set():
                await asyncio.sleep(0.02)
        return await original(self, path, scope)

    monkeypatch.setattr(viewer_server.RevalidatingStaticFiles, "get_response", response)

    async def scenario(tab, server):
        await tab.until(READY)
        armed.set()
        try:
            await reload_instrumented(tab, IMAGE_INSTRUMENT)
            await tab.until(READY)
            assert await tab.eval("window.__startup.hasGroundSource('procedural')")
            assert await tab.eval("window.__startup.decodes") == 0
            assert len(await tab.eval("window.__startup.rockWarnings")) == 1
        finally:
            release.set()
        await tab.until("window.__startup.decodes===2")
        await asyncio.sleep(0.15)
        assert await tab.eval("window.__startup.hasGroundSource('procedural')")
        server.pg.new_world("cached_import", floor="rock")
        version = server.pg.session.version
        await tab.until(f"({READY})&&window.__robotforge.scene.version==={version}")
        assert await tab.eval("window.__startup.hasGroundSource('natural-rock-face-03')")
        assert await tab.eval("window.__startup.decodes") == 2
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=rock_world)


MIXED_INSTRUMENT = """
p.heldStyles=[];p.releases=[];p.holdAll=false;p.failStyle=null;p.firstDecode=null;p.optionalWarnings=[];
const warn=console.warn;
console.warn=(...args)=>{const style=String(args[0]).match(/optional (wood|grass|rock) texture/);
  if(style)p.optionalWarnings.push(style[1]);return warn(...args)};
const decode=HTMLImageElement.prototype.decode;
HTMLImageElement.prototype.decode=async function(){
  if(!this.src.startsWith('data:image/'))return decode.call(this);
  p.firstDecode??=performance.now();await decode.call(this);
  if(!this.src.startsWith('data:image/jpeg;'))return;
  const payload=this.src.slice(this.src.indexOf(',')+1);
  const style=this.naturalWidth===512?'wood':payload.length===Math.ceil(15308/3)*4?'rock':'grass';
  if(p.holdAll){p.heldStyles.push(style);await new Promise(resolve=>p.releases.push(resolve));}
  if(p.failStyle===style)throw new DOMException('injected mixed failure','EncodingError');
};
p.releaseAll=()=>{p.holdAll=false;for(const resolve of p.releases.splice(0))resolve()};
p.mixedRows=()=>{const rows=[];window.__robotforge.three().scene.traverse(o=>{
  if(o.isMesh&&o.material?.userData?.groundSource)rows.push(o.material.userData.groundSource);
});return [...new Set(rows)].sort()};
"""


def mixed_world(pg):
    pg.set_floor("wood")
    pg.add_terrain("hills", name="grass_patch", style="grass", pos=[2, 0], params={"size": [2, 2]})
    pg.add_terrain("hills", name="rock_patch", style="rock", pos=[-2, 0], params={"size": [2, 2]})
    pg.add_robot("rover", template="rover")


def test_rock_failure_leaves_wood_and_grass_pairs_independent(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await reload_instrumented(tab, MIXED_INSTRUMENT + "p.failStyle='rock';")
        await tab.until(READY)
        assert await tab.eval("window.__startup.mixedRows()") == ["natural-leafy-grass", "natural-oak", "procedural"]
        assert await tab.eval("window.__startup.optionalWarnings") == ["rock"]
        assert await tab.eval("window.__robotforge.three().renderer.getContext().getError()") == 0
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=mixed_world)


def test_all_three_deadlines_run_concurrently_and_late_results_only_warm_caches(viewer):  # noqa: F811
    async def scenario(tab, server):
        await reload_instrumented(tab, MIXED_INSTRUMENT + "p.holdAll=true;")
        await tab.until(READY)
        assert await tab.eval("window.__startup.mixedRows()") == ["procedural"]
        await tab.run("""
          const p=window.__startup,g=await import('./static/ground-look.js');
          const desc={geoms:['wood','grass','rock'].map(style=>({type:'plane',material:'ground_'+style})),
            grounds:Object.fromEntries(['wood','grass','rock'].map(style=>['ground_'+style,{pattern:'noise'}]))};
          p.prepareElapsed=null;const started=performance.now();
          p.preparation=Promise.all([
            g.prepareGroundTextures(desc),g.prepareGrassTextures(desc),g.prepareRockTextures(desc)])
            .then(tiles=>{p.preparedTiles=tiles;p.prepareElapsed=performance.now()-started});
        """)
        await tab.until("window.__startup.heldStyles.length===3")
        assert await tab.eval("[...window.__startup.heldStyles].sort()") == ["grass", "rock", "wood"]
        await tab.until("window.__startup.prepareElapsed!==null")
        elapsed = await tab.eval("window.__startup.prepareElapsed")
        assert 800 < elapsed < 1700
        assert await tab.eval("window.__startup.preparedTiles") == [None, None, None]
        assert await tab.eval("window.__startup.mixedRows()") == ["procedural"]
        await tab.run("window.__startup.releaseAll()")
        await asyncio.sleep(0.15)
        assert await tab.eval("window.__startup.mixedRows()") == ["procedural"]
        mixed_world(server.pg)
        version = server.pg.session.version
        await tab.until(f"({READY})&&window.__robotforge.scene.version==={version}")
        assert await tab.eval("window.__startup.mixedRows()") == [
            "natural-leafy-grass",
            "natural-oak",
            "natural-rock-face-03",
        ]
        assert await tab.eval("[...window.__startup.optionalWarnings].sort()") == ["grass", "rock", "wood"]
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=lambda pg: pg.set_floor("concrete"))


def test_scene_change_disposes_owned_gpu_textures_and_reuses_only_cpu_sources(viewer):  # noqa: F811
    async def scenario(tab, server):
        await reload_instrumented(tab, IMAGE_INSTRUMENT)
        await tab.until(READY)
        assert await tab.eval("window.__startup.hasGroundSource('natural-rock-face-03')")
        await tab.run("""
          const p=window.__startup;p.textureDisposals=0;p.materialDisposals=0;p.oldTextures=new Set();
          window.__robotforge.three().scene.traverse(o=>{
            if(o.material?.userData?.groundSource!=='natural-rock-face-03')return;
            p.oldSource=o.material.map.source;p.oldDetailSource=o.material.bumpMap.source;
            o.material.addEventListener('dispose',()=>p.materialDisposals++);
            for(const tex of new Set([o.material.map,o.material.bumpMap,o.material.roughnessMap])){
              p.oldTextures.add(tex);tex.addEventListener('dispose',()=>p.textureDisposals++);
            }
          });
          p.oldMaterialCount=p.groundRows().length;
        """)
        assert await tab.eval("window.__startup.oldMaterialCount") > 0
        assert await tab.eval("window.__startup.oldTextures.size") > 0
        server.pg.new_world("plain_after_rock", floor="concrete")
        version = server.pg.session.version
        await tab.until(f"({READY})&&window.__robotforge.scene.version==={version}")
        assert await tab.eval("window.__startup.textureDisposals===window.__startup.oldTextures.size")
        assert await tab.eval("window.__startup.materialDisposals===window.__startup.oldMaterialCount")
        server.pg.new_world("rock_again", floor="rock")
        version = server.pg.session.version
        await tab.until(f"({READY})&&window.__robotforge.scene.version==={version}")
        result = await tab.run("""
          const p=window.__startup;let fresh=true,shared=true,count=0;
          window.__robotforge.three().scene.traverse(o=>{
            if(o.material?.userData?.groundSource!=='natural-rock-face-03')return;
            count++;
            fresh&&=!p.oldTextures.has(o.material.map)&&!p.oldTextures.has(o.material.bumpMap);
            shared&&=o.material.map.source===p.oldSource&&o.material.bumpMap.source===p.oldDetailSource;
          });return {fresh,shared,count,decodes:p.decodes};
        """)
        assert result.pop("count") > 0
        assert result == {"fresh": True, "shared": True, "decodes": 2}
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=rock_world)


def test_natural_rock_sampler_and_sources_survive_real_webgl_context_restoration(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await reload_instrumented(tab, IMAGE_INSTRUMENT)
        await tab.until(READY)
        assert await tab.eval("window.__startup.hasGroundSource('natural-rock-face-03')")
        await asyncio.sleep(0.3)
        before = await tab.run(r"""
          const p=window.__startup,{renderer,scene}=window.__robotforge.three();
          p.restoreCompiles=0;p.restorePatched=true;p.contextSources=[];
          scene.traverse(o=>{
            if(o.material?.userData?.groundSource!=='natural-rock-face-03')return;
            const m=o.material,hook=m.onBeforeCompile;
            p.contextSources.push([m.map.source,m.bumpMap.source]);
            m.onBeforeCompile=function(shader,r){
              hook.call(this,shader,r);p.restoreCompiles++;
              p.restorePatched&&=(shader.fragmentShader.match(/rfRockSample\(/g)||[]).length===7
                &&shader.fragmentShader.includes('rfMetricBump(');
            };
          });
          p.contextProbe=()=>{
            let count=0,sources=true,keys=true;scene.traverse(o=>{
              const m=o.material;if(m?.userData?.groundSource!=='natural-rock-face-03')return;
              count++;sources&&=p.contextSources.some(([c,d])=>m.map.source===c&&m.bumpMap.source===d);
              keys&&=m.customProgramCacheKey()==='rf-ground-metric-v1:square-quintic-rock-v1';
            });
            return {count,sources,keys,textures:renderer.info.memory.textures,
              programs:renderer.info.programs.length,gl:renderer.getContext().getError()};
          };
          p.loseRockContext=renderer.getContext().getExtension('WEBGL_lose_context');
          return {available:!!p.loseRockContext,...p.contextProbe()};
        """)
        assert before["count"] > 0 and before["sources"] and before["keys"] and before["gl"] == 0
        if not before["available"]:
            pytest.skip("driver does not expose WEBGL_lose_context")
        await tab.run("window.__startup.loseRockContext.loseContext()")
        await tab.until("window.__robotforge.renderedVersion===0")
        await tab.run("window.__startup.loseRockContext.restoreContext()")
        await tab.until(READY)
        await tab.until("window.__startup.restoreCompiles>0")
        after = await tab.run("return window.__startup.contextProbe()")
        assert after["count"] == before["count"] and after["count"] > 0
        assert after["sources"] and after["keys"] and after["gl"] == 0
        assert after["textures"] <= before["textures"] + 2
        assert after["programs"] <= before["programs"] + 1
        assert await tab.eval("window.__startup.restorePatched")
        assert await tab.eval("window.__startup.decodes") == 2
        assert await tab.eval("window.__startup.rockWarnings") == []
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=rock_world)
