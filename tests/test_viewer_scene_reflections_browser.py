"""GPU mechanism evidence for fixed scenery reflections, not a visual quality judgment."""

from __future__ import annotations

import json

import pytest
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_startup_browser import READY

SETUP = """
  const T=await import('three'),{createSceneReflections}=await import('./static/scene-reflections.js'),
    {surfaceMaterial}=await import('./static/look.js');
  const api=window.__robotforge,{renderer,scene:main,camera:mainCamera}=api.three();
  const scene=new T.Scene(),root=new T.Group();scene.add(root);
  scene.background=new T.Color(.1,.1,.1);scene.environment=main.environment;scene.environmentIntensity=.85;
  const camera=new T.PerspectiveCamera(35,1280/800,.01,100);
  camera.up.set(0,0,1);camera.position.set(0,-3,1);camera.lookAt(0,0,1);
  const wall=new T.Mesh(new T.BoxGeometry(4,.1,2),new T.MeshBasicMaterial({color:0xff0808}));
  wall.position.set(0,-3.6,1);wall.userData={body:0,geom:0};root.add(wall);
  const floor=new T.Mesh(new T.BoxGeometry(6,8,.1),new T.MeshBasicMaterial({color:0x333333}));
  floor.position.z=-.1;floor.userData={body:0,geom:1};root.add(floor);
  const sphere=new T.Mesh(new T.SphereGeometry(.55,40,24),new T.MeshStandardMaterial({metalness:1,roughness:.15}));
  sphere.position.set(0,0,1);sphere.userData={body:1,geom:2};root.add(sphere);
  const desc={version:1,stat:{center:[0,-1,1]},visibleGroupLimit:3,
    bodies:[{id:0,dynamic:false},{id:1,dynamic:true}],
    geoms:[{id:0,body:0,group:0,type:'box'},{id:1,body:0,group:0,type:'box'},{id:2,body:1,group:0,type:'sphere'}]};
  const lighting={studio:()=>main.environment,shadowReady:()=>true,preset:()=>preset};let preset='studio';
  const controller=createSceneReflections({scene,renderer,camera,lighting});
  const rt=new T.WebGLRenderTarget(320,200),gl=renderer.getContext();
  const pack=gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING);gl.bindBuffer(gl.PIXEL_PACK_BUFFER,null);
  const extraMaterials=[];
  function pixels(target=null,view=camera){
    renderer.setRenderTarget(target);renderer.render(scene,view);gl.finish();
    if(target){const p=new Uint8Array(target.width*target.height*4);
      renderer.readRenderTargetPixels(target,0,0,target.width,target.height,p);return p;}
    const p=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);
    gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,p);return p;
  }
  function different(a,b){if(a.length!==b.length)throw Error('pixel dimensions differ');let count=0;
    for(let i=0;i<a.length;i++)if(a[i]!==b[i])count++;return count;}
  function ready(){root.updateMatrixWorld(true);controller.posed();renderer.setRenderTarget(null);
    renderer.render(scene,camera);controller.afterMainRender();renderer.render(scene,camera);controller.afterMainRender();
    const s=controller.state();if(s.status!=='ready')throw Error(JSON.stringify(s));return s;}
  function bind(){controller.bind(root,desc,new Map([[1,'receiver']]));return ready();}
  function cleanup(renderMain=true){controller.dispose();rt.dispose();
    root.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});for(const m of extraMaterials)m.dispose();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER,pack);renderer.setRenderTarget(null);if(renderMain)renderer.render(main,mainCamera);}
"""


@pytest.mark.parametrize("surface", ["metal", "clearcoat", "grain", "weave", "diffuse", "explicit"])
def test_actual_scenery_changes_only_main_specular_and_off_restores_original(viewer, surface):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        result = await tab.run(
            SETUP
            + r"""
          try{
            const kind=SURFACE;extraMaterials.push(sphere.material);
            if(kind==='diffuse'||kind==='clearcoat')sphere.material=new T.MeshPhysicalMaterial({
              color:0xaaaaaa,roughness:.2,metalness:0,specularIntensity:0,clearcoat:kind==='clearcoat'?1:0,clearcoatRoughness:.1});
            if(kind==='grain'||kind==='weave')sphere.material=surfaceMaterial('authored',new T.Color(.6,.6,.6),1,
              {authored:{metalness:1,roughness:.2,env:1.45,texture:kind==='grain'?'brushed':'weave'}});
            if(kind==='explicit')sphere.material.envMap=main.environment;
            const baseline=pixels(),offscreenBefore=pixels(rt),other=camera.clone();other.updateMatrixWorld(true);
            const otherBefore=pixels(null,other);renderer.setRenderTarget(null);const first=bind();
            const red=pixels(),offscreenRed=pixels(rt),otherRed=pixels(null,other),redAgain=pixels();
            wall.material.color.set(0x0808ff);controller.invalidate('test-fixed-material');ready();
            const blue=pixels(),offscreenBlue=pixels(rt),blueAgain=pixels();
            controller.setEnabled(false);const off=pixels();
            controller.setEnabled(true);ready();controller.setEnabled(false);const restored=pixels();
            return {changed:different(red,blue),off:different(baseline,off),restored:different(baseline,restored),
              sensorRed:different(offscreenBefore,offscreenRed),sensorBlue:different(offscreenBefore,offscreenBlue),
              other:different(otherBefore,otherRed),alternateRed:different(red,redAgain),alternateBlue:different(blue,blueAgain),
              first,final:controller.state(),error:gl.getError()};
          }finally{cleanup();}
        """.replace("SURFACE", json.dumps(surface))
        )
        if surface in {"diffuse", "explicit"}:
            assert result["changed"] == 0, result
        else:
            assert result["changed"] > 100, result
        for field in ["off", "restored", "sensorRed", "sensorBlue", "other", "alternateRed", "alternateBlue", "error"]:
            assert result[field] == 0, (field, result)
        assert result["first"]["probe"]["center"] == [0, -1, 1]
        assert result["first"]["captureDraws"] > 6
        assert result["final"]["allocatedTargets"] == result["final"]["releasedTargets"]
        errors = await tab.eval("window.__qaErrors")
        assert errors == [], errors

    viewer("laptop", "en", scenario)


@pytest.mark.parametrize("fault", ["mesh", "compile"])
def test_internal_callback_failures_restore_state_and_render_depth(viewer, fault):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        result = await tab.run(
            SETUP
            + r"""
          const fault=FAULT,lists=renderer.renderLists,getList=lists.get,depths=[];
          const originalMeshHook=wall.onBeforeRender,originalCompile=wall.material.onBeforeCompile;
          lists.get=function(s,depth){if(s===scene)depths.push(depth);return getList.apply(this,arguments);};
          function state(){return {target:renderer.getRenderTarget()?.texture.uuid??null,
            face:renderer.getActiveCubeFace(),mip:renderer.getActiveMipmapLevel(),
            viewport:renderer.getViewport(new T.Vector4()).toArray(),
            scissor:renderer.getScissor(new T.Vector4()).toArray(),
            scissorTest:renderer.getScissorTest(),
            clear:renderer.getClearColor(new T.Color()).toArray(),alpha:renderer.getClearAlpha(),
            auto:renderer.autoClear,autoColor:renderer.autoClearColor,autoDepth:renderer.autoClearDepth,
            autoStencil:renderer.autoClearStencil,tone:renderer.toneMapping,exposure:renderer.toneMappingExposure,
            output:renderer.outputColorSpace,xr:renderer.xr.enabled,shadowAuto:renderer.shadowMap.autoUpdate,
            shadowNeeds:renderer.shadowMap.needsUpdate,background:scene.background.uuid??scene.background.getHex(),
            backgroundIntensity:scene.backgroundIntensity,backgroundBlurriness:scene.backgroundBlurriness,
            backgroundRotation:scene.backgroundRotation.toArray(),fog:scene.fog,
            visible:[wall.visible,floor.visible,sphere.visible],receive:[wall.receiveShadow,floor.receiveShadow]};}
          function restore(s){renderer.setRenderTarget(null,s.face,s.mip);
            renderer.setViewport(...s.viewport);renderer.setScissor(...s.scissor);renderer.setScissorTest(s.scissorTest);
            renderer.setClearColor(new T.Color().fromArray(s.clear),s.alpha);renderer.autoClearColor=s.autoColor;
            renderer.autoClearDepth=s.autoDepth;renderer.autoClearStencil=s.autoStencil;renderer.toneMappingExposure=s.exposure;
            scene.backgroundIntensity=s.backgroundIntensity;scene.backgroundBlurriness=s.backgroundBlurriness;
            scene.backgroundRotation.fromArray(s.backgroundRotation);}
          const initial=state();
          try{
            pixels(rt);const baseline=pixels();bind();controller.setEnabled(false);pixels();
            const memory={...renderer.info.memory},rows=[];let hookCalls=0;
            for(let i=0;i<10;i++){
              controller.setEnabled(true);renderer.render(scene,camera);
              const injected=function(...args){
                if(fault==='mesh'&&args[2]===camera)return originalMeshHook.apply(this,args);
                hookCalls++;renderer.setRenderTarget(rt);
                if(fault==='compile')args[0].fragmentShader+='\ninvalid_partial_shader';
                throw Error('injected internal '+fault+' callback');
              };
              if(fault==='mesh')wall.onBeforeRender=injected;
              else{wall.material.onBeforeCompile=injected;wall.material.needsUpdate=true;}
              renderer.autoClearColor=false;renderer.autoClearDepth=false;renderer.autoClearStencil=false;
              renderer.toneMappingExposure=.67;renderer.setClearColor(0x102030,.45);
              renderer.setViewport(3,4,100,110);renderer.setScissor(5,6,80,90);renderer.setScissorTest(true);
              renderer.setRenderTarget(null,3,1);
              scene.backgroundIntensity=.37;scene.backgroundBlurriness=.23;scene.backgroundRotation.set(.1,.2,.3);
              const before=state(),targetBefore=renderer.getRenderTarget(),callsBefore=hookCalls;
              controller.afterMainRender();const failed=controller.state(),after=state();
              const targetRestored=renderer.getRenderTarget()===targetBefore,
                hookRestored=(fault==='mesh'?wall.onBeforeRender:wall.material.onBeforeCompile)===injected;
              wall.onBeforeRender=originalMeshHook;wall.material.onBeforeCompile=originalCompile;
              if(fault==='compile')wall.material.dispose();
              restore(initial);const original=different(baseline,pixels());
              controller.invalidate('retry-recoverable-callback');ready();controller.setEnabled(false);pixels();
              rows.push({before,after,failed,targetRestored,hookRestored,original,hookCalls:hookCalls-callsBefore,
                memory:{...renderer.info.memory}});
            }
            return {rows,depths,memory,final:controller.state()};
          }finally{wall.onBeforeRender=originalMeshHook;wall.material.onBeforeCompile=originalCompile;
            lists.get=getList;restore(initial);cleanup();}
        """.replace("FAULT", json.dumps(fault))
        )
        assert result["depths"] and set(result["depths"]) == {0}, result
        for row in result["rows"]:
            assert row["before"] == row["after"] and row["targetRestored"] and row["hookRestored"], row
            assert row["hookCalls"] > 0 and row["original"] == 0, row
            assert row["failed"]["status"] == "disabled" and not row["failed"]["rendererRecoveryRequired"], row
            assert f"injected internal {fault} callback" in row["failed"]["reason"], row
            assert row["failed"]["allocatedTargets"] == row["failed"]["releasedTargets"], row
            assert row["memory"] == result["memory"], row
        errors = await tab.eval("window.__qaErrors")
        assert errors == [], errors

    viewer("laptop", "en", scenario)


def test_escaped_renderer_internal_failure_is_fatal_and_never_retried(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        result = await tab.run(
            SETUP
            + r"""
          const lists=renderer.renderLists,getList=lists.get;
          try{
            bind();controller.invalidate('fatal-case');renderer.render(scene,camera);
            // renderLists.get is called after Three pushes its render-state stack.
            // This is outside the guarded callbacks/draw method and must not claim recovery.
            lists.get=function(s,depth){if(s===scene)throw Error('injected internal traversal failure');
              return getList.apply(this,arguments);};
            controller.afterMainRender();const failure=controller.state();lists.get=getList;
            controller.invalidate('must-not-retry');controller.contextRestored();controller.setEnabled(false);
            controller.setEnabled(true);controller.bind(root,desc,new Map([[1,'receiver']]));
            controller.afterMainRender();
            return {failure,after:controller.state()};
          }finally{lists.get=getList;cleanup(false);}
        """
        )
        for state in [result["failure"], result["after"]]:
            assert state["status"] == "fatal" and state["rendererRecoveryRequired"], state
            assert "renderer-reload-required" in state["reason"], state
            assert state["activeTargets"] == state["buildingTargets"] == 0
            assert state["allocatedTargets"] == state["releasedTargets"]
        assert result["after"]["captures"] == result["failure"]["captures"]
        errors = await tab.eval("window.__qaErrors")
        assert errors == [], errors
        # This fixture closes the page; it does not render another image and certify recovery.

    viewer("laptop", "en", scenario)


def test_stale_capture_discards_old_content_and_matches_newest_reference(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        result = await tab.run(
            SETUP
            + r"""
          const hook=wall.onBeforeRender;
          try{
            bind();const red=pixels();controller.invalidate('older-red-edit');
            const oldGeneration=controller.state().generation;let fired=0,newGeneration=null;
            wall.onBeforeRender=function(r,s,c){hook.apply(this,arguments);
              if(c!==camera&&!fired){fired++;wall.material.color.set(0x0808ff);
                controller.invalidate('newest-blue-edit');newGeneration=controller.state().generation;}};
            renderer.render(scene,camera);controller.afterMainRender();const discarded=controller.state();
            wall.onBeforeRender=hook;ready();const repaired=controller.state(),blue=pixels();
            // A fresh controller observes only the already-edited blue scenery, without
            // sharing the old controller's generation or probe target.
            controller.dispose();const reference=createSceneReflections({scene,renderer,camera,lighting});
            let expected,referenceState;
            try{reference.bind(root,desc,new Map([[1,'receiver']]));root.updateMatrixWorld(true);reference.posed();
              renderer.render(scene,camera);reference.afterMainRender();renderer.render(scene,camera);reference.afterMainRender();
              expected=pixels();referenceState=reference.state();}finally{reference.dispose();}
            return {fired,oldGeneration,newGeneration,discarded,repaired,referenceState,
              changed:different(red,blue),matches:different(blue,expected),final:controller.state()};
          }finally{wall.onBeforeRender=hook;cleanup();}
        """
        )
        assert result["fired"] == 1 and result["newGeneration"] > result["oldGeneration"], result
        assert result["discarded"]["status"] == "dirty" and result["discarded"]["activeTargets"] == 0
        assert result["repaired"]["generation"] == result["newGeneration"]
        assert result["referenceState"]["status"] == result["repaired"]["status"] == "ready"
        assert result["changed"] > 100 and result["matches"] == 0, result
        assert result["final"]["allocatedTargets"] == result["final"]["releasedTargets"]
        errors = await tab.eval("window.__qaErrors")
        assert errors == [], errors

    viewer("laptop", "en", scenario)


def test_fixed_edits_worlds_presets_and_restore_have_bounded_resources(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        result = await tab.run(
            SETUP
            + r"""
          try{
            const first=bind(),steady=renderer.info.memory.textures;const cycles=[];
            // Camera/DPR have no invalidation hook; dynamic receiver motion also leaves generation alone.
            const generation=controller.state().generation,captures=controller.state().captures;
            sphere.position.x=.05;camera.position.x=.03;root.updateMatrixWorld(true);controller.posed();
            renderer.render(scene,camera);controller.afterMainRender();
            const moved=controller.state();sphere.position.x=0;camera.position.x=0;
            for(let i=0;i<10;i++){
              wall.position.x=i%2?.1:0;root.updateMatrixWorld(true);controller.posed();ready();
              const transformed=controller.state();
              preset=i%2?'daylight':'studio';ready();
              controller.contextLost();const lost=controller.state();controller.contextRestored();ready();
              desc.version++;controller.bind(root,desc,new Map([[1,'receiver']]));ready();
              const on=controller.state();controller.setEnabled(false);const off=controller.state();
              controller.setEnabled(true);ready();cycles.push({transformed,on,off,lost,textures:renderer.info.memory.textures});
            }
            controller.dispose();return {first,generation,captures,moved,steady,cycles,final:controller.state(),
              textures:renderer.info.memory.textures};
          }finally{cleanup();}
        """
        )
        assert result["moved"]["generation"] == result["generation"]
        assert result["moved"]["captures"] == result["captures"]
        for cycle in result["cycles"]:
            assert cycle["textures"] == result["steady"], result
            assert cycle["on"]["activeTargets"] == 1 and cycle["on"]["buildingTargets"] == 0
            assert cycle["off"]["allocatedTargets"] == cycle["off"]["releasedTargets"]
            assert cycle["lost"]["activeTargets"] == 0
        assert result["textures"] < result["steady"]
        assert result["final"]["allocatedTargets"] == result["final"]["releasedTargets"]
        errors = await tab.eval("window.__qaErrors")
        assert errors == [], errors

    viewer("laptop", "en", scenario)


def test_actual_sensor_modes_and_webgl_restore_preserve_original_images(viewer):  # noqa: F811
    def world(pg):
        pg.add_robot("post", build={"base": {"shape": "box", "size": [0.2, 0.2, 0.2], "fixed": True}})
        pg.add_object("wall", shape="wall", pos=[2.05, 0], yaw=90, size=[3, 0.1, 1.0], color="red")
        for mode in ["rgb", "depth", "segmentation"]:
            pg.add_sensor("post", {"type": "camera", "mode": mode, "name": mode, "width": 80, "height": 60})
        for kind in ["stereo", "event", "fisheye"]:
            pg.add_sensor("post", {"type": kind, "name": kind, "width": 80, "height": 60})

    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        await tab.until("window.__robotforge.sceneReflections.state().status==='ready'", timeout=15)
        modes = ["rgb", "depth", "segmentation", "stereo", "event", "fisheye"]
        peek = """
          const c=document.querySelector('.sensor-pip canvas');
          const p=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
          const colors=new Set();let drawn=0;
          for(let i=0;i<p.length;i+=4){colors.add(`${p[i]},${p[i+1]},${p[i+2]}`);if(p[i]||p[i+1]||p[i+2])drawn++;}
          return {picture:window.__robotforge.sensors().picture,width:c.width,height:c.height,
            colorCount:colors.size,drawn,url:c.toDataURL()};
        """
        for mode in modes:
            name = f"post/{mode}"
            await tab.until(f"window.__robotforge.sensors().sensors.includes({name!r})", timeout=10)
            await tab.run(
                f"window.__robotforge.showSensor({name!r});window.__robotforge.sceneReflections.setEnabled(false)"
            )
            await tab.until(f"window.__robotforge.sensors().picture==={name!r}", timeout=10)
            count = await tab.eval("window.__robotforge.sensors().pictures")
            await tab.until(f"window.__robotforge.sensors().pictures>{count + 2}", timeout=10)
            before = await tab.run(peek)
            assert before["picture"] == name and before["height"] == 60, before
            assert before["width"] == (160 if mode == "stereo" else 80), before
            assert before["drawn"] > 0, before
            if mode != "event":
                assert before["colorCount"] > 1, before
            # Event equality here covers the paused, settled image only, not transient events.
            await tab.run("window.__robotforge.sceneReflections.setEnabled(true)")
            await tab.until("window.__robotforge.sceneReflections.state().status==='ready'", timeout=15)
            count = await tab.eval("window.__robotforge.sensors().pictures")
            await tab.until(f"window.__robotforge.sensors().pictures>{count + 2}", timeout=10)
            after = await tab.run(peek)
            assert after == before, (mode, before, after)
            errors = await tab.eval("window.__qaErrors")
            assert errors == [], (mode, errors)
        available = await tab.run("""
          window.__reflectionContext=window.__robotforge.three().renderer.getContext().getExtension('WEBGL_lose_context');
          return !!window.__reflectionContext;
        """)
        if not available:
            pytest.skip("driver does not expose WEBGL_lose_context")
        previous = await tab.eval("window.__robotforge.sceneReflections.state()")
        await tab.run("window.__reflectionContext.loseContext()")
        await tab.until(
            "window.__robotforge.three().renderer.getContext().isContextLost() && "
            "window.__robotforge.sceneReflections.state().activeTargets===0"
        )
        await tab.run("window.__reflectionContext.restoreContext()")
        await tab.until(
            "!window.__robotforge.three().renderer.getContext().isContextLost() && "
            "window.__robotforge.sceneReflections.state().status==='ready'",
            timeout=20,
        )
        restored = await tab.eval("window.__robotforge.sceneReflections.state()")
        assert restored["captures"] > previous["captures"]
        assert restored["probe"] == previous["probe"]
        assert restored["activeTargets"] == 1 and restored["buildingTargets"] == 0
        count = await tab.eval("window.__robotforge.sensors().pictures")
        await tab.until(f"window.__robotforge.sensors().pictures>{count + 2}", timeout=10)
        assert await tab.run(peek) == before
        errors = await tab.eval("window.__qaErrors")
        assert errors == [], errors

    viewer("laptop", "en", scenario, world=world)
