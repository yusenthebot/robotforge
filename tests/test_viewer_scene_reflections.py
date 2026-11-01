"""Shader composition contracts; actual rendering is covered by the browser tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src/robotforge/viewer/static"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def run_js(tmp_path: Path, body: str) -> None:
    three = (STATIC / "vendor/three/three.module.min.js").as_uri()
    source = (STATIC / "look.js").read_text(encoding="utf-8")
    source = source.replace('from "three"', f"from {json.dumps(three)}")
    source = source.replace('from "./lighting-fit.js"', f"from {json.dumps((STATIC / 'lighting-fit.js').as_uri())}")
    module = tmp_path / "look.mjs"
    module.write_text(source, encoding="utf-8")
    script = f"""
      import assert from 'node:assert/strict';
      import * as T from {json.dumps(three)};
      import {{attachSceneReflectionMaterial, sceneReflectionCubeUV, surfaceMaterial}}
        from {json.dumps(module.as_uri())};
      {body}
    """
    result = subprocess.run(
        [NODE, "--experimental-default-type=module", "--input-type=module", "-"],
        input=script,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("texture", [None, "brushed", "wood", "weave", "cloth"])
def test_specular_extension_preserves_existing_hooks_and_original_variant(tmp_path, texture):
    run_js(
        tmp_path,
        r"""
      const material=surfaceMaterial('authored',new T.Color(.4,.5,.6),1,
        {authored:{roughness:.35,metalness:.6,env:1.45,clearcoat:.5,texture:TEXTURE}});
      let compileCalls=0,renderCalls=0;
      const authoredCompile=material.onBeforeCompile,authoredRender=material.onBeforeRender;
      material.onBeforeCompile=function(shader,renderer){assert.equal(this,material);compileCalls++;
        authoredCompile.call(this,shader,renderer);shader.fragmentShader+='\n// retained-authored-hook';};
      material.onBeforeRender=function(...args){
        assert.equal(this,material);renderCalls++;return authoredRender.apply(this,args);};
      const compile=material.onBeforeCompile, key=material.customProgramCacheKey,
        render=material.onBeforeRender, originalKey=key.call(material);
      function shader(){return {vertexShader:T.ShaderLib.physical.vertexShader,
        fragmentShader:T.ShaderLib.physical.fragmentShader,uniforms:{}};}
      const baseline=shader();compile.call(material,baseline,{});
      let main=false,rev=1;const failures=[];
      const undo=attachSceneReflectionMaterial(material,{isMainPass:()=>main,revision:()=>rev,
        uniforms:()=>({rfSceneEnvironment:{value:null}}),failed:e=>failures.push(e)});
      material.onBeforeRender({}, {}, {});
      const off=shader();material.onBeforeCompile(off,{});
      assert.equal(off.vertexShader,baseline.vertexShader);assert.equal(off.fragmentShader,baseline.fragmentShader);
      assert.equal(material.customProgramCacheKey(),originalKey);
      main=true;material.onBeforeRender({}, {}, {});
      const local=shader();material.onBeforeCompile(local,{});
      assert.equal(failures.length,0);assert(local.uniforms.rfEnvMultiplier);
      assert(local.fragmentShader.includes('// retained-authored-hook'));
      if(TEXTURE==='weave'||TEXTURE==='cloth')assert(local.fragmentShader.includes('vWeavePos'));
      if(TEXTURE==='brushed'||TEXTURE==='wood')assert(local.fragmentShader.includes('vGrainPos'));
      if(TEXTURE==='wood')assert(local.fragmentShader.includes('rfWoodAxis'));
      assert(local.fragmentShader.includes('rfSceneRadiance(reflectVec, roughness)'));
      assert(local.fragmentShader.includes('textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 )'));
      assert.equal((local.fragmentShader.match(/envMapIntensity \* rfEnvMultiplier/g)||[]).length,2);
      assert(local.vertexShader.includes('vRFWorldPosition = (modelMatrix * rfWorldPosition).xyz'));
      const cube=sceneReflectionCubeUV();assert(cube.includes('(1.0 / 384.0)'));
      assert(cube.includes('(1.0 / 512.0)'));assert(!/\bCUBEUV_|\btextureCubeUV\(/.test(cube));
      const v=material.version;rev++;material.onBeforeRender({}, {}, {});assert(material.version>v);
      main=false;material.onBeforeRender({}, {}, {});const restored=shader();material.onBeforeCompile(restored,{});
      assert.equal(restored.fragmentShader,baseline.fragmentShader);assert.equal(material.customProgramCacheKey(),originalKey);
      assert.equal(compileCalls,4);assert.equal(renderCalls,4);
      undo();assert.equal(material.onBeforeCompile,compile);assert.equal(material.onBeforeRender,render);
      assert.equal(material.customProgramCacheKey,key);
    """.replace("TEXTURE", json.dumps(texture)),
    )


def test_cached_uniforms_follow_rebound_controller_without_recompile(tmp_path: Path) -> None:
    run_js(
        tmp_path,
        r"""
      const material=new T.MeshStandardMaterial(),failures=[];
      const makeContext=()=>({
        rfSceneEnvironment:{value:new T.Texture()},rfProbeCenter:{value:new T.Vector3(1,2,3)},
        rfProbeMin:{value:new T.Vector3(-4,-5,-6)},rfProbeMax:{value:new T.Vector3(4,5,6)}});
      const first=makeContext(),second=makeContext();
      const attach=uniforms=>attachSceneReflectionMaterial(material,{isMainPass:()=>true,
        revision:()=>1,uniforms:()=>uniforms,failed:e=>failures.push(e)});
      const detach=attach(first);material.onBeforeRender({}, {}, {});
      const cached={vertexShader:T.ShaderLib.physical.vertexShader,
        fragmentShader:T.ShaderLib.physical.fragmentShader,uniforms:{}};
      material.onBeforeCompile(cached,{});
      const wrappers={...cached.uniforms},key=material.customProgramCacheKey();
      for(const name of Object.keys(first))assert.equal(cached.uniforms[name].value,first[name].value);
      detach();assert.equal(cached.uniforms.rfSceneEnvironment.value,null);
      const detachSecond=attach(second),version=material.version;
      // The program is already cached: no onBeforeCompile call occurs on this draw.
      material.onBeforeRender({}, {}, {});
      assert.equal(material.customProgramCacheKey(),key);assert(material.version>version);
      for(const name of Object.keys(second)){
        assert.equal(cached.uniforms[name],wrappers[name]);
        assert.equal(cached.uniforms[name].value,second[name].value);
      }
      detach();assert.equal(cached.uniforms.rfSceneEnvironment.value,second.rfSceneEnvironment.value);
      const replacement=new T.Texture();second.rfSceneEnvironment.value=replacement;
      material.onBeforeRender({}, {}, {});assert.equal(cached.uniforms.rfSceneEnvironment.value,replacement);
      detachSecond();assert.equal(cached.uniforms.rfSceneEnvironment.value,null);
      assert.deepEqual(failures,[]);
    """,
    )


def test_explicit_maps_and_unsupported_chunks_keep_original_shader(tmp_path):
    run_js(
        tmp_path,
        r"""
      const explicit=new T.MeshStandardMaterial();explicit.envMap=new T.Texture();
      assert.equal(attachSceneReflectionMaterial(explicit,{}),null);
      assert.equal(attachSceneReflectionMaterial(new T.MeshBasicMaterial(),{}),null);
      const material=new T.MeshStandardMaterial(),failures=[];
      attachSceneReflectionMaterial(material,{isMainPass:()=>true,revision:()=>1,
        uniforms:()=>({}),failed:e=>failures.push(e)});
      material.onBeforeRender({}, {}, {});
      const shader={vertexShader:T.ShaderLib.physical.vertexShader,fragmentShader:'unsupported',uniforms:{}};
      material.onBeforeCompile(shader,{});
      assert.equal(shader.fragmentShader,'unsupported');assert.equal(failures.length,1);
      assert(failures[0].message.includes('Unsupported scene reflection shader'));
      const authored=new T.MeshStandardMaterial();
      authored.onBeforeCompile=s=>{s.fragmentShader='half-edited';throw Error('authored compile failure');};
      attachSceneReflectionMaterial(authored,{isMainPass:()=>true,revision:()=>1,
        uniforms:()=>({}),failed:e=>failures.push(e)});
      authored.onBeforeRender({}, {}, {});
      const original={vertexShader:T.ShaderLib.physical.vertexShader,
        fragmentShader:T.ShaderLib.physical.fragmentShader,uniforms:{}};
      const attempted={...original,uniforms:{}};authored.onBeforeCompile(attempted,{});
      assert.equal(attempted.fragmentShader,original.fragmentShader);
      assert.equal(attempted.vertexShader,original.vertexShader);
      assert.equal(failures.length,2);assert.equal(failures[1].message,'authored compile failure');
    """,
    )
