"""Real GPU wood fibres follow the board length after local mesh axes are reordered."""

from __future__ import annotations

import pytest
from test_viewer_layout import pytestmark, viewer  # noqa: F401


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_wood_grain_stays_longitudinal_in_each_mesh_frame(viewer, axis):  # noqa: F811
    async def scenario(tab, _server):
        result = await tab.run(
            """
          const T = await import('three');
          const {surfaceMaterial,orientWoodGrain} = await import('./static/look.js');
          const {renderer} = window.__robotforge.three();
          const axis=AXIS, geometry=new T.PlaneGeometry(.12,.06);
          if(axis==='y')geometry.rotateZ(Math.PI/2);
          if(axis==='z')geometry.rotateY(Math.PI/2);
          const mat=surfaceMaterial('rf_wood',new T.Color(.4,.3,.2),1);
          orientWoodGrain(mat,geometry);
          const compile=mat.onBeforeCompile;
          let sharedUniform=false;
          mat.onBeforeCompile=shader=>{
            compile(shader);
            sharedUniform=shader.uniforms.rfWoodAxis.value===mat.userData.woodGrainAxis;
            shader.fragmentShader=shader.fragmentShader.replace('#include <opaque_fragment>',
              'gl_FragColor=vec4(vec3(grainH),1.0);return;');
          };
          const key=mat.customProgramCacheKey();mat.customProgramCacheKey=()=>key+'-axis-diagnostic';
          const scene=new T.Scene();scene.add(new T.Mesh(geometry,mat));
          const camera=new T.OrthographicCamera(-.07,.07,.035,-.035,.01,10);
          camera.position.set(...(axis==='z'?[1,0,0]:[0,0,1]));
          camera.up.set(...(axis==='y'?[-1,0,0]:[0,1,0]));camera.lookAt(0,0,0);
          const target=new T.WebGLRenderTarget(256,128,{type:T.FloatType});
          const previous=renderer.getRenderTarget();
          try {
            renderer.setRenderTarget(target);renderer.render(scene,camera);
            const data=new Float32Array(256*128*4);renderer.readRenderTargetPixels(target,0,0,256,128,data);
            let along=0,across=0,count=0,min=Infinity,max=-Infinity;
            for(let y=32;y<96;y++)for(let x=64;x<192;x++){
              const i=(y*256+x)*4,h=data[i];
              if(!Number.isFinite(h))throw Error('nonfinite wood height');
              min=Math.min(min,h);max=Math.max(max,h);
              along+=(data[i+4]-h)**2;across+=(data[i+256*4]-h)**2;count++;
            }
            return {axis:mat.userData.woodGrainAxis.toArray(),sharedUniform,min,max,
              along:Math.sqrt(along/count),across:Math.sqrt(across/count),gl:renderer.getContext().getError()};
          } finally {
            renderer.setRenderTarget(previous);target.dispose();geometry.dispose();mat.dispose();
          }
        """.replace("AXIS", repr(axis))
        )
        assert result["axis"] == {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
        assert result["sharedUniform"] and result["gl"] == 0, result
        assert 0.15 <= result["min"] < result["max"] <= 1, result
        assert result["across"] > 4 * result["along"] > 0, result
        errors = await tab.eval("window.__qaErrors")
        assert errors == [], errors

    viewer("laptop", "en", scenario, world=lambda pg: None)
