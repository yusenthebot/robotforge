"""Actual GPU normals retain their strength when only drawing resolution changes."""

from __future__ import annotations

import pytest
from test_viewer_layout import pytestmark, viewer  # noqa: F401


@pytest.mark.parametrize("kind", ["wood", "stone", "knurl", "ground_wood", "ground_concrete", "ground_grass"])
def test_resolved_bump_strength_is_independent_of_output_resolution(viewer, kind):  # noqa: F811
    async def scenario(tab, _server):
        return await tab.run(
            """
          const T = await import('three');
          const {surfaceMaterial} = await import('./static/look.js');
          const {groundMaterial} = await import('./static/ground-look.js');
          const {renderer} = window.__robotforge.three();
          const ground = KIND.startsWith('ground_'), kind = KIND.replace('ground_', '');
          const pitch = {wood:.012, stone:.025, knurl:.0016}[kind] || .012;
          const span = ground ? .06 : pitch * 2;
          const scene = new T.Scene();
          const camera = new T.OrthographicCamera(-span/2, span/2, span/2, -span/2, .001, 10);
          camera.position.z = 1;
          const geometry = new T.PlaneGeometry(span*4, span*4);
          const mat = ground ? groundMaterial(kind, new T.Color(.4,.45,.5), 1) :
            surfaceMaterial('sample', new T.Color(.4,.45,.5), 1,
              {sample:{roughness:.5,metalness:0,texture:kind}});
          const compile = mat.onBeforeCompile;
          mat.onBeforeCompile = shader => {
            compile(shader);
            shader.fragmentShader = shader.fragmentShader.replace('#include <opaque_fragment>',
              'gl_FragColor = vec4(normal, 1.0); return;');
          };
          const cache = mat.customProgramCacheKey();
          mat.customProgramCacheKey = () => cache + '-normal-diagnostic';
          const mesh = new T.Mesh(geometry, mat); scene.add(mesh);
          const previous = renderer.getRenderTarget(), targets = [];
          function draw(size) {
            const target = new T.WebGLRenderTarget(size,size,{type:T.FloatType});
            targets.push(target); renderer.setRenderTarget(target); renderer.render(scene,camera);
            const pixels = new Float32Array(size*size*4);
            renderer.readRenderTargetPixels(target,0,0,size,size,pixels);
            let energy=0, count=0;
            for(let y=size/4;y<size*3/4;y++) for(let x=size/4;x<size*3/4;x++) {
              const i=(y*size+x)*4;
              if(!Number.isFinite(pixels[i]+pixels[i+1]+pixels[i+2])) throw Error('nonfinite normal');
              energy += pixels[i]**2 + pixels[i+1]**2; count++;
            }
            return Math.sqrt(energy/count);
          }
          try {
            const low=draw(128), high=draw(512);
            return {low,high,ratio:low/high,gl:renderer.getContext().getError()};
          } finally {
            renderer.setRenderTarget(previous); targets.forEach(t=>t.dispose());
            geometry.dispose(); for(const t of new Set([mat.map,mat.bumpMap,mat.roughnessMap]))t?.dispose();
            mat.dispose();
          }
        """.replace("KIND", repr(kind))
        )

    result = viewer("laptop", "en", scenario, world=lambda pg: None)
    assert result["gl"] == 0, result
    assert result["low"] > 1e-5 and result["high"] > 1e-5, result
    assert 0.9 < result["ratio"] < 1.1, result
