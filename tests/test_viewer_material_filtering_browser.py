"""Actual GPU checks for resolved detail and minified grain, including shader cache reuse."""

from __future__ import annotations

import pytest
from test_viewer_layout import pytestmark, viewer  # noqa: F401


@pytest.mark.parametrize("kind,pitch", [("wood", 0.012), ("rubber", 0.0007), ("stone", 0.025)])
def test_filter_preserves_resolved_detail_mean_tone_and_cache(viewer, kind, pitch):  # noqa: F811
    async def scenario(tab, _server):
        result = await tab.run(
            """
          const THREE = await import('three');
          const {surfaceMaterial} = await import('./static/look.js');
          const {renderer} = window.__robotforge.three();
          const scene = new THREE.Scene();
          scene.add(new THREE.AmbientLight(0xffffff, 1));
          const light = new THREE.DirectionalLight(0xffffff, 2);
          light.position.set(1, 2, 3); scene.add(light);
          const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, .001, 100);
          camera.position.z = 10;
          const mesh = new THREE.Mesh(); scene.add(mesh);
          const target = new THREE.WebGLRenderTarget(128, 128);
          const previousTarget = renderer.getRenderTarget();
          const materials = [], geometries = [];
          const kind = KIND, pitch = PITCH;
          function material(strength, reference = false, scale = 1) {
            const mat = surfaceMaterial('sample', new THREE.Color(.4, .45, .5), 1,
              {sample: {roughness: .5, metalness: 0, texture: kind,
                textureStrength: strength, textureScale: scale}});
            if (reference) {
              // Same raw kernel and lighting, with octave filtering disabled. This is a diagnostic
              // control: resolved pixels must agree while minified fine octaves must differ.
              const compile = mat.onBeforeCompile;
              mat.onBeforeCompile = shader => {
                compile(shader);
                const source = shader.fragmentShader;
                shader.fragmentShader = source.replace(
                  'float fade = 1.0 - smoothstep(0.2, 0.5, footprint);', 'float fade = 1.0;');
                if (source === shader.fragmentShader) throw Error('unfiltered reference hook missing');
              };
              const key = mat.customProgramCacheKey();
              mat.customProgramCacheKey = () => key + '-unfiltered-reference';
            }
            materials.push(mat); return mat;
          }
          function draw(mat, footprint, tilt = 0) {
            const span = 128 * pitch * footprint;
            const geometry = new THREE.PlaneGeometry(span * 4, span * 4);
            geometries.push(geometry); mesh.geometry = geometry; mesh.material = mat;
            mesh.rotation.x = tilt;
            camera.left = camera.bottom = -span / 2;
            camera.right = camera.top = span / 2;
            camera.updateProjectionMatrix();
            renderer.setRenderTarget(target); renderer.render(scene, camera);
            const pixels = new Uint8Array(128 * 128 * 4);
            renderer.readRenderTargetPixels(target, 0, 0, 128, 128, pixels);
            return pixels;
          }
          function difference(a, b) {
            let total = 0, changed = 0;
            // Interior only: no silhouette or background pixels.
            for (let y = 16; y < 112; y++) for (let x = 16; x < 112; x++) {
              let d = 0;
              for (let c = 0; c < 3; c++) d += Math.abs(a[(y*128+x)*4+c]-b[(y*128+x)*4+c]);
              total += d; if (d > 3) changed++;
            }
            return {mae: total / (96*96*3), changed};
          }
          try {
            const filtered = material(1), unfiltered = material(1, true), zero = material(0);
            const plain = new THREE.MeshStandardMaterial({color: new THREE.Color(.4, .45, .5), roughness: .5});
            materials.push(plain);
            const near = draw(filtered, .02), nearReference = draw(unfiltered, .02);
            const transition = draw(filtered, .3), transitionReference = draw(unfiltered, .3);
            const oblique = draw(filtered, .18, .9), obliqueReference = draw(unfiltered, .18, .9);
            const scaled = draw(material(1, false, 2), .3);
            const flat = draw(plain, .3);
            const mean = kind === 'wood' ? .575 : .5;
            const average = new THREE.MeshStandardMaterial({
              color: new THREE.Color(.4, .45, .5).multiplyScalar(1 - .22 * (1 - mean)),
              roughness: .5 + (.5 - mean) * .3});
            materials.push(average);
            return {near: difference(near, nearReference), detail: difference(near, draw(zero, .02)),
              transition: difference(transition, transitionReference), oblique: difference(oblique, obliqueReference),
              scale: difference(transition, scaled), zero: difference(draw(zero, .3), flat),
              distant: difference(draw(filtered, 2), draw(average, 2)),
              repeat: difference(transition, draw(filtered, .3)), gl: renderer.getContext().getError()};
          } finally {
            renderer.setRenderTarget(previousTarget);
            for (const mat of materials) mat.dispose();
            for (const geometry of geometries) geometry.dispose();
            target.dispose();
          }
        """.replace("KIND", repr(kind)).replace("PITCH", str(pitch))
        )
        assert result["gl"] == 0, result
        assert result["near"]["mae"] < 0.05, result
        assert result["detail"]["changed"] > 1000, result
        assert result["transition"]["mae"] > 0.1, result
        assert result["oblique"]["mae"] > 0.05, result
        assert result["scale"]["changed"] > 1000, result
        assert result["zero"]["mae"] == result["repeat"]["mae"] == 0, result
        assert result["distant"]["mae"] < 0.5, result
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: None)
