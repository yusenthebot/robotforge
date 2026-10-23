"""The real viewer in headless Chrome draws every asset-engine surface: each procedural texture (brushed, speckle,
rubber, knurl) compiles as a shader and draws, and a material a plugin adds reaches the page through the scene
description with no change to the viewer.

A GLSL mistake in a texture is a console error and a black part, which no Python check can see. Marked slow, and run
where Chrome renders WebGL for sure (macOS, or anywhere with ROBOTFORGE_BROWSER_TESTS=1).
"""

from __future__ import annotations

import asyncio
import json

import pytest

# Chrome plumbing shared with the sensor views' browser checks
from test_viewer_sensors_browser import (  # noqa: F401  (a fixture)
    _page,
    _serve,
    chrome,
    pytestmark,  # noqa: F401  (slow, needs Chrome on a machine that renders)
)

from robotforge import assets
from robotforge.design.visuals import mesh as M
from robotforge.extend import generator as plugin_generator
from robotforge.extend import material as plugin_material
from robotforge.extend import registry
from robotforge.playground import Playground

READY = "!!(window.__robotforge?.scene && window.__robotforge.time > 0)"


@pytest.fixture
def served(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    with registry.isolated():
        plugin_material("woven_felt", roughness=0.95, texture="rubber", texture_scale=0.7, summary="a felt")

        @plugin_generator("knurled_knob", summary="a knurled knob", params={"radius": (0.02, 0.01, 0.05, "its radius")})
        def knob(a):
            body = M.cylinder(a.radius, 0.03, segments=32, z0=0.0)
            cap = M.cylinder(a.radius * 0.6, 0.006, segments=24, z0=0.03)
            return assets.Asset(
                "knurled_knob",
                (assets.Piece("metal", "knurled", body), assets.Piece("accent", "woven_felt", cap)),
                (assets.Collider("cylinder", (a.radius, 0.018), (0, 0, 0.018)),),
            )

        pg = Playground.with_demo()
        pg.new_world("surfaces")
        for i, style in enumerate(("cobot", "lab", "hobby", "rugged")):
            pg.add_robot(f"arm_{style}", template="arm", params={"dof": 3, "style": style}, pos=[0.0, 0.4 * (i - 1.5)])
        pg.add_robot("knob", template="asset", params={"generator": "knurled_knob"}, pos=[0.5, 0.0])
        pg.set_floor("wood")
        pg.add_terrain("stairs", pos=[1.5, 0], style="wood")
        pg.add_terrain("hills", pos=[-3, 0], params={"size": [2, 2]}, style="grass")
        server, thread, port = _serve(pg)
        yield pg, port
        server.should_exit = True
        thread.join(timeout=10)


def test_every_surface_compiles_draws_and_reaches_the_page(served, chrome):  # noqa: F811
    _, port = served

    async def run() -> dict:
        import websockets
        from test_viewer_sensors_browser import Tab, _wait

        version = json.loads(_wait(f"http://127.0.0.1:{chrome}/json/version"))
        async with websockets.connect(version["webSocketDebuggerUrl"], max_size=None) as ws:
            browser = Tab(ws)
            target = (await browser.call("Target.createTarget", url="about:blank"))["targetId"]
            tab = Tab(ws, (await browser.call("Target.attachToTarget", targetId=target, flatten=True))["sessionId"])
            await tab.call(
                "Emulation.setDeviceMetricsOverride", width=1280, height=800, deviceScaleFactor=1, mobile=False
            )
            await tab.call("Page.enable")
            await tab.call("Page.bringToFront")
            await tab.call(
                "Page.addScriptToEvaluateOnNewDocument",
                source="window.__errors = []; const e = console.error; "
                "console.error = (...a) => { window.__errors.push(a.map(String).join(' ')); e(...a); };",
            )
            await tab.call("Page.navigate", url=f"http://127.0.0.1:{port}/?lang=en")
            for _ in range(240):
                if await tab.eval(READY):
                    break
                await asyncio.sleep(0.25)
            await asyncio.sleep(2.0)  # shaders compile on the first frames
            return await tab.eval(
                """(() => {
                  const rf = window.__robotforge;
                  const geoms = rf.scene.geoms.filter((g) => g.material);
                  return {
                    errors: window.__errors,
                    surfaces: Object.keys(rf.scene.surfaces || {}),
                    meshes: geoms.length,
                  };
                })()"""
            )

    seen = asyncio.run(run())
    assert not [e for e in seen["errors"] if "Shader" in e or "WebGLProgram" in e or "GLSL" in e], seen["errors"][:3]
    names = {name.split("/")[-1] for name in seen["surfaces"]}
    assert {"rf_gloss", "rf_brushed", "rf_cast", "rf_tread", "rf_soft_touch", "rf_knurled", "rf_woven_felt"} <= names
    assert seen["meshes"] > 40


def test_surface_controls_pixels_and_shared_ground_lifetime(served, chrome):  # noqa: F811
    """Render small controlled samples on the real GPU; zero strength must actually be flat.

    Ground meshes need independent UV transforms but share allocations, which must be reclaimed on disposal.
    """
    _, port = served

    async def run() -> dict:
        async with _page(chrome, f"http://127.0.0.1:{port}/?lang=en") as tab:
            return await tab.eval(
                """(async () => {
                  const THREE = await import('three');
                  const {surfaceMaterial} = await import('./static/look.js');
                  const {groundMaterial, groundStyle} = await import('./static/ground-look.js');
                  const renderer = window.__robotforge.three().renderer;
                  const errors = [], originalError = console.error;
                  console.error = (...args) => { errors.push(args.map(String).join(' ')); originalError(...args); };
                  const originalTarget = renderer.getRenderTarget();
                  const scene = new THREE.Scene();
                  scene.environment = window.__robotforge.three().scene.environment;
                  scene.add(new THREE.AmbientLight(0xffffff, 1));
                  const light = new THREE.DirectionalLight(0xffffff, 2);
                  light.position.set(1, 2, 3); scene.add(light);
                  const camera = new THREE.OrthographicCamera(-0.04, 0.04, 0.04, -0.04, 0.01, 2);
                  camera.position.z = 1;
                  const geometry = new THREE.PlaneGeometry(0.08, 0.08);
                  const sample = new THREE.Mesh(geometry); scene.add(sample);
                  const target = new THREE.WebGLRenderTarget(256, 256);
                  const color = new THREE.Color(0.4, 0.45, 0.5);
                  const draw = (mat) => {
                    sample.material = mat;
                    renderer.setRenderTarget(target); renderer.render(scene, camera);
                    const pixels = new Uint8Array(256 * 256 * 4);
                    renderer.readRenderTargetPixels(target, 0, 0, 256, 256, pixels);
                    return pixels;
                  };
                  const diff = (a, b) => {
                    let changed = 0;
                    for (let i = 0; i < a.length; i += 4) {
                      if (Math.max(...[0, 1, 2].map(c => Math.abs(a[i+c] - b[i+c]))) > 2) changed++;
                    }
                    return changed;
                  };
                  const surface = (texture, textureStrength, textureScale = 1) =>
                    surfaceMaterial('sample', color, 1, {sample: {roughness: 0.5, metalness: 0,
                      texture, textureStrength, textureScale}});
                  const mats = [];
                  const picture = (...args) => {
                    const mat = surface(...args); mats.push(mat); return draw(mat);
                  };
                  const flat = picture('none', 1);
                  const zeroWeave = diff(flat, picture('weave', 0));
                  const zeroGrain = diff(flat, picture('speckle', 0));
                  const weaveScale = diff(picture('weave', 1, 0.5), picture('weave', 1, 2));
                  const grainStrength = diff(flat, picture('speckle', 1));
                  const samples = {};
                  for (const name of ['rf_wood', 'rf_bark', 'rf_stone', 'rf_fabric']) {
                    const mat = surfaceMaterial(name, color, 1); mats.push(mat);
                    const pixels = draw(mat);
                    samples[name] = pixels.reduce((total, value, i) => total + (i % 4 < 3 ? value : 0), 0);
                  }
                  for (const mat of mats) mat.dispose();
                  // Allocate the render target before taking a texture baseline.
                  const baseline = renderer.info.memory.textures;
                  let allocated = 0, remaining = 0;
                  const grounds = [];
                  for (let pass = 0; pass < 3; pass++) {
                    for (let i = 0; i < 20; i++) {
                      const mat = groundMaterial('sand', color, 1);
                      for (const tex of new Set([mat.map, mat.bumpMap, mat.roughnessMap])) tex.repeat.set(i + 1, 2);
                      grounds.push(mat); draw(mat);
                    }
                    allocated = Math.max(allocated, renderer.info.memory.textures - baseline);
                    for (const mat of grounds.splice(0)) {
                      for (const tex of new Set([mat.map, mat.bumpMap, mat.roughnessMap])) tex.dispose();
                      mat.dispose();
                    }
                    remaining = Math.max(remaining, renderer.info.memory.textures - baseline);
                  }
                  const legacy = ['planks', 'tiles', 'speckle'].map(pattern => groundStyle('external', {pattern}));
                  const namespaced = groundStyle('patch/ground_sand', {pattern: 'speckle'});
                  renderer.setRenderTarget(originalTarget);
                  geometry.dispose(); target.dispose();
                  console.error = originalError;
                  return {zeroWeave, zeroGrain, weaveScale, grainStrength, allocated, remaining, legacy, namespaced,
                    samples, errors};
                })()"""
            )

    seen = asyncio.run(run())
    assert seen["zeroWeave"] == seen["zeroGrain"] == 0, seen
    assert seen["weaveScale"] > 1000 and seen["grainStrength"] > 1000, seen
    assert seen["allocated"] == 2 and seen["remaining"] == 0, seen
    assert seen["legacy"] == ["wood", "tile", "concrete"]
    assert seen["namespaced"] == "sand"
    assert not seen["errors"], seen["errors"]
    assert all(total > 1_000_000 for total in seen["samples"].values()), seen["samples"]


@pytest.mark.parametrize("width", [1280, 390])
def test_ground_previews_switch_real_world_without_texture_growth(served, chrome, width):  # noqa: F811
    _, port = served

    async def run() -> dict:
        from test_viewer_sensors_browser import _until

        async with _page(chrome, f"http://127.0.0.1:{port}/?lang=zh-CN") as tab:
            await tab.call(
                "Emulation.setDeviceMetricsOverride", width=width, height=844, deviceScaleFactor=1, mobile=False
            )
            await tab.eval(
                "if (document.getElementById('world-panel').hidden) document.getElementById('world-toggle').click();"
                "document.querySelector('[data-tab=terrain]').click();"
            )
            assert await tab.eval(
                "!document.getElementById('world-panel').hidden && "
                "!document.querySelector('[data-body=terrain]').hidden"
            )
            marker = await tab.eval(
                """(() => {
                  const table = document.createElement('table'); table.className = 'tv-table';
                  const row = table.insertRow(), cell = row.insertCell();
                  const marker = document.createElement('span'); marker.className = 'swatch';
                  cell.append(marker); document.body.append(table);
                  const r = marker.getBoundingClientRect(); table.remove();
                  return {width: r.width, height: r.height};
                })()"""
            )
            previews = await tab.eval(
                """[...document.querySelectorAll('#ground-list button')].map(b => {
                  const c = b.querySelector('canvas'), r = b.getBoundingClientRect();
                  const pixels = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                  const reds = [...pixels].filter((_, i) => i % 4 === 0);
                  return {style: b.dataset.ground, variation: Math.max(...reds)-Math.min(...reds),
                    left: r.left, right: r.right, width: r.width, height: r.height, label: b.textContent.trim()};
                })"""
            )
            counts = []
            for style in ("tile", "grass", "wood", "tile", "grass", "wood"):
                version = await tab.eval("window.__robotforge.scene.version")
                await tab.eval(f"document.querySelector('[data-ground={style}]').click()")
                await _until(
                    tab,
                    f"window.__robotforge.scene.version > {version} && "
                    f"window.__robotforge.scene.floor === '{style}' && !window.__robotforge.busy().busy",
                    timeout=15,
                )
                counts.append(
                    await tab.eval(
                        "(() => { const {renderer, scene, camera} = window.__robotforge.three(); "
                        "renderer.render(scene, camera); return renderer.info.memory.textures; })()"
                    )
                )
            pressed = await tab.eval("document.querySelector('#ground-list [aria-pressed=true]').dataset.ground")
            return {"previews": previews, "counts": counts, "pressed": pressed, "marker": marker}

    seen = asyncio.run(run())
    assert len(seen["previews"]) == 9
    assert all(p["variation"] > 0 and p["label"] for p in seen["previews"]), seen
    assert all(p["left"] >= 0 and p["right"] <= width for p in seen["previews"]), seen
    assert all(p["width"] >= 64 and p["height"] >= 55 for p in seen["previews"]), seen
    assert seen["counts"][:3] == seen["counts"][3:], seen
    assert seen["pressed"] == "wood"
    assert seen["marker"] == {"width": 9, "height": 9}, seen
