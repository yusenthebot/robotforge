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
