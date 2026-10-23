"""Small creative scenes stay visible at the retained camera; water keeps its own fog."""

from __future__ import annotations

import pytest
from test_viewer_creation_browser import click
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_touch_build_browser import blue_at


async def fog(tab):
    return await tab.eval("""(()=>{const {scene,camera}=window.__robotforge.three();
      return {near:scene.fog.near,far:scene.fog.far,clip:camera.far};})()""")


def test_small_prop_remains_visible_far_and_grazing_and_fog_contracts_again(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await tab.eval("window.__robotforge.lookAt([.3,-.4,.5],[0,0,.18])")
        close = await fog(tab)
        await tab.run("window.__aboveFog=window.__robotforge.three().scene.fog")
        for eye in ([4, -5, 3], [0, -6.92, 0.4]):
            await tab.eval(f"window.__robotforge.lookAt({eye},[0,0,.18])")
            far = await fog(tab)
            assert far["near"] > close["near"]
            assert far["far"] > far["near"] and far["clip"] > far["far"]
            assert await blue_at(tab, [0, 0, 0.18]) > 5
        await tab.eval("window.__robotforge.lookAt([.3,-.4,.5],[0,0,.18])")
        assert await fog(tab) == pytest.approx(close)
        assert await tab.eval("window.__aboveFog === window.__robotforge.three().scene.fog")
        assert await tab.eval("document.getElementById('empty').hidden")
        assert not tab.errors

    def world(pg):
        pg.add_object("blue", shape="box", size=[0.3, 0.3, 0.36], pos=[0, 0], color="blue", fixed=True)

    viewer("phone", "en", scenario, world=world)


def test_camera_changes_underwater_preserve_attenuation_and_restore_above_fog(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await tab.eval("window.__robotforge.lookAt([3,-4,3],[0,0,1])")
        await tab.until("window.__robotforge.three().scene.fog.near > .05")
        await tab.run("window.__aboveFog=window.__robotforge.three().scene.fog")
        await tab.eval("window.__robotforge.lookAt([0,-2,.5],[0,0,.5])")
        await tab.until("window.__robotforge.three().scene.fog !== window.__aboveFog")
        under = await fog(tab)
        assert under["near"] == pytest.approx(0.05)
        await tab.eval("window.__robotforge.lookAt([0,-8,.5],[0,0,.5])")
        assert await fog(tab) == pytest.approx(under)
        await tab.eval("window.__robotforge.lookAt([0,-8,3],[0,0,1])")
        await tab.until("window.__robotforge.three().scene.fog === window.__aboveFog")
        above = await fog(tab)
        assert above["near"] > 8 and above["far"] > above["near"]
        assert await tab.eval("document.getElementById('empty').hidden")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.set_floor(water=1))


def test_empty_hint_returns_only_when_last_visible_terrain_is_removed(viewer):  # noqa: F811
    async def scenario(tab, _server):
        assert not await tab.eval("document.getElementById('empty').hidden")
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="terrain"]')
        index = await tab.eval("window.__robotforge.scene.catalog.terrains.indexOf('stairs')")
        await click(tab, f"#terrain-list .choice:nth-child({index + 1})")
        await tab.until("window.__robotforge.scene.terrain.includes('stairs') && !window.__robotforge.busy().busy")
        await click(tab, "#world-toggle")
        assert await tab.eval("document.getElementById('empty').hidden")
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="maps"]')
        await click(tab, "#map-list .choice:first-child")
        await tab.until("!window.__robotforge.scene.terrain.length && !window.__robotforge.busy().busy")
        await click(tab, "#world-toggle")
        assert not await tab.eval("document.getElementById('empty').hidden")
        assert not tab.errors

    viewer("phone", "zh-CN", scenario, world=lambda pg: None)
