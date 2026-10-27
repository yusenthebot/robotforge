"""Native prop inspection frames current geometry without changing the world or simulation."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_object_edit_browser import choose
from test_viewer_startup_browser import READY


def world(pg):
    pg.add_object("route", shape="box", size=[0.8, 0.12, 0.22], pos=[0, 0], fixed=True)
    pg.add_object("falling", shape="sphere", size=[0.18], pos=[0, 1, 1.5])
    pg.add_object("table", shape="table", size=[1.1, 0.6, 0.7], pos=[2, 0], fixed=True)
    pg.add_object("tiny", shape="box", size=[0.015, 0.02, 0.01], pos=[-2, 0], fixed=True)


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_native_focus_frames_fixed_free_compound_and_tiny_props(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await tab.until(READY)
        await asyncio.sleep(1.0)  # A free object falls from its saved spawn in the real simulation.
        await click(tab, "#btn-play")
        await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
        saved = server.pg.store.file.read_bytes()
        with server.pg.session.lock:
            data = server.pg.session.data
            before = [v.copy() for v in (data.qpos, data.qvel, data.ctrl)]
            stamp = data.time
        await tab.eval("window.__robotforge.lookAt([5,-8,4],[0,0,.2])")
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="objects"]')
        await click(tab, "#edit-objects")
        assert await tab.eval("document.getElementById('object-focus').disabled")
        await key(tab, "Escape")
        for name in ("route", "falling", "table", "tiny"):
            await tab.eval("window.__robotforge.lookAt([12,-16,10],[0,0,.2])")
            await choose(tab, name)
            initial = await tab.run("""
              const rf=window.__robotforge,{camera}=rf.three(),T=await import('three');
              const away=camera.position.clone().sub(new T.Vector3().fromArray(rf.cameraTarget()));
              return {distance:away.length(),direction:away.normalize().toArray()};
            """)
            await click(tab, "#object-focus")
            await tab.until("!window.__robotforge.tweening()")
            await asyncio.sleep(0.3)  # The panel's free-view projection also eases.
            framing = await tab.run("""
              const rf=window.__robotforge,{scene,camera}=rf.three(),T=await import('three');
              const highlight=scene.getObjectByName('object-edit-highlight'),box=highlight.box;
              const rect=rf.freeRect(),points=[];camera.updateMatrixWorld();
              for(const x of [box.min.x,box.max.x])for(const y of [box.min.y,box.max.y])
                for(const z of [box.min.z,box.max.z]){
                const p=new T.Vector3(x,y,z).project(camera);
                points.push([(p.x+1)*innerWidth/2,(1-p.y)*innerHeight/2,p.z]);
              }
              const direction=camera.position.clone().sub(new T.Vector3().fromArray(rf.cameraTarget())).normalize();
              return {points,rect,center:box.getCenter(new T.Vector3()).toArray(),target:rf.cameraTarget(),
                direction:direction.toArray(),
                distance:camera.position.distanceTo(new T.Vector3().fromArray(rf.cameraTarget())),
                gl:rf.three().renderer.getContext().getError()};
            """)
            assert framing["target"] == pytest.approx(framing["center"], abs=1e-6)
            assert framing["direction"] == pytest.approx(initial["direction"], abs=1e-5)
            assert framing["distance"] < initial["distance"] * 0.8
            r = framing["rect"]
            for x, y, z in framing["points"]:
                assert r["left"] <= x <= r["right"] and r["top"] <= y <= r["bottom"]
                assert -1 < z < 1
            if name == "falling":
                assert framing["center"][2] < 0.2
            assert framing["gl"] == 0
            await key(tab, "Escape")
        assert server.pg.store.file.read_bytes() == saved
        with server.pg.session.lock:
            current = server.pg.session.data
            assert current is data and server.pg.session.paused
            for actual, expected in zip((current.qpos, current.qvel, current.ctrl), before, strict=True):
                np.testing.assert_array_equal(actual, expected)
            assert current.time == stamp
        assert await tab.eval("window.__qaErrors") == []

    viewer(size, lang, scenario, world=world)
