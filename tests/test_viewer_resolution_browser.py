"""Actual Chrome events keep the drawing buffer in step with display density."""

from __future__ import annotations

import pytest
from test_viewer_creation_browser import click
from test_viewer_heading_browser import place, world
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_objects_browser import choose_object
from test_viewer_startup_browser import READY
from viewer_camera import settle_camera


async def metrics(tab, ratio, width=1280, height=800):
    await tab.page.call(
        "Emulation.setDeviceMetricsOverride",
        width=width,
        height=height,
        deviceScaleFactor=ratio,
        mobile=False,
    )
    # Chrome's headless compositor can defer resolution notifications until a paint.
    await tab.page.call("Page.captureScreenshot", format="png")
    await settle_camera(tab)


async def fresh(tab, server, ratio, width=1280, height=800):
    server.pg.session.paused = True
    server.pg.session.reset()
    await metrics(tab, ratio, width, height)
    await tab.page.call("Page.navigate", url=f"http://127.0.0.1:{tab.port}/?lang={tab.lang}")
    await tab.until(READY + " && window.__robotforge.time===0")
    await settle_camera(tab)


async def buffer(tab, ratio, width=1280, height=800):
    await tab.until(f"window.__robotforge.three().renderer.getPixelRatio()==={min(ratio, 2)}")
    actual = await tab.run("""
      const {renderer,camera}=window.__robotforge.three(), gl=renderer.getContext();
      return {device:devicePixelRatio,ratio:renderer.getPixelRatio(),
        buffer:[gl.drawingBufferWidth,gl.drawingBufferHeight],
        aspect:camera.aspect,gl:gl.getError()};
    """)
    assert actual == {
        "device": ratio,
        "ratio": min(ratio, 2),
        "buffer": [int(width * min(ratio, 2)), int(height * min(ratio, 2))],
        "aspect": width / height,
        "gl": 0,
    }


def test_delivered_resolution_events_change_density_without_reframing(viewer):  # noqa: F811
    async def scenario(tab, server):
        await fresh(tab, server, 1)
        await tab.run("window.__robotforge.lookAt([6,-8,8],[0,0,0])")
        await settle_camera(tab)
        await tab.run("""
          window.__densityEvents=[];
          window.__densityQueries=[1,1.25,2,3].map(ratio=>{
            const q=matchMedia(`(resolution: ${ratio}dppx)`);
            q.addEventListener('change',e=>window.__densityEvents.push({dpr:devicePixelRatio,trusted:e.isTrusted}));
            return q;
          });
        """)
        camera = """(() => {
          const {camera}=window.__robotforge.three();
          return {eye:camera.position.toArray(),target:window.__robotforge.cameraTarget(),
            projection:camera.projectionMatrix.toArray()};
        })()"""
        before = await tab.eval(camera)
        for ratio in [2, 1.25, 3]:
            await metrics(tab, ratio)
            await tab.until(f"window.__densityEvents.some(e=>e.dpr==={ratio}&&e.trusted)")
            await buffer(tab, ratio)
            after = await tab.eval(camera)
            assert after["projection"] == before["projection"]
            for key in ["eye", "target"]:
                assert after[key] == pytest.approx(before[key], rel=0, abs=1e-10)
        # Placement continues to use CSS coordinates after a density-only change.
        await choose_object(tab, "barrier")
        await click(tab, "#place-rotate-plus")
        await place(tab, [1, 1, 0])
        placed = server.pg.current_world().objects[-1]
        assert placed.yaw == 15
        assert placed.pos == pytest.approx([1, 1])
        assert not tab.errors

    viewer("laptop", "en", scenario, world=world)


def test_resize_and_density_change_updates_once_and_preserves_simulation(viewer):  # noqa: F811
    async def scenario(tab, server):
        await fresh(tab, server, 3)
        await buffer(tab, 3)
        original = server.pg.store.file.read_bytes()
        state = server.pg.session.data.qpos.copy()
        await tab.run("""
          const r=window.__robotforge.three().renderer, original=r.setDrawingBufferSize;
          window.__bufferChanges=0;
          r.setDrawingBufferSize=function(...args){window.__bufferChanges++;return original.apply(this,args)};
        """)
        # CDP's pure 3->1 override does not reliably emit media events. Exercise
        # a real CSS resize here, and keep media-only coverage in the test above.
        await metrics(tab, 1, 1281)
        await buffer(tab, 1, 1281)
        assert await tab.eval("window.__bufferChanges") == 1
        await metrics(tab, 1, 1281)
        assert await tab.eval("window.__bufferChanges") == 1
        await metrics(tab, 1)
        await buffer(tab, 1)
        assert await tab.eval("window.__bufferChanges") == 2
        assert server.pg.store.file.read_bytes() == original
        assert (server.pg.session.data.qpos == state).all()
        assert server.pg.session.paused and server.pg.session.data.time == 0 and not tab.errors

    viewer("laptop", "en", scenario)


@pytest.mark.parametrize("ratio", [1, 3])
def test_initial_default_canvas_dimensions_still_initialize_camera_aspect(viewer, ratio):  # noqa: F811
    async def scenario(tab, server):
        await fresh(tab, server, ratio, 300, 150)
        await buffer(tab, ratio, 300, 150)
        assert await tab.eval("window.__robotforge.three().camera.projectionMatrix.elements.every(Number.isFinite)")
        assert not tab.errors

    viewer("laptop", "en", scenario)
