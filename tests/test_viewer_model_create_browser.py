"""Native model selection, joint exploration and the existing saved-layout loop."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from test_viewer_creation_browser import click
from test_viewer_layout import pytestmark, viewer  # noqa: F401

from robotforge.design import combine
from robotforge.design.world import RobotEntry


def world(pg):
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    combine.save(RobotEntry(name="source", template="arm", params={"color": "red"}), "local_arm", pg.store.dir)


async def begin(tab, name="local_arm"):
    if await tab.eval("document.getElementById('drawer').hidden"):
        await click(tab, "#drawer-toggle")
    if await tab.eval("document.getElementById('create-options').hidden"):
        await click(tab, "#create-toggle")
    await click(tab, f'#model-list [data-model="{name}"]')
    await tab.until("!document.getElementById('placement-bar').hidden")
    await tab.run("window.__robotforge.lookAt([5,-7,5],[1,0,.1])")
    await asyncio.sleep(0.7)
    previous = None
    for _ in range(30):
        point = await tab.eval("window.__robotforge.screenOfPoint([2,0,0])")
        if previous is not None and point == pytest.approx(previous, abs=1e-7):
            break
        previous = point
        await asyncio.sleep(0.05)
    assert await tab.eval(f"document.elementFromPoint({point[0]},{point[1]}).id") == "view"
    await click(tab, point=point)
    await tab.until(
        f"window.__robotforge.selected()==='{name}' && !window.__robotforge.busy().busy && "
        "document.getElementById('placement-bar').hidden"
    )


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("laptop", "zh-CN"), ("phone", "en"), ("phone", "zh-CN")])
def test_native_model_controls_undo_save_restore_and_reload(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
        source_ctrl = server.pg.session.data.ctrl.copy()
        await begin(tab)
        assert server.pg.store.load().robots[-1].params == {"color": "red"}
        model, data = server.pg.session.model, server.pg.session.data
        source_ids = [a for a in range(model.nu) if model.actuator(a).name.startswith("rover/")]
        np.testing.assert_array_equal(data.ctrl[source_ids], source_ctrl)
        actuator = next(a for a in range(model.nu) if model.actuator(a).name.startswith("local_arm/"))
        joint = int(model.actuator_trnid[actuator, 0])
        adr = int(model.jnt_qposadr[joint])
        initial = float(data.qpos[adr])
        await click(tab, "#drawer-toggle")
        point = await tab.eval("""(() => {const e=document.querySelector('#joints input[type="range"]');
          e.scrollIntoView({block:'center'}); const r=e.getBoundingClientRect();
          return [r.x + r.width*.7, r.y+r.height/2];})()""")
        await click(tab, point=point)
        await asyncio.sleep(0.2)
        assert data.ctrl[actuator] != 0
        if not await tab.eval("document.getElementById('drawer').hidden"):
            await click(tab, "#drawer-toggle")
        await click(tab, "#btn-play")
        await asyncio.sleep(0.5)
        await click(tab, "#btn-play")
        assert abs(float(data.qpos[adr]) - initial) > 0.001
        np.testing.assert_array_equal(data.ctrl[source_ids], source_ctrl)
        await click(tab, "#undo-add")
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('undo-add').hidden")
        assert [r.name for r in server.pg.store.load().robots] == ["rover"]
        await begin(tab)
        saved = server.pg.store.load().model_dump(mode="json")
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="maps"]')
        await click(tab, "#layout-name")
        await tab.page.call("Input.insertText", text="My robot workspace")
        await click(tab, "#layout-save")
        await tab.until("document.querySelector('.layout-row') && !document.getElementById('layout-save').disabled")
        await click(tab, "#map-list .choice:nth-child(2)")
        await tab.until("!window.__robotforge.busy().busy && !document.getElementById('layout-save').disabled")
        assert server.pg.store.load().model_dump(mode="json") != saved
        await click(tab, '[data-layout-action="restore"]')
        await click(tab, "#layout-apply")
        await tab.until("!window.__robotforge.busy().busy && !document.getElementById('layout-save').disabled")
        assert server.pg.store.load().model_dump(mode="json") == saved
        await tab.page.call("Page.reload")
        await tab.until("!!window.__robotforge?.scene && document.getElementById('loading').hidden")
        assert server.pg.store.load().model_dump(mode="json") == saved
        if await tab.eval("document.getElementById('drawer').hidden"):
            await click(tab, "#drawer-toggle")
        await asyncio.sleep(0.3)
        lighting = "window.__robotforge.three().lighting"
        resources = """(() => {const {scene,renderer}=window.__robotforge.three();
          return [scene.environment.uuid,renderer.info.memory.textures];})()"""
        before = await tab.eval(resources)
        qpos = server.pg.session.data.qpos.copy()
        await click(tab, "#lighting-daylight")
        assert await tab.eval(f"{lighting}.preset()") == "daylight"
        assert await tab.eval("document.getElementById('lighting-daylight').getAttribute('aria-pressed')") == "true"
        assert await tab.eval(resources) == before
        np.testing.assert_array_equal(server.pg.session.data.qpos, qpos)
        assert server.pg.store.load().model_dump(mode="json") == saved
        await tab.page.call("Page.reload")
        await tab.until(f"!!window.__robotforge?.scene && {lighting}.preset()==='daylight'")
        assert not tab.errors

    viewer(size, lang, scenario, world=world)
