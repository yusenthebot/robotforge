"""Native robot-start editing, rotated replay and existing prop compatibility."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401


def world(pg):
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    pg.add_object("barrier", shape="box", size=[0.8, 0.12, 0.22], pos=[0, 0], fixed=True)


async def begin(tab):
    await tab.until(
        "!document.getElementById('robot-start').hidden && !document.getElementById('robot-start').disabled"
    )
    await click(tab, "#drawer-toggle")
    await click(tab, "#robot-start")
    await tab.until("!document.getElementById('placement-bar').hidden")
    await asyncio.sleep(0.5)


async def spot(tab, position):
    await tab.run("window.__robotforge.lookAt([5,-6,5],[0,0,.1])")
    await asyncio.sleep(0.6)
    previous = None
    for _ in range(20):
        point = await tab.eval(f"window.__robotforge.screenOfPoint({position!r})")
        if previous is not None and point == pytest.approx(previous, abs=1e-6):
            break
        previous = point
        await asyncio.sleep(0.05)
    assert await tab.eval(f"document.elementFromPoint({point[0]},{point[1]}).id") == "view"
    await click(tab, point=point)


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("laptop", "zh-CN"), ("phone", "en"), ("phone", "zh-CN")])
def test_native_reposition_rotate_reset_and_one_use_undo(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        saved = server.pg.store.load().model_dump()
        await begin(tab)
        assert await tab.eval("!document.getElementById('object-placement-controls').hidden")
        assert await tab.eval("document.querySelector('.repeat-placement').hidden")
        assert await tab.eval("Array.from(document.querySelectorAll('.prop-size-control')).every(e=>e.hidden)")
        await click(tab, "#place-rotate")
        await spot(tab, [2, 0, 0])
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden")
        entry = server.pg.store.load().robots[0]
        assert entry.pos == pytest.approx([2, 0], abs=0.015) and entry.yaw == 90
        assert server.pg.session.paused
        assert await tab.eval("!document.getElementById('undo-edit').hidden")
        qpos = server.pg.session.data.qpos.copy()
        server.pg.session.step(5)
        server.pg.session.reset()
        np.testing.assert_array_equal(server.pg.session.data.qpos, qpos)
        # Reset intentionally invalidates undo; make another real edit first.
        await tab.until("document.getElementById('undo-edit').hidden")
        await begin(tab)
        await spot(tab, [-2, 0, 0])
        await tab.until("!window.__robotforge.busy().busy && !document.getElementById('undo-edit').hidden")
        await click(tab, "#undo-edit")
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('undo-edit').hidden")
        assert server.pg.store.load().robots[0].pos == entry.pos
        assert server.pg.store.load().robots[0].yaw == entry.yaw
        assert server.pg.store.load().robots[0].model_dump(exclude={"pos", "yaw"}) == {
            k: v for k, v in saved["robots"][0].items() if k not in {"pos", "yaw"}
        }
        assert not tab.errors

    viewer(size, lang, scenario, world=world)


def test_cancel_and_occupied_rejection_preserve_world_and_selected_robot(viewer):  # noqa: F811
    async def scenario(tab, server):
        saved = server.pg.store.file.read_bytes()
        await begin(tab)
        await click(tab, "#place-rotate")
        await key(tab, "Escape")
        assert server.pg.store.file.read_bytes() == saved
        await begin(tab)
        await spot(tab, [0, 0, 0.22])
        await tab.until("!window.__robotforge.busy().busy")
        assert server.pg.store.file.read_bytes() == saved
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        server.pg.session.reset()
        await tab.until("document.getElementById('placement-bar').hidden")
        assert not tab.errors

    viewer("phone", "zh-CN", scenario, world=world)


@pytest.mark.parametrize("lang", ["en", "zh-CN"])
def test_phone_320_start_controls_fit_and_remain_native_clickable(viewer, lang):  # noqa: F811
    async def scenario(tab, server):
        saved = server.pg.store.file.read_bytes()
        await tab.page.call(
            "Emulation.setDeviceMetricsOverride", width=320, height=844, deviceScaleFactor=1, mobile=True
        )
        await begin(tab)
        controls = await tab.eval("""['place-rotate','place-cancel','place-here'].map(id=>{
          const e=document.getElementById(id),r=e.getBoundingClientRect();
          const hit=e.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2));
          return {id,left:r.left,right:r.right,top:r.top,bottom:r.bottom,hit};})""")
        assert all(
            c["left"] >= 0 and c["right"] <= 320 and c["top"] >= 0 and c["bottom"] <= 844 and c["hit"] for c in controls
        )
        await click(tab, "#place-rotate")
        await click(tab, "#place-cancel")
        assert server.pg.store.file.read_bytes() == saved and not tab.errors

    viewer("phone", lang, scenario, world=world)


def test_actual_fixed_arm_moves_onto_tabletop_and_keeps_its_configuration(viewer):  # noqa: F811
    def arm_world(pg):
        pg.add_robot("arm", template="arm", pos=[-2, 0])
        pg.add_object("table", shape="table", size=[2, 2, 0.8], pos=[1, 0], fixed=True)

    async def scenario(tab, server):
        before = server.pg.store.load().robots[0]
        await begin(tab)
        await click(tab, "#place-rotate")
        await spot(tab, [1, 0, 0.8])
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden")
        after = server.pg.store.load().robots[0]
        assert after.pos[:2] == pytest.approx([1, 0], abs=0.015) and after.pos[2] == pytest.approx(0.8)
        assert after.yaw == 90 and after.model_dump(exclude={"pos", "yaw"}) == before.model_dump(exclude={"pos", "yaw"})
        assert server.pg.session.paused and not tab.errors

    viewer("laptop", "en", scenario, world=arm_world)
