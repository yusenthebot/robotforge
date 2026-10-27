"""Native named and static-geom prop selection with reversible viewer edits."""

from __future__ import annotations

import asyncio

import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401


def world(pg):
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    pg.add_object("route", shape="box", size=[0.8, 0.12, 0.22], pos=[0, 0], fixed=True)
    pg.add_object("ball", shape="sphere", size=[0.18], pos=[0, 1])


async def settle_projection(tab):
    previous = None
    for _ in range(60):
        matrix = await tab.eval("window.__robotforge.three().camera.projectionMatrix.elements")
        if previous is not None and matrix == pytest.approx(previous, abs=1e-9):
            return
        previous = matrix
        await asyncio.sleep(0.05)
    raise AssertionError("Panel camera projection did not settle")


async def choose(tab, name):
    await click(tab, "#world-toggle")
    await click(tab, '[data-tab="objects"]')
    await click(tab, f'[data-edit-object="{name}"]')
    await tab.until("!document.getElementById('object-edit-bar').hidden")


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_named_move_rotate_remove_undo_and_static_scene_selection(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await tab.eval("window.__robotforge.lookAt([4,-6,5],[0,0,0])")
        before = server.pg.store.load().model_dump()
        await choose(tab, "route")
        assert await tab.eval("!!window.__robotforge.three().scene.getObjectByName('object-edit-highlight')")
        await click(tab, "#object-move")
        assert await tab.eval("document.querySelector('.repeat-placement').hidden")
        assert await tab.eval("document.getElementById('prop-material-control').hidden")
        await click(tab, "#place-rotate")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([1.5,0,0])"))
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden")
        route = next(o for o in server.pg.store.load().objects if o.name == "route")
        assert route.pos[0] == pytest.approx(1.5, abs=0.02) and route.yaw == 90
        assert [o.name for o in server.pg.store.load().objects] == ["route", "ball"]
        await click(tab, "#undo-edit")
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('undo-edit').hidden")
        assert server.pg.store.load().model_dump() == before
        await choose(tab, "ball")
        await click(tab, "#object-remove")
        await tab.until("!window.__robotforge.busy().busy && !document.getElementById('undo-edit').hidden")
        assert [o.name for o in server.pg.store.load().objects] == ["route"]
        await click(tab, "#undo-edit")
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('undo-edit').hidden")
        assert server.pg.store.load().model_dump() == before
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="objects"]')
        await click(tab, "#edit-objects")
        await settle_projection(tab)
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([0,0,.11])"))
        await tab.until("document.getElementById('object-edit-title').textContent==='route'", timeout=5)
        await key(tab, "Escape")
        assert await tab.eval("document.getElementById('object-edit-bar').hidden")
        assert not await tab.eval("!!window.__robotforge.three().scene.getObjectByName('object-edit-highlight')")
        assert server.pg.store.load().model_dump() == before
        assert await tab.eval("window.__qaErrors") == []

    viewer(size, lang, scenario, world=world)


def test_touch_selection_orbit_cancel_and_external_reset_do_not_edit(viewer):  # noqa: F811
    async def scenario(tab, server):
        from test_viewer_touch_build_browser import eye, touch

        await tab.eval("window.__robotforge.lookAt([4,-6,5],[0,0,0])")
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="objects"]')
        await click(tab, "#edit-objects")
        saved = server.pg.store.file.read_bytes()
        before = await eye(tab)
        await touch(tab, "touchStart", [(1, 170, 330)])
        await touch(tab, "touchMove", [(1, 235, 350)])
        await touch(tab, "touchEnd")
        assert await eye(tab) != pytest.approx(before)
        assert await tab.eval("document.getElementById('object-move').disabled")
        assert server.pg.store.file.read_bytes() == saved
        await tab.eval("window.__robotforge.lookAt([4,-6,5],[0,0,0])")
        await settle_projection(tab)
        x, y = await tab.eval("window.__robotforge.screenOfPoint([0,0,.11])")
        await touch(tab, "touchStart", [(1, x, y)])
        await touch(tab, "touchEnd")
        await tab.until("document.getElementById('object-edit-title').textContent==='route'", timeout=5)
        await click(tab, "#object-move")
        await click(tab, "#place-rotate")
        await key(tab, "Escape")
        assert server.pg.store.file.read_bytes() == saved
        await choose(tab, "route")
        server.pg.session.reset()
        await tab.until("document.getElementById('object-edit-bar').hidden", timeout=5)
        assert server.pg.store.file.read_bytes() == saved
        assert await tab.eval("window.__qaErrors") == []

    viewer("phone", "en", scenario, world=world)
