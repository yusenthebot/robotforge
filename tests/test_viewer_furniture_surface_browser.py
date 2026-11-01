"""Furniture finishes through native controls, mixed selection and reusable arrangements."""

from __future__ import annotations

import json

import pytest
from test_viewer_arrangements_browser import open_shelf, place, prepare, reload
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_object_surface_batch_browser import pause, select_group
from test_viewer_object_surface_browser import apply_surface
from test_viewer_startup_browser import READY

NAMES = ("desk", "tray", "block")


def furniture(pg):
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    pg.add_object("desk", shape="table", size=[1, 0.6, 0.7], pos=[0, 1.5], color="#a98a65")
    pg.add_object("tray", shape="bin", size=[0.4, 0.3, 0.2], pos=[1, 0], yaw=30, fixed=True)
    pg.add_object("block", size=[0.2] * 3, pos=[0, 0], material="steel")


async def choose(tab, name):
    await open_shelf(tab)
    await click(tab, f'[data-edit-object="{name}"]')
    await tab.until("!document.getElementById('object-edit-bar').hidden")
    assert not await tab.eval("document.getElementById('object-surface').hidden")
    await click(tab, "#object-surface")


async def drawn(tab, name):
    return await tab.run(
        """
      const rf=window.__robotforge,entry=rf.scene.object_edit.objects.find(e=>e.name===NAME),rows=[];
      rf.three().scene.traverse(o=>{if(o.isMesh&&o.visible&&entry.geoms.includes(o.userData.geom)){
        const g=rf.scene.geoms.find(g=>g.id===o.userData.geom);
        rows.push({preset:g.material,wood:!!o.material.userData.woodGrainAxis,
          metalness:o.material.metalness,color:o.material.color.toArray()});}});
      if(rf.three().renderer.getContext().getError()!==0)throw Error('Furniture GL error');
      return rows;
    """.replace("NAME", json.dumps(name))
    )


async def assert_finish(tab, name, finish):
    rows = await drawn(tab, name)
    surfaces = [r for r in rows if r["preset"] != "rf_rubber"]
    assert surfaces
    preset = {"plastic": "rf_shell", "wood": "rf_wood", "steel": "rf_metal"}[finish]
    assert all(r["preset"] == preset and r["wood"] == (finish == "wood") for r in surfaces)
    if name == "desk":
        assert any(r["preset"] == "rf_rubber" and not r["wood"] for r in rows)


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_furniture_single_batch_original_undo_reload_and_arrangement(viewer, size, lang):  # noqa: F811
    async def journey(tab, server):
        if size == "phone":
            await tab.page.call(
                "Emulation.setDeviceMetricsOverride", width=320, height=740, deviceScaleFactor=2, mobile=True
            )
        await pause(tab, server)
        initial = server.pg.store.load()
        for name, finish in [("desk", "wood"), ("tray", "steel")]:
            await choose(tab, name)
            original_drawn = await drawn(tab, name)
            await apply_surface(tab, finish)
            await assert_finish(tab, name, finish)
            current = next(o for o in server.pg.store.load().objects if o.name == name)
            assert current.surface_material == finish and current.material is None
            # Existing Original restores the authored default, and latest Undo restores the finish.
            await apply_surface(tab, "")
            assert await drawn(tab, name) == original_drawn
            assert next(o for o in server.pg.store.load().objects if o.name == name).surface_material is None
            await click(tab, "#object-edit-done")
            await click(tab, "#undo-edit")
            await tab.until(READY + " && !window.__robotforge.busy().busy")
            await assert_finish(tab, name, finish)
            assert next(o for o in server.pg.store.load().objects if o.name == name).surface_material == finish

        finished = server.pg.store.load()
        for finish in ("plastic", ""):
            await select_group(tab, NAMES)
            await click(tab, "#object-surface")
            assert await tab.eval("document.getElementById('object-surface-material').value") == "mixed"
            await apply_surface(tab, finish)
            assert [o.surface_material for o in server.pg.store.load().objects] == [finish or None] * 3
            assert server.pg.session.paused and server.pg.session.data.time == 0
            if finish:
                for name in NAMES:
                    await assert_finish(tab, name, finish)
            receipt, data = server.pg._play_edit, server.pg.session.data
            await click(tab, "#object-surface-apply")
            await tab.until("!window.__robotforge.busy().busy")
            assert server.pg._play_edit is receipt and server.pg.session.data is data
            await click(tab, "#object-edit-done")
            await click(tab, "#undo-edit")
            await tab.until(READY + " && !window.__robotforge.busy().busy")
            assert server.pg.store.load() == finished

        await reload(tab)
        assert server.pg.store.load() == finished
        for name, finish in [("desk", "wood"), ("tray", "steel"), ("block", "steel")]:
            await assert_finish(tab, name, finish)
        await select_group(tab, NAMES)
        await click(tab, "#arrangement-save-open")
        await tab.page.call("Input.insertText", text="家具外观" if lang == "zh-CN" else "Furniture finishes")
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        await tab.until("document.getElementById('arrangement-save-form').hidden")
        await click(tab, "#object-edit-done")
        await reload(tab)
        await open_shelf(tab)
        await tab.until("document.querySelectorAll('.arrangement-row').length===1")
        await click(tab, '[data-tab="maps"]')
        await click(tab, "#map-list .choice:first-child")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert not server.pg.current_world().objects
        await tab.eval("window.__robotforge.lookAt([6,-8,6],[0,0,0])")
        await prepare(tab)
        await place(tab, "[0,0,0]")
        inserted = server.pg.current_world().objects
        assert [(o.shape, o.surface_material, o.material) for o in inserted] == [
            (o.shape, o.surface_material, o.material) for o in finished.objects
        ]
        for actual, authored in zip(inserted, initial.objects, strict=True):
            assert actual.size == authored.size and actual.color == authored.color
            await assert_finish(tab, actual.name, actual.surface_material or actual.material)
        await click(tab, "#undo-edit")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert not server.pg.current_world().objects
        assert not tab.errors and await tab.eval("window.__qaErrors") == []

    viewer(size, lang, journey, world=furniture)
