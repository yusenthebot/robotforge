"""Native object building, preview controls and inverse edits in desktop/phone Chrome."""

from __future__ import annotations

import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401


async def choose_object(tab, name):
    await click(tab, "#world-toggle")
    await click(tab, '[data-tab="objects"]')
    await click(tab, f'[data-object="{name}"]')
    await tab.until("!document.getElementById('placement-bar').hidden")


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("laptop", "zh-CN"), ("phone", "en"), ("phone", "zh-CN")])
def test_native_object_preview_scale_rotate_place_undo_and_cancel(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await tab.eval("window.__robotforge.lookAt([4,-6,5],[0,0,0])")
        await choose_object(tab, "barrier")
        assert await tab.eval("document.getElementById('world-panel').hidden")
        await tab.eval("document.querySelector('[data-size=small]').focus()")
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        assert not server.pg.store.load().objects
        await click(tab, '[data-size="large"]')
        assert await tab.eval("document.querySelector('[data-size=large]').getAttribute('aria-pressed')") == "true"
        assert not server.pg.store.load().objects
        await click(tab, "#place-rotate")
        preview = await tab.eval("""(()=>{const scene=window.__robotforge.three().scene;
          const m=scene.getObjectByName('placement-size-preview');
          return {width:m.geometry.parameters.width,angle:m.rotation.z,visible:m.visible};})()""")
        assert preview["width"] == pytest.approx(1.2) and preview["angle"] == pytest.approx(1.5707963267948966)
        assert await tab.eval("""(()=>{const r=document.getElementById('placement-bar').getBoundingClientRect();
          const d=document.querySelector('.dock').getBoundingClientRect();
          return r.left>=0&&r.right<=innerWidth&&r.top>0&&r.bottom<=d.top;})()""")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([1.5,0,0])"))
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden")
        entry = server.pg.store.load().objects[0]
        assert entry.name == "barrier" and entry.yaw == 90 and entry.size == pytest.approx([1.2, 0.18, 0.33])
        assert not await tab.eval("!!window.__robotforge.three().scene.getObjectByName('placement-size-preview')")
        assert not await tab.eval("document.getElementById('undo-add').hidden")
        await choose_object(tab, "ball")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([0,0,0])"))
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden")
        assert [o.name for o in server.pg.store.load().objects] == ["barrier", "ball"]
        await click(tab, "#undo-add")
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('undo-add').hidden")
        assert [o.name for o in server.pg.store.load().objects] == ["barrier"]
        await choose_object(tab, "table")
        await key(tab, "Escape")
        await tab.until("document.getElementById('placement-bar').hidden")
        assert not await tab.eval("!!window.__robotforge.three().scene.getObjectByName('placement-size-preview')")
        assert not tab.errors

    viewer(size, lang, scenario, world=lambda pg: None)


def test_occupied_prop_placement_can_retry_and_drive_selection_survives(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.eval("window.__robotforge.select('rover');window.__robotforge.lookAt([4,-6,5],[0,0,0])")
        await choose_object(tab, "barrier")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([0.3,0,0])"))
        await tab.until(
            "!window.__robotforge.busy().busy && document.getElementById('toast').textContent.includes('occupied')",
            timeout=5,
        )
        assert not server.pg.store.load().objects
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([2,0,0])"))
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden")
        assert await tab.eval("window.__robotforge.selected()") == "rover"
        assert [o.name for o in server.pg.store.load().objects] == ["barrier"]
        # Reset changes generation without scene version; the UI must withdraw the stale undo affordance.
        await key(tab, " ", "Space")
        await key(tab, " ", "Space", kind="keyUp")
        await key(tab, "r", "KeyR")
        await key(tab, "r", "KeyR", kind="keyUp")
        await tab.until("document.getElementById('undo-add').hidden")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=lambda pg: pg.add_robot("rover", template="rover"))


@pytest.mark.parametrize("size", ["laptop", "phone"])
def test_objects_can_stack_on_loose_blocks_and_rest_on_tables(viewer, size):  # noqa: F811
    async def scenario(tab, server):
        await tab.eval("window.__robotforge.lookAt([.8,-1,1],[0,0,.1])")
        await choose_object(tab, "block")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([0,0,.18])"))
        await tab.until(
            "!window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden", timeout=5
        )
        entry = server.pg.store.load().objects[-1]
        assert entry.name == "block" and len(entry.pos) == 3 and 0.26 < entry.pos[2] < 0.28
        await click(tab, "#undo-add")
        await tab.until("!window.__robotforge.busy().busy && document.getElementById('undo-add').hidden", timeout=5)
        await tab.eval("window.__robotforge.lookAt([-1.2,-1.2,1.5],[-2,0,.5])")
        await choose_object(tab, "ball")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([-2,0,.5])"))
        await tab.until(
            "!window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden", timeout=5
        )
        entry = server.pg.store.load().objects[-1]
        assert entry.name == "ball" and entry.pos[2] == pytest.approx(0.59, abs=0.001)
        assert not tab.errors

    def world(pg):
        pg.add_object("base", shape="box", size=[0.18, 0.18, 0.18], pos=[0, 0])
        pg.add_object("table", shape="table", size=[0.8, 0.5, 0.5], pos=[-2, 0], fixed=True)

    viewer(size, "en", scenario, world=world)


def test_fixed_imported_robot_surfaces_are_excluded_from_prop_support(viewer):  # noqa: F811
    async def scenario(tab, server):
        assert any(r["name"] == "fixed" for r in await tab.eval("window.__robotforge.scene.robots"))
        bodies = await tab.eval("window.__robotforge.scene.bodies.filter(b=>b.name.startsWith('fixed/'))")
        assert bodies and all(not b["dynamic"] for b in bodies)
        await tab.eval("window.__robotforge.lookAt([1,-1,1],[0,0,.1])")
        await choose_object(tab, "block")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([0,0,.2])"))
        assert await tab.eval("document.getElementById('place-here').disabled")
        assert not server.pg.store.load().objects
        assert not tab.errors

    def world(pg):
        path = pg.store.dir / "fixed.xml"
        path.write_text(
            '<mujoco><worldbody><body name="base" pos="0 0 .1">'
            '<geom type="box" size=".2 .2 .1" mass="1"/></body></worldbody></mujoco>',
            encoding="utf-8",
        )
        pg.add_robot("fixed", file=str(path), pos=[0, 0])

    viewer("laptop", "en", scenario, world=world)
