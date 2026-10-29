"""Trusted multi-part selection through the existing paused editing lifecycle."""

import json

import pytest
from test_viewer_creation_browser import click
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_object_surface_browser import NOOP_WIRE
from test_viewer_robot_surface_batch import batch, parts
from test_viewer_robot_surface_browser import open_parts, select, world
from test_viewer_startup_browser import READY


async def choose_part(tab, body):
    selector = f"#robot-surface-part-list input[value={json.dumps(body)}]"
    await click(tab, selector)


async def apply(tab, value):
    if tab.size != "phone":
        await select(tab, "#robot-surface-material", value)
    else:
        # Native Home/Down skips the disabled mixed-state prompt.
        index = await tab.run(
            "const e=document.getElementById('robot-surface-material');e.focus();"
            "return Array.from(e.options).filter(o=>!o.disabled&&!o.hidden)"
            f".findIndex(o=>o.value==={json.dumps(value)})"
        )
        assert index >= 0
        for key, code in [("Home", 36), *[("ArrowDown", 40)] * index]:
            for kind in ["keyDown", "keyUp"]:
                await tab.page.call("Input.dispatchKeyEvent", type=kind, key=key, windowsVirtualKeyCode=code)
        assert await tab.eval("document.getElementById('robot-surface-material').value") == value
    await click(tab, "#robot-surface-apply")
    await tab.until(READY + " && !window.__robotforge.busy().busy")


def mixed_world(pg):
    world(pg)
    selected = parts(pg)
    batch(pg, selected[:1], "wood")
    batch(pg, selected[1:2], "steel")
    batch(pg, selected[-1:], "plastic")


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_native_batch_mixed_zero_noop_original_undo_and_reload(viewer, size, lang):  # noqa: F811
    async def journey(tab, server):
        server.pg.session.paused = True
        names = parts(server.pg)
        await open_parts(tab)
        await select(tab, "#robot-surface-body", names[0])
        await click(tab, "#robot-surface-multiple")
        await choose_part(tab, names[1])
        assert await tab.eval("document.getElementById('robot-surface-material').value") == "mixed"
        assert await tab.eval("document.getElementById('robot-surface-apply').disabled")
        await click(tab, "#robot-surface-clear")
        assert await tab.eval("document.getElementById('robot-surface-apply').disabled")
        for name in names[:2]:
            await choose_part(tab, name)
        version = server.pg.session.version
        await apply(tab, "wood")
        assert server.pg.session.version == version + 1
        expected = {**dict.fromkeys(names[:2], "wood"), names[-1]: "plastic"}
        assert server.pg.store.load().robots[0].body_surfaces == expected
        assert await tab.eval("document.getElementById('robot-surface-multiple').checked")
        checked = await tab.eval(
            "Array.from(document.querySelectorAll('#robot-surface-part-list input:checked')).map(e=>e.value)"
        )
        assert set(checked) == set(names[:2])
        receipt, version = server.pg._play_edit, server.pg.session.version
        await click(tab, "#robot-surface-apply")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert server.pg._play_edit is receipt and server.pg.session.version == version
        await apply(tab, "")
        assert server.pg.store.load().robots[0].body_surfaces == {names[-1]: "plastic"}
        await click(tab, "#drawer-toggle")
        await click(tab, "#undo-edit")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert server.pg.store.load().robots[0].body_surfaces == expected
        origin = await tab.eval("performance.timeOrigin")
        await tab.page.call("Page.reload")
        await tab.until(f"performance.timeOrigin!=={origin} && {READY}")
        assert server.pg.store.load().robots[0].body_surfaces == expected
        await open_parts(tab)
        assert not await tab.eval("document.getElementById('robot-surface-multiple').checked")
        assert await tab.eval("window.__qaErrors") == []
        assert await tab.eval("window.__robotforge.three().renderer.getContext().getError()") == 0

    viewer(size, lang, journey, world=mixed_world)


@pytest.mark.parametrize("reset", [False, True])
def test_batch_late_noop_restores_only_the_acknowledged_selection(viewer, reset):  # noqa: F811
    async def journey(tab, server):
        server.pg.session.paused = True
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=NOOP_WIRE)
        await tab.navigate()
        await open_parts(tab)
        names = parts(server.pg)[:2]
        await click(tab, "#robot-surface-multiple")
        await choose_part(tab, names[1])
        await apply(tab, "wood")
        await tab.run("window.__surfaceWire.hold=true")
        await click(tab, "#robot-surface-apply")
        await tab.until("!!window.__surfaceWire.waiting")
        if reset:
            generation = server.pg.session.generation
            server.pg.session.reset()
            await tab.until(f"window.__surfaceWire.generation>{generation}")
        await tab.run("window.__surfaceWire.hold=false;window.__surfaceWire.waiting.release()")
        await tab.until("!window.__robotforge.busy().busy")
        assert await tab.eval("document.getElementById('robot-surface-multiple').checked") is not reset
        assert await tab.eval("document.getElementById('undo-edit').hidden") is reset
        assert not tab.errors

    viewer("laptop", "en", journey, world=world)


def test_switching_robots_clears_group_and_does_not_style_the_new_robot(viewer):  # noqa: F811
    async def journey(tab, server):
        server.pg.session.paused = True
        await open_parts(tab)
        await click(tab, "#robot-surface-multiple")
        await choose_part(tab, parts(server.pg)[1])
        await click(tab, '#robot-list button[data-robot="other"]')
        await open_parts(tab)
        assert not await tab.eval("document.getElementById('robot-surface-multiple').checked")
        assert not server.pg.store.load().robots[1].body_surfaces
        await click(tab, '#robot-list button[data-robot="bot"]')
        await open_parts(tab)
        assert not await tab.eval("document.getElementById('robot-surface-multiple').checked")
        assert not server.pg.store.load().robots[0].body_surfaces
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", journey, world=world)
