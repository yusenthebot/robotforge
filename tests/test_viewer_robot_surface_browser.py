"""Native part styling and latest-only Undo, including display and stale-reply boundaries."""

import asyncio
import json

import pytest
from test_viewer_creation_browser import click
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_object_surface_browser import NOOP_WIRE, POSE_WIRE
from test_viewer_startup_browser import READY
from viewer_camera import settle_camera

from robotforge.playground import PlaygroundError


def world(pg):
    pg.add_robot("bot", template="rover")
    pg.add_robot("other", template="arm", pos=[-2, 0])


async def select(tab, selector, value):
    previous = await tab.eval(f"document.querySelector({json.dumps(selector)}).value")
    index = await tab.eval(
        f"Array.from(document.querySelector({json.dumps(selector)}).options).findIndex(o=>o.value==={json.dumps(value)})"
    )
    assert index >= 0
    await tab.run(
        f"document.querySelector({json.dumps(selector)}).focus();window.__partChanges=[];"
        f"document.querySelector({json.dumps(selector)}).onchange=e=>window.__partChanges.push(e.isTrusted)"
    )
    if tab.size == "phone":
        for key, code in [("Home", 36), *[("ArrowDown", 40)] * index]:
            for kind in ["keyDown", "keyUp"]:
                await tab.page.call("Input.dispatchKeyEvent", type=kind, key=key, windowsVirtualKeyCode=code)
    else:
        await asyncio.sleep(1.05)
        label = await tab.eval(f"document.querySelector({json.dumps(selector)}).options[{index}].textContent")
        for letter in label:
            for kind in ["keyDown", "keyUp"]:
                await tab.page.call(
                    "Input.dispatchKeyEvent", type=kind, key=letter, text=letter if kind == "keyDown" else ""
                )
    assert await tab.eval(f"document.querySelector({json.dumps(selector)}).value") == value
    changes = await tab.eval("window.__partChanges")
    if previous != value:
        assert changes and all(changes)


async def open_parts(tab):
    if await tab.eval("document.getElementById('drawer').hidden"):
        await click(tab, "#drawer-toggle")
    if await tab.eval("document.getElementById('robot-surface-controls').hidden"):
        await click(tab, "#robot-surface")
    await tab.until("!document.getElementById('robot-surface-controls').hidden")
    return await tab.eval("document.getElementById('robot-surface-body').value")


async def apply(tab, value):
    await select(tab, "#robot-surface-material", value)
    await click(tab, "#robot-surface-apply")
    await tab.until(READY + " && !window.__robotforge.busy().busy")
    await settle_camera(tab)


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_native_part_apply_original_noop_undo_and_reload(viewer, size, lang):  # noqa: F811
    async def journey(tab, server):
        server.pg.session.paused = True
        body = await open_parts(tab)
        initial = server.pg.store.load()
        await apply(tab, "wood")
        assert server.pg.store.load().robots[0].body_surfaces == {body: "wood"}
        assert server.pg.store.load().robots[1] == initial.robots[1]
        assert server.pg.session.paused and server.pg.session.data.time == 0
        assert not await tab.eval("document.getElementById('robot-surface-controls').hidden")
        assert await tab.eval("document.getElementById('robot-surface-body').value") == body
        assert await tab.eval("document.getElementById('undo-edit').textContent") == (
            "撤销部件外观" if lang.startswith("zh") else "Undo part surface"
        )
        receipt, version = server.pg._play_edit, server.pg.session.version
        await apply(tab, "wood")
        assert server.pg._play_edit is receipt and server.pg.session.version == version
        await apply(tab, "steel")
        await click(tab, "#drawer-toggle")
        await click(tab, "#undo-edit")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert server.pg.store.load().robots[0].body_surfaces == {body: "wood"}
        await open_parts(tab)
        await apply(tab, "")
        assert not server.pg.store.load().robots[0].body_surfaces
        await apply(tab, "plastic")
        saved = server.pg.store.load()
        origin = await tab.eval("performance.timeOrigin")
        await tab.page.call("Page.reload")
        await tab.until(f"performance.timeOrigin!=={origin} && {READY}")
        assert server.pg.store.load() == saved
        assert await tab.eval("window.__qaErrors") == []
        assert await tab.eval("window.__robotforge.three().renderer.getContext().getError()") == 0

    viewer(size, lang, journey, world=world)


def test_a_reset_during_display_wait_does_not_restore_stale_undo(viewer):  # noqa: F811
    async def journey(tab, server):
        server.pg.session.paused = True
        await open_parts(tab)
        await select(tab, "#robot-surface-material", "steel")
        await tab.run("""const fetch0=window.fetch;window.__heldPartPoses=false;window.__releasePartPoses=null;
          window.fetch=async(...args)=>{const r=await fetch0(...args);
            if(String(args[0]).includes('mesh/')){window.__heldPartPoses=true;
              await new Promise(resolve=>window.__releasePartPoses=resolve);}return r;};""")
        await click(tab, "#robot-surface-apply")
        await tab.until("window.__heldPartPoses")
        receipt = server.pg._play_edit
        assert receipt is not None
        server.pg.session.reset()
        await tab.run("window.__releasePartPoses()")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert await tab.eval("document.getElementById('undo-edit').hidden")
        assert receipt.generation != server.pg.session.generation
        with pytest.raises(PlaygroundError, match="changed|stale|no longer"):
            server.pg.edit_robot_surface_for_play(
                {
                    "action": "undo_robot_surface",
                    "token": receipt.token,
                    "version": receipt.version,
                    "generation": receipt.generation,
                }
            )
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", journey, world=world)


@pytest.mark.parametrize("reset", [False, True])
def test_late_noop_ack_retains_only_current_part_undo(viewer, reset):  # noqa: F811
    async def journey(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=NOOP_WIRE)
        await tab.navigate()
        await open_parts(tab)
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
        assert await tab.eval("document.getElementById('undo-edit').hidden") is reset
        assert not tab.errors

    viewer("laptop", "en", journey, world=world)


@pytest.mark.parametrize("reset", [False, True])
def test_context_loss_during_display_keeps_current_committed_part_undo(viewer, reset):  # noqa: F811
    async def journey(tab, server):
        await tab.page.call(
            "Page.addScriptToEvaluateOnNewDocument", source=POSE_WIRE.replace("m.data?.object", "m.data?.robot")
        )
        await tab.navigate()
        body = await open_parts(tab)
        await select(tab, "#robot-surface-material", "wood")
        await tab.run("window.__surfacePoses.afterVersion=window.__robotforge.scene.version")
        await click(tab, "#robot-surface-apply")
        await tab.until("!!window.__surfacePoses.reply && window.__surfacePoses.poses.length>0")
        extension = await tab.run("""window.__surfaceContext=window.__robotforge.three().renderer.getContext()
          .getExtension('WEBGL_lose_context');if(window.__surfaceContext)window.__surfaceContext.loseContext();
          return !!window.__surfaceContext;""")
        if not extension:
            pytest.skip("driver does not expose WEBGL_lose_context")
        await tab.until("window.__robotforge.three().renderer.getContext().isContextLost()")
        if reset:
            generation = server.pg.session.generation
            server.pg.session.reset()
            await tab.until(f"window.__surfacePoses.generation>{generation}")
        await tab.run(
            "const w=window.__surfacePoses;w.afterVersion=null;w.hold=false;for(const f of w.poses.splice(0))f()"
        )
        await tab.until("!window.__robotforge.busy().busy", timeout=20)
        assert await tab.eval("document.getElementById('undo-edit').hidden") is reset
        await tab.run("window.__surfaceContext.restoreContext()")
        await tab.until(READY)
        assert server.pg.current_world().robots[0].body_surfaces == {body: "wood"}
        if not reset:
            await click(tab, "#drawer-toggle")
            await click(tab, "#undo-edit")
            await tab.until(READY + " && !window.__robotforge.busy().busy")
            assert not server.pg.current_world().robots[0].body_surfaces
        assert not tab.errors

    viewer("laptop", "en", journey, world=world)
