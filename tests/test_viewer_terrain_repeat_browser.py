"""Repeated route building refreshes world guards and keeps one-use Undo honest."""

from __future__ import annotations

import asyncio

import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_placement_browser import spacing
from test_viewer_startup_browser import READY
from test_viewer_terrain_create_browser import HOLD, choose, course, prepare, tap


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_repeat_terrain_keeps_heading_spacing_retries_and_undoes_only_latest(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab)
        await tab.run(
            "window.__terrainSent=[];const old=WebSocket.prototype.send;"
            "WebSocket.prototype.send=function(s){const m=JSON.parse(s);"
            "if(m.action==='add_terrain')window.__terrainSent.push(m);return old.call(this,s)}"
        )
        await choose(tab, "ramp")
        await spacing(tab, "0.25")
        await click(tab, "#place-rotate-plus")
        await click(tab, "#place-repeat")
        for position in [[2.12, -2.12, 0], [2.12, 2.12, 0]]:
            digest = await tab.eval("window.__robotforge.scene.terrain_edit.world_sha256")
            version = server.pg.session.version
            generation = server.pg.session.generation
            await asyncio.sleep(0.55)
            await tap(tab, position)
            await tab.until(
                f"{READY} && window.__robotforge.scene.version>{version} && !window.__robotforge.busy().busy"
            )
            assert not await tab.eval("document.getElementById('placement-bar').hidden")
            assert await tab.eval("document.getElementById('place-spacing').value") == "0.25"
            assert (await tab.eval("document.getElementById('placement-heading').textContent")).count("15") == 1
            assert await tab.eval("document.getElementById('place-here').disabled")
            assert not await tab.eval(
                "window.__robotforge.three().scene.getObjectByName('placement-size-preview')?.visible"
            )
            assert ("保存起点" if lang == "zh-CN" else "saved starts") in await tab.eval(
                "document.getElementById('placement-hint').textContent"
            )
            sent = await tab.eval("window.__terrainSent.at(-1)")
            assert (sent["world_sha256"], sent["version"], sent["generation"]) == (digest, version, generation)
            assert server.pg.session.paused and server.pg.session.data.time == 0
            if len(server.pg.current_world().terrain) == 1:
                # Clear center, overlapping footprint: must preserve the last success and allow retry.
                await asyncio.sleep(0.55)
                receipt = server.pg._play_addition
                unchanged = server.pg.store.file.read_bytes()
                unchanged_version = (server.pg.session.version, server.pg.session.generation)
                await tap(tab, [4.62, -2.12, 0])
                await tab.until(
                    "!window.__robotforge.busy().busy && "
                    "document.getElementById('toast').textContent.includes('occupied')"
                )
                assert len(server.pg.current_world().terrain) == 1
                assert server.pg._play_addition is receipt
                assert server.pg.store.file.read_bytes() == unchanged
                assert (server.pg.session.version, server.pg.session.generation) == unchanged_version
        blocks = server.pg.current_world().terrain
        assert [b.pos for b in blocks] == [[2, -2], [2, 2]]
        assert all(b.yaw == 15 and b.style == "wood" for b in blocks)
        assert len({b.name for b in blocks}) == 2
        await click(tab, "#place-cancel")
        await click(tab, "#undo-add")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert server.pg.current_world().terrain == blocks[:1]
        await choose(tab, "stairs")
        assert await tab.eval("document.getElementById('place-spacing').value") == "free"
        await key(tab, "Escape")
        assert len(server.pg.current_world().terrain) == 1
        assert not await tab.eval("window.__qaErrors")

    viewer(size, lang, scenario, world=course)


@pytest.mark.parametrize("fault", ["missing", "version", "digest"])
def test_repeat_terrain_stops_on_mixed_metadata_but_keeps_committed_undo(viewer, monkeypatch, fault):  # noqa: F811
    from robotforge.viewer import terrain_edit

    original = terrain_edit.metadata

    def metadata(pg):
        result = original(pg)
        if result.get("terrains"):
            if fault == "missing":
                return {"terrains": []}
            result["version" if fault == "version" else "world_sha256"] = -1 if fault == "version" else "invalid"
        return result

    monkeypatch.setattr(terrain_edit, "metadata", metadata)

    async def scenario(tab, server):
        await prepare(tab)
        await choose(tab, "ramp")
        await click(tab, "#place-repeat")
        await tap(tab, [2, -2, 0])
        await tab.until(
            READY + " && !window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden"
        )
        assert len(server.pg.current_world().terrain) == 1
        assert not await tab.eval("document.getElementById('undo-add').hidden")
        await click(tab, "#undo-add")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert not server.pg.current_world().terrain
        assert not await tab.eval("window.__qaErrors")

    viewer("laptop", "en", scenario, world=course)


def test_reset_during_repeat_display_wait_cannot_restore_old_undo(viewer):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call("Page.addScriptToEvaluateOnNewDocument", source=HOLD)
        await tab.navigate()
        await prepare(tab)
        await choose(tab, "ramp")
        await click(tab, "#place-repeat")
        await tab.run("window.__terrainWire.afterVersion=window.__robotforge.scene.version")
        await tap(tab, [2, -2, 0])
        await tab.until("!!window.__terrainWire.reply && window.__terrainWire.poses.length>0")
        await tab.run(
            "window.__repeatSent=[];const send=WebSocket.prototype.send;"
            "WebSocket.prototype.send=function(s){const m=JSON.parse(s);"
            "if(m.action==='add_terrain')window.__repeatSent.push(m);return send.call(this,s)}"
        )
        assert await tab.eval("document.elementFromPoint(40,400)?.id") == "view"
        await click(tab, point=[40, 400])  # In-flight input cannot enqueue a second placement.
        assert await tab.eval("window.__repeatSent.length") == 0
        assert len(server.pg.current_world().terrain) == 1
        server.pg.session.reset()
        await asyncio.sleep(0.15)
        await tab.run(
            "const w=window.__terrainWire;w.afterVersion=null;w.hold=false;"
            "for(const release of w.poses.splice(0))release()"
        )
        await tab.until(
            READY + " && !window.__robotforge.busy().busy && document.getElementById('placement-bar').hidden",
            timeout=20,
        )
        assert len(server.pg.current_world().terrain) == 1
        assert await tab.eval("document.getElementById('undo-add').hidden")
        from robotforge.playground import PlaygroundError
        from robotforge.viewer import terrain_create

        receipt = server.pg._play_addition
        assert await tab.eval("window.__repeatSent.length") == 0
        assert receipt.generation != server.pg.session.generation
        with pytest.raises(PlaygroundError, match="simulation changed"):
            terrain_create.undo(server.pg, {"token": receipt.token, "version": receipt.version})
        assert len(server.pg.current_world().terrain) == 1
        assert not await tab.eval("window.__qaErrors")

    viewer("laptop", "en", scenario, world=course)
