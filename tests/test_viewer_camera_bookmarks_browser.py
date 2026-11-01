"""Named viewpoints use the real camera and native browser input, without changing the world."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_viewer_camera_views_browser import pose, world
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_startup_browser import READY
from viewer_camera import settle_camera

STORE = "robotforge.camera-bookmarks.v1"
COUNT = f"JSON.parse(localStorage.getItem('{STORE}'))?.sources[0]?.views.length"


async def open_views(tab):
    if await tab.eval("document.getElementById('view-panel').hidden"):
        await click(tab, "#view-toggle")


async def name_view(tab, name):
    await click(tab, ".camera-bookmarks-name")
    await tab.run("document.querySelector('.camera-bookmarks-name').select()")
    await tab.page.call("Input.insertText", text=name)


async def save_view(tab, name, count=1):
    await open_views(tab)
    await name_view(tab, name)
    await click(tab, ".camera-bookmarks-save")
    await tab.until(f"{COUNT} === {count}")


def same_pose(actual, expected):
    assert actual["eye"] == pytest.approx(expected["eye"], abs=1e-9)
    assert actual["target"] == pytest.approx(expected["target"], abs=1e-9)


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_named_view_restore_back_and_refresh_preserve_world_and_physics(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        await tab.run("window.__robotforge.lookAt([4,-6,3],[.2,0,.4])")
        await settle_camera(tab)
        original = await pose(tab)
        saved = server.pg.store.file.read_bytes()
        state = {name: getattr(server.pg.session.data, name).copy() for name in ("qpos", "qvel", "ctrl")}
        await save_view(tab, "工作台" if lang == "zh-CN" else "Workbench")
        same_pose(await pose(tab), original)
        await key(tab, "Escape")
        await tab.run("window.__robotforge.lookAt([-5,-4,4],[1,0,.6])")
        await settle_camera(tab)
        explored = await pose(tab)
        await open_views(tab)
        await click(tab, ".camera-bookmarks-restore")
        await settle_camera(tab)
        same_pose(await pose(tab), original)
        assert await tab.eval("document.getElementById('view-panel').hidden")
        assert await tab.eval("window.__robotforge.following()") is None
        await open_views(tab)
        await click(tab, "#view-back")
        await settle_camera(tab)
        same_pose(await pose(tab), explored)
        await tab.page.call("Page.reload")
        await tab.until(READY)
        await settle_camera(tab)
        await open_views(tab)
        assert await tab.eval("document.querySelectorAll('.camera-bookmarks-row').length") == 1
        await click(tab, ".camera-bookmarks-restore")
        await settle_camera(tab)
        same_pose(await pose(tab), original)
        assert server.pg.store.file.read_bytes() == saved
        for name, expected in state.items():
            np.testing.assert_array_equal(getattr(server.pg.session.data, name), expected)
        assert not tab.errors

    viewer(size, lang, scenario, world=world)


def test_names_limits_safe_text_and_explicit_keyboard_deletion(viewer):  # noqa: F811
    async def scenario(tab, _server):
        await tab.until(READY)
        await settle_camera(tab)
        malicious = "<img src=x onerror=window.__bookmarkXss=true>"
        await save_view(tab, malicious)
        assert await tab.eval("document.querySelector('.camera-bookmarks-view-name').textContent") == malicious
        assert not await tab.eval("Boolean(window.__bookmarkXss || document.querySelector('#camera-bookmarks img'))")
        await name_view(tab, " " + malicious + " ")
        await click(tab, ".camera-bookmarks-save")
        assert await tab.eval(COUNT) == 1
        assert "already" in await tab.eval("document.querySelector('.camera-bookmarks-status').textContent")
        for index in range(2, 9):
            await save_view(tab, f"View {index}", index)
        await name_view(tab, "Ninth")
        await click(tab, ".camera-bookmarks-save")
        assert await tab.eval(COUNT) == 8
        assert "full" in await tab.eval("document.querySelector('.camera-bookmarks-status').textContent")
        await click(tab, ".camera-bookmarks-row:first-child .camera-bookmarks-remove")
        assert await tab.eval("document.activeElement.classList.contains('camera-bookmarks-cancel')")
        assert await tab.eval(COUNT) == 8
        await key(tab, "Escape")
        assert not await tab.eval("document.getElementById('view-panel').hidden")
        assert await tab.eval("document.activeElement.classList.contains('camera-bookmarks-remove')")
        await click(tab, ".camera-bookmarks-row:first-child .camera-bookmarks-remove")
        await click(tab, ".camera-bookmarks-confirm")
        await tab.until(f"{COUNT} === 7")
        assert await tab.eval("document.activeElement.matches('.camera-bookmarks-restore,.camera-bookmarks-name')")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=world)


@pytest.mark.parametrize("failure", ["corrupt", "quota", "locks", "legacy"])
def test_storage_and_older_scene_failures_are_visible_without_overwriting_data(viewer, failure):  # noqa: F811
    async def scenario(tab, _server):
        await tab.until(READY)
        await settle_camera(tab)
        original = await pose(tab)
        if failure == "corrupt":
            await tab.run(f"localStorage.setItem('{STORE}','not valid JSON')")
        elif failure == "quota":
            await tab.run(f"""
              const native=Storage.prototype.setItem;
              Storage.prototype.setItem=function(k,v){{if(k==={json.dumps(STORE)})
                throw new DOMException('injected quota','QuotaExceededError');return native.call(this,k,v);}};
            """)
        elif failure == "locks":
            await tab.run("Object.defineProperty(navigator,'locks',{value:undefined,configurable:true})")
        else:
            await tab.run("delete window.__robotforge.scene.view_scope")
        await open_views(tab)
        if failure == "quota":
            await name_view(tab, "Unsaved")
            await click(tab, ".camera-bookmarks-save")
            await tab.until(
                "document.querySelector('.camera-bookmarks-status').textContent.includes('could not be saved')"
            )
        else:
            assert await tab.eval("document.querySelector('.camera-bookmarks-save').disabled")
        expected = "not valid JSON" if failure == "corrupt" else None
        assert await tab.eval(f"localStorage.getItem('{STORE}')") == expected
        same_pose(await pose(tab), original)
        assert not tab.errors

    viewer("laptop", "en", scenario, world=world)


@pytest.mark.parametrize("change", ["scene", "dismiss"])
def test_queued_save_is_cancelled_after_scene_change_or_dismissal(viewer, change):  # noqa: F811
    async def scenario(tab, server):
        await tab.until(READY)
        await settle_camera(tab)
        await tab.run(f"""
          const manager=navigator.locks,native=manager.request.bind(manager);
          manager.request=(name,options,callback)=>name==={json.dumps(STORE)}
            ?new Promise((resolve,reject)=>{{window.__releaseBookmark=()=>
              native(name,options,callback).then(resolve,reject)}})
            :native(name,options,callback);
        """)
        await open_views(tab)
        await name_view(tab, "Old viewpoint")
        await click(tab, ".camera-bookmarks-save")
        await tab.until("typeof window.__releaseBookmark==='function'")
        if change == "scene":
            version = server.pg.session.version
            server.pg.add_object("new_block", shape="box", pos=[2, 2])
            await tab.until(f"{READY} && window.__robotforge.scene.version>{version}")
        else:
            await key(tab, "Escape")
        await tab.run("await window.__releaseBookmark(); await new Promise(requestAnimationFrame)")
        assert await tab.eval(f"localStorage.getItem('{STORE}')") is None
        assert not tab.errors

    viewer("laptop", "en", scenario, world=world)


def test_restoring_ends_follow_but_saving_and_old_rows_are_guarded(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        await settle_camera(tab)
        await save_view(tab, "Overview")
        await key(tab, "Escape")
        await click(tab, "#camera-view")
        await settle_camera(tab)
        assert await tab.eval("window.__robotforge.following()") is not None
        await open_views(tab)
        assert await tab.eval("document.querySelector('.camera-bookmarks-save').disabled")
        assert not await tab.eval("document.querySelector('.camera-bookmarks-restore').disabled")
        await click(tab, ".camera-bookmarks-restore")
        await settle_camera(tab)
        assert await tab.eval("window.__robotforge.following()") is None
        await open_views(tab)
        await tab.run("window.__oldBookmarkButton=document.querySelector('.camera-bookmarks-restore')")
        version = server.pg.session.version
        server.pg.add_object("another_block", shape="box", pos=[2, 2])
        await tab.until(f"{READY} && window.__robotforge.scene.version>{version}")
        await settle_camera(tab)
        camera = await pose(tab)
        await tab.run("window.__oldBookmarkButton.click()")
        await settle_camera(tab)
        same_pose(await pose(tab), camera)
        assert not tab.errors

    viewer("laptop", "en", scenario, world=world)


def test_removing_last_view_while_following_keeps_keyboard_focus_in_the_shelf(viewer):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        await tab.until(READY)
        await settle_camera(tab)
        await save_view(tab, "Overview")
        await key(tab, "Escape")
        await click(tab, "#camera-view")
        await settle_camera(tab)
        await open_views(tab)
        assert await tab.eval("document.querySelector('.camera-bookmarks-name').disabled")
        await click(tab, ".camera-bookmarks-remove")
        await click(tab, ".camera-bookmarks-confirm")
        await tab.until(f"JSON.parse(localStorage.getItem('{STORE}')).sources.length===0")
        assert await tab.eval("document.activeElement.id") == "camera-bookmarks"
        assert await tab.eval("window.__robotforge.following()") is not None
        assert not tab.errors

    viewer("laptop", "en", scenario, world=world)
