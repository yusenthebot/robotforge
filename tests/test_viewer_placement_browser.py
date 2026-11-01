"""Native shared coordinate placement and rejection of stale or unsupported support."""

from __future__ import annotations

import asyncio
import json

import pytest
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_startup_browser import READY
from viewer_camera import settle_camera

from robotforge.design import combine
from robotforge.design.world import RobotEntry


def world(pg):
    pg.add_robot("rover", template="rover", pos=[-3, -2])
    pg.add_object("desk", shape="table", size=[1, 0.6, 0.7], pos=[0, 0], fixed=True)
    pg.add_object("ledge", shape="box", size=[0.18, 0.3, 0.2], pos=[0.16, 1.2], fixed=True)
    pg.add_object("loose", shape="box", size=[0.4, 0.4, 0.2], pos=[1.5, 0])
    pg.add_object("route", shape="box", size=[0.8, 0.12, 0.22], pos=[-2, 1.5], fixed=True)
    pg.add_terrain("ramp", name="dock", pos=[3, 2], params={"length": 0.6, "width": 0.8, "angle": 8, "landing": 0.4})
    combine.save(RobotEntry(name="source", template="arm"), "local_arm", pg.store.dir)


async def prepare(tab, server):
    server.pg.session.paused = True
    server.pg.session.reset()
    await tab.until(READY + " && window.__robotforge.time===0")
    # Read-only catalog refreshes are not placement submissions.
    await tab.run(
        "window.__placementMessages=[];const send=WebSocket.prototype.send;"
        "WebSocket.prototype.send=function(s){const m=JSON.parse(s);"
        "if(m.type==='world'&&!['layout_list','arrangement_list'].includes(m.action))"
        "window.__placementMessages.push(m);return send.call(this,s)}"
    )
    await tab.run("window.__robotforge.lookAt([4,-6,5],[0,0,-.3])")
    await settle_camera(tab)


async def spacing(tab, value):
    await tab.run(
        "document.getElementById('place-spacing').focus();window.__spacingInput=[];"
        "document.getElementById('place-spacing').onchange=e=>window.__spacingInput.push(e.isTrusted)"
    )
    if tab.size == "phone":
        steps = ["free", "0.1", "0.25"].index(value)
        for name, code in [("Home", 36), *[("ArrowDown", 40)] * steps]:
            for event in ["keyDown", "keyUp"]:
                await tab.page.call("Input.dispatchKeyEvent", type=event, key=name, windowsVirtualKeyCode=code)
    else:
        await asyncio.sleep(1.05)
        text = (
            "0.10"
            if value == "0.1"
            else "0.25"
            if value == "0.25"
            else await tab.eval("document.querySelector('#place-spacing option[value=free]').textContent[0]")
        )
        for char in text:
            for event in ["keyDown", "keyUp"]:
                await tab.page.call(
                    "Input.dispatchKeyEvent", type=event, key=char, text=char if event == "keyDown" else ""
                )
    assert await tab.eval("document.getElementById('place-spacing').value") == value
    changes = await tab.eval("window.__spacingInput")
    assert changes and all(changes)


async def choose_prop(tab, name="pin"):
    if await tab.eval("document.getElementById('world-panel').hidden"):
        await click(tab, "#world-toggle")
    await click(tab, '[data-tab="objects"]')
    await click(tab, f'[data-object="{name}"]')
    await tab.until("!document.getElementById('placement-bar').hidden")
    await settle_camera(tab)


async def enter(tab, kind):
    if kind == "object":
        await choose_prop(tab)
    elif kind in ["copy", "edit"]:
        from test_viewer_object_edit_browser import choose

        await choose(tab, "route")
        await click(tab, "#object-copy" if kind == "copy" else "#object-move")
    elif kind in ["robot", "model"]:
        await click(tab, "#drawer-toggle")
        if await tab.eval("document.getElementById('create-options').hidden"):
            await click(tab, "#create-toggle")
        await click(tab, '[data-template="rover"]' if kind == "robot" else '#model-list [data-model="local_arm"]')
    elif kind == "start":
        await click(tab, "#drawer-toggle")
        await click(tab, "#robot-start")
    else:
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="terrain"]')
        await click(tab, '[data-terrain="ramp"]' if kind == "terrain" else '[data-edit-terrain="dock"]')
    await tab.until("!document.getElementById('placement-bar').hidden")


async def hover(tab, point):
    p = await tab.eval(f"window.__robotforge.screenOfPoint({point})")
    assert await tab.eval(f"document.elementFromPoint({p[0]},{p[1]})?.id") == "view"
    await tab.page.call("Input.dispatchMouseEvent", type="mouseMoved", x=p[0], y=p[1])
    await asyncio.sleep(0.08)


async def candidate(tab):
    return await tab.run(
        "const m=window.__robotforge.three().scene.getObjectByName('placement-point-marker');"
        "return {visible:m.visible,point:m.position.toArray(),"
        "text:document.getElementById('placement-position').textContent,"
        "disabled:document.getElementById('place-here').disabled,"
        "status:document.getElementById('placement-coordinate-status').textContent}"
    )


def test_option_changes_keep_original_anchor_and_enter_submits_one_displayed_point(viewer):  # noqa: F811
    async def journey(tab, server):
        await prepare(tab, server)
        await choose_prop(tab)
        await hover(tab, [0.1234, -0.1134, 0.7])
        free = await candidate(tab)
        assert free["visible"] and free["point"][:2] == pytest.approx([0.1234, -0.1134], abs=1e-4)
        await spacing(tab, "0.1")
        snapped = await candidate(tab)
        assert snapped["point"] == pytest.approx([0.1, -0.1, 0.708], abs=1e-6)
        assert "0.1000" in snapped["text"] and "0.7000" in snapped["text"]
        preview = await tab.eval(
            "window.__robotforge.three().scene.getObjectByName('placement-size-preview').position.toArray()"
        )
        assert preview[:2] == [0.1, -0.1]
        await spacing(tab, "0.25")
        assert (await candidate(tab))["point"][:2] == [0, 0]
        await spacing(tab, "free")
        assert (await candidate(tab))["point"] == free["point"]
        await spacing(tab, "0.1")
        # Native Enter on the focused select must not place; canvas Enter submits exactly the marker point.
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        assert await tab.eval("window.__placementMessages") == []
        await tab.run("document.getElementById('view').focus()")
        version = server.pg.session.version
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        await tab.until(f"window.__robotforge.scene.version>{version} && !window.__robotforge.busy().busy")
        messages = await tab.eval("window.__placementMessages")
        assert len(messages) == 1 and messages[0]["pos"] == [0.1, -0.1, 0.7]
        pin = server.pg.current_world().objects[-1]
        assert pin.pos == [0.1, -0.1, 0.82]
        assert not await tab.eval("window.__qaErrors")

    viewer("laptop", "en", journey, world=world)


def test_grid_outside_support_and_dynamic_surface_reject_without_falling_to_floor(viewer):  # noqa: F811
    async def journey(tab, server):
        await prepare(tab, server)
        saved = server.pg.store.file.read_bytes()
        await choose_prop(tab)
        await hover(tab, [0.08, 1.2, 0.2])
        free = await candidate(tab)
        assert free["visible"]
        await spacing(tab, "0.25")
        rejected = await candidate(tab)
        assert rejected["disabled"] and not rejected["visible"] and "outside" in rejected["status"]
        await spacing(tab, "free")
        assert (await candidate(tab))["point"] == free["point"]
        await hover(tab, [1.5, 0, 0.2])
        assert (await candidate(tab))["visible"]  # Legacy free placement still accepts a loose support.
        await spacing(tab, "0.1")
        rejected = await candidate(tab)
        assert rejected["disabled"] and "stationary" in rejected["status"]
        await spacing(tab, "free")
        # The actual authored ramp's top normal is sloping, even though free placement accepts it.
        target = await tab.run(
            "const e=window.__robotforge.scene.terrain_edit.terrains.find(e=>e.name==='dock'),"
            "b=e.boxes.find(b=>b.pitch!==0);return [3+b.center[0]+Math.sin(b.pitch)*b.half[2],"
            "2+b.center[1],b.center[2]+Math.cos(b.pitch)*b.half[2]]"
        )
        await tab.run("window.__robotforge.lookAt([7,-6,6],[3,2,-.3])")
        await settle_camera(tab)
        await hover(tab, target)
        assert (await candidate(tab))["visible"]
        await spacing(tab, "0.25")
        assert "stationary" in (await candidate(tab))["status"]
        assert server.pg.store.file.read_bytes() == saved and not await tab.eval("window.__placementMessages.length")
        assert not await tab.eval("window.__qaErrors")

    viewer("laptop", "en", journey, world=world)


@pytest.mark.parametrize("mutation", ["matrix", "geometry"])
def test_submit_rechecks_support_and_never_changes_the_displayed_coordinate_silently(viewer, mutation):  # noqa: F811
    async def journey(tab, server):
        await prepare(tab, server)
        await choose_prop(tab)
        await hover(tab, [0.1234, -0.1134, 0.7])
        await spacing(tab, "0.1")
        saved, data = server.pg.store.file.read_bytes(), server.pg.session.data
        # Deliberate rendered-support faults, without modifying the saved/physics world.
        shifts = await tab.run(
            """const THREE=await import('three'),rf=window.__robotforge;
          const ids=rf.scene.object_edit.objects.find(o=>o.name==='desk').geoms,result=[];
          rf.three().scene.traverse(o=>{if(o.isMesh&&ids.includes(o.userData.geom)){
            const before=new THREE.Box3().setFromObject(o).max.z;
            if(MUTATION==='matrix')o.position.z+=.01;
            else {const inverse=new THREE.Matrix3().setFromMatrix4(o.matrixWorld).invert();
              const delta=new THREE.Vector3(0,0,.0003).applyMatrix3(inverse);
              o.geometry.translate(...delta.toArray());}
            o.updateWorldMatrix(true,false);result.push(new THREE.Box3().setFromObject(o).max.z-before);}});
          return result;
        """.replace("MUTATION", json.dumps(mutation))
        )
        assert shifts and shifts == pytest.approx([0.01 if mutation == "matrix" else 0.0003] * len(shifts), abs=1e-7)
        await click(tab, "#place-here")
        result = await candidate(tab)
        assert result["disabled"] and not result["visible"] and "changed" in result["status"]
        assert server.pg.store.file.read_bytes() == saved and server.pg.session.data is data
        assert await tab.eval("window.__placementMessages") == []

    viewer("laptop", "en", journey, world=world)


@pytest.mark.parametrize("kind", ["object", "copy", "edit", "robot", "model", "start", "terrain", "terrain-edit"])
def test_every_placement_entry_exposes_common_controls_and_cancel_preserves_world(viewer, kind):  # noqa: F811
    async def journey(tab, server):
        await prepare(tab, server)
        saved = server.pg.store.file.read_bytes()
        await enter(tab, kind)
        await spacing(tab, "0.25")
        assert await tab.eval("""(()=>{const b=document.getElementById('place-spacing').getBoundingClientRect();
          return b.width>0&&b.height>0&&b.top>=0&&b.bottom<innerHeight;})()""")
        assert not await tab.eval("document.getElementById('place-spacing').disabled")
        await key(tab, "Escape")
        assert server.pg.store.file.read_bytes() == saved
        assert await tab.eval("window.__placementMessages") == []
        await choose_prop(tab)
        assert await tab.eval("document.getElementById('place-spacing').value") == "free"

    viewer("laptop", "en", journey, world=world)


@pytest.mark.parametrize("kind", ["object", "copy", "edit", "robot", "model", "start", "terrain", "terrain-edit"])
def test_all_entry_submissions_use_the_displayed_point_and_existing_inverse(viewer, kind):  # noqa: F811
    async def journey(tab, server):
        await prepare(tab, server)
        original = server.pg.current_world()
        await enter(tab, kind)
        await spacing(tab, "0.25")
        await tab.run("window.__robotforge.lookAt([5,-8,6],[1,-2,-.6])")
        await settle_camera(tab)
        await hover(tab, [1.02, -2.02, 0])
        shown = await candidate(tab)
        assert shown["visible"] and shown["point"] == [1, -2, 0.008]
        version = server.pg.session.version
        await click(tab, "#place-here")
        await tab.until(
            f"window.__robotforge.scene.version>{version} && !window.__robotforge.busy().busy && "
            "document.getElementById('placement-bar').hidden"
        )
        messages = await tab.eval("window.__placementMessages")
        assert len(messages) == 1 and messages[0]["pos"] == [1, -2, 0]
        expected_action = {
            "object": "add_object",
            "copy": "duplicate_object",
            "edit": "edit_object",
            "robot": "add_robot",
            "model": "add_model",
            "start": "edit_robot_start",
            "terrain": "add_terrain",
            "terrain-edit": "edit_terrain",
        }[kind]
        assert messages[0]["action"] == expected_action
        current = server.pg.current_world()
        if kind in ["robot", "model"]:
            assert current.robots[-1].pos[:2] == [1, -2]
        elif kind == "start":
            assert current.robots[0].pos[:2] == [1, -2]
        elif kind in ["terrain", "terrain-edit"]:
            entry = current.terrain[-1] if kind == "terrain" else current.terrain[0]
            assert entry.pos == [1, -2]
        else:
            entry = next(o for o in current.objects if o.name == "route") if kind == "edit" else current.objects[-1]
            assert entry.pos[:2] == [1, -2]
        inverse = "#undo-edit" if kind in ["copy", "edit", "start", "terrain-edit"] else "#undo-add"
        await click(tab, inverse)
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert server.pg.current_world() == original
        assert not await tab.eval("window.__qaErrors")

    viewer("laptop", "en", journey, world=world)


@pytest.mark.parametrize("width,height", [(390, 844), (320, 568)])
def test_native_touch_repeat_preserves_spacing_clears_point_and_stays_reachable(viewer, width, height):  # noqa: F811
    async def journey(tab, server):
        await tab.page.call(
            "Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=2, mobile=True
        )
        await prepare(tab, server)
        await enter(tab, "copy")
        await spacing(tab, "0.25")
        await click(tab, "#place-repeat")
        bounds = await tab.run(
            "const r=document.getElementById('placement-bar').getBoundingClientRect();"
            "return {top:r.top,bottom:r.bottom,height:innerHeight}"
        )
        assert bounds["top"] >= 0 and bounds["bottom"] <= height
        await tab.run("window.__robotforge.lookAt([5,-8,6],[1.5,-2,-1])")
        await settle_camera(tab)
        for target in [[1, -2, 0], [2, -2, 0]]:
            await asyncio.sleep(0.6)
            p = await tab.eval(f"window.__robotforge.screenOfPoint({target})")
            assert await tab.eval(f"document.elementFromPoint({p[0]},{p[1]})?.id") == "view", (bounds, p)
            version = server.pg.session.version
            await click(tab, point=p)
            await tab.until(
                f"window.__robotforge.scene.version>{version} && !window.__robotforge.busy().busy && "
                "!document.getElementById('place-cancel').disabled"
            )
            assert server.pg.current_world().objects[-1].pos[:2] == target[:2]
            assert await tab.eval("document.getElementById('place-spacing').value") == "0.25"
            shown = await candidate(tab)
            assert not shown["visible"] and shown["disabled"]
        await click(tab, "#place-cancel")
        await click(tab, "#undo-edit")
        await tab.until(READY + " && !window.__robotforge.busy().busy")
        assert [o.name for o in server.pg.current_world().objects][-1] == "route_2"
        await choose_prop(tab)
        assert await tab.eval("document.getElementById('place-spacing').value") == "free"
        assert not await tab.eval("window.__qaErrors")

    viewer("phone", "zh-CN", journey, world=world)


@pytest.mark.parametrize("boundary", ["cancel", "reset", "edit"])
def test_snapped_touch_cannot_survive_cancel_or_scene_change(viewer, boundary):  # noqa: F811
    from test_viewer_touch_build_browser import touch

    async def journey(tab, server):
        await prepare(tab, server)
        await choose_prop(tab)
        await spacing(tab, "0.1")
        await tab.run("window.__robotforge.lookAt([5,-8,6],[1,-2,-1])")
        await settle_camera(tab)
        original = server.pg.current_world()
        p = await tab.eval("window.__robotforge.screenOfPoint([1,-2,0])")
        assert await tab.eval(f"document.elementFromPoint({p[0]},{p[1]})?.id") == "view"
        await touch(tab, "touchStart", [(1, *p)])
        await tab.run("document.getElementById('view').focus()")
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        assert server.pg.current_world() == original
        if boundary == "cancel":
            await touch(tab, "touchCancel")
        else:
            if boundary == "reset":
                server.pg.session.reset()
            else:
                server.pg.add_object("external", shape="sphere", size=[0.1], pos=[4, 0])
            await tab.until("document.getElementById('placement-bar').hidden")
            await touch(tab, "touchEnd")
            await tab.until(READY)
            await choose_prop(tab)
            assert await tab.eval("document.getElementById('place-spacing').value") == "free"
            await spacing(tab, "0.1")
        assert await tab.eval("window.__placementMessages") == []
        assert len(server.pg.current_world().objects) == len(original.objects) + (boundary == "edit")
        await asyncio.sleep(0.6)
        p = await tab.eval("window.__robotforge.screenOfPoint([1,-2,0])")
        version = server.pg.session.version
        await click(tab, point=p)
        await tab.until(f"window.__robotforge.scene.version>{version} && " + READY)
        assert server.pg.current_world().objects[-1].pos[:2] == [1, -2]
        assert len(await tab.eval("window.__placementMessages")) == 1
        assert not tab.errors

    viewer("phone", "en", journey, world=world)


@pytest.mark.parametrize("lang", ["en", "zh-CN"])
def test_short_phone_scroll_and_coordinate_feedback_keep_the_placement_view_stable(viewer, lang):  # noqa: F811
    from test_viewer_touch_build_browser import touch

    async def layout(tab):
        return await tab.run("""const rf=window.__robotforge,b=document.getElementById('placement-bar');
          return {bar:b.getBoundingClientRect().toJSON(),projection:rf.three().camera.projectionMatrix.elements,
            eye:rf.three().camera.position.toArray(),target:rf.cameraTarget(),
            centre:rf.screenOfPoint(rf.cameraTarget())};""")

    async def journey(tab, server):
        await tab.page.call(
            "Emulation.setDeviceMetricsOverride", width=320, height=568, deviceScaleFactor=2, mobile=True
        )
        await prepare(tab, server)
        original = server.pg.current_world()
        await choose_prop(tab)
        initial = await candidate(tab)
        assert initial["visible"]
        marker = await tab.eval(f"window.__robotforge.screenOfPoint({initial['point']})")
        assert await tab.eval(f"document.elementFromPoint({marker[0]},{marker[1]})?.id") == "view"
        first = await layout(tab)
        bar = first["bar"]
        assert bar["top"] > 136 and bar["bottom"] < 568
        scroll = await tab.run(
            "const b=document.getElementById('placement-bar');return [b.scrollTop,b.scrollHeight,b.clientHeight]"
        )
        assert scroll[1] > scroll[2]
        x, bottom, top = bar["right"] - 8, bar["bottom"] - 20, bar["top"] + 20
        assert await tab.eval(f"!!document.elementFromPoint({x},{bottom})?.closest('#placement-bar')")
        await touch(tab, "touchStart", [(1, x, bottom)])
        for i in range(1, 6):
            await touch(tab, "touchMove", [(1, x, bottom + (top - bottom) * i / 5)])
        await touch(tab, "touchEnd")
        await tab.until(f"document.getElementById('placement-bar').scrollTop>{scroll[0]}")
        assert server.pg.current_world() == original
        await spacing(tab, "0.1")
        await tab.run("window.__robotforge.lookAt([4,-4,3],[.08,1.2,.2])")
        await settle_camera(tab)
        await hover(tab, [0.08, 1.2, 0.2])
        assert (await candidate(tab))["visible"]
        await settle_camera(tab, quiet_ms=200)
        valid = await layout(tab)
        for mode, visible in [("0.25", False), ("free", True), ("0.1", True)]:
            await spacing(tab, mode)
            await settle_camera(tab, quiet_ms=200)
            current = await layout(tab)
            assert (await candidate(tab))["visible"] is visible
            assert current["bar"] == pytest.approx(valid["bar"], abs=1e-6)
            assert current["projection"] == pytest.approx(valid["projection"], abs=1e-9)
            assert await tab.run("""const box=document.querySelector('.placement-coordinate-readout');
              return [...box.children].filter(e=>!e.hidden).every(e=>e.scrollHeight<=box.clientHeight);""")
        await key(tab, "Escape")
        await settle_camera(tab)
        closed = await layout(tab)
        assert closed["eye"] == pytest.approx(valid["eye"], abs=1e-9)
        assert closed["target"] == pytest.approx(valid["target"], abs=1e-9)
        assert closed["centre"][1] > valid["centre"][1]
        assert server.pg.current_world() == original
        assert await tab.eval("window.__placementMessages") == []
        assert not tab.errors

    viewer("phone", lang, journey, world=world)
