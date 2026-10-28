"""Explicit surface inspection owns its tap and changes only the camera."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from test_viewer_camera_views_browser import pose, view
from test_viewer_creation_browser import click, key
from test_viewer_layout import pytestmark, viewer  # noqa: F401
from test_viewer_object_edit_browser import choose
from test_viewer_startup_browser import READY
from viewer_camera import settle_camera


def world(pg):
    pg.add_robot("rover", template="rover", pos=[-2, -1])
    pg.add_object("block", shape="box", size=[0.4, 0.4, 0.3], pos=[0, 0], fixed=True)
    pg.add_terrain("ramp", name="dock", pos=[2, 0], params={"length": 1, "width": 1, "angle": 12})


async def prepare(tab, server):
    server.pg.session.paused = True
    server.pg.session.reset()
    await tab.until(f"{READY} && window.__robotforge.time===0")
    eye = [12, -16, 12] if tab.size == "phone" else [5, -7, 5]
    await tab.run(f"window.__robotforge.lookAt({eye},[.3,0,.3])")
    await settle_camera(tab)
    await tab.run("""
      window.__inspectMessages=[];
      const send=WebSocket.prototype.send;
      WebSocket.prototype.send=function(s){const m=JSON.parse(s);
        if(m.type!=='seen')window.__inspectMessages.push(m);return send.call(this,s)};
      const T=await import('three');
      window.addEventListener('pointerup',e=>{
        if(document.getElementById('inspection-bar').hidden||e.target.id!=='view')return;
        const rf=window.__robotforge,{scene,camera}=rf.three();
        scene.updateMatrixWorld(true);camera.updateMatrixWorld();const r=new T.Raycaster();
        r.setFromCamera(new T.Vector2(e.clientX/innerWidth*2-1,1-e.clientY/innerHeight*2),camera);
        const h=r.intersectObjects(scene.children,true).find(h=>h.object.visible&&h.object.isMesh
          &&(Number.isInteger(h.object.userData.geom)||h.object.userData.flex));
        if(h)window.__inspectRelease={point:h.point.toArray(),distance:camera.position.distanceTo(h.point),
          rayOrigin:r.ray.origin.toArray(),rayDirection:r.ray.direction.toArray(),
          direction:camera.position.clone().sub(new T.Vector3().fromArray(rf.cameraTarget())).normalize().toArray()};
      },true);
    """)


async def begin(tab):
    await click(tab, "#view-toggle")
    await click(tab, "#view-inspect")
    await tab.until("!document.getElementById('inspection-bar').hidden")
    await settle_camera(tab)


async def ray(tab, xyz):
    return await tab.run(f"""
      const rf=window.__robotforge,T=await import('three'),{{scene,camera}}=rf.three();
      scene.updateMatrixWorld(true);camera.updateMatrixWorld();
      const screen=rf.screenOfPoint({json.dumps(xyz)}),r=new T.Raycaster();
      r.setFromCamera(new T.Vector2(screen[0]/innerWidth*2-1,1-screen[1]/innerHeight*2),camera);
      const h=r.intersectObjects(scene.children,true).find(h=>h.object.visible&&h.object.isMesh
        &&(Number.isInteger(h.object.userData.geom)||h.object.userData.flex));
      const g=rf.scene.geoms.find(g=>g.id===h?.object.userData.geom);
      const owners=[...(rf.scene.object_edit?.objects||[]),...(rf.scene.terrain_edit?.terrains||[])]
        .filter(e=>e.geoms.includes(g?.id)).map(e=>e.name);
      return {{screen,point:h?.point.toArray(),body:h?.object.userData.body,
        bodyName:rf.scene.bodies[h?.object.userData.body]?.name,geom:g?.name,owners,type:g?.type,flex:!!h?.object.userData.flex}};
    """)


async def named_ray(tab, name):
    found = await tab.run(f"""
      const rf=window.__robotforge,T=await import('three'),{{scene,camera}}=rf.three(),points=[],candidates=[];
      const ids=new Set([...(rf.scene.object_edit?.objects||[]),...(rf.scene.terrain_edit?.terrains||[])]
        .filter(e=>e.name==={name!r}).flatMap(e=>e.geoms));
      scene.updateMatrixWorld(true);camera.updateMatrixWorld();scene.traverse(o=>{{
        const g=rf.scene.geoms.find(g=>g.id===o.userData?.geom);
        if(!o.isMesh||!o.visible||!g||!(ids.has(g.id)||((g.name||'')+(rf.scene.bodies[g.body]?.name||'')).includes({name!r})))return;
        const p=new T.Box3().setFromObject(o).getCenter(new T.Vector3()),q=p.clone().project(camera);
        const r=new T.Raycaster();
        if(Math.abs(q.x)>.85||Math.abs(q.y)>.75)return;
        r.setFromCamera(new T.Vector2(q.x,q.y),camera);
        const hit=r.intersectObjects(scene.children,true)
          .find(h=>h.object.visible&&Number.isInteger(h.object.userData.geom));
        candidates.push({{name:g.name,body:rf.scene.bodies[g.body]?.name,point:p.toArray(),screen:q.toArray(),hit:hit?.object.userData.geom,own:o.userData.geom}});
        if(hit?.object===o)points.push(p.toArray());
      }});return {{point:points[0],candidates}};
    """)
    assert found.get("point"), (name, found)
    return await ray(tab, found["point"])


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_robot_fixed_terrain_and_floor_inspection_keeps_physics_and_returns(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        saved = server.pg.store.file.read_bytes()
        data = server.pg.session.data
        state = {n: getattr(data, n).copy() for n in ("qpos", "qvel", "ctrl")}
        resources = await tab.eval("window.__robotforge.three().renderer.info.memory")
        for xyz, identity in (
            ([-2, -1, 0.2], "rover"),
            ([0, 0, 0.3], "block"),
            ([2, 0, 0.15], "dock"),
            ([0, -2, 0], "floor"),
        ):
            original = await pose(tab)
            await begin(tab)
            expected = await ray(tab, xyz) if identity == "floor" else await named_ray(tab, identity)
            assert expected["point"]
            if identity == "floor":
                assert expected["type"] == "plane"
            else:
                assert identity in (expected.get("bodyName") or "") + (expected.get("geom") or "") + ",".join(
                    expected["owners"]
                )
            await click(tab, point=expected["screen"])
            await tab.until("document.getElementById('inspection-bar').hidden")
            await settle_camera(tab)
            result = await pose(tab)
            released = await tab.eval("window.__inspectRelease")
            assert released["point"] == pytest.approx(expected["point"], abs=1e-6)
            assert result["target"] == pytest.approx(released["point"], abs=1e-8)
            assert result["direction"] == pytest.approx(released["direction"], abs=1e-8)
            assert result["distance"] == pytest.approx(released["distance"], abs=1e-8)
            assert await tab.eval("window.__robotforge.following()") is None
            # Subsequent axis orbit and dolly retain the chosen 3D point, rather than an old scene aim.
            await view(tab, "side")
            assert (await pose(tab))["target"] == pytest.approx(released["point"], abs=1e-8)
            await tab.page.call("Input.dispatchMouseEvent", type="mouseWheel", x=100, y=300, deltaX=0, deltaY=-120)
            await settle_camera(tab)
            assert (await pose(tab))["target"] == pytest.approx(released["point"], abs=1e-8)
            assert (await pose(tab))["distance"] < result["distance"]
            returned = await view(tab, "back")
            assert returned["eye"] == pytest.approx(original["eye"], abs=1e-8)
            assert returned["target"] == pytest.approx(original["target"], abs=1e-8)
        assert await tab.eval("window.__inspectMessages") == []
        assert await tab.eval("window.__robotforge.three().renderer.info.memory") == resources
        assert server.pg.store.file.read_bytes() == saved
        assert server.pg.session.data is data and server.pg.session.paused and data.time == 0
        for n, expected in state.items():
            np.testing.assert_array_equal(getattr(data, n), expected)
        assert not tab.errors

    viewer(size, lang, scenario, world=world)


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("phone", "zh-CN")])
def test_inspection_does_not_consume_editor_or_submit_on_enter_cancel_or_reentry(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="objects"]')
        await click(tab, '[data-object="ball"]')
        saved = server.pg.store.file.read_bytes()
        await tab.run("window.__inspectMessages=[]")
        await begin(tab)
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        assert not await tab.eval("document.getElementById('inspection-bar').hidden")
        # A keyboard-native activation also exits the old mode before another inspection begins.
        await tab.run("document.getElementById('view-toggle').focus()")
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        assert await tab.eval("document.getElementById('inspection-bar').hidden")
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        await click(tab, "#view-inspect")
        await key(tab, "Escape")
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        await begin(tab)
        hit = await ray(tab, [0, 0, 0.3])
        await click(tab, point=hit["screen"])
        await tab.until("document.getElementById('inspection-bar').hidden")
        await settle_camera(tab)
        assert not await tab.eval("document.getElementById('placement-bar').hidden")
        assert await tab.eval("document.getElementById('place-here').disabled")
        assert server.pg.store.file.read_bytes() == saved
        assert not any(
            m["type"] in {"world", "drag_start", "go_to"} for m in await tab.eval("window.__inspectMessages")
        )
        await key(tab, "Escape")
        assert await tab.eval("document.getElementById('placement-bar').hidden")
        await choose(tab, "block")
        await tab.run("window.__inspectMessages=[]")
        await begin(tab)
        assert await tab.eval("document.getElementById('object-edit-bar').hidden")
        await click(tab, "#inspection-cancel")
        assert not await tab.eval("document.getElementById('object-edit-bar').hidden")
        await begin(tab)
        hit = await named_ray(tab, "block")
        await click(tab, point=hit["screen"])
        await tab.until("document.getElementById('inspection-bar').hidden")
        await settle_camera(tab)
        assert not await tab.eval("document.getElementById('object-edit-bar').hidden")
        assert await tab.eval("document.getElementById('object-edit-title').textContent") == "block"
        assert await tab.eval("window.__inspectMessages") == []
        assert server.pg.store.file.read_bytes() == saved
        assert not tab.errors

    viewer(size, lang, scenario, world=world)


def test_drag_multitouch_stale_generation_and_blur_cannot_pick(viewer):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        await begin(tab)
        for kind, points in [
            ("touchStart", [{"id": 1, "x": 150, "y": 400}]),
            ("touchMove", [{"id": 1, "x": 190, "y": 410}]),
            ("touchEnd", []),
            ("touchStart", [{"id": 1, "x": 150, "y": 400}, {"id": 2, "x": 230, "y": 420}]),
            ("touchEnd", []),
        ]:
            await tab.page.call("Input.dispatchTouchEvent", type=kind, touchPoints=points)
        await settle_camera(tab)
        assert not await tab.eval("document.getElementById('inspection-bar').hidden")
        hit = await ray(tab, [0, 0, 0.3])
        x, y = hit["screen"]
        await tab.page.call("Input.dispatchTouchEvent", type="touchStart", touchPoints=[{"x": x, "y": y}])
        server.pg.session.reset()
        await tab.until("document.getElementById('inspection-bar').hidden")
        await tab.page.call("Input.dispatchTouchEvent", type="touchEnd", touchPoints=[])
        await asyncio.sleep(0.1)  # Let the deliberately stale release callback run.
        assert await tab.eval("document.getElementById('view-back').disabled")
        await begin(tab)
        await tab.run("window.dispatchEvent(new Event('blur'))")
        assert await tab.eval("document.getElementById('inspection-bar').hidden")
        await begin(tab)
        hit = await ray(tab, [0, 0, 0.3])
        x, y = hit["screen"]
        await tab.page.call("Input.dispatchTouchEvent", type="touchStart", touchPoints=[{"x": x, "y": y}])
        version = server.pg.session.version
        server.pg.set_floor("grass")
        await tab.until(f"{READY} && window.__robotforge.scene.version>{version}")
        assert await tab.eval("document.getElementById('inspection-bar').hidden")
        await tab.page.call("Input.dispatchTouchEvent", type="touchEnd", touchPoints=[])
        await asyncio.sleep(0.1)  # A held old-scene release cannot create camera history.
        assert await tab.eval("document.getElementById('view-back').disabled")
        assert not any(
            m["type"] in {"world", "drag_start", "go_to"} for m in await tab.eval("window.__inspectMessages")
        )
        assert not tab.errors

    viewer("phone", "zh-CN", scenario, world=world)


def water_and_cloth(pg):
    world(pg)
    pg.set_floor(water={"level": 0.4, "area": [0.5, -1, 2, 1], "waves": 0.02})
    pg.add_object("cloth", shape="cloth", size=[0.6, 0.6], pos=[-1, 0, 0.8])


def test_water_uses_bounded_still_level_and_deformed_cloth_is_not_skipped(viewer):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        await begin(tab)
        point = await tab.eval("window.__robotforge.screenOfPoint([1,0,.4])")
        await click(tab, point=point)
        await tab.until("document.getElementById('inspection-bar').hidden")
        await settle_camera(tab)
        actual = await tab.eval("window.__inspectRelease")
        origin, direction = np.array(actual["rayOrigin"]), np.array(actual["rayDirection"])
        expected_water = origin + direction * ((0.4 - origin[2]) / direction[2])
        assert expected_water == pytest.approx([1, 0, 0.4], abs=1e-6)
        assert (await pose(tab))["target"] == pytest.approx(expected_water, abs=1e-8)
        await view(tab, "back")
        await begin(tab)
        expected = await ray(tab, [-1, 0, 0.8])
        assert expected["flex"]
        await click(tab, point=expected["screen"])
        await tab.until("document.getElementById('inspection-bar').hidden")
        await settle_camera(tab)
        assert (await pose(tab))["target"] == pytest.approx(
            (await tab.eval("window.__inspectRelease"))["point"], abs=1e-8
        )
        assert not any(
            m["type"] in {"world", "drag_start", "go_to"} for m in await tab.eval("window.__inspectMessages")
        )
        assert not tab.errors

    viewer("laptop", "en", scenario, world=water_and_cloth)


def layers(pg):
    pg.add_object("front", shape="box", size=[1, 1, 0.2], pos=[0, 0, 4], fixed=True)
    pg.add_object("back", shape="box", size=[1, 1, 0.2], pos=[0, 0, 2], fixed=True)


def test_clipped_hidden_and_transparent_foreground_does_not_hide_visible_surface(viewer):  # noqa: F811
    async def scenario(tab, server):
        await prepare(tab, server)
        for excluded in ("clipped", "hidden", "transparent"):
            await tab.run("""
              const rf=window.__robotforge,{scene,camera}=rf.three();
              const ids=new Set(rf.scene.object_edit.objects.find(e=>e.name==='front').geoms);
              scene.traverse(o=>{if(o.isMesh&&ids.has(o.userData.geom)){o.visible=true;o.material.opacity=1}});
              camera.near=.01;camera.far=30;camera.updateProjectionMatrix();
              rf.lookAt([0,-.001,5],[0,0,2]);
            """)
            await settle_camera(tab)
            await tab.run(f"""
              const rf=window.__robotforge,{{scene,camera}}=rf.three();
              const ids=new Set(rf.scene.object_edit.objects.find(e=>e.name==='front').geoms);
              if({excluded!r}==='clipped')camera.near=1.5;
              scene.traverse(o=>{{if(!o.isMesh||!ids.has(o.userData.geom))return;
                if({excluded!r}==='hidden')o.visible=false;
                if({excluded!r}==='transparent')o.material.opacity=0;
              }});camera.updateProjectionMatrix();
            """)
            await begin(tab)
            point = await tab.eval("window.__robotforge.screenOfPoint([0,0,2])")
            await click(tab, point=point)
            await tab.until("document.getElementById('inspection-bar').hidden")
            await settle_camera(tab)
            assert 1.9 <= (await pose(tab))["target"][2] <= 2.2
            await view(tab, "back")
        # Both source boxes are outside the visible depth interval: keep inspecting, never pick the floor.
        await tab.run("const c=window.__robotforge.three().camera;c.near=1.5;c.far=2;c.updateProjectionMatrix()")
        await begin(tab)
        original = await pose(tab)
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([0,0,2])"))
        await tab.until("document.getElementById('toast').classList.contains('show')")
        assert not await tab.eval("document.getElementById('inspection-bar').hidden")
        assert (await pose(tab))["target"] == pytest.approx(original["target"], abs=1e-8)
        await key(tab, "Escape")
        assert not tab.errors

    viewer("laptop", "en", scenario, world=layers)
