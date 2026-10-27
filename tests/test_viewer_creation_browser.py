"""Native Chrome pointer/keyboard/touch journeys through the local creation loop."""

from __future__ import annotations

import asyncio
import json

import pytest
from test_viewer_layout import pytestmark, viewer  # noqa: F401, F811


async def click(tab, selector=None, point=None):
    if selector:
        point = await tab.eval(f"""(() => {{const e=document.querySelector({json.dumps(selector)});
          e.scrollIntoView({{block:'center'}}); const r=e.getBoundingClientRect();
          if(!r.width||!r.height)throw Error('hidden '+{json.dumps(selector)});
          return [r.x+r.width/2,r.y+r.height/2];}})()""")
    x, y = point
    if tab.size == "phone":
        await tab.page.call("Input.dispatchTouchEvent", type="touchStart", touchPoints=[{"x": x, "y": y}])
        await tab.page.call("Input.dispatchTouchEvent", type="touchEnd", touchPoints=[])
    else:
        await tab.page.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        for kind in ("mousePressed", "mouseReleased"):
            await tab.page.call("Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1)
    await asyncio.sleep(0.15)


async def key(tab, value, code=None, kind="keyDown"):
    await tab.page.call(
        "Input.dispatchKeyEvent",
        type=kind,
        key=value,
        code=code or value,
        windowsVirtualKeyCode={"Enter": 13, "Tab": 9, "Escape": 27}.get(value, 0),
        text="\r" if value == "Enter" and kind == "keyDown" else "",
    )


async def choose(tab, template):
    await click(tab, "#drawer-toggle")
    if await tab.eval("document.getElementById('create-options').hidden"):
        await click(tab, "#create-toggle")
    await click(tab, f'.create-card[data-template="{template}"]')
    await tab.until("!document.getElementById('placement-bar').hidden")
    await asyncio.sleep(0.4)


@pytest.mark.parametrize("size,lang", [("laptop", "en"), ("laptop", "zh-CN"), ("phone", "en"), ("phone", "zh-CN")])
def test_empty_world_to_local_robot_placement_cancel_and_map_loop(viewer, size, lang):  # noqa: F811
    async def scenario(tab, server):
        assert await tab.eval("document.getElementById('drawer-label').textContent") in ["Robots", "机器人"]
        await click(tab, "#world-toggle")
        assert await tab.eval("document.getElementById('empty').hidden")
        await click(tab, "#world-toggle")
        await click(tab, "#empty-add")
        assert await tab.eval("document.querySelectorAll('#create-list .create-card').length") == 4
        await click(tab, '.create-card[data-template="rover"]')
        assert await tab.eval("""(()=>{const r=document.getElementById('placement-bar').getBoundingClientRect(),
          d=document.querySelector('.dock').getBoundingClientRect();
          return r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=d.top;})()""")
        # Native Tab must leave the canvas; focused Cancel must keep its native Enter behavior.
        await key(tab, "Tab")
        assert await tab.eval("document.activeElement.id !== 'view'")
        await tab.eval("document.getElementById('place-cancel').focus()")
        await key(tab, "Enter")
        await key(tab, "Enter", kind="keyUp")
        await tab.until("document.getElementById('placement-bar').hidden", timeout=3)
        assert not server.pg.store.load().robots
        for template, pos, expected in [("rover", [0, 0, 0], "rover"), ("arm", [2, 0, 0], "arm")]:
            await tab.eval("window.__robotforge.lookAt([6,-10,8],[0,0,0])")
            await choose(tab, template)
            point = await tab.eval(f"window.__robotforge.screenOfPoint({pos})")
            await click(tab, point=point)
            try:
                await tab.until(
                    f"window.__robotforge.selected() === {expected!r} && !window.__robotforge.busy().busy", timeout=5
                )
            except TimeoutError:
                await tab.capture(tab.home / "creation-failure.png")
                pytest.fail(
                    str(
                        await tab.eval(f"""({{point:{point},target:document.elementFromPoint(...{point})?.id,
                  text:document.body.innerText,selected:window.__robotforge.selected(),busy:window.__robotforge.busy()}})""")
                    )
                )
            await tab.until("document.getElementById('placement-bar').hidden")
        assert len(server.pg.store.load().robots) == 2
        # A static arm can be added again away from its saved origin, with a unique name.
        await tab.eval("window.__robotforge.lookAt([6,-10,8],[0,0,0])")
        await choose(tab, "arm")
        await click(tab, point=await tab.eval("window.__robotforge.screenOfPoint([-2,0,0])"))
        await tab.until("window.__robotforge.selected() === 'arm_2' && !window.__robotforge.busy().busy")
        assert [r.name for r in server.pg.store.load().robots] == ["rover", "arm", "arm_2"]
        await click(tab, "#world-toggle")
        await click(tab, '[data-tab="terrain"]')
        index = await tab.eval("window.__robotforge.scene.catalog.terrains.indexOf('stairs')")
        await click(tab, f"#terrain-list .choice:nth-child({index + 1})")
        await tab.until("window.__robotforge.scene.terrain.includes('stairs') && !window.__robotforge.busy().busy")
        assert server.pg.store.load().terrain[0].type == "stairs"
        await click(tab, '[data-tab="maps"]')
        await click(tab, "#map-list .choice:first-child")
        await tab.until("!window.__robotforge.scene.terrain.length && !window.__robotforge.busy().busy")
        assert len(server.pg.store.load().robots) == 3
        assert not tab.errors

    viewer(size, lang, scenario, world=lambda pg: None)


def test_entering_placement_stops_held_keyboard_drive_and_suppresses_arm_pad(viewer):  # noqa: F811
    from test_viewer_gamepad import PAD, axes

    async def scenario(tab, server):
        await tab.eval("""window.__creationSent=[]; const real=WebSocket.prototype.send;
          WebSocket.prototype.send=function(data){try{window.__creationSent.push(JSON.parse(data))}catch{}
            return real.call(this,data)}; window.__robotforge.select('rover');""")
        await key(tab, "w", "KeyW")
        await tab.until("window.__creationSent.some(m=>m.type==='teleop'&&m.speed>0)")
        await choose(tab, "arm")
        await tab.until("window.__creationSent.filter(m=>m.type==='teleop').at(-1)?.speed===0")
        await key(tab, "w", "KeyW", kind="keyUp")
        await key(tab, "Escape")
        await tab.until("document.getElementById('placement-bar').hidden")
        await tab.eval("window.__robotforge.select('arm')")
        await tab.run(PAD)
        await tab.run(axes(a1=-1))
        await tab.until("window.__creationSent.some(m=>m.type==='teleop_arm'&&m.velocity.some(v=>v!==0))")
        await choose(tab, "rover")
        await tab.until("window.__robotforge.pad().profile === null")
        mark = await tab.eval("window.__creationSent.length")
        await asyncio.sleep(0.5)
        assert not await tab.eval(f"""window.__creationSent.slice({mark}).some(m=>
          m.type==='teleop_arm' && (m.velocity?.some(v=>v!==0)||m.gripper))""")
        assert not tab.errors

    def world(pg):
        pg.add_robot("rover", template="rover")
        pg.add_robot("arm", template="arm", pos=[2, 0])

    viewer("laptop", "en", scenario, world=world)
