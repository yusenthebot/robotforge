"""Actual late lesson-open replies retain deferred fitting and explicit camera intent."""

from __future__ import annotations

import json
import math

import pytest
from test_viewer_build_flow_browser import instrument_wire
from test_viewer_creation_browser import click
from test_viewer_layout import RIG_ACROSS, pytestmark, viewer  # noqa: F401
from test_viewer_touch_build_browser import touch
from viewer_camera import settle_camera


@pytest.mark.parametrize("manual", [False, True])
def test_late_lesson_open_reply_refits_unless_user_moves_camera(viewer, monkeypatch, manual):  # noqa: F811
    async def scenario(tab, server):
        original_start = server.pg.lessons.start

        def paused_start(lesson):
            result = original_start(lesson)
            server.pg.session.paused = True  # Hold initial poses while the real reply is delayed.
            return result

        monkeypatch.setattr(server.pg.lessons, "start", paused_start)
        await instrument_wire(tab)
        await click(tab, "#learn-toggle")
        await tab.until("!!document.querySelector('[data-lesson=pid_twins]') && window.__robotforge.busy().pending===0")
        await tab.run("window.__flow.holdReply=true")
        await click(tab, '[data-lesson="pid_twins"]')
        await tab.until("""
          window.__flow.heldMessages.some(m=>m.data?.id==='pid_twins')
          && window.__robotforge.lesson()?.id==='pid_twins'
          && window.__robotforge.renderedVersion===window.__robotforge.scene.version
          && document.getElementById('btn-play').classList.contains('is-paused')
        """)
        trace = {"first_poses_before_reply": await tab.eval("window.__robotforge.time")}
        trace["requested_camera"] = await tab.eval(
            "window.__flow.heldMessages.find(m=>m.data?.id==='pid_twins').data.camera"
        )
        await tab.run("window.__flow.releaseReplies()")
        await tab.until("window.__robotforge.busy().pending===0")
        if manual:
            x, y = await tab.run("""
              const rf=window.__robotforge,r=rf.freeRect();
              for(let j=1;j<4;j++)for(let i=1;i<4;i++){
                const x=r.left+(r.right-r.left)*i/5,y=r.top+(r.bottom-r.top)*j/5;
                if(document.elementFromPoint(x,y)?.id==='view'&&rf.pick(x,y)===null)return [x,y];
              }throw Error('no clear canvas point for native camera gesture');
            """)
            await touch(tab, "touchStart", [(1, x, y)])
            await touch(tab, "touchMove", [(1, x + 25, y + 15)])
            await touch(tab, "touchEnd")
            await settle_camera(tab)
            trace["manual_camera"] = await tab.eval("""(()=>{const rf=window.__robotforge;
              return [...rf.three().camera.position.toArray(),...rf.cameraTarget()]})()""")
        await click(tab, "#btn-play")
        await tab.until("window.__robotforge.time>=1.3")
        await settle_camera(tab, quiet_ms=1000)
        trace["seen"] = await tab.eval(RIG_ACROSS)
        (tab.home / "late-camera.json").write_text(json.dumps(trace, indent=2) + "\n")
        if manual:
            camera = await tab.eval("""(()=>{const rf=window.__robotforge;
              return [...rf.three().camera.position.toArray(),...rf.cameraTarget()]})()""")
            assert camera == pytest.approx(trace["manual_camera"], abs=1e-7)
            requested = trace["requested_camera"]
            requested_direction = [a - b for a, b in zip(requested["eye"], requested["target"], strict=True)]
            actual = [a - b for a, b in zip(camera[:3], camera[3:], strict=True)]
            dot = sum(a * b for a, b in zip(requested_direction, actual, strict=True))
            cosine = dot / math.sqrt(sum(v * v for v in requested_direction) * sum(v * v for v in actual))
            assert cosine < 0.9999  # A deferred automatic fit would restore the requested lesson direction.
        else:
            seen = trace["seen"]
            assert sorted(seen["robots"]) == ["a", "b", "c"], seen
            assert all(left >= 0 and right <= seen["width"] for left, right in seen["robots"].values()), seen
        assert await tab.eval("window.__qaErrors") == []

    viewer("phone", "en", scenario)
