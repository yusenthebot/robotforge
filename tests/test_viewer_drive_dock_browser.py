"""Inventory and driving controls must leave the simulation dock usable."""

from __future__ import annotations

import pytest
from test_viewer_creation_browser import click
from test_viewer_layout import pytestmark, viewer  # noqa: F401


@pytest.mark.parametrize("lang", ["en", "zh-CN"])
@pytest.mark.parametrize("width", [320, 700, 820, 821, 1280, 1920])
def test_inventory_keeps_reset_and_dock_reachable(viewer, width, lang):  # noqa: F811
    async def scenario(tab, server):
        await tab.page.call(
            "Emulation.setDeviceMetricsOverride", width=width, height=800, deviceScaleFactor=2, mobile=False
        )
        await tab.run("window.__robotforge.select('rover')")
        await tab.until("!document.getElementById('drive').hidden")
        await click(tab, "#btn-play")
        await tab.until("document.getElementById('btn-play').classList.contains('is-paused')")
        await tab.run("""window.__dockResets=[];window.__dockTrusted=0;
          const send=WebSocket.prototype.send;
          WebSocket.prototype.send=function(s){const m=JSON.parse(s);
            if(m.type==='cmd'&&m.cmd==='reset')window.__dockResets.push(m);return send.call(this,s)};
          document.getElementById('btn-reset').addEventListener('click',e=>{
            if(e.isTrusted)window.__dockTrusted++;
          });""")
        for opened in [True, False]:
            await click(tab, "#world-toggle")
            assert await tab.eval("!document.getElementById('world-panel').hidden") is opened
            controls = await tab.eval("""(()=>{
              const shown=e=>e.getClientRects().length&&getComputedStyle(e).display!=='none';
              const obstacles=[...document.querySelectorAll('.drive-hud > *,#world-panel')].filter(shown);
              return [...document.querySelectorAll('.dock button')].filter(shown).map(e=>{
                const r=e.getBoundingClientRect(),hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
                return {id:e.id||e.dataset.speed,hit:e.contains(hit),
                  inside:r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=innerHeight,
                  overlaps:obstacles.filter(o=>{const b=o.getBoundingClientRect();
                    return Math.min(r.right,b.right)>Math.max(r.left,b.left)
                      &&Math.min(r.bottom,b.bottom)>Math.max(r.top,b.top);
                  }).map(o=>o.id)};
              });})()""")
            assert controls and all(c["hit"] and c["inside"] and not c["overlaps"] for c in controls), controls
            generation = server.pg.session.generation
            count = len(await tab.eval("window.__dockResets"))
            await click(tab, "#btn-reset")
            await tab.until(f"window.__dockTrusted==={count + 1} && window.__dockResets.length==={count + 1}")
            assert server.pg.session.generation == generation + 1
            assert server.pg.session.data.time == 0
            assert server.pg.session.paused
            assert await tab.eval("!document.getElementById('world-panel').hidden") is opened
        assert not tab.errors

    viewer("laptop", lang, scenario, world=lambda pg: pg.add_robot("rover", template="rover"))
