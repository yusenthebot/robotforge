"""The native browser decoder retains the old packed PNG pixels exactly."""

from __future__ import annotations

import pytest
from test_viewer_layout import pytestmark, viewer  # noqa: F401


@pytest.mark.parametrize(
    "style,expected",
    [
        ("wood", "48039c51a4638ac3a7e213fbd61125e3da47b5de91471c7d8aa95d7f6bc54dfa"),
        ("grass", "40f0ec20a41e4b4f80e1ce699dee876753dde9e60db330daafd413737ec84442"),
    ],
)
def test_browser_decodes_compact_detail_to_previous_exact_rgba(viewer, style, expected):  # noqa: F811
    async def scenario(tab, _server):
        result = await tab.run(f"""
          const module=await import('./static/ground-{style}.js');
          const tile=await module.{style}Tiles();
          const canvas=document.createElement('canvas');canvas.width=canvas.height=256;
          const ctx=canvas.getContext('2d');ctx.drawImage(tile.detail,0,0);
          const pixels=ctx.getImageData(0,0,256,256).data;
          const digest=await crypto.subtle.digest('SHA-256',pixels);
          return {{hash:Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join(''),
            size:[tile.detail.naturalWidth,tile.detail.naturalHeight],
            lossless:tile.detail.src.startsWith('data:image/webp;base64,')}};
        """)
        assert result == {"hash": expected, "size": [256, 256], "lossless": True}
        assert await tab.eval("window.__qaErrors") == []

    viewer("laptop", "en", scenario, world=lambda pg: None)
