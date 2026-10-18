"""The viewer's sensor drawings: the fields sensors.js reads are the ones the server sends, and the plain helpers
(sensorkit.js, plot.js) agree with the server's own colors and names. The helpers run in Node as the page loads
them (skipped without Node), the way tests/test_viewer_js.py checks the scripts parse."""

from __future__ import annotations

import json
import shutil
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pytest
from mcp import Client

from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sensors import camera
from robotforge.viewer import scene

STATIC = Path(__file__).resolve().parents[1] / "src" / "robotforge" / "viewer" / "static"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


def js(body: str) -> object:
    """Run `body` (an async function body that returns something JSON can carry) with the viewer's helpers
    imported as `kit` (sensorkit.js) and `plot` (plot.js)."""
    source = (
        f"import * as kit from {json.dumps((STATIC / 'sensorkit.js').as_uri())};\n"
        f"import * as plot from {json.dumps((STATIC / 'plot.js').as_uri())};\n"
        f"const out = await (async () => {{ {body} }})();\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    run = subprocess.run(
        [NODE, "--input-type=module"], input=source, capture_output=True, text=True, encoding="utf-8", timeout=60
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground.with_demo()
    playground.new_world("viewing")
    return playground


# ---- the protocol ---------------------------------------------------------------------------------------------


@needs_node
def test_every_field_the_viewer_reads_is_sent(pg):
    reads = js("return kit.READS;")
    pg.add_robot("post", build={"base": {"shape": "box", "size": [0.2, 0.2, 0.2], "fixed": True}})
    pg.add_robot("panda", library="panda")
    pg.add_object("wall", shape="wall", pos=[2.05, 0], yaw=90, size=[3, 0.1, 1.0])
    for fields in (
        {"type": "lidar"},
        {"type": "lidar3d", "name": "lidar3d"},
        {"type": "camera", "mode": "depth"},
        {"type": "imu"},
        {"type": "rangefinder"},
        {"type": "touch", "rows": 2, "cols": 2},
    ):
        pg.add_sensor("post", fields)
    pg.add_sensor("panda", {"type": "force_torque", "on": "hand"})
    pg.add_sensor("panda", {"type": "encoders"})
    msg = pg.sensors.viewer_message()
    assert msg["version"] == pg.session.version  # the page draws on the model it has loaded, and only that
    seen = set()
    for entry in msg["sensors"]:
        missing = [f for f in [*reads["all"], *reads.get(entry["type"], [])] if f not in entry]
        assert not missing, f"{entry['name']} ({entry['type']}) lacks {missing}"
        seen.add(entry["type"])
    assert seen >= set(reads) - {"all"}
    by_name = {e["name"]: e for e in msg["sensors"]}
    assert by_name["post/lidar"]["added"] is True
    # The body and the place on it put a drawing where the sensor is: a world pose from them matches the sensor's.
    lidar = by_name["post/lidar"]
    body = pg.session.data.xpos[lidar["body"]]
    assert np.allclose(np.array(body) + lidar["local_pos"], lidar["pos"], atol=1e-3)


def test_a_robot_s_own_sensors_are_not_marked_added(pg):
    pg.add_robot("dog", template="quadruped")
    entries = pg.sensors.viewer_message()["sensors"]
    assert entries and not any(e["added"] for e in entries)  # the template's IMU, camera and feet start hidden


def test_touch_pads_are_described_on_their_body(pg):
    """A pad's zones lie on the foot's surface in the body's frame, so the page lights them where the foot is now."""
    pg.add_robot("box", build={"base": {"shape": "box", "size": [0.2, 0.2, 0.1], "mass": 2.0}})
    pg.add_sensor("box", {"type": "touch", "rows": 2, "cols": 2})
    pg.run_fast(lambda: False, 0.5)  # it settles on the floor
    pad = next(e for e in pg.sensors.viewer_message()["sensors"] if e["type"] == "touch")
    assert pad["cell_shape"] == "box" and len(pad["cells_local"]) == 4
    data = pg.session.data
    xpos, xmat = data.xpos[pad["body"]], data.xmat[pad["body"]].reshape(3, 3)
    bottom = [xpos + xmat @ np.array(p) for p in pad["cells_local"]]
    assert all(abs(p[2]) < 0.004 for p in bottom)  # on the block's bottom face, which rests on the floor
    assert pad["total_n"] == pytest.approx(2.0 * 9.81, rel=0.05)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_open_viewer_shows_a_robot_s_own_camera(pg, monkeypatch):
    """open_viewer(show=...) asks the page to draw sensors it would keep hidden (a template's camera) and says which;
    a robot's name stands for all of its sensors, and a wrong name is refused with the ones there are."""
    monkeypatch.setenv("ROBOTFORGE_NO_BROWSER", "1")
    pg.add_robot("dog", template="quadruped")
    server = build_server(pg, 0)
    async with Client(server) as client:
        opened = await client.call_tool("open_viewer", {"show": ["dog/front cam"]})
        assert not opened.is_error, opened.content[0].text
        assert json.loads(opened.content[0].text)["showing"] == ["dog/front_cam"]
        first = pg.sensors.viewer_message()["show"]
        assert first["names"] == ["dog/front_cam"]
        everything = await client.call_tool("open_viewer", {"show": ["dog"]})
        assert set(json.loads(everything.content[0].text)["showing"]) >= {"dog/front_cam", "dog/imu"}
        assert pg.sensors.viewer_message()["show"]["seq"] == first["seq"] + 1  # each request applied once by the page
        wrong = await client.call_tool("open_viewer", {"show": ["dog/lidar"]})
        assert wrong.is_error and "front_cam" in wrong.content[0].text
        plain = await client.call_tool("open_viewer", {})
        assert "showing" not in json.loads(plain.content[0].text)


# ---- the helpers agree with the server -----------------------------------------------------------------------


@needs_node
def test_depth_colors_match_the_server_s_depth_picture():
    depth = np.array([[0.5, 1.0, 2.5, np.nan], [4.0, 6.0, 9.0, 10.0]])
    server = camera.depth_picture(depth, 10.0)
    flat = [None if np.isnan(v) else float(v) for v in depth.ravel()]
    page = js(
        f"const d = Float32Array.from({json.dumps(flat)}.map((v) => v ?? NaN));"
        "const out = new Uint8ClampedArray(4 * d.length);"
        "kit.depthPicture(d, 0.5, 10.0, out, { width: 4 });"
        "return Array.from(out);"
    )
    page = np.array(page, dtype=int).reshape(2, 4, 4)
    assert (page[..., 3] == 255).all()
    assert np.abs(page[..., :3] - server.astype(int)).max() <= 3  # a table of 1024 steps against the polynomial
    assert (page[0, 3, :3] == 0).all()  # nothing in range is black


@needs_node
def test_segmentation_colors_and_names_match_the_server(pg):
    pg.add_robot("dog", template="quadruped")
    pg.add_object("crate", shape="box", size=[0.2, 0.2, 0.2], pos=[1.0, 0.0], color="red")
    pg.load_scene("warehouse", keep_robots=True, starter=False)
    model = pg.session.model
    desc = scene.describe(model, 1, [])
    shown = desc["geoms"][:: max(1, len(desc["geoms"]) // 80)]  # a spread of the world's shapes
    server = {g["id"]: camera.thing_name(model, g["id"]) for g in shown}
    names = js(
        f"const bodies = {json.dumps(desc['bodies'])}; const geoms = {json.dumps(shown)};"
        "return Object.fromEntries(geoms.map((g) => [g.id, kit.thingName(bodies, g)]));"
    )
    assert {int(k): v for k, v in names.items()} == server
    things = sorted({*server.values(), "dog", "crate", "floor", "机器人"})
    colors = js(f"return {json.dumps(things)}.map((n) => [kit.crc32(n), kit.thingColor(n)]);")
    for name, (crc, color) in zip(things, colors, strict=True):
        code = zlib.crc32(name.encode("utf-8"))
        assert crc == code
        hue = (code % 360) / 360.0
        assert color == camera._hsv(hue, 0.55 + (code >> 9) % 40 / 100, 0.75 + (code >> 17) % 25 / 100)


@needs_node
def test_depth_unpacks_as_the_shader_packs_it():
    """sensors.js's depth pass writes distance / range in 16 bits over red and green, blue 255 where drawn."""
    rng = 10.0
    meters = [0.0, 0.0153, 1.5, 7.25, 9.9999]
    packed = []
    for d in meters:
        q = int(np.floor(min(max(d / rng, 0), 1) * 65535 + 0.5))
        packed += [q // 256, q % 256, 255, 255]
    packed += [0, 0, 0, 0]  # sky: nothing drawn
    out = js(
        f"const b = Uint8Array.from({json.dumps(packed)}); const d = new Float32Array(b.length / 4);"
        f"kit.unpackDepth(b, {rng}, d); return Array.from(d).map((v) => (Number.isNaN(v) ? null : v));"
    )
    assert out[-1] is None
    assert np.allclose(out[:-1], meters, atol=rng / 65535)


# ---- the helpers on their own ---------------------------------------------------------------------------------


@needs_node
def test_frustum_corners_span_the_field_of_view():
    corners = np.array(js("return kit.frustumCorners(60, 4 / 3, 0.2);"))
    assert np.allclose(corners[:, 2], -0.2)
    top = corners[2]
    assert np.degrees(2 * np.arctan2(top[1], 0.2)) == pytest.approx(60)
    assert top[0] / top[1] == pytest.approx(4 / 3)


@needs_node
def test_joint_groups_split_legs_and_runs():
    go2 = [f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")]
    panda = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]
    out = js(
        f"return [kit.jointGroups({json.dumps(go2)}), kit.jointGroups({json.dumps(panda)}),"
        " kit.jointGroups(['a', 'b'])];"
    )
    legs, runs, few = out
    assert [g["label"] for g in legs] == ["FL", "FR", "RL", "RR"]
    assert [j["short"] for j in legs[0]["joints"]] == ["hip", "thigh", "calf"]
    assert [g["label"] for g in runs] == ["1-6", "7-9"] and len(runs[0]["joints"]) == 6
    assert few == [{"label": "", "joints": [{"name": "a", "short": "a"}, {"name": "b", "short": "b"}]}]


@needs_node
def test_a_plot_does_not_magnify_noise_at_rest():
    """An IMU at rest reads 9.81 with a few hundredths of noise: the axis stays at least its span tall."""
    out = js(
        "let r = null; const seen = [];"
        "for (let k = 0; k < 50; k++) { const noise = 0.05 * Math.sin(k * 1.7);"
        "  r = plot.plotRange(r, 9.81 - Math.abs(noise), 9.81 + Math.abs(noise), 2); seen.push(r); }"
        "const big = plot.plotRange(r, 0, 30, 2);"  # a jolt widens it at once
        "const after = plot.plotRange(big, 9.8, 9.82, 2);"  # and it narrows back slowly
        "const labels = [0.04, -0.0001].map((v) => plot.formatTick(v, [0.02, 0.04]));"
        "return { seen, big, after, ticks: plot.niceTicks(-0.02, 0.06, 3), labels };"
    )
    assert all(hi - lo >= 2.0 for lo, hi in out["seen"])
    assert out["big"][0] <= 0 and out["big"][1] >= 30
    assert out["after"][1] > 25  # 5 % of the way back per draw
    assert len(out["ticks"]) >= 2 and all(-0.02 <= v <= 0.06 for v in out["ticks"])
    assert out["labels"] == ["0.04", "0"]


@needs_node
def test_lidar_color_span_and_rays():
    out = js(
        "const near = Array.from({ length: 95 }, (_, i) => 1 + i * 0.01); const far = [20, 25, 28, 29, 30];"
        "const span = kit.quantileSpan([...near, ...far, NaN]);"
        "const steady = [kit.steadySpan(null, [1, 5]), kit.steadySpan([1, 5], [2, 9]),"
        " kit.steadySpan([1, 5], [0.5, 4])];"
        "const many = kit.rayPicks(1200, 150);"
        "const arrow = [kit.arrowLength(5, 10, 0.2), kit.arrowLength(50, 10, 0.2), kit.arrowLength(1, 0)];"
        "return { span, steady, picks: [kit.rayPicks(5, 10), many.length, many.at(-1)], arrow,"
        "  back: kit.pullBack([0, 0, 0], [3, 4, 0], 0.5) };"
    )
    low, high = out["span"]
    assert low == pytest.approx(1.05, abs=0.02) and high < 2.0  # the farthest few do not wash the near ones out
    assert out["steady"][0] == [1, 5]
    assert out["steady"][1][1] == 9 and out["steady"][1][0] == pytest.approx(1.08)  # widens at once, narrows slowly
    assert out["steady"][2][0] == 0.5
    assert out["picks"][0] == [0, 1, 2, 3, 4] and out["picks"][1] == 150 and out["picks"][2] < 1200
    assert out["back"] == pytest.approx([2.7, 3.6, 0])
    assert out["arrow"] == pytest.approx([0.1, 0.2, 0])
