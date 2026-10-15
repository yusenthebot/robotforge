"""Colors as agents write them: names, hex codes and [r, g, b] lists, everywhere a color is accepted.

A Codex session called add_object(color="#9C6B3F"), was told the color was unknown and retried with "brown".
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robotforge.design import components
from robotforge.design.builder.guide import guide
from robotforge.design.world import World, compile_world
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import SimSession


def _hex(code: str) -> list[float]:
    """The rgba a hex code stands for, written out independently of the parser under test."""
    return [int(code[i : i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0]


def _colors(model: mujoco.MjModel, body: str) -> list[list[float]]:
    """The rgba of every geom on a body (collision shapes and the visual meshes drawn over them)."""
    b = model.body(body).id
    return [model.geom_rgba[g].tolist() for g in range(model.ngeom) if model.geom_bodyid[g] == b]


def _wears(model: mujoco.MjModel, body: str, rgba: list[float]) -> bool:
    return any(np.allclose(c, rgba, atol=1e-6) for c in _colors(model, body))


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("paint")
    return playground


# ---- parsing --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, rgba",
    [
        ("#9C6B3F", _hex("#9c6b3f")),
        ("#9c6b3f", _hex("#9c6b3f")),
        ("#0f0", [0.0, 1.0, 0.0, 1.0]),
        ("#AbC", _hex("#aabbcc")),
        (" #ffffff ", [1.0, 1.0, 1.0, 1.0]),
        ("brown", list(components.COLORS["brown"])),
        ("Brown", list(components.COLORS["brown"])),
        ("grey", list(components.COLORS["gray"])),
        ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 1.0]),
        ((0.1, 0.2, 0.3, 0.5), [0.1, 0.2, 0.3, 0.5]),
    ],
)
def test_a_color_is_a_name_a_hex_code_or_numbers(value, rgba):
    assert components.color(value) == pytest.approx(rgba)


@pytest.mark.parametrize(
    "value, message",
    [
        ("chartreuse", r"unknown color 'chartreuse'; use one of red, .*a hex code like \"#9c6b3f\""),
        ("#12345", r"unknown color '#12345': a hex color is # and 6 or 3 digits"),
        ("#9c6b3f80", "6 or 3 digits"),
        ("#ggg", "6 or 3 digits"),
        ("9c6b3f", "unknown color '9c6b3f'"),
        ([156, 107, 63], r"in 0\.\.1 \(or a hex code"),
        ([0.1, 0.2], "3 or 4 numbers"),
    ],
)
def test_a_color_that_does_not_parse_says_what_works(value, message):
    with pytest.raises(ValueError, match=message):
        components.color(value)


def test_the_builder_guide_says_hex_works():
    assert '"#9c6b3f"' in guide()["robot"]


# ---- world.yaml -----------------------------------------------------------------------------------

HEX_WORLD = {
    "robots": [
        {"name": "dog", "template": "quadruped", "params": {"color": "#FF8800"}},
        {"name": "bot", "build": {"color": "#336699", "base": {"size": [0.2, 0.1, 0.05], "color": "#fff"}}},
    ],
    "objects": [
        {"name": "crate", "color": "#9C6B3F", "pos": [0.5, 0]},
        {"name": "ball", "shape": "sphere", "color": "brown", "pos": [0.8, 0]},
        {"name": "tile", "color": [0.1, 0.2, 0.3], "pos": [1.1, 0]},
    ],
}


def test_hex_colors_are_saved_quoted_and_load_back_unchanged():
    world = World.model_validate(HEX_WORLD)
    text = world.to_yaml()
    # Unquoted, YAML would read "#..." as a comment and the color as missing.
    assert "color: '#9C6B3F'" in text and "color: '#FF8800'" in text and "color: '#fff'" in text
    again = World.from_yaml(text)
    assert again == world
    assert again.objects[0].color == "#9C6B3F"  # kept as the agent wrote it

    model = compile_world(again).model
    assert _colors(model, "crate")[0] == pytest.approx(_hex("#9c6b3f"))
    assert _colors(model, "ball")[0] == pytest.approx(list(components.COLORS["brown"]))
    assert _colors(model, "tile")[0] == pytest.approx([0.1, 0.2, 0.3, 1.0])
    assert _wears(model, "dog/trunk", _hex("#ff8800"))
    assert _wears(model, "bot/base", [1.0, 1.0, 1.0, 1.0])


# world.yaml as robotforge wrote it before hex colors existed: names and lists only.
OLD_WORLD_YAML = """\
name: colors
floor:
  style: checker
  friction: 1.0
terrain: []
robots:
- name: dog
  template: quadruped
  file: null
  library: null
  build: null
  params:
    color: orange
  pos:
  - 0.0
  - 1.5
  yaw: 0.0
objects:
- name: old
  shape: box
  size:
  - 0.05
  - 0.05
  - 0.05
  pos:
  - 0.5
  - -0.5
  yaw: 0.0
  color: brown
  mass: null
  fixed: false
- name: lst
  shape: box
  size:
  - 0.05
  - 0.05
  - 0.05
  pos:
  - 0.8
  - -0.5
  yaw: 0.0
  color:
  - 0.1
  - 0.2
  - 0.3
  mass: null
  fixed: false
"""


def test_worlds_saved_before_hex_colors_still_load():
    world = World.from_yaml(OLD_WORLD_YAML)
    assert world.to_yaml() == OLD_WORLD_YAML
    model = compile_world(world).model
    assert _colors(model, "old")[0] == pytest.approx(list(components.COLORS["brown"]))
    assert _colors(model, "lst")[0] == pytest.approx([0.1, 0.2, 0.3, 1.0])
    assert _wears(model, "dog/trunk", list(components.COLORS["orange"]))


@pytest.mark.parametrize(
    "text, line",
    [
        ("name: w\nobjects:\n- name: crate\n  color: #9C6B3F\n", 4),
        ("name: w\nobjects:\n- name: crate\n  color: #fff   # white\n", 4),
        ("name: w\nobjects: [{name: crate, color: #9C6B3F}]\n", 2),
        ("name: w\nrobots:\n- name: bot\n  build:\n    base:\n      size: [0.2, 0.1, 0.05]\n      color: #fff\n", 7),
    ],
)
def test_an_unquoted_hex_color_in_a_hand_written_world_says_to_quote_it(text, line):
    # Without the check a part's color would silently fall back to its default.
    with pytest.raises(ValueError, match=f'world.yaml line {line}: put the hex color in quotes, color: "#'):
        World.from_yaml(text)


def test_a_hex_code_inside_a_comment_is_left_alone():
    world = World.from_yaml("name: w\nobjects:\n- name: crate  # brown like #9C6B3F\n  color: brown\n")
    assert world.objects[0].color == "brown"


def test_a_world_yaml_that_is_not_yaml_is_a_value_error():
    with pytest.raises(ValueError, match="not valid YAML"):
        World.from_yaml("name: w\nobjects: [{name: crate\n")


# ---- through the Playground -----------------------------------------------------------------------


def test_add_object_and_add_robot_take_hex_colors(pg):
    pg.add_object("crate", size=[0.2, 0.2, 0.2], pos=[0.6, 0], color="#9C6B3F")
    pg.add_robot("dog", template="quadruped", params={"color": "#FF8800"}, pos=[-0.4, 0])
    m = pg.session.model
    assert _colors(m, "crate")[0] == pytest.approx(_hex("#9c6b3f"))
    assert _wears(m, "dog/trunk", _hex("#ff8800"))

    pg.update_object("crate", color="#0f0")
    pg.update_robot("dog", params={"color": "#336699"})
    m = pg.session.model
    assert _colors(m, "crate")[0] == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert _wears(m, "dog/trunk", _hex("#336699"))
    saved = pg.store.load()
    assert saved.objects[0].color == "#0f0"
    assert saved.robots[0].params == {"color": "#336699"}


def test_robots_built_from_parts_take_hex_colors(pg):
    build = {
        "color": "#336699",
        "base": {"shape": "box", "size": [0.3, 0.14, 0.06]},
        "parts": [
            {"name": "left_wheel", "at": [0, 0.09, -0.02], "shape": "wheel", "size": [0.1, 0.03], "mirror": True},
            {"name": "mast", "at": "top", "joint": "fixed", "shape": "box", "size": [0.03, 0.03, 0.1]},
        ],
    }
    pg.add_robot("cart", build=build)
    pg.update_part("cart", "mast", {"color": "#C0FFEE"})
    m = pg.session.model
    assert _wears(m, "cart/base", _hex("#336699"))
    assert _wears(m, "cart/mast", _hex("#c0ffee"))
    assert _wears(m, "cart/right_wheel", _hex("#336699"))  # the rims wear the robot's color, mirror included
    saved = pg.store.load().robots[0].build
    assert saved["color"] == "#336699" and saved["parts"][1]["color"] == "#C0FFEE"


def test_a_color_that_does_not_parse_changes_nothing_and_reads_plainly(pg):
    pg.add_object("crate", pos=[0.6, 0], color="#9C6B3F")
    before = pg.store.file.read_text(encoding="utf-8")
    with pytest.raises(PlaygroundError, match="unknown color 'chartreuse'.*hex code") as raised:
        pg.add_object("ball", shape="sphere", color="chartreuse")
    # pydantic's own wording ("1 validation error for ObjectEntry ... errors.pydantic.dev") buries the fix.
    assert "pydantic" not in str(raised.value) and "validation error" not in str(raised.value)
    with pytest.raises(PlaygroundError, match="6 or 3 digits"):
        pg.update_object("crate", color="#9C6B3")
    with pytest.raises(ValueError, match="robot 'dog': unknown color '#nothex'"):
        pg.add_robot("dog", template="quadruped", params={"color": "#nothex"})
    assert pg.store.file.read_text(encoding="utf-8") == before
