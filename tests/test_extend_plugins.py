"""Plugins (robotforge.extend): loading files, reporting their errors, trusting a world's own, and plugin parts."""

from pathlib import Path

import mujoco
import pytest

from robotforge.design import props
from robotforge.design.store import WorldStore
from robotforge.design.world import World, compile_world
from robotforge.extend import REGISTRY, registry

SIGNPOST = """
from robotforge.extend import part, Box, Cylinder

@part("signpost", size="[height]", default_size=[1.2], fixed=True, color="brown", summary="a post with a board")
def signpost(size, rgba):
    h = size[0]
    return [Cylinder(0.05, h, at=[0, 0, h / 2]), Box([0.4, 0.03, 0.2], at=[0, 0, h - 0.1], color=1.3)]
"""


CLASH = """
from robotforge.extend import part, Box

@part("box", size="[a]", default_size=[1])
def f(size, rgba):
    return [Box([1, 1, 1])]
"""
BAD = """
from robotforge.extend import part, Box

@part("bad", size="[a]", default_size=[1])
def f(size, rgba):
    return [Box([1, 1])]
"""


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """A user plugin folder and a robotforge home of its own, with the registry as it was handed over."""
    user = tmp_path / "user_plugins"
    user.mkdir()
    monkeypatch.setenv("ROBOTFORGE_PLUGINS", str(user))
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    with registry.isolated():
        REGISTRY.user_loaded = False
        yield user


def write(folder: Path, name: str, code: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(code, encoding="utf-8", newline="\n")
    return path


def test_a_part_becomes_a_shape_like_any_other(folders, tmp_path):
    write(folders, "sign.py", SIGNPOST)
    REGISTRY.load_user()
    assert "signpost" in props.SHAPES
    world = World.model_validate(
        {
            "objects": [
                {"name": "a", "shape": "signpost", "pos": [1, 0]},
                {"name": "b", "shape": "Signpost", "size": [2.0], "pos": [2, 0], "yaw": 90},
            ]
        }
    )
    assert world.objects[0].fixed and world.objects[0].size == [1.2]  # the shape's defaults
    model = compile_world(world, tmp_path).model
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a")
    assert model.body_jntnum[body] == 0  # fixed: no free joint
    post = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "a")
    assert model.geom_pos[post][2] == pytest.approx(0.6)  # its post stands on the floor
    assert world.objects[1].shape == "signpost"  # "Signpost" is the shape it names


def test_a_shape_that_is_not_loaded_is_unknown_like_any_other_and_loaded_it_is_listed(folders):
    yaml_text = "name: w\nobjects:\n- {name: a, shape: signpost}\n"
    with pytest.raises(ValueError, match=r"shape 'signpost' is not one of: box, sphere") as unknown:
        World.from_yaml(yaml_text)
    assert "signpost," not in str(unknown.value) and "plugin" not in str(unknown.value)
    write(folders, "sign.py", SIGNPOST)
    REGISTRY.load_user()
    assert World.from_yaml(yaml_text).objects[0].shape == "signpost"
    with pytest.raises(ValueError, match=r"is not one of: box.*cloth.*signpost"):
        World.from_yaml(yaml_text.replace("signpost", "pyramid"))


def test_unload_takes_the_shape_away_and_keeps_the_built_ins(folders):
    path = write(folders, "sign.py", SIGNPOST)
    REGISTRY.load_user()
    before = props.PROPS
    path.unlink()
    REGISTRY.reload()
    assert "signpost" not in props.SHAPES
    assert "signpost" not in props.PROPS and "wall" in props.PROPS
    assert before != props.PROPS


def test_a_loose_part_has_the_mass_it_was_given(folders, tmp_path):
    code = SIGNPOST.replace("fixed=True", "fixed=False, mass=2.5")
    write(folders, "sign.py", code)
    REGISTRY.load_user()
    model = compile_world(World.model_validate({"objects": [{"name": "a", "shape": "signpost"}]}), tmp_path).model
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a")
    assert model.body_mass[body] == pytest.approx(2.5, rel=1e-6)


def test_a_file_that_fails_is_reported_with_its_line_and_the_others_still_load(folders):
    write(folders, "a_good.py", SIGNPOST)
    write(folders, "b_syntax.py", "x = (\n")
    write(folders, "c_runtime.py", "from robotforge.extend import part\n\nraise RuntimeError('boom')\n")
    write(folders, "d_clash.py", CLASH)
    REGISTRY.load_user()
    problems = REGISTRY.problems()
    assert "signpost" in props.SHAPES
    assert any("SyntaxError" in why and why.startswith("line ") for key, why in problems.items() if "b_syntax" in key)
    assert any("line 3" in why and "boom" in why for key, why in problems.items() if "c_runtime" in key)
    assert any(
        "already a shape" in why or "already has a shape" in why for key, why in problems.items() if "d_clash" in key
    )
    assert not any("a_good" in key for key in problems)


def test_a_failing_file_leaves_nothing_half_registered(folders):
    code = SIGNPOST + "\nraise RuntimeError('late')\n"
    write(folders, "late.py", code)
    REGISTRY.load_user()
    assert "signpost" not in props.SHAPES
    assert REGISTRY.of("part") == {}


def test_a_part_that_cannot_be_built_is_refused_when_registered(folders):
    code = BAD
    write(folders, "bad.py", code)
    REGISTRY.load_user()
    assert "bad" not in props.SHAPES
    assert "three numbers" in " ".join(REGISTRY.problems().values())


def test_two_files_cannot_add_the_same_name(folders):
    write(folders, "one.py", SIGNPOST)
    write(folders, "two.py", SIGNPOST)
    REGISTRY.load_user()
    assert "already added by" in " ".join(REGISTRY.problems().values())
    assert REGISTRY.of("part")["signpost"].plugin.endswith("one.py")


def test_reload_picks_up_an_edit(folders):
    path = write(folders, "sign.py", SIGNPOST)
    REGISTRY.load_user()
    assert props.SHAPES["signpost"].size == (1.2,)
    path.write_text(SIGNPOST.replace("[1.2]", "[2.0]"), encoding="utf-8")
    REGISTRY.reload()
    assert props.SHAPES["signpost"].size == (2.0,)


# ---- a world's own plugins ---------------------------------------------------------------------------------


def make_world(parent: Path) -> WorldStore:
    store = WorldStore.create("plugged", parent=parent)
    return store


def test_a_worlds_plugin_is_not_loaded_until_it_is_trusted(folders, tmp_path):
    store = make_world(tmp_path / "worlds")
    write(store.dir / "plugins", "sign.py", SIGNPOST)
    store.load()
    assert "signpost" not in props.SHAPES
    assert any("trust" in why for why in REGISTRY.problems().values())
    assert REGISTRY.trust(store.dir) == ["sign.py"]
    assert "signpost" in props.SHAPES
    assert not REGISTRY.problems()


def test_a_trusted_file_that_changes_is_not_trusted_any_more(folders, tmp_path):
    store = make_world(tmp_path / "worlds")
    path = write(store.dir / "plugins", "sign.py", SIGNPOST)
    REGISTRY.trust(store.dir)
    assert "signpost" in props.SHAPES
    path.write_text(SIGNPOST + "\n# someone changed this\n", encoding="utf-8")
    store.load()
    assert "signpost" not in props.SHAPES
    assert any("trust" in why for why in REGISTRY.problems().values())


def test_a_copied_world_is_not_trusted(folders, tmp_path):
    store = make_world(tmp_path / "worlds")
    write(store.dir / "plugins", "sign.py", SIGNPOST)
    REGISTRY.trust(store.dir)
    copy = tmp_path / "downloaded" / "plugged"
    (copy / "plugins").mkdir(parents=True)
    (copy / "plugins" / "sign.py").write_text(SIGNPOST, encoding="utf-8")
    (copy / "world.yaml").write_text(store.file.read_text(encoding="utf-8"), encoding="utf-8")
    REGISTRY.reload()  # forgets the original's trust-loaded file; the copy was never trusted
    for folder in list(REGISTRY.worlds):
        REGISTRY._forget_folder(folder)
    WorldStore(copy).load()
    assert "signpost" not in props.SHAPES
    assert REGISTRY.untrusted


def test_deleting_a_worlds_plugin_folder_forgets_its_shapes(folders, tmp_path):
    store = make_world(tmp_path / "worlds")
    path = write(store.dir / "plugins", "sign.py", SIGNPOST)
    REGISTRY.trust(store.dir)
    path.unlink()
    store.load()
    assert "signpost" not in props.SHAPES


def test_a_saved_world_with_a_plugin_part_reopens_when_the_plugin_is_there(folders, tmp_path):
    write(folders, "sign.py", SIGNPOST)
    REGISTRY.load_user()
    store = make_world(tmp_path / "worlds")
    sign = World.model_validate({"objects": [{"name": "s", "shape": "signpost"}]}).objects[0]
    store.apply(lambda w: w.objects.append(sign))
    again = WorldStore(store.dir).load()
    assert again.objects[0].shape == "signpost"
    assert "signpost" in store.file.read_text(encoding="utf-8")
