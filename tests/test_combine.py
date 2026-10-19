"""Mount anything on anything (combine.py, mount_checks.py): one call, an adapter plate, checks once it is on, quick
tool changes and saved combinations.

Offline against the Menagerie stand-ins in tests/fixtures/menagerie, like test_mounting.py.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.design import combine, mount_checks, mounting
from robotforge.mcp_server import build_server
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import SimSession

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"
BIG_BASE = {"length": 0.7, "width": 0.5, "wheel_radius": 0.1, "motor": "jga25"}


@pytest.fixture(autouse=True)
def offline_menagerie(monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab", "checker")
    return playground


@pytest.fixture
def cart(pg):
    pg.add_robot("cart", template="rover", params=BIG_BASE)
    return pg


def _body(pg: Playground, name: str) -> int:
    return mujoco.mj_name2id(pg.session.model, mujoco.mjtObj.mjOBJ_BODY, name)


def _entry(pg: Playground, robot: str):
    return next(r for r in pg.current_world().robots if r.name == robot)


# ---- what goes on and where ----------------------------------------------------------------------------------------


def test_words_say_what_is_mounted():
    assert combine.classify("lidar") == ("sensor", {"type": "lidar"})
    assert combine.classify("depth camera") == ("sensor", {"type": "depth_camera"})
    assert combine.classify("suction")[1] == {"template": "suction_cup"}
    assert combine.classify("Robotiq 2F-85") == ("tool", {"library": "robotiq_2f85"})
    assert combine.classify("ur5e") == ("robot", {"library": "ur5e"})
    assert combine.classify("arm") == ("robot", {"template": "arm"})
    with pytest.raises(ValueError, match="did you mean"):
        combine.classify("ur5")


def test_an_arm_goes_on_a_wheeled_base_on_an_adapter_plate_and_is_checked(cart):
    out = cart.mount("ur5e", "cart")
    assert out["mounted"] == "cart/arm"
    plate = _body(cart, "cart/arm/adapter_plate")
    assert plate >= 0 and cart.session.model.body_mass[plate] > 0.02  # a real aluminium plate, not a drawing
    checks = out["checks"]
    assert checks["carries_kg"]["arm"] > 1.0
    assert checks["stands"]["margin_m"] > 0 and checks["stands"]["tilt_deg"] < 5
    (arm,) = checks["arms"]
    assert arm["arm"] == "cart/arm" and arm["reach_m"] > 0.3
    assert {"self_collision", "holds", "tips_robot", "reaches_floor"} <= set(arm)
    mount = _entry(cart, "cart").mounts[0]
    assert mount.library == "ur5e" and mount.adapter is True


def test_a_base_too_small_for_its_arm_is_refused_and_keeps_its_world(pg):
    pg.add_robot("cart", template="rover")
    with pytest.raises(PlaygroundError, match="does not stand|would tip"):
        pg.mount("ur5e", "cart")
    assert _entry(pg, "cart").mounts == []


def test_without_an_adapter_there_is_no_plate(cart):
    cart.mount("ur5e", "cart", adapter=False)
    assert _body(cart, "cart/arm/adapter_plate") < 0
    assert _body(cart, "cart/arm/base") >= 0


def test_worlds_saved_before_mounts_had_plates_save_as_before(cart):
    cart.add_robot("dog", library="go2", mounts=[{"library": "piper"}], pos=[2.0, 0.0])
    text = cart.store.file.read_text(encoding="utf-8")
    assert "adapter" not in text and "design" not in text.split("robots:", 1)[1]
    assert _body(cart, "dog/arm/adapter_plate") < 0


# ---- end effectors: a quick change ---------------------------------------------------------------------------------


def test_an_end_effector_goes_on_the_arms_flange_and_a_second_one_swaps_it(cart):
    cart.mount("ur5e", "cart")
    first = cart.mount("robotiq_2f85", "cart/arm")
    assert first["mounted"] == "cart/gripper"
    model = cart.session.model
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart/arm/gripper")
    assert grip >= 0 and model.actuator_trntype[grip] != mujoco.mjtTrn.mjTRN_BODY  # the fingers
    swapped = cart.mount("suction", "cart/arm")
    assert swapped["swapped"] == "cart/gripper" and swapped["mounted"] == "cart/suction_cup"
    assert swapped["took_off"] == "Robotiq 2F-85"
    assert [m.template or m.library for m in _entry(cart, "cart").mounts] == ["ur5e", "suction_cup"]
    model = cart.session.model
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart/arm/gripper")
    assert model.actuator_trntype[grip] == mujoco.mjtTrn.mjTRN_BODY  # an adhesion pad now
    assert _body(cart, "cart/gripper/base") < 0  # the Robotiq is gone


def test_a_cup_on_a_mounted_arm_with_fingers_goes_in_front_of_them(cart):
    cart.mount("piper", "cart")
    out = cart.mount("magnet", "cart/arm")
    assert _entry(cart, "cart").end_effector == "magnet"
    model = cart.session.model
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart/arm/gripper")
    assert model.actuator_trntype[grip] == mujoco.mjtTrn.mjTRN_BODY
    assert out["checks"]["arms"][0]["arm"] == "cart/arm"


def test_a_magnet_on_a_template_arm_riding_a_template_dog_goes_in_front_of_its_fingers(pg):
    pg.add_robot("dog", template="quadruped")
    pg.mount("arm", "dog")
    pg.mount("magnet", "dog/arm")
    assert _entry(pg, "dog").end_effector == "magnet"
    model = pg.session.model
    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "dog/arm/gripper")
    assert model.actuator_trntype[grip] == mujoco.mjtTrn.mjTRN_BODY


def test_a_plate_on_a_flange_is_a_flange_adapter_not_a_dinner_plate(cart):
    cart.mount("ur5e", "cart")
    cart.mount("robotiq_2f85", "cart/arm")
    model = cart.session.model
    plate = _body(cart, "cart/gripper/adapter_plate")
    plates = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == plate]
    radius = max(float(model.geom_size[g][0]) for g in plates if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER)
    assert radius <= mounting.FLANGE_PLATE_RADIUS + 1e-9


def test_a_cup_given_to_a_robot_whose_arm_is_mounted_works_through_update_robot(cart):
    cart.update_robot("cart", mounts=[{"library": "piper"}], end_effector="suction")
    grip = mujoco.mj_name2id(cart.session.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart/arm/gripper")
    assert grip >= 0


def test_an_end_effector_needs_an_arm_with_a_flange(cart):
    with pytest.raises(PlaygroundError, match="flange"):
        cart.mount("robotiq_2f85", "cart")


# ---- sensors -------------------------------------------------------------------------------------------------------


def test_a_lidar_goes_on_an_arms_wrist(cart):
    cart.mount("ur5e", "cart")
    out = cart.mount("lidar", "cart/arm")
    assert out["sensor"] == "cart/lidar" and out["on"] == "arm/wrist_3_link"
    assert _entry(cart, "cart").sensors[0].on == "arm/wrist_3_link"


def test_a_sensor_goes_on_a_body_named(cart):
    out = cart.mount("camera", "cart", at="chassis", name="deck_cam")
    assert out["sensor"] == "cart/deck_cam"


# ---- mounting on mounts, taking off --------------------------------------------------------------------------------


def test_mounting_on_a_mount_that_is_not_there_says_what_is(cart):
    cart.mount("ur5e", "cart")
    with pytest.raises(PlaygroundError, match="cart/arm"):
        cart.mount("robotiq_2f85", "cart/arms2")


def test_remove_takes_an_arm_off_with_what_was_on_it(cart):
    cart.mount("ur5e", "cart")
    cart.mount("robotiq_2f85", "cart/arm")
    cart.mount("lidar", "cart/arm")
    out = cart.remove("cart/arm")
    assert sorted(out["removed"]) == ["cart/arm", "cart/gripper", "cart/lidar"]
    entry = _entry(cart, "cart")
    assert entry.mounts == [] and entry.sensors == []


# ---- saved combinations ------------------------------------------------------------------------------------------


def test_a_combination_saves_as_a_design_that_adds_and_mounts_again(cart):
    cart.mount("ur5e", "cart")
    cart.mount("robotiq_2f85", "cart/arm", save_as="picker_cart")
    assert "picker_cart" in combine.designs()
    cart.add_robot("cart2", design="picker_cart", pos=[2.5, 0.0])
    again = _entry(cart, "cart2")
    assert again.template == "rover" and [m.library for m in again.mounts] == ["ur5e", "robotiq_2f85"]
    assert _body(cart, "cart2/arm/adapter_plate") >= 0


def test_a_saved_arm_with_its_gripper_is_mounted_whole_on_another_robot(pg):
    pg.add_robot("arm", library="ur5e", mounts=[{"library": "robotiq_2f85"}])
    pg.mount(None, "arm", save_as="ur5e_robotiq")
    pg.remove("arm")
    pg.add_robot("cart", template="rover", params=BIG_BASE)
    out = pg.mount("ur5e_robotiq", "cart")
    assert out["mounted"] == "cart/ur5e_robotiq"
    model = pg.session.model
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cart/ur5e_robotiq/tcp") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cart/ur5e_robotiq/gripper") >= 0
    assert out["checks"]["arms"][0]["arm"] == "cart/ur5e_robotiq"


def test_a_design_name_taken_by_the_library_is_refused(cart):
    with pytest.raises(PlaygroundError, match="taken"):
        cart.mount(None, "cart", save_as="go2")


def test_a_refused_save_name_leaves_the_world_as_it_was(cart):
    # The mount used to go on first and the name was judged after: the refusal read as "nothing happened".
    with pytest.raises(PlaygroundError, match="taken"):
        cart.mount("ur5e", "cart", save_as="a1")
    assert not _entry(cart, "cart").mounts
    assert cart.mount("ur5e", "cart", save_as="cart_arm")["mounted"] == "cart/arm"


def test_a_design_is_not_overwritten_unless_asked(cart):
    cart.mount("ur5e", "cart", save_as="cart_arm")
    first = (combine.designs_home() / "cart_arm.yaml").read_text(encoding="utf-8")
    cart.add_robot("cart2", template="rover", params=BIG_BASE, pos=[3.0, 0.0])
    with pytest.raises(PlaygroundError, match="overwrite"):
        cart.mount("panda", "cart2", save_as="cart_arm")
    assert not _entry(cart, "cart2").mounts
    assert (combine.designs_home() / "cart_arm.yaml").read_text(encoding="utf-8") == first
    cart.mount("panda", "cart2", save_as="cart_arm", overwrite=True)
    assert "panda" in (combine.designs_home() / "cart_arm.yaml").read_text(encoding="utf-8")


def test_a_design_cannot_be_saved_under_the_name_of_one_it_carries(cart):
    cart.mount("ur5e", "cart", save_as="cart_arm")
    cart.add_robot("cart2", template="rover", params=BIG_BASE, pos=[3.0, 0.0])
    before = (combine.designs_home() / "cart_arm.yaml").read_text(encoding="utf-8")
    with pytest.raises(PlaygroundError, match="carries"):
        cart.mount("cart_arm", "cart2", save_as="cart_arm", overwrite=True)
    assert not _entry(cart, "cart2").mounts  # refused before anything went on
    cart.mount("cart_arm", "cart2")
    with pytest.raises(PlaygroundError, match="carries"):
        cart.mount(None, "cart2", save_as="cart_arm", overwrite=True)  # and when saving what is already on
    assert (combine.designs_home() / "cart_arm.yaml").read_text(encoding="utf-8") == before


def test_a_design_that_mounts_itself_on_disk_is_named_not_recursed(cart):
    home = combine.designs_home()
    home.mkdir(parents=True, exist_ok=True)
    text = "design: loop\nrobot:\n  template: rover\n  mounts:\n  - name: m\n    design: loop\n"
    (home / "loop.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="loop -> loop"):
        combine.build_design("loop")
    with pytest.raises(PlaygroundError, match="loop"):
        cart.mount("loop", "cart")


@pytest.mark.parametrize("bad", [[float("nan"), 0.0, 0.0], [1e9, 0.0, 0.0], [0.0, 0.0, -50.0]])
def test_a_mount_shift_that_is_not_a_place_is_refused_before_compiling(cart, bad):
    with pytest.raises(PlaygroundError, match="shift"):
        cart.mount("ur5e", "cart", shift=bad)
    assert not _entry(cart, "cart").mounts


def test_a_mount_rotation_that_is_not_an_angle_is_refused(cart):
    with pytest.raises(PlaygroundError, match="rotate"):
        cart.mount("ur5e", "cart", rotate=[0.0, float("inf"), 0.0])


def test_the_shadow_hand_is_refused_saying_why_and_what_to_use(cart):
    with pytest.raises(ValueError, match="forearm.*leap_hand"):
        combine.classify("shadow_hand")
    cart.mount("ur5e", "cart")
    with pytest.raises(PlaygroundError, match="forearm"):
        cart.mount("shadow_hand", "cart/arm")


# ---- the checks ------------------------------------------------------------------------------------------------------


def test_support_polygon_margin_is_inside_positive_and_outside_negative():
    square = mount_checks._hull(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], float))
    assert len(square) == 4
    assert mount_checks._margin(square, np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert mount_checks._margin(square, np.array([1.5, 0.5])) == pytest.approx(-0.5)


def test_checks_report_a_fixed_arm_without_a_stance(pg):
    pg.add_robot("panda", library="panda")
    found = mount_checks.check(pg.session.model, "panda", "arm")
    assert "stands" not in found.report and found.refusal is None
    assert found.report["arms"][0]["arm"] == "panda"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_mount_puts_an_arm_on_a_base(cart):
    server = build_server(cart, find_free_port(8812))
    async with Client(server) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert "mount" in tools
        result = await client.call_tool("mount", {"what": "ur5e", "on": "cart"})
        out = json.loads(result.content[0].text)
        assert out["mounted"] == "cart/arm" and out["checks"]["arms"]
