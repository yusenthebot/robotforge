"""Putting a held peg in a chamfered hole it is not told the place of exactly: which controllers go in, and how hard."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

import mujoco
import pytest

from robotforge.control import TaskError, compare_methods, make_task
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

MENAGERIE = Path(__file__).parent / "fixtures" / "menagerie"
HOLE = [0.35, -0.2]


def _world(tmp_path, monkeypatch, peg_yaw: float = 0.0) -> Playground:
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab", "checker")
    pg.add_robot("arm", library="ur5e", mounts=[{"library": "robotiq_2f85"}])
    pg.add_object("socket1", shape="socket", pos=HOLE, size=[0.028, 0.12, 0.04])
    pg.add_object(
        "peg", shape="box", size=[0.024, 0.024, 0.08], pos=[0.45, 0.2], yaw=peg_yaw, mass=0.05, color="orange"
    )
    pg.run_fast(lambda: False, 0.5)
    return pg


def _move(pg: Playground, position: list[float], grip: str) -> None:
    arm = pg.behaviors.move_arm("arm", position, grip)
    pg.run_fast(lambda: arm.done, 20.0)


def _hold_the_peg_over_the_socket(pg: Playground) -> None:
    _move(pg, [0.45, 0.2, 0.2], "open")
    _move(pg, [0.45, 0.2, 0.07], "open")
    pg.behaviors.set_gripper("arm", "close")
    pg.run_fast(lambda: False, 1.2)
    _move(pg, [0.45, 0.2, 0.25], "close")
    _move(pg, [HOLE[0] + 0.02, HOLE[1] + 0.05, 0.25], "close")


def test_a_peg_dropped_over_the_socket_goes_in_within_the_funnel_and_stays_out_beyond_it(tmp_path, monkeypatch):
    depths = {}
    for offset in (0.0, 0.004, 0.03):
        os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp()
        pg = Playground(session=SimSession.from_xml("<mujoco/>"))
        pg.new_world("lab", "checker")
        pg.add_object("hole", shape="socket", pos=[0.0, 0.0], size=[0.028, 0.12, 0.04])
        pg.add_object("peg", shape="box", size=[0.024, 0.024, 0.08], pos=[offset, 0.0, 0.1], mass=0.05)
        pg.run_fast(lambda: False, 2.0)
        m, d = pg.session.model, pg.session.data
        depths[offset] = float(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "peg")][2])
    assert depths[0.0] == pytest.approx(0.048, abs=2e-3) and depths[0.004] == pytest.approx(0.048, abs=2e-3)
    assert depths[0.03] > 0.07  # on the block's top, not in the hole


def test_gentle_control_puts_the_peg_in_and_stiff_control_puts_it_in_too_hard(tmp_path, monkeypatch):
    pg = _world(tmp_path, monkeypatch)
    _hold_the_peg_over_the_socket(pg)
    model, data = pg.session.model, copy.copy(pg.session.data)
    params = {"hole": "socket1", "error": [0.006, 0.0], "force": 8.0, "limit": 40.0}
    cmp = compare_methods(
        "insert", ["ik", "cartesian_impedance", "osc"], 14.0, task_params=params, model=model, robot="arm", data=data
    )
    by = {r.method: r.report | {"ok": r.success} for r in cmp.results}
    for method in ("cartesian_impedance", "osc"):
        assert by[method]["ok"], (method, by[method])
        assert by[method]["depth_mm"] > 30 and by[method]["peak_down_n"] < 40 and by[method]["tilt_deg"] < 4
    assert by["ik"]["depth_mm"] > 30 and by["ik"]["damaged"] and not by["ik"]["ok"]  # in, with 70 N behind it


def test_a_peg_gripped_turned_to_the_hole_is_turned_before_it_goes_in(tmp_path, monkeypatch):
    pg = _world(tmp_path, monkeypatch, peg_yaw=30.0)
    _hold_the_peg_over_the_socket(pg)
    model, data = pg.session.model, copy.copy(pg.session.data)
    cmp = compare_methods("insert", ["osc"], 16.0, task_params={"hole": "socket1"}, model=model, robot="arm", data=data)
    result = cmp.results[0]
    assert result.success and result.report["depth_mm"] > 30, result.report


def test_insert_says_what_is_missing(tmp_path, monkeypatch):
    pg = _world(tmp_path, monkeypatch)
    model, data = pg.session.model, copy.copy(pg.session.data)
    mujoco.mj_forward(model, data)
    with pytest.raises(TaskError, match="holds nothing"):
        make_task("insert", model, "arm", {"hole": "socket1"}).begin(data)
    with pytest.raises(TaskError, match="no socket named"):
        make_task("insert", model, "arm", {"hole": "sockett"})
    assert json.dumps(make_task("insert", model, "arm", {"hole": "socket1", "peg": "peg"}).describe())
