"""Carried-prop selection follows saved/live support edges without editing either state."""

from __future__ import annotations

import copy
import math
from collections import Counter

import mujoco
import numpy as np
import pytest
from test_viewer_creation import pg  # noqa: F401
from test_viewer_object_copy import state, unchanged
from test_viewer_object_group import guard, prepare

from robotforge.design.layouts import LayoutError
from robotforge.design.world import ObjectEntry
from robotforge.playground import PlaygroundError
from robotforge.sim.session import fresh_data
from robotforge.viewer import object_selection
from robotforge.viewer.object_edit import _owners, _support_context, _SupportContext, _supported


def select(pg, name="base", **patch):  # noqa: F811
    return pg.edit_object_for_play({"action": "select_object_group", "name": name, **guard(pg), **patch})


def four(pg):  # noqa: F811
    pg._edit(
        lambda world: world.objects.extend(
            [
                ObjectEntry(name="base", shape="table", size=[1.2, 0.8, 0.5], pos=[-2, 0], fixed=True),
                ObjectEntry(name="box", size=[0.2, 0.2, 0.2], pos=[-2.3, 0, 0.6]),
                ObjectEntry(name="bin", shape="bin", size=[0.3, 0.3, 0.2], pos=[-1.7, 0, 0.5], fixed=True),
                ObjectEntry(name="top", size=[0.1, 0.1, 0.1], pos=[-2.3, 0, 0.75]),
            ]
        )
    )
    pg.session.paused = True


def native_arrays(native):
    arrays, detached = {}, {}
    for name in dir(native):
        value = getattr(native, name)
        if not isinstance(value, np.ndarray):
            continue
        # MuJoCo 3.15 arena getters pass absent pointers through InitPyArray.
        # The pinned pybind11 constructor allocates an unrelated array when given
        # nullptr. Compare its detached classification/shape/type, rather than
        # uninitialized allocation bytes. All native-backed data and all model
        # arrays remain exact, regardless of field name or solver configuration.
        if isinstance(native, mujoco.MjData) and value.flags.owndata and value.base is None:
            detached[name] = value.shape, value.dtype.str
        else:
            arrays[name] = value.copy()
    return arrays, detached


def integration_state(pg):  # noqa: F811
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    result = np.empty(mujoco.mj_stateSize(pg.session.model, specification))
    mujoco.mj_getState(pg.session.model, pg.session.data, result, specification)
    return result


def immutable_query(pg, monkeypatch, *, error=None, error_type=PlaygroundError, name="base", patch=None):  # noqa: F811
    from robotforge.design import store as store_module
    from robotforge.design import world as world_module

    before = state(pg)
    receipts = pg._play_group, pg._play_edit
    model, data = native_arrays(pg.session.model), native_arrays(pg.session.data)
    integration = integration_state(pg)
    real_forward = mujoco.mj_forward

    def private_forward(query_model, query_data):
        assert query_model is not pg.session.model and query_data is not pg.session.data
        return real_forward(query_model, query_data)

    def forbidden(*args, **kwargs):
        raise AssertionError("selection must not compile, load, save, or invoke move admission")

    with monkeypatch.context() as scoped:
        scoped.setattr(mujoco, "mj_forward", private_forward)
        scoped.setattr(store_module, "compile_world", forbidden)
        scoped.setattr(world_module, "compile_world", forbidden)
        scoped.setattr(pg, "_load_world", forbidden)
        scoped.setattr(pg.store, "apply", forbidden)
        scoped.setattr("robotforge.viewer.object_group._movable", forbidden)
        arguments = {"name": name, **(patch or {})}
        if error is None:
            result = select(pg, **arguments)
        else:
            with pytest.raises(error_type, match=error):
                select(pg, **arguments)
            result = None
    unchanged(pg, before)
    assert pg._play_group is receipts[0] and pg._play_edit is receipts[1]
    np.testing.assert_array_equal(integration_state(pg), integration)
    for native, (expected, detached) in ((pg.session.model, model), (pg.session.data, data)):
        current, current_detached = native_arrays(native)
        assert current.keys() == expected.keys()
        assert current_detached == detached
        for field, values in expected.items():
            np.testing.assert_array_equal(current[field], values, err_msg=field)
    return result


def test_recursive_table_box_fixed_bin_and_top_keep_every_state_and_receipt(pg, monkeypatch):  # noqa: F811
    four(pg)
    prepare(pg, ["base", "box", "bin", "top"])
    pg._play_edit = object()
    pg.session.data.qvel[:] = 0.4
    pg.session.data.time = 3.0
    out = immutable_query(pg, monkeypatch)
    assert out == {"name": "base", "names": ["base", "box", "bin", "top"], **guard(pg)}
    assert select(pg, "box")["names"] == ["box", "top"]
    assert select(pg, "bin")["names"] == ["bin"]


def test_result_uses_world_order_instead_of_traversal_order(pg):  # noqa: F811
    four(pg)
    pg._edit(lambda world: world.objects.reverse())
    pg.session.paused = True
    assert select(pg)["names"] == ["top", "bin", "box", "base"]


def test_real_diamond_visits_shared_carried_table_once_per_state(pg, monkeypatch):  # noqa: F811
    pg._edit(
        lambda world: world.objects.extend(
            [
                ObjectEntry(name="base", shape="table", size=[1.8, 1.0, 0.2], pos=[-2, 0]),
                ObjectEntry(name="left", size=[0.7, 0.8, 0.2], pos=[-2.45, 0, 0.3], fixed=True),
                ObjectEntry(name="right", size=[0.7, 0.8, 0.2], pos=[-1.55, 0, 0.3], fixed=True),
                ObjectEntry(name="bridge", shape="table", size=[1.2, 0.6, 0.4], pos=[-2, 0, 0.4]),
            ]
        )
    )
    pg.session.paused = True
    calls = Counter()
    real = _SupportContext.supported

    def counted(context, name):
        calls[name] += 1
        result = real(context, name)
        if name in {"left", "right"}:
            assert result == {"bridge"}
        return result

    monkeypatch.setattr(_SupportContext, "supported", counted)
    assert select(pg)["names"] == ["base", "left", "right", "bridge"]
    assert calls == Counter({name: 2 for name in ("base", "left", "right", "bridge")})


@pytest.mark.parametrize("count", [1, 16, 17])
def test_member_limit_is_atomic_and_stops_before_scanning_seventeenth_owner(pg, monkeypatch, count):  # noqa: F811
    entries = [ObjectEntry(name="base", shape="table", size=[1, 1, 0.5], pos=[-2, 0])]
    entries.extend(
        ObjectEntry(name=f"p{i:02d}", size=[0.1, 0.1, 0.1], pos=[-2, 0, 0.55 + i * 0.1], fixed=True)
        for i in range(count - 1)
    )
    pg._edit(lambda world: world.objects.extend(entries))
    pg.session.paused = True
    seen = []
    original = _SupportContext.supported

    def counted(context, name):
        seen.append(name)
        return original(context, name)

    monkeypatch.setattr(_SupportContext, "supported", counted)
    if count == 17:
        immutable_query(pg, monkeypatch, error="more than 16")
        assert len(seen) == 16 and "p15" not in seen
    else:
        assert immutable_query(pg, monkeypatch)["names"] == [entry.name for entry in entries]
        assert len(seen) == 2 * count


@pytest.mark.parametrize(
    "patch",
    [
        {"name": "missing"},
        {"name": "bas"},
        {"name": " base"},
        {"name": ""},
        {"name": True},
        {"version": True},
        {"version": 0},
        {"generation": -1},
        {"generation": "0"},
        {"world_sha256": "0" * 64},
        {"world_sha256": "G" * 64},
        {"names": ["base"]},
        {"mode": "move"},
    ],
)
def test_invalid_or_stale_request_cannot_return_partial_selection(pg, monkeypatch, patch):  # noqa: F811
    four(pg)
    immutable_query(pg, monkeypatch, error="invalid|changed|editable", patch=patch)


@pytest.mark.parametrize("reason", ["running", "tilted", "overhang", "floating", "custom_hook"])
def test_unavailable_or_incomplete_live_layout_is_read_only(pg, monkeypatch, reason):  # noqa: F811
    four(pg)
    if reason == "running":
        pg.session.paused = False
    elif reason == "custom_hook":
        pg.session.reset_hooks.append(lambda: None)
    else:
        adr = pg.session.model.jnt_qposadr[pg.session.model.joint("top").id]
        if reason == "tilted":
            pg.session.data.qpos[adr + 3 : adr + 7] = [math.cos(0.1), math.sin(0.1), 0, 0]
        elif reason == "overhang":
            pg.session.data.qpos[adr] += 0.08
        else:
            pg.session.data.qpos[adr + 2] += 0.3
        mujoco.mj_forward(pg.session.model, pg.session.data)
    immutable_query(
        pg,
        monkeypatch,
        error="pause|tilted|support|Stop",
        error_type=LayoutError if reason == "custom_hook" else PlaygroundError,
    )


def test_same_members_with_different_saved_live_edges_are_rejected(pg, monkeypatch):  # noqa: F811
    pg._edit(
        lambda world: world.objects.extend(
            [
                ObjectEntry(name="base", shape="table", size=[1.2, 0.8, 0.5], pos=[-2, 0]),
                ObjectEntry(name="left", size=[0.2, 0.2, 0.2], pos=[-2.3, 0, 0.6]),
                ObjectEntry(name="right", size=[0.2, 0.2, 0.2], pos=[-1.7, 0, 0.6]),
                ObjectEntry(name="top", size=[0.1, 0.1, 0.1], pos=[-2.3, 0, 0.75]),
            ]
        )
    )
    pg.session.paused = True
    adr = pg.session.model.jnt_qposadr[pg.session.model.joint("top").id]
    pg.session.data.qpos[adr] = -1.7
    mujoco.mj_forward(pg.session.model, pg.session.data)
    private = copy.copy(pg.session.model)
    world = pg.store.load()
    owners = _owners(private, world)
    saved = object_selection._reachable(_support_context(private, fresh_data(private), owners), world, "base")
    live = object_selection._reachable(_support_context(private, copy.copy(pg.session.data), owners), world, "base")
    assert set(saved) == set(live) and saved != live
    immutable_query(pg, monkeypatch, error="saved and live support layouts differ")


@pytest.mark.parametrize("owner", ["robot", "unsupported", "unknown", "cycle"])
def test_any_unselectable_carried_owner_or_cycle_rejects_whole_query(pg, monkeypatch, owner):  # noqa: F811
    four(pg)
    if owner == "robot":
        pg.add_robot("robot", template="rover", pos=[7, 0])
    elif owner == "unsupported":
        pg.add_object("unsupported", shape="drawer", pos=[7, 0])
    pg.session.paused = True
    real = _SupportContext.supported

    def injected(context, name):
        # Native positive-height geometry cannot form a support cycle. Inject at the
        # owner boundary, keeping all request, guard, copy and traversal code real.
        edges = set(real(context, name))
        if owner == "cycle" and name == "top":
            edges.add("base")
        elif owner != "cycle" and name == "base":
            edges.add(owner)
        return frozenset(edges)

    monkeypatch.setattr(_SupportContext, "supported", injected)
    immutable_query(pg, monkeypatch, error="editable rigid|cyclic")


def test_actual_robot_on_support_rejects_without_selecting_only_props(pg, monkeypatch):  # noqa: F811
    from robotforge.design.world import RobotEntry, spawn_height

    four(pg)
    pg.add_robot("robot", template="rover", pos=[-2, 0, 0.5 + spawn_height(RobotEntry(name="robot", template="rover"))])
    pg.session.paused = True
    immutable_query(pg, monkeypatch, error="robot.*editable rigid")


def test_execution_admission_and_late_guard_are_checked(pg, monkeypatch):  # noqa: F811
    from robotforge.sim.execution import ExecutionError, RunLimits

    four(pg)
    before = state(pg)
    with pg.session.execution.run("selection-test", RunLimits()), pytest.raises(ExecutionError):
        select(pg)
    unchanged(pg, before)
    checks = []
    real = object_selection._check

    def invalidate_before_return(*args):
        checks.append(1)
        if len(checks) == 2:
            pg.session.paused = False
        return real(*args)

    monkeypatch.setattr(object_selection, "_check", invalidate_before_return)
    immutable_query(pg, monkeypatch, error="pause")
    assert len(checks) == 2


def test_task_bound_selection_and_copy_remain_available_while_move_stays_guarded(pg):  # noqa: F811
    from robotforge.design.tasks.spec import TaskEntry

    four(pg)
    pg.add_robot("arm", template="arm", pos=[7, 0])
    pg._edit(lambda world: world.tasks.append(TaskEntry(name="pick", kind="pick_up", robot="arm", object="top")))
    pg.session.paused = True
    out = select(pg)
    assert out["names"] == ["base", "box", "bin", "top"]
    assert prepare(pg, out["names"])["mode"] == "copy"
    with pytest.raises(PlaygroundError, match="used by task"):
        prepare(pg, out["names"], mode="move")


@pytest.mark.parametrize("gap,expected", [(0.0, {"load"}), (0.0079, {"load"}), (0.0081, set())])
def test_reusable_context_retains_static_unmasked_eight_mm_detection(gap, expected):
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><geom name="base" type="box" size=".2 .2 .1" pos="0 0 .1" '
        'contype="0" conaffinity="1"/><geom name="load" type="box" size=".1 .1 .1" '
        f'pos="0 0 {0.3 + gap}" contype="0" conaffinity="1"/></worldbody></mujoco>'
    )
    data = fresh_data(model)
    owners = {0: "base", 1: "load"}
    context = _support_context(model, data, owners)
    assert data.ncon == 0
    assert context.supported("base") == expected
    assert context.supported("base") is context.supported("base")
    assert _supported(model, data, "base", owners) == expected
