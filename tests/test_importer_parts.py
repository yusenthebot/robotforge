"""Unit tests for the import doctor's building blocks: URI resolution, mesh repair, Collada, xacro, reports."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import trimesh
from importer_fixtures import (
    ascii_stl,
    box_stl,
    dae_text,
    flat_quad_stl,
    multi_object_obj,
    write,
)

from robotforge.design.importing import meshio, xacro
from robotforge.design.importing.dae import DaeError, read_dae
from robotforge.design.importing.report import DoctorReport
from robotforge.design.importing.resolve import MeshResolver


def _loads_in_mujoco(path) -> mujoco.MjModel:
    xml = (
        f'<mujoco><asset><mesh name="m" file="{path}"/></asset>'
        '<worldbody><geom type="mesh" mesh="m"/></worldbody></mujoco>'
    )
    return mujoco.MjModel.from_xml_string(xml)


# ---- mesh URI resolution ------------------------------------------------------------------------


def test_package_uri_found_by_folder_name(tmp_path):
    mesh = box_stl(tmp_path / "ws" / "my_pkg" / "meshes" / "a.stl")
    urdf = write(tmp_path / "ws" / "my_pkg" / "urdf" / "r.urdf", "<robot/>")
    res = MeshResolver(urdf).resolve("package://my_pkg/meshes/a.stl")
    assert res.path == mesh.resolve()
    assert not res.guessed


def test_package_uri_found_by_package_xml_when_folder_is_named_differently(tmp_path):
    root = tmp_path / "repo-master"
    write(root / "package.xml", "<package><name>real_name</name></package>")
    mesh = box_stl(root / "meshes" / "a.stl")
    urdf = write(root / "urdf" / "r.urdf", "<robot/>")
    res = MeshResolver(urdf).resolve("package://real_name/meshes/a.stl")
    assert res.path == mesh.resolve()
    assert not res.guessed


def test_package_uri_found_in_a_sibling_package(tmp_path):
    mesh = box_stl(tmp_path / "src" / "other_pkg" / "meshes" / "a.stl")
    urdf = write(tmp_path / "src" / "my_pkg" / "urdf" / "r.urdf", "<robot/>")
    assert MeshResolver(urdf).resolve("package://other_pkg/meshes/a.stl").path == mesh.resolve()


def test_missing_package_falls_back_to_a_guess_and_says_so(tmp_path):
    mesh = box_stl(tmp_path / "repo" / "meshes" / "base.stl")
    urdf = write(tmp_path / "repo" / "urdf" / "r.urdf", "<robot/>")
    res = MeshResolver(urdf).resolve("package://ghost_pkg/meshes/base.stl")
    assert res.path == mesh.resolve()
    assert res.guessed and "ghost_pkg" in res.note


def test_unresolvable_mesh_reports_why(tmp_path):
    urdf = write(tmp_path / "r.urdf", "<robot/>")
    res = MeshResolver(urdf).resolve("package://nowhere/meshes/none.stl")
    assert res.path is None
    assert "nowhere" in res.note


def test_relative_absolute_and_file_uris(tmp_path):
    mesh = box_stl(tmp_path / "meshes" / "a b.stl")
    urdf = write(tmp_path / "urdf" / "r.urdf", "<robot/>")
    resolver = MeshResolver(urdf)
    assert resolver.resolve("../meshes/a b.stl").path == mesh.resolve()
    assert resolver.resolve(str(mesh)).path == mesh.resolve()
    assert resolver.resolve(f"file://{str(mesh).replace(' ', '%20')}").path == mesh.resolve()


def test_unsupported_scheme_and_non_mesh_files_are_refused(tmp_path):
    secret = write(tmp_path / "secret.txt", "password")
    urdf = write(tmp_path / "r.urdf", "<robot/>")
    resolver = MeshResolver(urdf)
    assert resolver.resolve("https://example.com/a.stl").path is None
    assert resolver.resolve(str(secret)).path is None  # only mesh formats are ever read


def test_package_paths_cannot_escape_the_package_folder(tmp_path):
    box_stl(tmp_path / "outside.stl")
    pkg = tmp_path / "ws" / "pkg"
    urdf = write(pkg / "urdf" / "r.urdf", "<robot/>")
    from robotforge.design.importing.resolve import _confined

    assert _confined(pkg, "../../../outside.stl") is None
    res = MeshResolver(urdf).resolve("package://pkg/../../../outside.stl")
    # The traversal itself is rejected; at most a by-name guess inside the workspace may be offered, and it says so.
    assert res.path is None or res.guessed


# ---- mesh repair --------------------------------------------------------------------------------


def test_binary_stl_is_used_as_it_is(tmp_path):
    src = box_stl(tmp_path / "a.stl")
    asset = meshio.prepare_mesh(src, tmp_path / "out")
    assert asset.merged == src.resolve()
    assert asset.notes == []
    assert asset.out_faces == 12


def test_ascii_stl_is_converted_to_binary(tmp_path):
    src = ascii_stl(tmp_path / "a.stl")
    asset = meshio.prepare_mesh(src, tmp_path / "out")
    assert asset.merged != src.resolve()
    assert any("ASCII" in n for n in asset.notes)
    _loads_in_mujoco(asset.merged)


def test_multi_object_obj_is_merged_so_mujoco_sees_everything(tmp_path):
    src = multi_object_obj(tmp_path / "two.obj")
    asset = meshio.prepare_mesh(src, tmp_path / "out")
    assert any("several objects" in n for n in asset.notes)
    assert _loads_in_mujoco(asset.merged).mesh_vertnum[0] == 8
    assert asset.out_faces == 8


def test_flat_sheet_is_given_thickness_so_mujoco_accepts_it(tmp_path):
    src = flat_quad_stl(tmp_path / "quad.stl")
    with pytest.raises(ValueError, match="coplanar"):
        _loads_in_mujoco(src)  # the problem the doctor exists to fix
    asset = meshio.prepare_mesh(src, tmp_path / "out")
    assert any("thickness" in n for n in asset.notes)
    _loads_in_mujoco(asset.merged)


def test_huge_meshes_are_simplified_below_the_mujoco_limit(tmp_path):
    big = trimesh.creation.icosphere(subdivisions=7)  # 327k triangles
    src = tmp_path / "big.stl"
    big.export(src)
    asset = meshio.prepare_mesh(src, tmp_path / "out")
    assert asset.faces > 300_000
    assert asset.out_faces <= meshio.VISUAL_FACE_CAP
    _loads_in_mujoco(asset.merged)


def test_nan_vertices_are_removed(tmp_path):
    verts = np.asarray(trimesh.creation.box((0.1, 0.1, 0.1)).vertices).copy()
    mesh = trimesh.creation.box((0.1, 0.1, 0.1))
    extra_v = np.vstack([verts, [[np.nan, 0, 0], [0, 0, 1], [1, 0, 0]]])
    extra_f = np.vstack([mesh.faces, [[8, 9, 10]]])
    bad = trimesh.Trimesh(extra_v, extra_f, process=False)
    src = tmp_path / "nan.stl"
    bad.export(src)
    asset = meshio.prepare_mesh(src, tmp_path / "out")
    assert np.isfinite(asset.mesh.vertices).all()
    _loads_in_mujoco(asset.merged)


def test_glb_ply_and_off_meshes_convert_to_stl(tmp_path):
    box = trimesh.creation.box((0.2, 0.1, 0.05))
    for ext in ("glb", "ply", "off"):
        src = tmp_path / f"m.{ext}"
        box.export(src)
        asset = meshio.prepare_mesh(src, tmp_path / "out")
        assert any(f"converted from {ext.upper()}" in n for n in asset.notes)
        assert _loads_in_mujoco(asset.merged).nmesh == 1


def test_unreadable_mesh_raises_a_clear_error(tmp_path):
    bad = write(tmp_path / "broken.stl", "this is not a mesh")
    with pytest.raises(meshio.MeshError):
        meshio.prepare_mesh(bad, tmp_path / "out")


def test_solid_properties_match_a_known_box():
    box = trimesh.creation.box((0.2, 0.4, 0.6))
    props = meshio.solid_properties(box, np.array([1.0, 1.0, 1.0]))
    assert props.volume == pytest.approx(0.048)
    assert props.inertia[0, 0] == pytest.approx(0.048 / 12 * (0.4**2 + 0.6**2))
    scaled = meshio.solid_properties(box, np.array([2.0, 2.0, 2.0]))
    assert scaled.volume == pytest.approx(0.048 * 8)


# ---- Collada --------------------------------------------------------------------------------------


def test_dae_reader_reads_geometry_color_and_unit(tmp_path):
    src = write(tmp_path / "cube.dae", dae_text(unit=0.01, size=(10.0, 20.0, 30.0), color=(0.2, 0.4, 0.6, 1.0)))
    (part,) = read_dae(src)
    assert part.rgba == pytest.approx((0.2, 0.4, 0.6, 1.0))
    assert np.ptp(part.vertices, axis=0) == pytest.approx([0.1, 0.2, 0.3])  # centimeters became meters
    assert len(part.faces) == 12


def test_dae_y_up_is_rotated_to_z_up(tmp_path):
    src = write(tmp_path / "cube.dae", dae_text(up="Y_UP", size=(1.0, 2.0, 3.0)))
    (part,) = read_dae(src)
    assert np.ptp(part.vertices, axis=0) == pytest.approx([1.0, 3.0, 2.0])  # old y (2) is now z... after rotation: z=2


def test_dae_without_scene_graph_still_loads(tmp_path):
    src = write(tmp_path / "cube.dae", dae_text(nodes=False))
    assert len(read_dae(src)[0].faces) == 12


def test_dae_garbage_is_rejected(tmp_path):
    with pytest.raises(DaeError):
        read_dae(write(tmp_path / "bad.dae", "<html/>"))
    with pytest.raises(DaeError):
        read_dae(write(tmp_path / "bad2.dae", "not xml at all"))


def test_dae_mesh_converts_without_pycollada_and_keeps_its_color(tmp_path):
    src = write(tmp_path / "part.dae", dae_text(color=(0.0, 1.0, 0.0, 1.0)))
    asset = meshio.prepare_mesh(src, tmp_path / "out")
    assert asset.pieces[0].rgba == pytest.approx((0.0, 1.0, 0.0, 1.0))
    assert any("converted from DAE" in n for n in asset.notes)
    _loads_in_mujoco(asset.merged)


# ---- xacro ----------------------------------------------------------------------------------------

XACRO_NS = 'xmlns:xacro="http://www.ros.org/wiki/xacro"'


def _expand(tmp_path, body: str, args=None, name="r.xacro", extra_files=None) -> str:
    for rel, text in (extra_files or {}).items():
        write(tmp_path / rel, text)
    src = write(tmp_path / name, f'<robot name="r" {XACRO_NS}>{body}</robot>')
    text, _ = xacro.expand(src, args)
    return text


def test_xacro_properties_math_and_macros(tmp_path):
    body = """
    <xacro:property name="r" value="0.05"/>
    <xacro:property name="d" value="${2*r}"/>
    <xacro:macro name="wheel" params="prefix y *origin side:=left">
      <link name="${prefix}_wheel_${side}"><visual><xacro:insert_block name="origin"/>
        <geometry><cylinder radius="${r}" length="${d + 0.01}"/></geometry></visual></link>
    </xacro:macro>
    <xacro:wheel prefix="a" y="1" side="right"><origin xyz="0 ${r} 0"/></xacro:wheel>
    <xacro:wheel prefix="b" y="2"><origin xyz="1 2 3"/></xacro:wheel>
    """
    out = _expand(tmp_path, body)
    assert 'name="a_wheel_right"' in out and 'name="b_wheel_left"' in out
    assert 'radius="0.05"' in out and 'length="0.11"' in out
    assert 'xyz="0 0.05 0"' in out and 'xyz="1 2 3"' in out


def test_xacro_args_include_and_conditionals(tmp_path):
    body = """
    <xacro:arg name="mode" default="simple"/>
    <xacro:include filename="$(dirname)/parts.xacro"/>
    <xacro:if value="${'$(arg mode)' == 'fancy'}"><link name="fancy"/></xacro:if>
    <xacro:unless value="${'$(arg mode)' == 'fancy'}"><link name="plain"/></xacro:unless>
    """
    parts = f'<robot {XACRO_NS}><link name="from_include"/></robot>'
    plain = _expand(tmp_path, body, extra_files={"parts.xacro": parts})
    assert "from_include" in plain and 'name="plain"' in plain and "fancy" not in plain
    fancy = _expand(tmp_path, body, args={"mode": "fancy"}, extra_files={"parts.xacro": parts})
    assert 'name="fancy"' in fancy and "plain" not in fancy


def test_xacro_find_resolves_packages_next_to_the_file(tmp_path):
    write(tmp_path / "pkg" / "package.xml", "<package><name>pkg</name></package>")
    write(tmp_path / "pkg" / "inc.xacro", f'<robot {XACRO_NS}><link name="inc"/></robot>')
    src = write(
        tmp_path / "pkg" / "urdf" / "r.xacro",
        f'<robot name="r" {XACRO_NS}><xacro:include filename="$(find pkg)/inc.xacro"/></robot>',
    )
    text, files = xacro.expand(src)
    assert 'name="inc"' in text and any(f.name == "inc.xacro" for f in files)


def test_xacro_python_helpers_are_refused_with_an_explanation(tmp_path):
    with pytest.raises(xacro.XacroError, match="Python helper"):
        _expand(tmp_path, '<link name="${made_up_helper(1)}"/>')
    with pytest.raises(xacro.XacroError, match="Python helper|cannot evaluate"):
        _expand(tmp_path, "<link name=\"${open('/etc/passwd').read()}\"/>")
    with pytest.raises(xacro.XacroError, match="never defined"):
        _expand(tmp_path, '<link name="${missing}"/>')


def test_xacro_load_yaml_and_scope_parent(tmp_path):
    write(tmp_path / "params.yaml", "links:\n  arm:\n    mass: 2.5\n    names: [a, b]\n")
    body = """
    <xacro:macro name="load" params="file">
      <xacro:property name="data" value="${xacro.load_yaml(file)}" scope="parent"/>
    </xacro:macro>
    <xacro:load file="params.yaml"/>
    <link name="${data['links']['arm']['names'][1]}">
      <inertial><mass value="${data.links.arm.mass * 2}"/></inertial></link>
    """
    out = _expand(tmp_path, body)
    assert 'name="b"' in out and 'value="5.0"' in out


def test_xacro_params_with_quotes_and_spaces_and_booleans(tmp_path):
    body = """
    <xacro:macro name="m" params="path:='$(find nothing_here)/x y.yaml' empty:='' flag:=false">
      <link name="n"><xacro:if value="${flag}"><visual/></xacro:if>
        <xacro:unless value="${flag}"><collision/></xacro:unless></link>
    </xacro:macro>
    <xacro:m path="plain"/>
    """
    out = _expand(tmp_path, body)
    assert "<collision" in out and "<visual" not in out


def test_xacro_double_star_blocks_string_methods_and_numeric_strings(tmp_path):
    body = """
    <xacro:macro name="part" params="name xyz **extras">
      <xacro:property name="x" value="${xyz.split(' ')[0]}"/>
      <link name="${name}"><origin xyz="${x + 0.5} 0 0"/><xacro:insert_block name="extras"/></link>
    </xacro:macro>
    <xacro:part name="p" xyz="1 2 3"><visual/><collision/></xacro:part>
    """
    out = _expand(tmp_path, body)
    assert 'xyz="1.5 0 0"' in out  # "1" became a number, so the formula could add to it
    assert "<visual" in out and "<collision" in out


def test_xacro_yaml_degrees_tag_and_eager_properties(tmp_path):
    write(tmp_path / "limits.yaml", "lower: !degrees -90\nupper: !radians 1.5\n")
    body = """
    <xacro:property name="limits" value="${xacro.load_yaml('limits.yaml')}" lazy_eval="false"/>
    <joint name="j"><limit lower="${limits.lower}" upper="${limits.upper}"/></joint>
    """
    out = _expand(tmp_path, body)
    assert 'lower="-1.57079' in out and 'upper="1.5"' in out


def test_xacro_undeclared_namespace_and_bom_are_tolerated(tmp_path):
    text = '\ufeff<robot name="r"><xacro:property name="a" value="2"/><link name="l${a}"/></robot>'
    src = tmp_path / "r.xacro"
    src.write_text(text, encoding="utf-8")
    out, _ = xacro.expand(src)
    assert 'name="l2"' in out


def test_xacro_cannot_read_files_outside_its_project_or_environment_variables(tmp_path):
    outside = write(tmp_path / "secret" / "data.yaml", "key: topsecret\n")
    project = tmp_path / "proj"
    write(project / "package.xml", "<package><name>proj</name></package>")
    src = write(
        project / "r.xacro",
        f'<robot name="r" {XACRO_NS}><xacro:property name="d" value="${{xacro.load_yaml(\'{outside}\')}}"/>'
        '<link name="${d.key}"/></robot>',
    )
    with pytest.raises(xacro.XacroError, match="outside the project folder"):
        xacro.expand(src)
    include = write(project / "i.xacro", f'<robot {XACRO_NS}><xacro:include filename="{outside}"/></robot>')
    with pytest.raises(xacro.XacroError, match="outside the project folder"):
        xacro.expand(include)
    env = write(project / "e.xacro", f'<robot name="r" {XACRO_NS}><link name="$(env HOME)"/></robot>')
    with pytest.raises(xacro.XacroError, match="not allowed"):
        xacro.expand(env)
    optenv = write(project / "o.xacro", f'<robot name="r" {XACRO_NS}><link name="$(optenv HOME fallback)"/></robot>')
    assert 'name="fallback"' in xacro.expand(optenv)[0]


def test_xacro_expressions_are_sandboxed(tmp_path):
    for evil in ("__import__('os').system('echo hi')", "().__class__", "[x for x in range(3)]", "exec('1')"):
        with pytest.raises(xacro.XacroError):
            _expand(tmp_path, f'<link name="${{{evil}}}"/>')


def test_xacro_unknown_macro_argument_and_missing_include_are_clear(tmp_path):
    with pytest.raises(xacro.XacroError, match="has no parameter"):
        _expand(tmp_path, '<xacro:macro name="m" params="a"><link name="${a}"/></xacro:macro><xacro:m a="1" b="2"/>')
    with pytest.raises(xacro.XacroError, match="does not exist"):
        _expand(tmp_path, '<xacro:include filename="nope.xacro"/>')


def test_needs_expansion_only_looks_at_the_parts_we_read():
    plain = '<robot name="r"><link name="a"/><gazebo><plugin filename="$(find x)/lib.so"/></gazebo></robot>'
    assert not xacro.needs_expansion(plain)  # Gazebo blocks are dropped anyway
    assert xacro.needs_expansion(f'<robot {XACRO_NS}><xacro:property name="a" value="1"/></robot>')
    assert xacro.needs_expansion('<robot name="r"><link name="${ns}base"/></robot>')
    assert xacro.needs_expansion("<robot><xacro:foo/></robot>")  # undeclared prefix still counts


# ---- report ---------------------------------------------------------------------------------------


def test_report_roundtrips_and_shortens_long_item_lists():
    report = DoctorReport("x.urdf")
    report.fixed("meshes", "bad paths", "fixed them", [f"m{i}" for i in range(10)])
    report.unfixed("meshes", "gone", "skipped")
    report.info("base", "fixed base")
    short = report.to_dict()
    assert short["fixed"] == 1 and short["unfixed"] == 1
    assert short["findings"][0]["items"][-1] == "... and 4 more"
    again = DoctorReport.from_dict(report.to_dict(full=True))
    assert again.findings == report.findings
    assert "1 thing could not be fixed" in report.summary
    assert "NOT FIXED" in report.to_text()
    assert DoctorReport("y").summary == "No problems found."
