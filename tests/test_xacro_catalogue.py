"""xacro constructs that real ROS packages use and the built-in reader first could not read (found by adding the
catalogue's Toyota HSR-C and OpenArm for real)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from importer_fixtures import write

from robotforge.design.importing import xacro

HEAD = '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="r">'


def _expand(tmp_path, body: str) -> ET.Element:
    path = write(tmp_path / "r.xacro", f"{HEAD}{body}</robot>")
    return ET.fromstring(xacro.expand(path).text)


def test_a_block_handed_on_to_another_macro_means_what_it_meant_where_it_was_written(tmp_path):
    """The HSR's macros pass their own origin block to the macro they call: `<xacro:insert_block name="origin"/>` inside
    a call to a macro whose parameter is also called origin. Read in the callee's scope it inserted itself for ever."""
    robot = _expand(
        tmp_path,
        """
        <xacro:macro name="inner" params="name *origin">
          <joint name="${name}" type="fixed"><xacro:insert_block name="origin"/></joint>
        </xacro:macro>
        <xacro:macro name="outer" params="name *origin">
          <xacro:inner name="${name}_inner"><xacro:insert_block name="origin"/></xacro:inner>
        </xacro:macro>
        <xacro:outer name="arm"><origin xyz="0 0 1"/></xacro:outer>
        """,
    )
    joint = robot.find("joint")
    assert joint is not None and joint.get("name") == "arm_inner"
    origin = joint.find("origin")
    assert origin is not None and origin.get("xyz") == "0 0 1"


def test_formulas_with_string_methods_comprehensions_and_dicts(tmp_path):
    """OpenArm's: `prefix.find('right_') != -1` and a list of the components a preset enables, built in a loop."""
    robot = _expand(
        tmp_path,
        """
        <xacro:property name="prefix" value="right_arm_"/>
        <xacro:property name="components" value="${['arm', 'hand', 'head']}"/>
        <xacro:property name="enabled" value="${['arm', 'head']}"/>
        <joint name="${'R' if prefix.find('right_') != -1 else 'L'}" type="fixed"/>
        <link name="${','.join([c.upper() for c in components if c in enabled and c != 'head'])}"/>
        """,
    )
    assert robot.find("joint").get("name") == "R"  # type: ignore[union-attr]
    assert robot.find("link").get("name") == "ARM"  # type: ignore[union-attr]


def test_nested_loops_are_bounded_by_their_passes_not_only_their_length(tmp_path):
    """The files come from third-party repositories: [[a for a in X] for b in X] over 100000 items is 1e10 passes."""
    import pytest

    from robotforge.design.importing.xacro import XacroError

    with pytest.raises(XacroError, match="loops too long"):
        _expand(
            tmp_path,
            """
            <xacro:property name="items" value="${[0] * 50000}"/>
            <link name="${len([[a for a in items] for b in items])}"/>
            """,
        )
