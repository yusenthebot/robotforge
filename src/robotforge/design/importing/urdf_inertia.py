"""Give every link that needs one a valid mass and inertia, derived from its shapes where the file's are bad."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import numpy as np

from robotforge.design.importing import inertia
from robotforge.design.importing.inertia import Inertial, Shape
from robotforge.design.importing.report import DoctorReport
from robotforge.design.importing.xmlnum import floats

if TYPE_CHECKING:
    from robotforge.design.importing.urdf import JointInfo


def _parse_inertial(link: ET.Element) -> Inertial | None:
    el = link.find("inertial")
    if el is None:
        return None
    mass_el = el.find("mass")
    mass = floats(mass_el.get("value"), 1)[0] if mass_el is not None else 0.0
    origin = el.find("origin")
    xyz = np.array(floats(origin.get("xyz") if origin is not None else None, 3))
    rot = inertia.rpy_matrix(*floats(origin.get("rpy") if origin is not None else None, 3))
    i = el.find("inertia")
    vals = {k: floats(i.get(k) if i is not None else None, 1)[0] for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
    tensor = np.array(
        [
            [vals["ixx"], vals["ixy"], vals["ixz"]],
            [vals["ixy"], vals["iyy"], vals["iyz"]],
            [vals["ixz"], vals["iyz"], vals["izz"]],
        ]
    )
    return Inertial(mass, xyz, rot @ tensor @ rot.T)


def _write_inertial(link: ET.Element, value: Inertial) -> None:
    for old in link.findall("inertial"):
        link.remove(old)
    el = ET.Element("inertial")
    ET.SubElement(el, "origin", {"xyz": " ".join(f"{v:.9g}" for v in value.com), "rpy": "0 0 0"})
    ET.SubElement(el, "mass", {"value": f"{value.mass:.9g}"})
    t = (value.tensor + value.tensor.T) / 2
    ET.SubElement(
        el,
        "inertia",
        {
            k: f"{v:.9g}"
            for k, v in zip(
                ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"),
                (t[0, 0], t[0, 1], t[0, 2], t[1, 1], t[1, 2], t[2, 2]),
                strict=True,
            )
        },
    )
    link.insert(0, el)


def fix_inertials(
    links: dict[str, ET.Element],
    joints: list[JointInfo],
    roots: list[str],
    link_shapes: dict[str, list[Shape]],
    report: DoctorReport,
) -> dict[str, float]:
    parent_joint = {j.child: j for j in joints}
    masses: dict[str, float] = {}
    missing, bad_mass, bad_inertia, nudged, implausible, defaulted = [], [], [], [], [], []
    for name, link in links.items():
        needs = name in roots or parent_joint[name].type != "fixed"
        shapes = link_shapes[name]
        given = _parse_inertial(link)
        result: Inertial | None = given

        if given is None:
            if needs:
                result = inertia.inertial_from_shapes(shapes)
                if result is not None and result.mass >= inertia.TINY_MASS:
                    missing.append(name)
                else:  # no shape, or only a marker-sized one: a placeholder is all the simulator needs
                    result = inertia.default_inertial()
                    if name not in roots:
                        defaulted.append(name)
        else:
            result = _check_given(given, shapes, bad_mass, bad_inertia, nudged, implausible, name, needs, defaulted)

        if result is not None:
            _write_inertial(link, result)
            masses[name] = result.mass
        else:
            for old in link.findall("inertial"):  # a rigid frame with a bogus inertial: let MuJoCo infer it
                link.remove(old)
            masses[name] = 0.0

    if missing:
        report.fixed(
            "inertia", "some links had no mass or inertia", "computed them from the link shapes (1000 kg/m^3)", missing
        )
    if bad_mass:
        report.fixed(
            "inertia",
            "some links had zero, negative or tiny mass",
            "computed mass and inertia from the link shapes",
            bad_mass,
        )
    if bad_inertia:
        report.fixed(
            "inertia",
            "some links had an unusable inertia (zero or not positive)",
            "rebuilt it from the link shapes and kept the mass",
            bad_inertia,
        )
    if nudged:
        report.fixed(
            "inertia",
            "some inertias were physically impossible (one axis heavier than the other two together)",
            "nudged them to the nearest valid values",
            nudged,
        )
    if implausible:
        report.fixed(
            "inertia",
            "some inertias do not match the size of their link (probably wrong units or placeholders)",
            "rebuilt them from the link shapes and kept the mass",
            implausible,
        )
    if defaulted:
        report.fixed(
            "inertia",
            "some moving links have no shape and no mass",
            f"gave them a tiny placeholder mass ({inertia.TINY_MASS} kg)",
            defaulted,
        )
    return masses


def _check_given(
    given: Inertial,
    shapes: list[Shape],
    bad_mass: list[str],
    bad_inertia: list[str],
    nudged: list[str],
    implausible: list[str],
    name: str,
    needs: bool,
    defaulted: list[str],
) -> Inertial | None:
    problem = inertia.inertia_problem(given.mass, given.tensor)
    if not problem:
        text = inertia.inertia_implausible(given.mass, given.tensor, inertia.shapes_radius(shapes, given.com))
        if text:
            fixed = inertia.scaled_to_mass(shapes, given.mass)
            if fixed is not None:
                implausible.append(f"{name}: {text}")
                return Inertial(given.mass, given.com, fixed.tensor)
        return given
    if "mass" in problem:
        derived = inertia.inertial_from_shapes(shapes)
        if derived is not None and derived.mass >= inertia.TINY_MASS:
            bad_mass.append(name)
            return derived
        if needs:
            defaulted.append(name)
            return inertia.default_inertial()
        return None
    if "impossible" in problem:
        nudged.append(name)
        return Inertial(given.mass, given.com, inertia.repair_triangle(given.tensor))
    fixed = inertia.scaled_to_mass(shapes, given.mass)
    if fixed is None:
        fixed = Inertial(given.mass, given.com, np.eye(3) * (given.mass * 1e-3))
    bad_inertia.append(name)
    return Inertial(given.mass, given.com, fixed.tensor)
