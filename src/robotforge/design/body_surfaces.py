"""Explicit finishes on direct body visuals, preserving colour and ray visibility."""

from __future__ import annotations

import hashlib
import json

import mujoco
import numpy as np

from robotforge.design.visuals import DEFAULT_RGBA, PRESETS
from robotforge.sensors import declare

FINISHES = {"plastic": "shell", "steel": "metal", "wood": "wood"}


def _sensor_prefixes(texts: list[tuple[str, str]]) -> tuple[str, ...]:
    prefixes = []
    for name, raw in texts:
        at = name.rfind(declare.TEXT_PREFIX)
        if at < 0:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("v") == declare.VERSION and isinstance(data.get("name"), str):
            owner = (data["name"] + "_sensor").replace("/", "_")
            prefixes.append(f"{name[:at]}rf_{owner}_")
    return tuple(prefixes)


def _opaque(rgba, material_rgba) -> bool:
    effective = material_rgba if material_rgba is not None and np.allclose(rgba, DEFAULT_RGBA) else rgba
    return float(effective[3]) == 1.0


def apply(spec: mujoco.MjSpec, overrides: dict[str, str]) -> None:
    """No-op for old worlds. Never mutate a shared source material or sensor housing."""
    if not overrides:
        return
    materials = {m.name: m for m in spec.materials}
    sensors = _sensor_prefixes([(t.name, t.data) for t in spec.texts])
    bodies = {b.name: b for b in spec.bodies if b.name and b.name != spec.worldbody.name}
    created = set()
    for body_name, finish in sorted(overrides.items()):
        body = bodies.get(body_name)
        geoms = [] if body is None else [g for g in body.geoms if g.group < 3 and not g.meshname.startswith(sensors)]
        if not geoms or any(
            not _opaque(g.rgba, materials[g.material].rgba if g.material else None)
            or (g.material and any(materials[g.material].textures))
            for g in geoms
        ):
            raise ValueError(f"body {body_name!r}: choose an opaque, untextured part with direct visible geometry")
        preset = FINISHES[finish]
        p = PRESETS[preset]
        for g in geoms:
            # Transparent RF materials make visual dressing invisible to rays even with opaque geom colour.
            # Default geom RGBA instead inherits material colour. Preserve both independently of the finish.
            rgba = (
                [float(c) for c in materials[g.material].rgba]
                if g.material
                else ([float(c) for c in g.rgba] if np.allclose(g.rgba, DEFAULT_RGBA) else [1.0, 1.0, 1.0, 1.0])
            )
            key = hashlib.sha256(json.dumps([body_name, finish, rgba]).encode()).hexdigest()[:16]
            name = f"rf_body_surface_{key}/rf_{preset}"
            if name in materials and name not in created:
                raise ValueError("source model uses a reserved body-surface material name")
            if name not in materials:
                materials[name] = spec.add_material(
                    name=name,
                    rgba=rgba,
                    specular=p.specular,
                    shininess=p.shininess,
                    reflectance=p.reflectance,
                    emission=p.emission,
                    metallic=p.metallic,
                    roughness=p.roughness,
                )
                created.add(name)
            g.material = name


def parts(model: mujoco.MjModel, robot: str) -> list[dict]:
    """Same direct-visual eligibility, excluding declared sensor dressing, on the published model."""
    sensors = tuple(f"{s.prefix}rf_{(s.data['name'] + '_sensor').replace('/', '_')}_" for s in declare.read(model))
    prefix = robot + "/"
    out = []
    for b in range(1, model.nbody):
        name = model.body(b).name
        if not name.startswith(prefix):
            continue
        geoms = []
        for g in range(int(model.body_geomadr[b]), int(model.body_geomadr[b] + model.body_geomnum[b])):
            mesh = int(model.geom_dataid[g]) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else -1
            if model.geom_group[g] < 3 and not (mesh >= 0 and model.mesh(mesh).name.startswith(sensors)):
                geoms.append(g)
        if geoms and all(
            _opaque(model.geom_rgba[g], model.mat_rgba[model.geom_matid[g]] if model.geom_matid[g] >= 0 else None)
            and (model.geom_matid[g] < 0 or not np.any(model.mat_texid[model.geom_matid[g]] >= 0))
            for g in geoms
        ):
            out.append({"body": name[len(prefix) :], "geoms": geoms})
    return out
