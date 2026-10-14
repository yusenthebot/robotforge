"""Read any common mesh format and turn it into something MuJoCo loads without complaints.

MuJoCo reads binary STL and OBJ. Everything else (Collada, glTF/GLB, PLY, OFF, ASCII STL) is
converted to binary STL in a cache folder. Meshes that would make MuJoCo fail (too many
triangles, flat sheets, NaN vertices) are repaired on the way, and every repair is reported.
"""

from __future__ import annotations

import hashlib
import logging
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from robotforge.design.importing.dae import DaeError, read_dae

log = logging.getLogger(__name__)

TOOL_VERSION = "1"  # bump when conversion output changes, so cached files are rebuilt
# MuJoCo refuses STL/OBJ files above 200k triangles, and the viewer ships every triangle to the browser,
# so anything above this cap is simplified.
VISUAL_FACE_CAP = 100_000
FLAT_THICKNESS = 0.002  # meters; flat sheets are given this much depth so MuJoCo can build a hull
_NATIVE = (".stl", ".obj")


class MeshError(ValueError):
    """A mesh file that cannot be used at all; the message says why."""


@dataclass
class Piece:
    """One output file of a mesh and the flat color it should be drawn with (None = unknown)."""

    path: Path
    rgba: tuple[float, float, float, float] | None
    faces: int


@dataclass
class MeshAsset:
    """A mesh ready for MuJoCo, plus what the doctor did to it."""

    source: Path
    pieces: list[Piece]
    merged: Path  # a single file holding all pieces (used where color does not matter)
    faces: int  # triangles in the original file
    out_faces: int  # triangles MuJoCo will see
    bounds: np.ndarray  # (2, 3) min and max corner in the mesh's own units
    mesh: trimesh.Trimesh  # cleaned geometry, kept for mass properties
    notes: list[str] = field(default_factory=list)  # fixes applied, plain language


@dataclass(frozen=True)
class SolidProperties:
    volume: float
    center: np.ndarray  # center of mass, mesh frame
    inertia: np.ndarray  # (3, 3) about the center of mass, for density 1


def prepare_mesh(source: Path, out_dir: Path) -> MeshAsset:
    """Load `source`, repair it and return files MuJoCo can read."""
    suffix = source.suffix.lower()
    loaded = _load_pieces(source)
    notes: list[str] = []
    cleaned: list[tuple[trimesh.Trimesh, tuple[float, ...] | None]] = []
    for mesh, rgba in loaded:
        mesh, note = _clean(mesh)
        notes += note
        if len(mesh.faces):
            cleaned.append((mesh, rgba))
    if not cleaned:
        raise MeshError(f"{source.name} contains no usable triangles")
    original_faces = sum(len(m.faces) for m, _ in cleaned)

    digest = hashlib.sha1(
        f"{source.resolve()}|{source.stat().st_size}|{source.stat().st_mtime_ns}|{TOOL_VERSION}".encode()
    )
    tag = digest.hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem)[:40]

    pieces: list[tuple[trimesh.Trimesh, tuple[float, ...] | None]] = []
    rewrite = suffix not in _NATIVE or len(cleaned) > 1 or any(rgba for _, rgba in cleaned) or bool(notes)
    for mesh, rgba in cleaned:
        mesh, fixes = _make_loadable(mesh)
        notes += fixes
        rewrite = rewrite or bool(fixes)
        pieces.append((mesh, rgba))
    if suffix in _NATIVE and not rewrite and not _native_file_ok(source):
        rewrite = True
        notes.append(_native_problem(source))

    merged_mesh = trimesh.util.concatenate([m for m, _ in pieces]) if len(pieces) > 1 else pieces[0][0]
    out_pieces: list[Piece] = []
    if rewrite:
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (mesh, rgba) in enumerate(pieces):
            name = f"{stem}-{tag}" + (f"-c{i}" if len(pieces) > 1 else "") + ".stl"
            _write_stl(mesh, out_dir / name)
            out_pieces.append(Piece(out_dir / name, rgba, len(mesh.faces)))
        if len(pieces) > 1:
            merged, extra = _make_loadable(merged_mesh)
            notes += extra
            merged_path = out_dir / f"{stem}-{tag}-all.stl"
            _write_stl(merged, merged_path)
        else:
            merged_path = out_pieces[0].path
        if suffix not in _NATIVE:
            notes.insert(0, f"converted from {suffix[1:].upper()} to STL")
    else:
        merged_path = source.resolve()
        out_pieces.append(Piece(merged_path, None, len(pieces[0][0].faces)))

    out_faces = sum(p.faces for p in out_pieces)
    return MeshAsset(
        source=source.resolve(),
        pieces=out_pieces,
        merged=merged_path,
        faces=original_faces,
        out_faces=out_faces,
        bounds=np.array(merged_mesh.bounds),
        mesh=merged_mesh,
        notes=_dedupe(notes),
    )


def solid_properties(mesh: trimesh.Trimesh, scale: np.ndarray) -> SolidProperties:
    """Volume, center of mass and inertia (density 1) of `mesh` after per-axis `scale`."""
    scaled = mesh.copy()
    scaled.vertices = np.asarray(mesh.vertices) * scale
    box = float(np.prod(np.maximum(scaled.extents, 1e-9)))
    volume = abs(float(scaled.volume)) if scaled.is_watertight else 0.0
    if volume < 0.02 * box:
        # Open or paper-thin meshes have no trustworthy volume; fall back to a filled fraction of the box.
        return SolidProperties(0.3 * box, np.array(scaled.bounds).mean(axis=0), _box_inertia(scaled.extents, 0.3 * box))
    inertia = np.asarray(scaled.moment_inertia)
    if scaled.volume < 0:  # inside-out mesh: mass properties come out negative
        inertia = -inertia
    return SolidProperties(volume, np.asarray(scaled.center_mass), inertia)


def _box_inertia(extents: np.ndarray, mass: float) -> np.ndarray:
    x, y, z = (float(e) for e in extents)
    return np.diag([y * y + z * z, x * x + z * z, x * x + y * y]) * mass / 12.0


# ---- loading ---------------------------------------------------------------------------------


def _load_pieces(source: Path) -> list[tuple[trimesh.Trimesh, tuple[float, ...] | None]]:
    suffix = source.suffix.lower()
    if suffix == ".dae":
        try:
            return [(trimesh.Trimesh(p.vertices, p.faces, process=True), p.rgba) for p in read_dae(source)]
        except DaeError as exc:
            log.info("built-in Collada reader failed on %s: %s", source.name, exc)
            return _load_with_trimesh(source, f"The Collada file could not be read ({exc})")
    return _load_with_trimesh(source, "")


def _load_with_trimesh(source: Path, prefix: str) -> list[tuple[trimesh.Trimesh, tuple[float, ...] | None]]:
    try:
        loaded = trimesh.load(source, force="mesh", allow_remote=False)
    except ModuleNotFoundError as exc:
        hint = ""
        if source.suffix.lower() == ".dae" or "collada" in str(exc).lower():
            hint = " DAE meshes need pycollada for this file; ask the user to approve installing it."
        raise MeshError(f"{prefix or source.name}: a reader library is missing ({exc.name}).{hint}".strip()) from exc
    except (ValueError, OSError, KeyError, IndexError, TypeError) as exc:
        reason = f"{prefix}; " if prefix else ""
        raise MeshError(f"{source.name}: {reason}trimesh could not read it ({exc})") from exc
    if not isinstance(loaded, trimesh.Trimesh):
        raise MeshError(f"{source.name} did not contain a triangle mesh")
    # Only glTF and PLY carry colors worth keeping; OBJ/STL would report a default gray.
    colored = source.suffix.lower() in (".glb", ".gltf", ".ply")
    return [(loaded, _mean_color(loaded) if colored else None)]


def _mean_color(mesh: trimesh.Trimesh) -> tuple[float, ...] | None:
    try:
        visual = mesh.visual
        if visual.kind == "vertex":
            colors = np.asarray(visual.vertex_colors, dtype=float)
            return tuple(float(c) / 255.0 for c in colors.mean(axis=0)[:4])
        if visual.kind == "face":
            colors = np.asarray(visual.face_colors, dtype=float)
            return tuple(float(c) / 255.0 for c in colors.mean(axis=0)[:4])
        material = getattr(visual, "material", None)
        main = getattr(material, "main_color", None)
        if main is not None:
            return tuple(float(c) / 255.0 for c in np.asarray(main)[:4])
    except (AttributeError, ValueError, TypeError):
        return None
    return None


# ---- repair ----------------------------------------------------------------------------------


def _clean(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    notes = []
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    bad = ~np.isfinite(verts).all(axis=1)
    if bad.any():
        keep = ~bad[np.asarray(mesh.faces)].any(axis=1)
        mesh = trimesh.Trimesh(np.nan_to_num(verts), np.asarray(mesh.faces)[keep], process=False)
        notes.append(f"removed {int(bad.sum())} broken (NaN) vertices")
    valid = mesh.nondegenerate_faces()
    if not valid.all() and valid.any():
        mesh = trimesh.Trimesh(mesh.vertices, np.asarray(mesh.faces)[valid], process=False)
    return mesh, notes


def _make_loadable(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    """Make sure MuJoCo can load `mesh`: not too big, not flat, enough vertices."""
    notes = []
    if len(mesh.faces) > VISUAL_FACE_CAP:
        before = len(mesh.faces)
        mesh = decimate_clusters(mesh, VISUAL_FACE_CAP)
        notes.append(f"simplified from {before:,} to {len(mesh.faces):,} triangles")
    if _is_flat(mesh):
        mesh = _thicken(mesh)
        notes.append(f"gave a flat sheet {FLAT_THICKNESS * 1000:.0f} mm of thickness")
    return mesh, notes


def _is_flat(mesh: trimesh.Trimesh) -> bool:
    verts = np.asarray(mesh.vertices)
    if len(verts) < 4:
        return True
    centered = verts - verts.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    return bool(s[0] <= 0 or s[2] / s[0] < 1e-6 or s[2] < 1e-7)


def _thicken(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    centered = verts - verts.mean(axis=0)
    # The sheet normal is the direction of least spread; a line or point falls back to z.
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[2] if len(vt) == 3 and s[1] > 1e-9 else np.array([0.0, 0.0, 1.0])
    offset = normal * FLAT_THICKNESS / 2
    n = len(verts)
    top, bottom = verts + offset, verts - offset
    all_faces = [faces, faces[:, ::-1] + n]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keys = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    boundary = edges[counts[inverse.ravel()] == 1]
    if len(boundary):
        a, b = boundary[:, 0], boundary[:, 1]
        all_faces.append(np.stack([a, b, b + n], axis=1))
        all_faces.append(np.stack([a, b + n, a + n], axis=1))
    return trimesh.Trimesh(np.vstack([top, bottom]), np.vstack(all_faces), process=False)


def decimate_clusters(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Cheap simplification by snapping vertices to a grid and merging those that share a cell."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    span = float(np.max(hi - lo)) or 1.0
    best: tuple[np.ndarray, np.ndarray] | None = None
    # Larger cells give fewer triangles, so search for the smallest cell that fits the budget.
    low, high = span / 4000.0, span / 4.0
    for _ in range(14):
        cell = (low + high) / 2
        new_verts, new_faces = _cluster(verts, faces, lo, cell)
        if len(new_faces) > target_faces:
            low = cell
        else:
            best, high = (new_verts, new_faces), cell
    if best is None:
        best = _cluster(verts, faces, lo, high)
    return trimesh.Trimesh(best[0], best[1], process=False)


def _cluster(verts: np.ndarray, faces: np.ndarray, origin: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.floor((verts - origin) / cell).astype(np.int64)
    _, index, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    sums = np.zeros((len(index), 3))
    np.add.at(sums, inverse, verts)
    counts = np.bincount(inverse, minlength=len(index))[:, None]
    new_faces = inverse[faces]
    keep = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    return sums / counts, new_faces[keep]


# ---- native file checks ----------------------------------------------------------------------


def _native_file_ok(path: Path) -> bool:
    """True when MuJoCo can read the STL/OBJ as it is (binary STL; OBJ with a single object)."""
    return _native_problem(path) == ""


def _native_problem(path: Path) -> str:
    try:
        size = path.stat().st_size
        if path.suffix.lower() == ".stl":
            with path.open("rb") as f:
                head = f.read(84)
            if len(head) < 84:
                return "STL file is too short"
            (count,) = struct.unpack("<I", head[80:84])
            if size != 84 + 50 * count:
                return "ASCII STL converted to binary STL (MuJoCo only reads binary STL)"
            return ""
        if size > 200_000_000:
            return "OBJ file is too large to inspect"
        objects = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith(("o ", "g ")):
                    objects += 1
                    if objects > 1:
                        return "OBJ file holds several objects; merged them into one mesh (MuJoCo reads only the first)"
        return ""
    except OSError as exc:
        return f"file could not be read ({exc})"


def _write_stl(mesh: trimesh.Trimesh, path: Path) -> None:
    if path.exists():
        return
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(trimesh.exchange.stl.export_stl(mesh))
    tmp.replace(path)


def _dedupe(notes: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for n in notes:
        seen.setdefault(n, None)
    return list(seen)
