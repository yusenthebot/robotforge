"""Split a concave collision mesh into convex pieces with CoACD.

MuJoCo collides meshes as their convex hull, so a bracket or a bowl behaves like a solid
blob. Splitting the mesh into several convex pieces keeps its real shape. The work is slow
(seconds per mesh), so results are cached on disk and the step is opt-in.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import trimesh

from robotforge.design.importing.meshio import TOOL_VERSION, MeshAsset

log = logging.getLogger(__name__)

DECOMPOSE_MIN_FACES = 300  # smaller meshes are cheap and usually simple enough
MAX_PIECES = 16
_SETTINGS = f"th0.06-n{MAX_PIECES}-seed0"


class DecomposeError(RuntimeError):
    """CoACD is missing or failed on this mesh."""


def decompose(asset: MeshAsset, out_dir: Path) -> list[Path]:
    """Convex pieces of `asset` as STL files (cached by content; one file if already convex)."""
    key = hashlib.sha1(
        f"{asset.merged.resolve()}|{asset.merged.stat().st_mtime_ns}|{_SETTINGS}|{TOOL_VERSION}".encode()
    )
    tag = key.hexdigest()[:10]
    stem = asset.source.stem[:30]
    done = out_dir / f"{stem}-{tag}-convex.json"
    if done.exists():
        names = done.read_text(encoding="utf-8").split()
        paths = [out_dir / n for n in names]
        if paths and all(p.exists() for p in paths):
            return paths

    try:
        import coacd
    except ImportError as exc:  # coacd is an approved dependency but may be absent in slim installs
        raise DecomposeError("the 'coacd' package is not installed") from exc

    mesh = asset.mesh
    coacd.set_log_level("error")
    try:
        result = coacd.run_coacd(
            coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)),
            threshold=0.06,
            max_convex_hull=MAX_PIECES,
            preprocess_mode="auto",
            mcts_iterations=100,
            seed=0,
        )
    except Exception as exc:  # the native library raises plain RuntimeError or aborts with a message
        raise DecomposeError(f"CoACD failed ({exc})") from exc
    if not result:
        raise DecomposeError("CoACD returned no pieces")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (verts, faces) in enumerate(result):
        piece = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=False)
        path = out_dir / f"{stem}-{tag}-p{i}.stl"
        path.write_bytes(trimesh.exchange.stl.export_stl(piece))
        paths.append(path)
    done.write_text("\n".join(p.name for p in paths), encoding="utf-8")
    return paths
