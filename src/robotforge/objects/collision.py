"""Convex collision for a mesh a generator only drew: CoACD cuts it into convex pieces, once.

MuJoCo collides a mesh as its convex hull, so a mug drawn as one mesh would collide like a solid blob (a ball could
not be dropped in it). Generators that know their shape give convex pieces directly (`kit.vessel`, capsules along a
handle); for a shape they only draw (a sculpted or imported mesh), `decompose(mesh)` finds the pieces. CoACD takes
seconds to a minute per mesh, so the answer is kept on disk by the mesh's content and the parameters:
`$ROBOTFORGE_HOME/cache/coacd`. coacd and its import are slow, so nothing here runs unless it is called.

    g.part(mesh, "ceramic", collision.decompose(mesh, max_pieces=8))
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from robotforge.design.visuals import mesh as M
from robotforge.objects import kit

DEFAULT_THRESHOLD = 0.06  # CoACD's concavity: lower follows the shape more closely, with more pieces
MAX_PIECES = 16


class DecomposeError(RuntimeError):
    """CoACD is missing or failed on this mesh."""


def cache_dir() -> Path:
    root = Path(os.environ.get("ROBOTFORGE_HOME", Path.home() / "robotforge")).expanduser()
    return root / "cache" / "coacd"


def _key(mesh: M.Mesh, threshold: float, max_pieces: int, seed: int) -> str:
    h = hashlib.sha1()
    h.update(np.round(mesh.vertices, 6).tobytes())
    h.update(mesh.faces.astype(np.int32).tobytes())
    h.update(f"|{threshold:.4f}|{max_pieces}|{seed}".encode())
    return h.hexdigest()[:20]


def convex_pieces(
    mesh: M.Mesh,
    threshold: float = DEFAULT_THRESHOLD,
    max_pieces: int = 8,
    seed: int = 0,
    cache: Path | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The convex pieces of a closed mesh as (vertices, faces), found by CoACD and kept in the cache."""
    max_pieces = max(1, min(int(max_pieces), MAX_PIECES))
    folder = cache_dir() if cache is None else cache
    path = folder / f"{_key(mesh, threshold, max_pieces, seed)}.npz"
    if path.is_file():
        try:
            with np.load(path) as stored:
                count = int(stored["count"])
                return [(stored[f"v{i}"], stored[f"f{i}"]) for i in range(count)]
        except (OSError, KeyError, ValueError):
            path.unlink(missing_ok=True)  # a half-written file: find them again
    try:
        import coacd
    except ImportError as exc:  # approved dependency, but a slim install may lack it
        raise DecomposeError("the 'coacd' package is not installed") from exc
    coacd.set_log_level("error")
    try:
        result = coacd.run_coacd(
            coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)),
            threshold=threshold,
            max_convex_hull=max_pieces,
            preprocess_mode="auto",
            mcts_iterations=60,
            seed=seed,
        )
    except Exception as exc:  # the native library raises plain RuntimeError or aborts with a message
        raise DecomposeError(f"CoACD failed on this mesh ({exc})") from exc
    if not result:
        raise DecomposeError("CoACD returned no pieces")
    pieces = [(np.asarray(v, dtype=float), np.asarray(f, dtype=np.int32)) for v, f in result]
    folder.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"count": np.array(len(pieces))}
    for i, (v, f) in enumerate(pieces):
        arrays[f"v{i}"], arrays[f"f{i}"] = v, f
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return pieces


def decompose(mesh: M.Mesh, **options: float | int) -> list[kit.Piece]:
    """Collision pieces for `mesh` (see convex_pieces for the options): one convex hull per piece CoACD finds."""
    out = []
    for vertices, faces in convex_pieces(mesh, **options):  # type: ignore[arg-type]
        zeros = np.zeros_like(vertices)
        out.append(kit.hull(M.Mesh(vertices, zeros, faces, np.arange(len(vertices), dtype=np.int64))))
    return out
