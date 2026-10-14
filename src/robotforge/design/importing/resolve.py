"""Find mesh files that a URDF refers to: package://, model://, file://, absolute and relative paths.

URDFs from GitHub rarely sit inside a ROS workspace, so a plain lookup often fails. The
resolver searches upward from the URDF for the named package and, if that fails, falls back
to progressively looser guesses and says which guess it used.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

# Formats we can read (MuJoCo natively: stl, obj; the rest are converted).
MESH_SUFFIXES = frozenset({".stl", ".obj", ".dae", ".glb", ".gltf", ".ply", ".off"})
_MAX_LEVELS_UP = 8
_MAX_SIBLINGS = 300
_MAX_INDEXED_FILES = 40_000
_SKIP_DIRS = frozenset({"node_modules", "__pycache__", "venv", "site-packages", "build", "install"})
_PACKAGE_NAME = re.compile(r"<name>\s*([^<\s]+)\s*</name>")


@dataclass(frozen=True)
class Resolution:
    """Where a mesh URI points. `note` is empty for a clean hit and explains any guess otherwise."""

    path: Path | None
    note: str = ""
    guessed: bool = False


class MeshResolver:
    """Resolve mesh URIs for one URDF file."""

    def __init__(self, urdf_path: Path):
        self.base = urdf_path.resolve().parent
        self.ancestors = [self.base, *list(self.base.parents)[:_MAX_LEVELS_UP]]
        self._packages: dict[str, Path | None] = {}
        self._index: dict[str, list[Path]] | None = None
        self._cache: dict[str, Resolution] = {}

    def resolve(self, uri: str) -> Resolution:
        key = uri.strip()
        if key not in self._cache:
            self._cache[key] = self._resolve(key)
        return self._cache[key]

    # ---- resolution ------------------------------------------------------------------

    def _resolve(self, uri: str) -> Resolution:
        if not uri:
            return Resolution(None, "empty mesh filename")
        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()
        if len(scheme) == 1:  # a Windows drive letter such as C:\meshes\a.stl, not a URI scheme
            scheme = ""
        if scheme in ("package", "model"):
            return self._resolve_package(uri, parsed.netloc, unquote(parsed.path).lstrip("/"))
        if scheme == "file":
            path = Path(unquote(parsed.path))
            if parsed.netloc and parsed.netloc != "localhost":
                path = Path(f"//{parsed.netloc}{unquote(parsed.path)}")
            return self._resolve_plain(uri, path)
        if scheme:
            return Resolution(None, f"unsupported address '{scheme}://' (download the mesh and use a local path)")
        return self._resolve_plain(uri, Path(unquote(uri)))

    def _resolve_plain(self, uri: str, path: Path) -> Resolution:
        candidate = path if path.is_absolute() else self.base / path
        if _is_mesh_file(candidate):
            return Resolution(candidate.resolve())
        guess = self._guess_by_suffix(path)
        if guess is not None:
            return Resolution(guess, f"'{uri}' does not exist; used {self._pretty(guess)} instead", guessed=True)
        return Resolution(None, f"'{uri}' does not exist")

    def _resolve_package(self, uri: str, package: str, rest: str) -> Resolution:
        root = self.find_package(package)
        if root is not None:
            hit = _confined(root, rest)
            if hit is not None:
                return Resolution(hit)
        guess = self._guess_by_suffix(Path(rest))
        if guess is not None:
            why = "is not on this computer" if root is None else "has no such file"
            return Resolution(guess, f"package '{package}' {why}; used {self._pretty(guess)} instead", guessed=True)
        return Resolution(None, f"package '{package}' was not found near the URDF, or has no file '{rest}'")

    def find_package(self, name: str) -> Path | None:
        """The folder of ROS package `name`, searching upward from the URDF, then ROS environment paths."""
        if name not in self._packages:
            self._packages[name] = self._find_package(name)
        return self._packages[name]

    def _find_package(self, name: str) -> Path | None:
        for ancestor in self.ancestors:
            if ancestor.name == name or _package_xml_name(ancestor) == name:
                return ancestor
            sibling = ancestor / name
            if sibling.is_dir():
                return sibling
        for ancestor in self.ancestors[: _MAX_LEVELS_UP // 2]:
            for child in _children(ancestor):
                if child.name == name or _package_xml_name(child) == name:
                    return child
                for grandchild in _children(child):
                    if grandchild.name == name or _package_xml_name(grandchild) == name:
                        return grandchild
        for var, sub in (("ROS_PACKAGE_PATH", ""), ("AMENT_PREFIX_PATH", "share"), ("GAZEBO_MODEL_PATH", "")):
            for entry in os.environ.get(var, "").split(os.pathsep):
                cand = Path(entry) / sub / name if entry else None
                if cand is not None and cand.is_dir():
                    return cand
        return None

    # ---- guessing ----------------------------------------------------------------------

    def _guess_by_suffix(self, rest: Path) -> Path | None:
        """Look for `rest` (or its tail) under the URDF's folder and parents, then by file name."""
        parts = [p for p in rest.parts if p not in ("", ".", "..", "/")]
        if not parts:
            return None
        for ancestor in self.ancestors[:4]:
            for i in range(len(parts)):
                cand = ancestor.joinpath(*parts[i:])
                if _is_mesh_file(cand):
                    return cand.resolve()
        matches = self._file_index().get(parts[-1].lower(), [])
        if not matches:
            return None
        # With several files of that name, prefer the one sharing the most trailing path parts.
        wanted = [p.lower() for p in parts]

        def overlap(path: Path) -> int:
            have = [p.lower() for p in path.parts]
            n = 0
            while n < min(len(have), len(wanted)) and have[-1 - n] == wanted[-1 - n]:
                n += 1
            return n

        return max(matches, key=overlap).resolve()

    def _file_index(self) -> dict[str, list[Path]]:
        if self._index is None:
            self._index = {}
            top = self._search_root()
            seen = 0
            for dirpath, dirnames, filenames in os.walk(top):
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS]
                for fn in filenames:
                    if Path(fn).suffix.lower() in MESH_SUFFIXES:
                        self._index.setdefault(fn.lower(), []).append(Path(dirpath) / fn)
                seen += len(filenames)
                if seen > _MAX_INDEXED_FILES:
                    break
        return self._index

    def _search_root(self) -> Path:
        """Where to look for files by name: the enclosing repository or package, else just next to the URDF's folder.

        Staying inside one project avoids picking up an unrelated mesh that happens to share a file name.
        """
        for ancestor in self.ancestors[:5]:
            if (ancestor / ".git").exists() or (ancestor / "package.xml").is_file():
                return ancestor
        return self.ancestors[min(1, len(self.ancestors) - 1)]

    def _pretty(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.ancestors[min(1, len(self.ancestors) - 1)]))
        except ValueError:
            return str(path)


def _is_mesh_file(path: Path) -> bool:
    return path.suffix.lower() in MESH_SUFFIXES and path.is_file()


def _confined(root: Path, rest: str) -> Path | None:
    """`root/rest` if it is a mesh file that really lies inside `root` (blocks `..` and symlink escapes)."""
    base = root.resolve()
    cand = (base / rest).resolve()
    if base != cand and base not in cand.parents:
        return None
    return cand if _is_mesh_file(cand) else None


def _package_xml_name(folder: Path) -> str | None:
    xml = folder / "package.xml"
    try:
        if not xml.is_file():
            return None
        match = _PACKAGE_NAME.search(xml.read_text(encoding="utf-8", errors="replace")[:4000])
    except OSError:
        return None
    return match.group(1) if match else None


def _children(folder: Path) -> list[Path]:
    try:
        entries = sorted(folder.iterdir())[:_MAX_SIBLINGS]
    except OSError:
        return []
    return [e for e in entries if e.is_dir() and not e.name.startswith(".") and e.name not in _SKIP_DIRS]
