"""A world on disk: world.yaml plus undo history and the compiled MJCF.

Edits are atomic: a change is validated and compiled before world.yaml is
touched, so a bad edit never leaves a broken world behind.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

from robotforge.design.world import CompiledWorld, World, compile_world

HISTORY_LIMIT = 50
_SAFE = re.compile(r"[^a-z0-9_]+")


def worlds_home() -> Path:
    """Where worlds live: $ROBOTFORGE_HOME/worlds, default ~/robotforge/worlds."""
    root = Path(os.environ.get("ROBOTFORGE_HOME", Path.home() / "robotforge")).expanduser()
    return root / "worlds"


CODE_LENGTH = 6  # hex digits of the code that tells apart worlds whose names are not in Latin letters


def slug(name: str) -> str:
    """The folder a world lives in: a-z, 0-9 and underscores. A name in another script (a Chinese world name) would
    clean to nothing, and two such names must not be one world, so a short code of the name stands in for it."""
    s = _SAFE.sub("_", name.strip().lower()).strip("_")
    if any(c.isalnum() and not c.isascii() for c in name):
        code = hashlib.sha1(name.strip().encode("utf-8")).hexdigest()[:CODE_LENGTH]
        s = f"{s[:24]}_{code}" if s else f"world_{code}"
    if not s or not s[0].isalpha():
        s = f"world_{s}" if s else "world"
    return s[:32]


def _legacy_slug(name: str) -> str:
    """The folder robotforge used before names in other scripts got their code ("Café" was saved as caf), so worlds
    saved then still open by the name they were given."""
    s = _SAFE.sub("_", name.strip().lower()).strip("_")
    if not s or not s[0].isalpha():
        s = f"world_{s}" if s else "world"
    return s[:32]


def _folder(name: str, parent: Path | None = None) -> Path:
    """Where the world called `name` lives: its own folder, or the one an older robotforge saved it in."""
    home = parent or worlds_home()
    own, legacy = home / slug(name), home / _legacy_slug(name)
    # Every name without Latin letters used to share "world": that one opens by its own name only.
    shared = legacy.name == "world"
    if not (own / "world.yaml").exists() and legacy != own and not shared and (legacy / "world.yaml").exists():
        world = legacy / "world.yaml"
        with contextlib.suppress(OSError, ValueError):  # a world whose file cannot be read is not this one
            if World.from_yaml(world.read_text(encoding="utf-8")).name in (name.strip(), _legacy_slug(name)):
                return legacy
    return own


class WorldStore:
    def __init__(self, directory: Path):
        self.dir = directory
        self.file = directory / "world.yaml"
        self.history = directory / ".history"
        self.build_dir = directory / "build"
        self.reopened = False  # set by create() when the world was already on disk

    @classmethod
    def create(cls, name: str, floor_style: str = "checker", parent: Path | None = None) -> WorldStore:
        directory = _folder(name, parent)
        store = cls(directory)
        if store.file.exists():
            store.reopened = True  # reopening an existing world is friendlier than failing
            return store
        world = World.model_validate({"name": slug(name), "floor": {"style": floor_style}})
        compiled = compile_world(world, directory)
        directory.mkdir(parents=True, exist_ok=True)
        store._write(world, compiled)
        return store

    @classmethod
    def open(cls, name_or_path: str) -> WorldStore:
        p = Path(name_or_path).expanduser()
        candidates = [p, p / "world.yaml"] if p.is_absolute() or p.exists() else []
        candidates.append(_folder(name_or_path))
        for c in candidates:
            directory = c.parent if c.name == "world.yaml" else c
            if (directory / "world.yaml").exists():
                return cls(directory)
        known = sorted(d.name for d in worlds_home().glob("*") if (d / "world.yaml").exists())
        raise ValueError(f"no world named {name_or_path!r}. Existing worlds: {', '.join(known) or 'none yet'}")

    def load(self) -> World:
        return World.from_yaml(self.file.read_text(encoding="utf-8"))

    def compile(self) -> CompiledWorld:
        return compile_world(self.load(), self.dir)

    def apply(self, change: Callable[[World], None], snapshot: bool = True) -> tuple[World, CompiledWorld]:
        """Run `change` on a copy, validate and compile it, then save. Raises without saving on error. snapshot=False
        saves no undo step: the change finishes the one before it (undo takes back both)."""
        current = self.load()
        candidate = current.model_copy(deep=True)
        change(candidate)
        candidate = World.model_validate(candidate.model_dump())  # re-run validators after mutation
        compiled = compile_world(candidate, self.dir)
        if snapshot:
            self._snapshot()
        self._write(candidate, compiled)
        return candidate, compiled

    def undo(self) -> World:
        snapshots = sorted(self.history.glob("*.yaml"))
        if not snapshots:
            raise ValueError("nothing to undo")
        last = snapshots[-1]
        world = World.from_yaml(last.read_text(encoding="utf-8"))
        compiled = compile_world(world, self.dir)
        self._write(world, compiled)
        last.unlink()
        return world

    def _snapshot(self) -> None:
        self.history.mkdir(exist_ok=True)
        taken = [int(p.stem) for p in self.history.glob("*.yaml") if p.stem.isdigit()]
        # The Windows clock ticks every ~15 ms, so two quick edits can read the same time; keep names increasing.
        stamp = max(time.time_ns(), max(taken, default=0) + 1)
        _write_text(self.history / f"{stamp}.yaml", self.file.read_text(encoding="utf-8"))
        for old in sorted(self.history.glob("*.yaml"))[:-HISTORY_LIMIT]:
            old.unlink()

    def _write(self, world: World, compiled: CompiledWorld) -> None:
        _write_text(self.file, world.to_yaml())
        self.build_dir.mkdir(exist_ok=True)
        _write_text(self.build_dir / "world.xml", compiled.to_xml())


def _write_text(path: Path, text: str) -> None:
    """UTF-8 with LF endings on every OS, so a world folder copied between machines does not churn."""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
