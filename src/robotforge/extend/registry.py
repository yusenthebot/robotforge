"""What plugins have added, where it came from, and loading plugin files.

A plugin is a Python file that registers new parts, sensors, controllers or behaviors with the decorators in
robotforge.extend. Files are looked for in two places:

- the user's own folder (`~/.robotforge/plugins`, or the folders `ROBOTFORGE_PLUGINS` lists), which is code the user
  put on this computer: loaded when robotforge starts and on request;
- a world's `plugins/` folder, which travels with the world. A world can come from somebody else, and plugin code is
  ordinary Python that runs with the user's rights, so these files load only once they have been trusted: written
  here by write_script(plugin=True), or approved with plugins(action="trust") after the user has seen them. A world
  copied or downloaded to another place, or a file changed since, is not trusted again.

Nothing here imports mujoco or the design layer: a plugin's decorators do that when they register something, so
importing robotforge does not pay for plugins nobody has.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
import traceback
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Kind = Literal["part", "sensor", "controller", "behavior"]
KINDS: tuple[Kind, ...] = ("part", "sensor", "controller", "behavior")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
ENV_FOLDERS = "ROBOTFORGE_PLUGINS"
WORLD_FOLDER = "plugins"
TRUST_FILE = "plugins_trusted.json"


class PluginError(ValueError):
    """A plugin registered something wrongly; the message says what to change."""


@dataclass
class Item:
    kind: Kind
    name: str
    summary: str
    obj: Any  # the part's builder, the sensor's reader, the controller's or behavior's function or class
    plugin: str = "<code>"  # the file it came from, "geiger.py" or "my_world/plugins/geiger.py"
    path: Path | None = None  # that file
    meta: dict[str, Any] = field(default_factory=dict)  # what the kind needs (a part's sizes, a sensor's options)
    undo: Callable[[], None] | None = None  # takes it out of the tables it was put in

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "summary": self.summary, "from": self.plugin, **self.meta.get("describe", {})}


@dataclass
class Loaded:
    """One plugin file as last loaded."""

    path: Path
    label: str
    origin: Literal["user", "world"]
    signature: tuple[int, int]  # mtime_ns, size: loaded again when it changes
    items: list[tuple[Kind, str]] = field(default_factory=list)
    error: str | None = None


class Registry:
    def __init__(self) -> None:
        self.items: dict[tuple[Kind, str], Item] = {}
        self.files: dict[Path, Loaded] = {}
        self.untrusted: dict[Path, str] = {}  # world plugin file -> why it was not loaded
        self.worlds: set[Path] = set()  # plugins/ folders of worlds that were opened, for a reload
        self.user_loaded = False
        self._loading: Loaded | None = None

    # ---- registering ---------------------------------------------------------------------------------------

    def add(self, item: Item) -> None:
        if not NAME_RE.match(item.name):
            raise PluginError(
                f"{item.kind} name {item.name!r} must use lowercase letters, digits and _, start with a letter and "
                "have at most 32 characters, e.g. 'signpost'"
            )
        taken = self.items.get((item.kind, item.name))
        if taken is None and item.kind in ("controller", "behavior"):  # plugins(run) takes either by name
            taken = self.items.get(("behavior" if item.kind == "controller" else "controller", item.name))
        if taken is not None:
            raise PluginError(f"{taken.kind} {item.name!r} is already added by {taken.plugin}; pick another name")
        if self._loading is not None:
            item.plugin, item.path = self._loading.label, self._loading.path
        self.items[(item.kind, item.name)] = item
        if self._loading is not None:
            self._loading.items.append((item.kind, item.name))

    def remove(self, kind: Kind, name: str) -> None:
        item = self.items.pop((kind, name), None)
        if item is not None and item.undo is not None:
            item.undo()

    def get(self, kind: Kind, name: str) -> Item | None:
        return self.items.get((kind, name))

    def of(self, kind: Kind) -> dict[str, Item]:
        return {name: item for (k, name), item in self.items.items() if k == kind}

    # ---- loading -------------------------------------------------------------------------------------------

    def load_user(self) -> None:
        """The user's own folders, once; reload() reads them again."""
        if self.user_loaded:
            return
        self.user_loaded = True
        for folder in user_folders():
            self._load_folder(folder, "user", str(folder.name))

    def load_world(self, world_dir: Path) -> None:
        """A world's plugins/ folder: the trusted files load, the others are listed as not loaded."""
        self.load_user()
        folder = (world_dir / WORLD_FOLDER).resolve()
        for other in [w for w in self.worlds if w != folder]:  # one world's plugins at a time: the open world's
            self._forget_folder(other)
        if not folder.is_dir():
            if folder in self.worlds:
                self._forget_folder(folder)
            return
        self.worlds.add(folder)
        self._load_folder(folder, "world", f"{world_dir.name}/{WORLD_FOLDER}", trusted=_trusted(folder))

    def trust(self, world_dir: Path) -> list[str]:
        """Approve every file now in the world's plugins/ folder and load it. Returns the file names."""
        folder = (world_dir / WORLD_FOLDER).resolve()
        files = _py_files(folder)
        _record_trust(folder, {p.name: _digest(p) for p in files})
        self.load_world(world_dir)
        return [p.name for p in files]

    def written(self, world_dir: Path, path: Path) -> dict[str, Any]:
        """A plugin file just written here (write_script): trusted, loaded, and what came of it."""
        folder = (world_dir / WORLD_FOLDER).resolve()
        _record_trust(folder, {path.name: _digest(path)})
        self.load_world(world_dir)
        label = next((s.label for p, s in self.files.items() if p == path.resolve()), path.name)
        state = self.files.get(path.resolve())
        if state is None or state.error:
            return {"loaded": False, "error": state.error if state else "not loaded"}
        added = [{"kind": k, "name": n} for k, n in state.items]
        return {"loaded": True, "file": label, "adds": added}

    def reload(self) -> None:
        """Forget every plugin and read the folders again (a file edited, added or removed)."""
        worlds = sorted(self.worlds)
        for path in list(self.files):
            self._unload(path)
        self.untrusted.clear()
        self.user_loaded = False
        self.worlds = set()
        self.load_user()
        for folder in worlds:
            self.load_world(folder.parent)

    def load_file(self, path: Path, origin: Literal["user", "world"], label: str) -> Loaded:
        """Run one plugin file. Whatever it registered is taken back out if it fails, and the error is kept."""
        path = path.resolve()
        self._unload(path)
        state = Loaded(path, label, origin, _signature(path))
        self.files[path] = state
        previous, self._loading = self._loading, state
        sys.path.insert(0, str(path.parent))  # a plugin may import a helper that sits beside it
        try:
            module = types.ModuleType(f"robotforge_plugin_{path.stem}")
            module.__file__ = str(path)
            code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
            exec(code, module.__dict__)  # noqa: S102 (the user's own code, run by their choice)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # a plugin must not take robotforge down, not even by SystemExit
            state.error = _explain(exc, path)
            for kind, name in reversed(state.items):
                self.remove(kind, name)
            state.items = []
        finally:
            self._loading = previous
            with contextlib.suppress(ValueError):
                sys.path.remove(str(path.parent))
        return state

    def _load_folder(
        self, folder: Path, origin: Literal["user", "world"], label: str, trusted: dict[str, str] | None = None
    ) -> None:
        here = folder.resolve()
        files = [p.resolve() for p in _py_files(here)]
        for path in [p for p in self.files if p.parent == here and p not in files]:
            self._unload(path)  # the file was deleted
        for path in [p for p in self.untrusted if p.parent == here and p not in files]:
            del self.untrusted[path]
        for path in files:
            if origin == "world" and (trusted or {}).get(path.name) != _digest(path):
                self._unload(path)
                self.untrusted[path] = (
                    "not loaded: this plugin file was not written on this computer or has changed since, and it is "
                    "Python that would run with your rights. Read it (read_script), and if you trust it, "
                    'plugins(action="trust")'
                )
                continue
            self.untrusted.pop(path, None)
            have = self.files.get(path)
            if have is None or have.signature != _signature(path):
                self.load_file(path, origin, f"{label}/{path.name}")

    def _forget_folder(self, folder: Path) -> None:
        for path in [p for p in self.files if p.parent == folder]:
            self._unload(path)
        for path in [p for p in self.untrusted if p.parent == folder]:
            del self.untrusted[path]
        self.worlds.discard(folder)

    def _unload(self, path: Path) -> None:
        state = self.files.pop(path, None)
        if state is not None:
            for kind, name in reversed(state.items):
                self.remove(kind, name)

    def world_notes(self, world_dir: Path) -> list[str]:
        """What to tell the user when a world opens: its own Python that runs, and its files that do not."""
        folder = (world_dir / WORLD_FOLDER).resolve()
        notes = []
        for path, state in self.files.items():
            if path.parent != folder and not (state.origin == "user" and state.error):
                continue
            if state.error:
                notes.append(f"plugin {state.label} failed to load: {state.error}")
            elif state.items:
                what = ", ".join(f"{k} {n}" for k, n in state.items)
                notes.append(f"this world runs its own Python plugin {path.name} (adds {what})")
        for path, why in self.untrusted.items():
            if path.parent == folder:
                notes.append(f"plugin {path.name} {why}")
        return notes

    # ---- reporting -----------------------------------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """Everything plugins added and every file that did not load, for plugins(action="list")."""
        out: dict[str, Any] = {}
        for kind in KINDS:
            found = self.of(kind)
            if found:
                out[f"{kind}s"] = [item.describe() for item in found.values()]
        bad = self.problems()
        if bad:
            out["not_loaded"] = bad
        out["folders"] = [str(f) for f in user_folders()]
        return out

    def problems(self) -> dict[str, str]:
        """Plugin files that did not load, by label, with why."""
        out = {state.label: state.error for state in self.files.values() if state.error}
        for path, why in self.untrusted.items():
            out[f"{path.parent.parent.name}/{WORLD_FOLDER}/{path.name}"] = why
        return out


REGISTRY = Registry()


# ---- the folders and the trust record -----------------------------------------------------------------------


def user_folders() -> list[Path]:
    """Where the user's own plugins are: ROBOTFORGE_PLUGINS (folders joined with the system's path separator), else
    ~/.robotforge/plugins."""
    given = os.environ.get(ENV_FOLDERS)
    if given is not None:
        return [Path(p).expanduser() for p in given.split(os.pathsep) if p.strip()]
    return [Path.home() / ".robotforge" / "plugins"]


def trust_file() -> Path:
    root = Path(os.environ.get("ROBOTFORGE_HOME", Path.home() / "robotforge")).expanduser()
    return root / TRUST_FILE


def _py_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.py") if p.is_file() and not p.name.startswith("_"))


def _signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_trust() -> dict[str, Any]:
    try:
        record = json.loads(trust_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _trusted(folder: Path) -> dict[str, str]:
    found = _read_trust().get(str(folder), {})
    return found if isinstance(found, dict) else {}


def _record_trust(folder: Path, files: dict[str, str]) -> None:
    record = _read_trust()
    record[str(folder)] = {**_trusted(folder), **files}
    path = trust_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=1, sort_keys=True)


def _explain(exc: BaseException, path: Path) -> str:
    """The error as a plugin's author needs it: the line in their file, then what went wrong."""
    if isinstance(exc, SyntaxError):
        where = f"line {exc.lineno}: " if exc.lineno else ""
        return f"{where}SyntaxError: {exc.msg}"
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if Path(frame.filename).resolve() == path:
            line = frame.lineno
    where = f"line {line}: " if line else ""
    kind = "" if isinstance(exc, PluginError) else f"{type(exc).__name__}: "
    return f"{where}{kind}{exc}"


@contextlib.contextmanager
def isolated() -> Iterator[Registry]:
    """For tests and examples: whatever is added inside the block is taken out again at its end, and the files and
    folders the registry knew before are as they were."""
    before = (dict(REGISTRY.items), dict(REGISTRY.files), dict(REGISTRY.untrusted), set(REGISTRY.worlds))
    loaded = REGISTRY.user_loaded
    try:
        yield REGISTRY
    finally:
        for key in [k for k in REGISTRY.items if k not in before[0]]:
            REGISTRY.remove(*key)
        REGISTRY.files, REGISTRY.untrusted, REGISTRY.worlds = before[1], before[2], before[3]
        REGISTRY.user_loaded = loaded
