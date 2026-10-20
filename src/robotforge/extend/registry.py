"""What plugins have added, where it came from, and loading plugin files.

A plugin is a Python file that registers new parts, sensors, controllers, behaviors, training algorithms or tasks
with the decorators in robotforge.extend. Files are looked for in two places:

- the user's own folder (`~/.robotforge/plugins`, or the folders `ROBOTFORGE_PLUGINS` lists), which is code the user
  put on this computer: loaded when robotforge starts and on request;
- a world's `plugins/` folder, which travels with the world. A world can come from somebody else, and plugin code is
  ordinary Python that runs with the user's rights, so these files load only once they have been trusted: written
  here by write_script(plugin=True), or approved by the user in a terminal with `robotforge plugins trust WORLD`
  after they have seen them (an agent cannot approve: it is a decision about what runs on this computer). Trust covers
  every file under the folder, helpers and subfolders included, as one set: a plugin may import what sits beside
  it, so one file nobody looked at would be code that runs. A world copied or downloaded to another place, or
  any file added or changed since, is not trusted again.

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

Kind = Literal["part", "sensor", "controller", "behavior", "algorithm", "task", "generator", "material", "style"]
KINDS: tuple[Kind, ...] = (
    "part",
    "sensor",
    "controller",
    "behavior",
    "algorithm",
    "task",
    "generator",
    "material",
    "style",
)
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
ENV_FOLDERS = "ROBOTFORGE_PLUGINS"
WORLD_FOLDER = "plugins"
TRUST_FILE = "plugins_trusted.json"
_HOW = (
    'Plugin files are Python that would run with your rights. Read them (plugins(action="show")); only if the user '
    'trusts them, they approve them in a terminal: robotforge plugins trust "{world}"'
)
MAX_TREE_FILES = 300  # more than a person reads before trusting
MAX_SHOWN_BYTES = 200_000


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
    module: str = ""  # its name in sys.modules while it is loaded (a dataclass or pickle looks it up there)
    imported: list[str] = field(default_factory=list)  # helper modules from its folder it imported, dropped with it


class Registry:
    def __init__(self) -> None:
        self.items: dict[tuple[Kind, str], Item] = {}
        self.files: dict[Path, Loaded] = {}
        self.untrusted: dict[Path, str] = {}  # world plugin file -> why it was not loaded
        self.worlds: set[Path] = set()  # plugins/ folders of worlds that were opened, for a reload
        self.folder_state: dict[Path, tuple] = {}  # a folder's files (name, mtime, size) as last loaded
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
        """A world's plugins/ folder: when every file in it is trusted they load, otherwise none do and the files
        that are not trusted are listed."""
        self.load_user()
        folder = (world_dir / WORLD_FOLDER).resolve()
        for other in [w for w in self.worlds if w != folder]:  # one world's plugins at a time: the open world's
            self._forget_folder(other)
        if not folder.is_dir():
            if folder in self.worlds:
                self._forget_folder(folder)
            return
        self.worlds.add(folder)
        self._load_folder(folder, "world", f"{world_dir.name}/{WORLD_FOLDER}")

    def trust(self, world_dir: Path) -> list[str]:
        """Approve every file now under the world's plugins/ folder as one set and load it. Returns the file names.
        Only for the user's own decision (robotforge plugins trust): robotforge's tools do not call it."""
        folder = (world_dir / WORLD_FOLDER).resolve()
        tree = read_tree(folder)
        if tree.flaws:
            raise PluginError(f"{folder} cannot be trusted: {'; '.join(tree.flaws)}")
        _record_trust(folder, tree.files)
        self.load_world(world_dir)
        return sorted(tree.files)

    def written(self, world_dir: Path, path: Path) -> dict[str, Any]:
        """A plugin file just written here (write_script): trusted when the rest of the folder is, loaded, and what
        came of it. Beside files nobody approved it waits with them."""
        folder = (world_dir / WORLD_FOLDER).resolve()
        tree = read_tree(folder)
        rel = path.resolve().relative_to(folder).as_posix()
        record = _trusted(folder)
        rest = {k: v for k, v in tree.files.items() if k != rel}
        waiting = sorted(k for k, v in rest.items() if record.get(k) != v)
        if not tree.flaws and not waiting:
            _record_trust(folder, {**record, rel: tree.files[rel]})
        self.load_world(world_dir)
        state = self.files.get(path.resolve())
        if state is None:
            if waiting:
                return {
                    "loaded": False,
                    "error": "saved, but not loaded: the folder loads as one set and these files in it are not "
                    f"trusted yet ({', '.join(waiting)}). {_HOW.format(world=world_dir)}",
                }
            return {"loaded": False, "error": self.untrusted.get(path.resolve(), "not loaded")}
        if state.error:
            return {"loaded": False, "error": state.error}
        added = [{"kind": k, "name": n} for k, n in state.items]
        return {"loaded": True, "file": state.label, "adds": added}

    def reload(self) -> None:
        """Forget every plugin and read the folders again (a file edited, added or removed)."""
        worlds = sorted(self.worlds)
        for path in list(self.files):
            self._unload(path)
        self.untrusted.clear()
        self.folder_state.clear()
        self.user_loaded = False
        self.worlds = set()
        self.load_user()
        for folder in worlds:
            self.load_world(folder.parent)

    def load_file(self, path: Path, origin: Literal["user", "world"], label: str) -> Loaded:
        """Run one plugin file. Whatever it registered is taken back out if it fails, and the error is kept."""
        path = path.resolve()
        self._unload(path)
        state = Loaded(path, label, origin, _signature(path), module=self._module_name(path))
        self.files[path] = state
        previous, self._loading = self._loading, state
        folder = str(path.parent)
        known = set(sys.modules)
        # A user's folder comes first, as it would on any python path; a world's last, so that its files can never
        # stand in for a module of the standard library or of robotforge that something else imports.
        if origin == "user":
            sys.path.insert(0, folder)
        else:
            sys.path.append(folder)
        try:
            module = types.ModuleType(state.module)
            module.__file__ = str(path)
            sys.modules[state.module] = module  # before it runs: a @dataclass and pickle look the module up by name
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
                sys.path.remove(folder)
            state.imported = [m for m in sys.modules if m not in known and _comes_from(sys.modules[m], path.parent)]
            if state.error:
                self._drop_modules(state)
        return state

    def _module_name(self, path: Path) -> str:
        name = f"robotforge_plugin_{path.stem}"
        taken = next((s for s in self.files.values() if s.module == name and s.path != path), None)
        return name if taken is None else f"{name}_{hashlib.sha256(str(path).encode()).hexdigest()[:8]}"

    def _load_folder(self, folder: Path, origin: Literal["user", "world"], label: str) -> None:
        here = folder.resolve()
        files = [p.resolve() for p in _py_files(here)]

        def inside(p: Path) -> bool:
            return here in p.parents

        for path in [p for p in self.files if inside(p) and p not in files]:
            self._unload(path)  # the file was deleted
        for path in [p for p in self.untrusted if inside(p)]:
            del self.untrusted[path]
        state = _folder_state(here)
        if self.folder_state.get(here) != state:  # a file anywhere in it changed: a helper too is read again
            for path in [p for p in self.files if inside(p)]:
                self._unload(path)
            self.folder_state[here] = state
        if origin == "world":
            tree = read_tree(here)
            why = self._untrusted_reasons(here, tree, files)
            if why:
                for path in [p for p in self.files if inside(p)]:
                    self._unload(path)
                self.untrusted.update(why)
                return
        for path in files:
            have = self.files.get(path)
            if have is None or have.signature != _signature(path):
                self.load_file(path, origin, f"{label}/{path.name}")

    def _untrusted_reasons(self, folder: Path, tree: Tree, entries: list[Path]) -> dict[Path, str]:
        """Why the folder does not load, by file (empty when every file in it is trusted)."""
        record = _trusted(folder)
        bad = sorted(rel for rel, digest in tree.files.items() if record.get(rel) != digest)
        if not bad and not tree.flaws:
            return {}
        how = _HOW.format(world=folder.parent)
        if tree.flaws:
            return {p: f"not loaded: {'; '.join(tree.flaws)}, so the folder cannot be trusted" for p in entries}
        out = {}
        for rel in bad:
            out[folder / rel] = f"not loaded: this file was not written on this computer or has changed since. {how}"
        others = ", ".join(bad)
        for path in entries:
            out.setdefault(path, f"not loaded: the files beside it are not trusted yet ({others}). {how}")
        return out

    def _forget_folder(self, folder: Path) -> None:
        for path in [p for p in self.files if folder in p.parents]:
            self._unload(path)
        for path in [p for p in self.untrusted if folder in p.parents]:
            del self.untrusted[path]
        self.folder_state.pop(folder, None)
        self.worlds.discard(folder)

    def _unload(self, path: Path) -> None:
        state = self.files.pop(path, None)
        if state is not None:
            for kind, name in reversed(state.items):
                self.remove(kind, name)
            self._drop_modules(state)

    @staticmethod
    def _drop_modules(state: Loaded) -> None:
        """The plugin's module and the helpers it imported from its folder leave sys.modules, so that the next load
        reads them again (an edited helper, or another world's helper of the same name)."""
        for name in [state.module, *state.imported]:
            sys.modules.pop(name, None)
        state.imported = []

    def label_of(self, path: Path) -> str:
        """`world/plugins/file.py` for a file in a world's plugins/ folder."""
        for folder in self.worlds:
            if folder in path.parents:
                return f"{folder.parent.name}/{WORLD_FOLDER}/{path.relative_to(folder).as_posix()}"
        return path.name

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
            if folder in path.parents:
                notes.append(f"plugin {path.relative_to(folder).as_posix()} {why}")
        return notes

    def show(self, world_dir: Path) -> dict[str, Any]:
        """The files of a world's plugins/ folder as text, for the user (or an agent for the user) to read before
        trusting them: the source of every Python file and the names and sizes of the rest."""
        folder = (world_dir / WORLD_FOLDER).resolve()
        tree = read_tree(folder)
        record = _trusted(folder)
        out: dict[str, Any] = {
            "folder": str(folder),
            "trusted": bool(tree.files) and not tree.flaws and all(record.get(k) == v for k, v in tree.files.items()),
            "files": {},
        }
        left = MAX_SHOWN_BYTES
        for rel in sorted(tree.files):
            path = folder / rel
            size = path.stat().st_size
            if rel.endswith(".py") and size <= left:
                out["files"][rel] = path.read_text(encoding="utf-8", errors="replace")
                left -= size
            else:
                out["files"][rel] = f"(not shown: {size} bytes)"
        if tree.flaws:
            out["cannot_be_trusted"] = tree.flaws
        return out

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
            out[self.label_of(path)] = why
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


@dataclass
class Tree:
    """Every file under a plugins/ folder, by path inside it, with its digest; flaws say why it cannot be trusted."""

    files: dict[str, str] = field(default_factory=dict)
    flaws: list[str] = field(default_factory=list)


def read_tree(folder: Path) -> Tree:
    tree = Tree()
    if not folder.is_dir():
        return tree
    for here, dirs, names in os.walk(folder, followlinks=False):
        base = Path(here)
        for d in list(dirs):
            if d == "__pycache__":
                dirs.remove(d)
            elif (base / d).is_symlink():  # an import would follow it into files nobody hashed
                tree.flaws.append(f"{(base / d).relative_to(folder).as_posix()} is a link to a folder")
                dirs.remove(d)
        for name in names:
            if name.endswith(".pyc") or name == ".DS_Store":
                continue
            path = base / name
            rel = path.relative_to(folder).as_posix()
            try:
                tree.files[rel] = _digest(path)
            except OSError as exc:
                tree.flaws.append(f"{rel} cannot be read ({exc.strerror or exc})")
        if len(tree.files) > MAX_TREE_FILES:
            tree.flaws.append(f"it has more than {MAX_TREE_FILES} files, more than anyone reviews")
            break
    return tree


def _folder_state(folder: Path) -> tuple:
    """(path, mtime, size) of every file under a folder, to see that something changed."""
    out = []
    for here, dirs, names in os.walk(folder, followlinks=False):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in names:
            if not name.endswith(".pyc"):
                with contextlib.suppress(OSError):
                    stat = (Path(here) / name).stat()
                    out.append((os.path.join(here, name), stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(out))


def _comes_from(module: Any, folder: Path) -> bool:
    file = getattr(module, "__file__", None)
    if not file:
        return False
    try:
        return folder in Path(file).resolve().parents
    except (OSError, ValueError):
        return False


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
    record[str(folder)] = dict(files)
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
    states = dict(REGISTRY.folder_state)
    loaded = REGISTRY.user_loaded
    try:
        yield REGISTRY
    finally:
        for key in [k for k in REGISTRY.items if k not in before[0]]:
            REGISTRY.remove(*key)
        for path in [p for p in REGISTRY.files if p not in before[1]]:
            REGISTRY._unload(path)
        REGISTRY.files, REGISTRY.untrusted, REGISTRY.worlds = before[1], before[2], before[3]
        REGISTRY.folder_state = states
        REGISTRY.user_loaded = loaded
