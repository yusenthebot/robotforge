"""The open robot catalogue: over two hundred open robots (arms, dogs, humanoids, hands, wheeled bases, drones) that
`add_robot(library=<name>)` loads, on top of the hand-tuned library (library.py).

The list comes from the robot_descriptions package (catalogue.json, made by scripts/catalogue_gen.py): which
repository and commit each robot lives at, which file is its model and what its makers call it. Nothing is
downloaded until a robot is added; then just its folder is fetched (catalogue_fetch.py), cached and run through the
import doctor (URDF, xacro) or read as MJCF, and catalogue_inspect.py says from its structure what it is and what it
can do.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robotforge import wording
from robotforge.design import catalogue_fetch, library, menagerie

log = logging.getLogger(__name__)
DATA = Path(__file__).with_name("catalogue.json")
KINDS = ("arm", "quadruped", "humanoid", "wheeled", "hand", "aircraft", "other")
# What robot_descriptions calls a robot, as one of ours. Several tags: the first that is here wins (a "biped" that is
# also "wheeled" is a legged robot first). The structure of the loaded model has the last word (catalogue_inspect.py).
TAG_KIND = (
    ("quadruped", "quadruped"),
    ("humanoid", "humanoid"),
    ("biped", "humanoid"),
    ("drone", "aircraft"),
    ("mobile_manipulator", "wheeled"),
    ("wheeled", "wheeled"),
    ("end_effector", "hand"),
    ("dual_arm", "arm"),
    ("arm", "arm"),
)


class CatalogueError(ValueError):
    """User-facing: says what went wrong and what to do about it."""


@dataclass(frozen=True)
class Entry:
    id: str
    robot: str  # what its makers call it ("Go2", "Franka FR3")
    maker: str | None
    dof: int | None
    tags: tuple[str, ...]
    license: str | None
    repo: str
    url: str
    commit: str
    package: str  # folder of the repository the model lives in ("" for the repository itself)
    format: str  # "mjcf", "urdf" or "xacro"
    model: str  # the model file, relative to the repository
    variants: tuple[str, ...]  # other model files of the same robot ("MJCF_PATH_LEFT:path")
    xacro_args: tuple[tuple[str, str], ...]

    @property
    def kind(self) -> str:
        """What it is by its tags; the loaded model's structure may say otherwise (catalogue_inspect.py)."""
        return next((kind for tag, kind in TAG_KIND if tag in self.tags), "other")

    @property
    def title(self) -> str:
        return f"{self.maker} {self.robot}" if self.maker and not self.robot.startswith(self.maker) else self.robot

    @property
    def in_library(self) -> str | None:
        """The library robot that is this same model (the library's is tuned; use it), or None."""
        if self.repo != "mujoco_menagerie":
            return None
        folder = self.model.split("/", 1)[0]
        return next((r.name for r in library.CATALOG.values() if r.folder == folder), None)

    @property
    def summary(self) -> str:
        """One line before it is downloaded: what it is, from its tags and size; downloading refines it."""
        joints = f" with {self.dof} joints" if self.dof else ""
        what = {
            "arm": f"a robot arm{joints}",
            "quadruped": f"a four-legged robot{joints}",
            "humanoid": f"a walking robot on two legs{joints}",
            "wheeled": "a wheeled robot" + (" that carries an arm" if "mobile_manipulator" in self.tags else ""),
            "hand": "a gripper or robot hand",
            "aircraft": "a drone",
            "other": "a robot",
        }[self.kind]
        if "dual_arm" in self.tags:
            what = f"a two-armed robot{joints}"
        if self.kind == "hand":
            joints = f" with {self.dof} joints" if self.dof else ""
            what += joints
        return f"{what[0].upper()}{what[1:]} ({self.format.upper()}, {self.license or 'see the repository'})"

    def row(self) -> dict[str, Any]:
        """The entry as list_library shows it."""
        return {
            "name": self.id,
            "title": self.title,
            "kind": self.kind,
            "summary": self.summary,
            "maker": self.maker,
            "license": self.license,
            "source": "catalogue",
            "downloaded": self.is_downloaded(),
            **({"library": self.in_library} if self.in_library else {}),
        }

    def is_downloaded(self) -> bool:
        return catalogue_fetch.folder_path(self.repo, self.commit, self.package) is not None

    def model_file(self) -> Path | None:
        """The model file on disk if the robot was downloaded, else None."""
        folder = catalogue_fetch.folder_path(self.repo, self.commit, self.package)
        if folder is None:
            return None
        root = folder.parents[len(Path(self.package).parts) - 1] if self.package else folder
        path = root / self.model
        if not path.is_file():
            return None
        if self.format == "mjcf" and not path.stem.endswith(("_pos", "_position")):
            # Menagerie's torque-motor models have a sibling driven by joint targets (toddlerbot): use that
            for suffix in ("_pos", "_position"):
                sibling = path.with_name(f"{path.stem}{suffix}{path.suffix}")
                if sibling.is_file():
                    return sibling
        return path


@functools.cache
def entries() -> dict[str, Entry]:
    data = json.loads(DATA.read_text(encoding="utf-8"))["robots"]
    out = {}
    for e in data:
        out[e["id"]] = Entry(
            id=e["id"],
            robot=e["robot"],
            maker=e["maker"],
            dof=e["dof"],
            tags=tuple(e["tags"]),
            license=e["license"],
            repo=e["repo"]["name"],
            url=e["repo"]["url"],
            commit=e["repo"]["commit"],
            package=e["package"],
            format=e["format"],
            model=e["model"],
            variants=tuple(e["variants"]),
            xacro_args=tuple(sorted(e["xacro_args"].items())),
        )
    return out


def _titles() -> dict[str, str]:
    """Names for robots by what their makers call them, where that is unique ("Franka FR3" is fr3)."""
    count: dict[str, int] = {}
    for e in entries().values():
        count[wording.squash(e.title)] = count.get(wording.squash(e.title), 0) + 1
    return {wording.squash(e.title): e.id for e in entries().values() if count[wording.squash(e.title)] == 1}


def find(name: str) -> Entry | None:
    """The catalogue robot `name` can only mean, or None."""
    found = wording.match(name, entries(), _titles())
    return entries()[found] if found else None


def get(name: str) -> Entry:
    found = find(name)
    if found is None:
        close = wording.near(name, entries(), _titles())
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise CatalogueError(
            f"no robot {name!r} in the library or the open catalogue.{hint} "
            "list_library(search=..., kind=...) finds robots by name and kind"
        )
    return found


def tuned(entry: Entry) -> bool:
    """Whether the hand-tuned library already has this robot (the same model, or a robot of the same name)."""
    return entry.in_library is not None or entry.id in library.CATALOG


KIND_ALIASES = {  # what an agent (or the tuned library's own rows) may call a kind that is called otherwise here
    "gripper": "hand",
    "end_effector": "hand",
    "dog": "quadruped",
    "legged": "quadruped",
    "biped": "humanoid",
    "drone": "aircraft",
    "uav": "aircraft",
    "mobile": "wheeled",
    "rover": "wheeled",
    "manipulator": "arm",
}


def canonical_kind(kind: str | None) -> str | None:
    """`kind` as one of KINDS (aliases and capitals allowed), or None for no kind. Raises CatalogueError otherwise."""
    if kind is None or not kind.strip():
        return None
    key = kind.strip().lower()
    key = KIND_ALIASES.get(key, key)
    if key not in KINDS:
        aliases = ", ".join(sorted(KIND_ALIASES))
        raise CatalogueError(f"kind must be one of {', '.join(KINDS)} (or {aliases}), got {kind!r}")
    return key


def _tokens(*texts: str | None) -> list[str]:
    return re.findall(r"[a-z0-9]+", " ".join(t for t in texts if t).lower())


def search(kind: str | None = None, query: str | None = None) -> list[Entry]:
    """Catalogue robots of a kind whose id, name, maker or tags have a word starting with every word of `query`
    ("ur" finds ur5e and UR10e, not "urdf"), best match first. Robots the tuned library has are left out:
    list_library shows them from there."""
    kind = canonical_kind(kind)
    words = _tokens(query)
    scored = []
    for e in entries().values():
        if (kind is not None and e.kind != kind) or tuned(e):
            continue
        tokens = _tokens(e.id, e.robot, e.maker, " ".join(e.tags))
        if not all(any(t.startswith(w) for t in tokens) for w in words):
            continue
        names = _tokens(e.id, e.robot)
        exact = sum(w in names for w in words)
        scored.append((-exact, e.in_library is not None, e.id, e))
    return [e for *_, e in sorted(scored, key=lambda t: t[:3])]


def counts() -> dict[str, int]:
    out = dict.fromkeys(KINDS, 0)
    for e in entries().values():
        out[e.kind] += not tuned(e)
    return {k: v for k, v in out.items() if v}


def listing(kind: str | None = None, query: str | None = None, limit: int = 40) -> dict[str, Any]:
    """list_library's answer: the tuned library's robots and the open catalogue's that match `kind` and `query`.
    Without either it is the tuned library plus a count of what the open catalogue has."""
    kind = canonical_kind(kind)
    found = search(kind, query)
    if kind is None and not query:
        return {
            "robots": library.catalog(),
            "open_catalogue": {
                "robots": len(found),
                "by_kind": counts(),
                "how": "list_library(kind='arm', search='kinova') searches them; add_robot(library=<name>) loads one "
                "(downloads its folder on first use, imports it and says what it is and what it can do)",
            },
        }
    words = _tokens(query)
    tuned_rows = [
        r
        for r in library.catalog()
        if (kind is None or canonical_kind(r["kind"]) == kind)
        and all(any(t.startswith(w) for t in _tokens(r["name"], r["title"], r["maker"])) for w in words)
    ]
    out: dict[str, Any] = {
        "robots": [*tuned_rows, *(e.row() for e in found[:limit])],
        "matches": len(tuned_rows) + len(found),
    }
    if len(found) > limit:
        out["more"] = f"{len(found) - limit} more match: narrow it with search=... or raise limit"
    if not out["robots"]:
        out["note"] = f"nothing matches; kinds are {', '.join(KINDS)}"
    return out


# ---- fetching ---------------------------------------------------------------------------------------


def model_path(name: str, progress: menagerie.Progress | None = None) -> Path:
    """The model file of a catalogue robot, downloading its folder first if needed."""
    entry = get(name)
    found = entry.model_file()
    if found is not None:
        return found
    try:
        catalogue_fetch.fetch(entry.repo, entry.url, entry.commit, entry.package, progress)
    except catalogue_fetch.CatalogueFetchError as exc:
        raise CatalogueError(f"{entry.title}: {exc}") from exc
    found = entry.model_file()
    if found is None:
        raise CatalogueError(f"{entry.title}: the download has no {entry.model} (the repository changed?)")
    return found


def source_of(name: str) -> tuple[str | None, str | None]:
    """Where `add_robot(library=name)` gets the robot: (library name, None) for the tuned library, which also covers
    a catalogue robot that is the same model, or (None, catalogue name) for an open one."""
    curated = wording.match(name, library.CATALOG, library._titles())
    if curated is not None:
        return curated, None
    entry = get(name)
    return (entry.in_library, None) if entry.in_library else (None, entry.id)


TOO_BIG = frozenset({"drake", "mujoco_menagerie"})  # repositories of hundreds of robots: never downloaded whole


def widen(name: str, progress: menagerie.Progress | None = None) -> bool:
    """Download the whole repository of a robot whose files refer to folders outside its own (a xacro that includes
    another package, meshes shared with a sibling). False when all of it is here already, it is too big or it failed."""
    entry = get(name)
    if entry.repo in TOO_BIG or catalogue_fetch.is_whole(entry.repo, entry.commit):
        return False
    try:
        catalogue_fetch.fetch(entry.repo, entry.url, entry.commit, "", progress, whole=True)
    except catalogue_fetch.CatalogueFetchError as exc:
        log.warning("could not download all of %s for %s: %s", entry.repo, entry.id, exc)
        return False
    return True
