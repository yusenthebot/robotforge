"""Save supported live selections and insert guarded, server-owned reusable records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from robotforge.design import arrangements as library
from robotforge.design.layouts import LayoutError
from robotforge.design.world import ObjectEntry
from robotforge.playground import PlaygroundError
from robotforge.sim.execution import ExecutionError
from robotforge.viewer import object_group as group
from robotforge.viewer.object_edit import Digest

if TYPE_CHECKING:
    from robotforge.playground import Playground


class Empty(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Save(group.Prepare):
    name: str = Field(min_length=1, max_length=256)
    # Saving has no copy/move mode; the handler forbids the inherited field.


class Target(group.Guard):
    arrangement_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    sha256: Digest


@dataclass
class PreparedArrangement:
    store: object
    version: int
    generation: int
    world_sha256: str
    token: str
    arrangement_id: str
    sha256: str
    name: str
    root: Path
    root_identity: dict[Path, tuple[int, int, int]]
    anchor: tuple[float, float, float]
    members: tuple[group.Member, ...]
    mode: str = "arrangement"
    invalidated: bool = False


def _response(pg, entries, **extra):
    return {
        "available": True,
        "version": pg.session.version,
        "generation": pg.session.generation,
        "world_sha256": pg._play_world_sha256,
        "entries": entries,
        "limit": library.MAX_RECORDS,
        **extra,
    }


def _current(pg, request):
    from robotforge.viewer.object_edit import _guard as current

    if pg.store is None:
        raise library.ArrangementError("no_world", "Choose a map before using arrangements")
    return current(pg, pg.store, request.version, request.generation, request.world_sha256, "arrangement")


def _guard(pg, request):
    if pg.store is None:
        raise library.ArrangementError("no_world", "Choose a map before using arrangements")
    return group._check(pg, pg._require_store(), request)


def _destination(pg, prepared):
    """Consume an observed stale destination even if an outside writer later restores its bytes."""
    from robotforge.viewer.object_edit import _guard as current

    try:
        return current(pg, prepared.store, prepared.version, prepared.generation, prepared.world_sha256, "arrangement")
    except (ValueError, OSError) as exc:
        prepared.invalidated = True
        raise library.ArrangementError("stale", "The destination world changed; prepare the arrangement again") from exc


def _eligible_destination(pg, prepared, request):
    _destination(pg, prepared)
    try:
        return _guard(pg, request)
    except (PlaygroundError, LayoutError):
        # Eligibility failures keep their existing classification; an observed world change consumes the token.
        _destination(pg, prepared)
        raise


def _check_record(saved, prepared):
    try:
        if saved.root != prepared.root:
            raise library.ArrangementError("changed", "The arrangement library changed; prepare it again")
        saved.check(prepared.root_identity)
        return saved.record(prepared.arrangement_id, prepared.sha256)
    except (library.ArrangementError, OSError):
        prepared.invalidated = True
        raise


def _insert(pg, request):
    prepared = pg._play_group
    if (
        not isinstance(prepared, PreparedArrangement)
        or prepared.invalidated
        or (prepared.token, prepared.version, prepared.generation, prepared.world_sha256)
        != (request.token, request.version, request.generation, request.world_sha256)
    ):
        raise library.ArrangementError("stale", "This arrangement preparation expired; select it again")
    world = _eligible_destination(pg, prepared, request)
    with library.locked() as saved:
        _check_record(saved, prepared)
        entries = group._clones(world, prepared, request)
        if any(abs(entry.yaw) > 3600 for entry in entries):
            raise library.ArrangementError(
                "invalid_request", "The inserted orientation exceeds the supported full turns"
            )
        names = {entry.name for entry in entries}

        def check():
            _eligible_destination(pg, prepared, request)
            _check_record(saved, prepared)

        def change(candidate):
            check()
            taken = {
                obj.name for objects in (candidate.objects, candidate.robots, candidate.terrain) for obj in objects
            }
            if names & taken:
                raise library.ArrangementError("changed", "The inserted prop names are no longer available")
            candidate.objects.extend(entries)

        def validate(candidate, compiled):
            check()
            try:
                group._validate_targets(pg, candidate, compiled, names, pos=request.pos)
            except group.OccupiedError as exc:
                raise library.ArrangementError("occupied", str(exc)) from exc
            except PlaygroundError as exc:
                raise library.ArrangementError("target", str(exc)) from exc
            check()

        try:
            return group._commit(
                pg, prepared.store, request, change, validate, entries, before_publish=check, check_current=check
            )
        except (ValueError, OSError):
            # The shared commit has its own world guards before/after our target validation.
            # Classify any destination staleness observed there before rollback can restore a token.
            _destination(pg, prepared)
            raise


def handle(pg: Playground, message: dict[str, Any]) -> dict[str, Any]:
    action = message.get("action")
    schema = {
        "arrangement_list": Empty,
        "arrangement_save": Save,
        "arrangement_remove": Target,
        "arrangement_prepare": Target,
        "arrangement_insert": group.Duplicate,
    }.get(action)
    if schema is None:
        raise library.ArrangementError("invalid_request", "Unknown arrangement action")
    try:
        values = {key: value for key, value in message.items() if key not in {"type", "action", "id"}}
        if action == "arrangement_save" and "mode" in values:
            raise library.ArrangementError("invalid_request", "Saving arrangements does not accept a mode")
        request = schema.model_validate(values)
    except ValidationError as exc:
        raise library.ArrangementError("invalid_request", f"Invalid arrangement request: {exc}") from exc
    try:
        with pg.session.execution.command("viewer_arrangements"), pg.lessons._lock, pg.session.lock:
            if action == "arrangement_list":
                if pg.store is None or pg.lessons.lesson is not None:
                    return {
                        **_response(pg, []),
                        "available": False,
                        "reason": "arrangement_lesson" if pg.lessons.lesson is not None else "arrangement_no_world",
                    }
                # Listing is available while running; verify only the authoritative saved-world guard.
                from robotforge.viewer.object_edit import _guard as current

                current(pg, pg.store, pg.session.version, pg.session.generation, pg._play_world_sha256, "arrangement")
                with library.locked() as saved:
                    return _response(pg, saved.entries())
            if action == "arrangement_insert":
                return _insert(pg, request)
            world = _current(pg, request) if action == "arrangement_remove" else _guard(pg, request)
            with library.locked() as saved:
                if action == "arrangement_save":
                    label = library.name(request.name)
                    _, _, members, anchor = group._capture(pg, request, world)
                    entries = []
                    for member in members:
                        yaw = member.entry.yaw + (member.yaw - member.entry.yaw + 180) % 360 - 180
                        entries.append(
                            {
                                **member.entry.model_dump(),
                                "pos": [member.pos[i] - anchor[i] for i in range(3)],
                                "yaw": yaw,
                            }
                        )
                    _guard(pg, request)
                    entry = saved.save(label, entries)
                    return _response(pg, saved.entries(), saved=entry)
                if action == "arrangement_remove":
                    saved.remove(request.arrangement_id, request.sha256)
                    return _response(pg, saved.entries())
                value = saved.record(request.arrangement_id, request.sha256)
                members = tuple(
                    group.Member(ObjectEntry.model_validate(entry), tuple(entry["pos"]), entry["yaw"], ())
                    for entry in value["members"]
                )
                _guard(pg, request)
                prepared = PreparedArrangement(
                    pg.store,
                    request.version,
                    request.generation,
                    request.world_sha256,
                    uuid4().hex,
                    request.arrangement_id,
                    request.sha256,
                    value["name"],
                    saved.root,
                    dict(saved.identities),
                    (0.0, 0.0, 0.0),
                    members,
                )
                pg._play_group = prepared
                return {
                    "mode": "arrangement",
                    "token": prepared.token,
                    "arrangement_id": prepared.arrangement_id,
                    "sha256": prepared.sha256,
                    "name": prepared.name,
                    "version": prepared.version,
                    "generation": prepared.generation,
                    "world_sha256": prepared.world_sha256,
                    "members": [
                        {
                            "name": m.entry.name,
                            "shape": m.entry.shape,
                            "size": m.entry.size,
                            "offset": list(m.pos),
                            "yaw": m.yaw,
                        }
                        for m in members
                    ],
                }
    except library.ArrangementError as exc:
        if (
            action == "arrangement_insert"
            and exc.code in {"arrangement_changed", "arrangement_damaged", "arrangement_storage"}
            and isinstance(pg._play_group, PreparedArrangement)
        ):
            pg._play_group.invalidated = True
        raise
    except ExecutionError as exc:
        raise library.ArrangementError("busy", str(exc)) from exc
    except (PlaygroundError, LayoutError) as exc:
        raise library.ArrangementError("invalid", str(exc)) from exc
    except OSError as exc:
        raise library.ArrangementError("storage", "The arrangement files could not be accessed safely") from exc
