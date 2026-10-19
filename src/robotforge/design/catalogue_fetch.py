"""Download a robot's folder from the GitHub repository it lives in, pinned to one commit, on first use.

The catalogue (catalogue.py) lists open robots from many repositories. Each download takes just the robot's folder:
with git a shallow, sparse checkout (a few requests), without git a streamed archive of the commit of which only that
folder is written to disk. A finished folder is moved into the cache in one step, so a half-finished download never
looks complete, and the folders of one repository share a directory, so a URDF finds its sibling packages.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from http.client import HTTPException
from pathlib import Path, PurePosixPath

from robotforge.design import menagerie

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://codeload.github.com/{owner}/{repo}/tar.gz/{commit}"
FETCH_ATTEMPTS = 3


class CatalogueFetchError(ValueError):
    """User-facing: says what went wrong and what to do about it."""


def repo_dir(name: str, commit: str) -> Path:
    """Where the downloaded folders of a repository at a commit sit, in the same layout as in the repository."""
    return menagerie.cache_root() / "catalogue" / f"{_safe(name)}-{commit[:12]}"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def folder_path(name: str, commit: str, folder: str) -> Path | None:
    """The folder on disk if it was downloaded (or is in a local checkout named by $ROBOTFORGE_REPOS), else None."""
    for root in _local_roots(name, commit):
        path = root / folder if folder else root
        if path.is_dir():
            return path
    return None


def _local_roots(name: str, commit: str) -> list[Path]:
    roots = [repo_dir(name, commit)]
    override = os.environ.get("ROBOTFORGE_REPOS")  # a folder of checkouts named like the repositories, for tests
    if override:
        roots.insert(0, Path(override).expanduser() / name)
    full = Path(os.environ.get("ROBOT_DESCRIPTIONS_CACHE", "~/.cache/robot_descriptions")).expanduser()
    for clone in full.glob("*"):  # a full clone the robot_descriptions package made, at the same commit
        head = clone / ".git" / "HEAD"
        try:
            if clone.name.replace("_", "-") == name.replace("_", "-") and head.read_text().strip() == commit:
                roots.append(clone)
        except OSError:
            continue
    return roots


def fetch(
    name: str, url: str, commit: str, folder: str, progress: menagerie.Progress | None = None, whole: bool = False
) -> Path:
    """The folder of a repository on disk (the repository itself for ""), downloading it first if needed."""
    found = None if whole else folder_path(name, commit, folder)
    if found is not None:
        return found
    if os.environ.get("ROBOTFORGE_REPOS") and (Path(os.environ["ROBOTFORGE_REPOS"]).expanduser() / name).is_dir():
        raise CatalogueFetchError(f"{folder or name} is not in ROBOTFORGE_REPOS={os.environ['ROBOTFORGE_REPOS']}")
    report = progress or (lambda _f, _m: None)
    root = repo_dir(name, commit)
    root.mkdir(parents=True, exist_ok=True)
    menagerie._sweep_stale(root.parent)
    target = root / folder if folder else root
    log.info("downloading %s from %s (first use only)", folder or name, url)
    wanted = "" if whole else folder
    with tempfile.TemporaryDirectory(dir=root.parent, prefix=".fetch-", ignore_cleanup_errors=True) as tmp:
        fetched = _download(url, commit, wanted, Path(tmp), report)
        _place(fetched, target if not whole else root)
    report(1.0, "done")
    return target


def _place(fetched: Path, target: Path) -> None:
    """Move a downloaded folder into the cache. A whole repository merges into what is there."""
    if target.is_dir() and any(target.iterdir()):
        shutil.copytree(fetched, target, dirs_exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_dir():
        target.rmdir()
    menagerie._move_into_place(fetched, target)


def _download(url: str, commit: str, folder: str, work: Path, report: menagerie.Progress) -> Path:
    git = menagerie.usable_git()
    git_error: CatalogueFetchError | None = None
    if git is not None:
        try:
            return _git_fetch(git, url, commit, folder, work / "git", report)
        except menagerie.MenagerieError as exc:  # git may lack the proxy settings the system has; try plain HTTPS
            log.warning("git download failed, trying direct download: %s", exc)
            git_error = CatalogueFetchError(str(exc))
    try:
        return _archive_fetch(url, commit, folder, work / "http", report)
    except CatalogueFetchError as exc:
        if git_error is not None:
            raise CatalogueFetchError(f"{exc} (git also failed: {git_error})") from exc
        if git is None:
            raise CatalogueFetchError(f"{exc}. {menagerie._git_hint()}") from exc
        raise


def _git_fetch(git: str, url: str, commit: str, folder: str, work: Path, report: menagerie.Progress) -> Path:
    menagerie._git(git, ["init", "-q", str(work)])
    menagerie._git(git, ["-C", str(work), "remote", "add", "origin", url])
    if folder:
        menagerie._git(git, ["-C", str(work), "sparse-checkout", "set", "--cone", folder])
    report(0.05, "looking up the files")
    menagerie._git(git, ["-C", str(work), "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit], True)
    report(0.3, "downloading the files")
    menagerie._git(git, ["-C", str(work), "checkout", "-q", "FETCH_HEAD"], True)
    shutil.rmtree(work / ".git", onexc=menagerie._make_writable_and_retry)
    return work / folder if folder else work


def _archive_fetch(url: str, commit: str, folder: str, work: Path, report: menagerie.Progress) -> Path:
    match = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if match is None:
        raise CatalogueFetchError(f"cannot download from {url} without git")
    archive = ARCHIVE_URL.format(owner=match[1], repo=match[2], commit=commit)
    out = work / folder if folder else work
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"{folder.strip('/')}/" if folder else ""
    last: Exception | None = None
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(archive, timeout=menagerie.HTTP_TIMEOUT) as response:  # noqa: S310
                report(0.1, "downloading the files")
                _extract(response, prefix, work)
            if not any(out.iterdir()):
                raise CatalogueFetchError(f"{url} has no folder {folder!r} at {commit[:12]}")
            return out
        except (urllib.error.URLError, HTTPException, OSError, tarfile.TarError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise CatalogueFetchError(f"{url} has no commit {commit[:12]}") from exc
            log.warning("download failed (attempt %d of %d): %s", attempt, FETCH_ATTEMPTS, exc)
            time.sleep(2.0 * attempt)
    raise CatalogueFetchError(f"could not reach GitHub ({url}); check your internet connection or proxy: {last}")


def _extract(stream, prefix: str, work: Path) -> None:
    """Write the files of a streamed .tar.gz that lie under `prefix` (inside the archive's one top folder)."""
    with tarfile.open(fileobj=stream, mode="r|gz") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts[1:]  # the archive's top folder is <repo>-<commit>
            relative = "/".join(parts)
            if not member.isfile() or not relative.startswith(prefix) or ".." in parts:
                continue
            file = tar.extractfile(member)
            if file is None:
                continue
            destination = work.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(file.read())
