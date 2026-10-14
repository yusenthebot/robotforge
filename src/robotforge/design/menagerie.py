"""Fetch robot model folders from MuJoCo Menagerie, pinned to one commit, on first use.

Two routes give the same result. With a recent git installed, a shallow, sparse
checkout downloads just the folder in a few requests. Without git (common on
Windows, and on Macs without the command line tools) each file is downloaded
over HTTPS from raw.githubusercontent.com, using the file list in
menagerie_files.json, so no rate-limited GitHub API call is needed. Both routes
check every file against its git hash and move the finished folder into the
cache in one step, so a half-finished download never looks complete.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

REPOSITORY = "google-deepmind/mujoco_menagerie"
# The commit robot_descriptions 3.3.0 pins. Changing it means regenerating menagerie_files.json
# (python -m robotforge.design.menagerie <checkout>) and re-running the network tests.
COMMIT = "feadf76d42f8a2162426f7d226a3b539556b3bf5"
GIT_URL = f"https://github.com/{REPOSITORY}.git"
RAW_HOST = "https://raw.githubusercontent.com"
MANIFEST = Path(__file__).with_name("menagerie_files.json")

MIN_GIT = (2, 35)  # `sparse-checkout set --cone`
GIT_TIMEOUT = 900  # s, a hard cap; a stalled link fails after 30 s (see GIT_OPTIONS)
GIT_OPTIONS = [
    *("-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=30"),  # below 1 kB/s for 30 s: give up
    *("-c", "core.autocrlf=false", "-c", "core.eol=lf"),  # byte-exact files, so their hashes check out
]
HTTP_TIMEOUT = 30  # s without data before a request fails
HTTP_WORKERS = 6
ATTEMPTS = 3  # proxies and flaky Wi-Fi drop connections now and then
RENAME_ATTEMPTS = 6  # Windows: antivirus and the search indexer briefly hold new files open
STALE_FETCH = 3600  # s; leftovers of interrupted downloads older than this are removed

Progress = Callable[[float, str], None]  # (fraction done 0..1, what is happening)

_TRANSIENT = re.compile(
    r"could not resolve|unable to access|ssl|tls|timed out|timeout|connection|early eof|rpc failed|"
    r"too slow|hung up|remote end|\b50[234]\b|recv failure|empty reply|network|proxy",
    re.IGNORECASE,
)
_MISSING = re.compile(r"not our ref|couldn't find remote ref|not found|does not exist|no such", re.IGNORECASE)


class MenagerieError(ValueError):
    """User-facing: says what went wrong and what to do about it."""


class FileEntry(NamedTuple):
    path: str  # relative to the robot's folder, with forward slashes
    size: int
    sha: str  # git blob hash


# ---- what to download ------------------------------------------------------------------------


@functools.cache
def _manifest() -> dict[str, list[FileEntry]]:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data["commit"] != COMMIT:
        raise MenagerieError(f"{MANIFEST.name} lists files for {data['commit'][:12]}, not the pinned {COMMIT[:12]}")
    return {folder: [FileEntry(*f) for f in files] for folder, files in data["folders"].items()}


def files(folder: str) -> list[FileEntry]:
    try:
        return _manifest()[folder]
    except KeyError:
        raise MenagerieError(f"{folder!r} is not in robotforge's list of Menagerie folders") from None


def download_size(folder: str) -> int:
    """Bytes a folder takes on disk once downloaded."""
    return sum(f.size for f in files(folder))


# ---- where it lives -----------------------------------------------------------------------------


def cache_root() -> Path:
    """$ROBOTFORGE_CACHE, else the platform's user cache directory."""
    if os.environ.get("ROBOTFORGE_CACHE"):
        return Path(os.environ["ROBOTFORGE_CACHE"]).expanduser()
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "robotforge" / "cache"
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base).expanduser() / "robotforge"


def _menagerie_dir() -> Path:
    return cache_root() / f"menagerie-{COMMIT[:12]}"


def local_folder(folder: str) -> Path | None:
    """Where a Menagerie folder already is on disk, without downloading anything."""
    override = os.environ.get("ROBOTFORGE_MENAGERIE")  # any Menagerie checkout, for offline use and tests
    if override:
        path = Path(override).expanduser() / folder
        return path if path.is_dir() else None
    # A full clone made by the robot_descriptions package at the same commit is just as good.
    full = Path(os.environ.get("ROBOT_DESCRIPTIONS_CACHE", "~/.cache/robot_descriptions")).expanduser()
    full = full / "mujoco_menagerie"
    head = full / ".git" / "HEAD"
    if (full / folder).is_dir() and head.is_file() and head.read_text(errors="replace").strip() == COMMIT:
        return full / folder
    ours = _menagerie_dir() / folder
    return ours if ours.is_dir() else None


# ---- fetching -----------------------------------------------------------------------------------


def fetch(folder: str, progress: Progress | None = None) -> Path:
    """The folder on disk, downloading it first if needed."""
    found = local_folder(folder)
    if found is not None:
        return found
    if os.environ.get("ROBOTFORGE_MENAGERIE"):
        raise MenagerieError(f"{folder} is not in ROBOTFORGE_MENAGERIE={os.environ['ROBOTFORGE_MENAGERIE']}")
    report = progress or (lambda _f, _m: None)
    files(folder)  # fail early for a folder we know nothing about
    root = _menagerie_dir()
    root.mkdir(parents=True, exist_ok=True)
    _sweep_stale(root)
    target = root / folder
    log.info("downloading %s from MuJoCo Menagerie (first use only)", folder)
    # Cleanup errors are ignored: files held open on Windows are swept on a later download.
    with tempfile.TemporaryDirectory(dir=root, prefix=".fetch-", ignore_cleanup_errors=True) as tmp:
        fetched = _download(folder, Path(tmp), report)
        _move_into_place(fetched, target)
    report(1.0, "done")
    return target


def _download(folder: str, work: Path, report: Progress) -> Path:
    git = usable_git()
    git_error: MenagerieError | None = None
    if git is not None:
        try:
            fetched = _git_fetch(git, folder, work / "git", report)
            _verify(folder, fetched)
            return fetched
        except MenagerieError as exc:  # git may lack the proxy settings the system has; try plain HTTPS
            log.warning("git download failed, trying direct download: %s", exc)
            git_error = exc
    try:
        fetched = _http_fetch(folder, work / "http", report)
        _verify(folder, fetched)
        return fetched
    except MenagerieError as exc:
        if git_error is not None:
            raise MenagerieError(f"{exc} (git also failed: {git_error})") from exc
        if git is None:
            raise MenagerieError(f"{exc}. {_git_hint()}") from exc
        raise


def _git_hint() -> str:
    if sys.platform == "darwin":
        how = "run `xcode-select --install`"
    elif sys.platform == "win32":
        how = "install it from https://git-scm.com"
    else:
        how = "install it with your package manager"
    return f"Installing git gives robotforge a second, faster way to download ({how})"


def _verify(folder: str, path: Path) -> None:
    for entry in files(folder):
        file = path / entry.path
        if not file.is_file():
            raise MenagerieError(f"the download of {folder} is missing {entry.path}; please try again")
        if _git_hash(file.read_bytes()) != entry.sha:
            raise MenagerieError(f"the download of {folder} has a damaged file ({entry.path}); please try again")


def _git_hash(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def _move_into_place(source: Path, target: Path) -> None:
    for attempt in range(RENAME_ATTEMPTS):
        try:
            source.rename(target)
            return
        except OSError as exc:
            if target.is_dir():  # another download of the same robot finished first; folders move whole
                return
            if attempt == RENAME_ATTEMPTS - 1:
                raise MenagerieError(
                    f"could not move the downloaded robot into {target.parent} ({exc.strerror or exc}). "
                    "Another program (often antivirus) may be scanning the new files; try again in a moment."
                ) from exc
            time.sleep(0.2 * 2**attempt)


def _sweep_stale(root: Path) -> None:
    cutoff = time.time() - STALE_FETCH
    for leftover in root.glob(".fetch-*"):
        try:
            if leftover.stat().st_mtime < cutoff:
                shutil.rmtree(leftover, onexc=_make_writable_and_retry)
        except OSError as exc:  # still in use or not ours to remove: leave it for next time
            log.debug("could not remove %s: %s", leftover, exc)


def _make_writable_and_retry(func, path, _exc) -> None:
    os.chmod(path, 0o700)  # git writes read-only pack files, which Windows refuses to delete
    func(path)


# ---- route 1: git -------------------------------------------------------------------------------


@functools.cache
def usable_git() -> str | None:
    """Path of a git that can do sparse checkouts, or None (then files are downloaded one by one)."""
    git = shutil.which("git")
    if git is None:
        return None
    if sys.platform == "darwin" and Path(git).resolve() == Path("/usr/bin/git"):
        # Apple's placeholder git pops up an installer dialog unless the command line tools exist.
        try:
            tools = subprocess.run(["xcode-select", "-p"], capture_output=True, stdin=subprocess.DEVNULL, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if tools.returncode != 0:
            log.info("git needs the command line tools here (xcode-select --install); downloading without git")
            return None
    try:
        done = subprocess.run(
            [git, "--version"], capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"(\d+)\.(\d+)", done.stdout)
    if done.returncode != 0 or match is None or (int(match[1]), int(match[2])) < MIN_GIT:
        log.info("git %s is too old for sparse checkouts; downloading without git", done.stdout.strip())
        return None
    return git


def _git_fetch(git: str, folder: str, work: Path, report: Progress) -> Path:
    _git(git, ["init", "-q", str(work)])
    _git(git, ["-C", str(work), "remote", "add", "origin", GIT_URL])
    _git(git, ["-C", str(work), "sparse-checkout", "set", "--cone", folder])
    report(0.05, "looking up the files")
    _git(git, ["-C", str(work), "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT], network=True)
    report(0.3, "downloading the files")
    # The checkout is what downloads the folder's files: the clone above holds only their names.
    _git(git, ["-C", str(work), "checkout", "-q", "FETCH_HEAD"], network=True)
    return work / folder


def _git(git: str, args: list[str], network: bool = False) -> None:
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",  # never wait for a password nobody can type
        "GIT_ASKPASS": "",
        "SSH_ASKPASS": "",
        "GCM_INTERACTIVE": "never",
        "LC_ALL": "C",  # English messages, which the error checks below understand
    }
    attempts = ATTEMPTS if network else 1
    for attempt in range(1, attempts + 1):
        try:
            # stdin is closed and output captured: on a stdio MCP server both belong to the protocol.
            done = subprocess.run(
                [git, *GIT_OPTIONS, *args],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=GIT_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MenagerieError("the download timed out; check your internet connection and try again") from exc
        except OSError as exc:
            raise MenagerieError(f"could not run git: {exc}") from exc
        if done.returncode == 0:
            return
        detail = " / ".join((done.stderr or done.stdout).strip().splitlines()[-3:]) or f"exit code {done.returncode}"
        if _MISSING.search(detail):  # retrying will not make it appear
            raise MenagerieError(f"MuJoCo Menagerie does not have this robot at {COMMIT[:12]}: {detail}")
        if not (network and _TRANSIENT.search(detail)):
            raise MenagerieError(f"git failed: {detail}")
        if attempt < attempts:
            log.warning("download failed (attempt %d of %d), retrying: %s", attempt, attempts, detail)
            time.sleep(2.0 * attempt)
    raise MenagerieError(
        f"could not reach GitHub (MuJoCo Menagerie); check your internet connection or proxy. git said: {detail}"
    )


# ---- route 2: plain HTTPS -----------------------------------------------------------------------


def _http_fetch(folder: str, work: Path, report: Progress) -> Path:
    entries = files(folder)
    total = max(1, sum(f.size for f in entries))
    dest = work / folder
    done = 0
    lock = threading.Lock()

    def one(entry: FileEntry) -> None:
        nonlocal done
        path = (dest / entry.path).resolve()
        if not path.is_relative_to(dest.resolve()):
            raise MenagerieError(f"refusing to write outside the download folder: {entry.path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{RAW_HOST}/{REPOSITORY}/{COMMIT}/{folder}/{urllib.parse.quote(entry.path)}"
        path.write_bytes(_get(url, entry))
        with lock:
            done += entry.size
            report(0.02 + 0.97 * done / total, f"downloaded {done / 1e6:.1f} of {total / 1e6:.1f} MB")

    report(0.02, f"downloading {len(entries)} files")
    with ThreadPoolExecutor(max_workers=HTTP_WORKERS) as pool:
        list(pool.map(one, entries))  # re-raises the first failure
    return dest


def _get(url: str, entry: FileEntry) -> bytes:
    """One file, retried on dropped connections and checked against its git hash."""
    problem = ""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "robotforge"})
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:  # noqa: S310 - fixed https URL
                data = response.read()
            if _git_hash(data) == entry.sha:
                return data
            problem = f"{entry.path} arrived damaged ({len(data)} of {entry.size} bytes)"
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):  # retrying will not make it appear
                missing = f"GitHub has no {entry.path} at {COMMIT[:12]}; robotforge needs an update"
                raise MenagerieError(missing) from exc
            if exc.code == 403:
                raise MenagerieError(
                    f"GitHub refused the download (HTTP 403); a proxy or firewall may block {RAW_HOST}"
                ) from exc
            problem = f"HTTP {exc.code} for {entry.path}"  # 429 and 5xx pass with time
        except (urllib.error.URLError, HTTPException, TimeoutError, ConnectionError, ssl.SSLError) as exc:
            problem = f"{getattr(exc, 'reason', exc)}"
        if attempt < ATTEMPTS:
            log.warning("download failed (attempt %d of %d), retrying: %s", attempt, ATTEMPTS, problem)
            time.sleep(2.0 * attempt)
    raise MenagerieError(
        f"could not download from GitHub (MuJoCo Menagerie); check your internet connection or proxy ({problem})"
    )


# ---- maintenance ----------------------------------------------------------------------------------


def write_manifest(checkout: Path, folders: list[str]) -> None:
    """Regenerate menagerie_files.json from a Menagerie checkout at COMMIT (macOS or Linux line endings)."""
    out = {"repository": REPOSITORY, "commit": COMMIT, "folders": {}}
    for folder in folders:
        root = checkout / folder
        entries = sorted(p for p in root.rglob("*") if p.is_file())
        out["folders"][folder] = [
            [p.relative_to(root).as_posix(), p.stat().st_size, _git_hash(p.read_bytes())] for p in entries
        ]
    lines = [f'{{"repository": "{REPOSITORY}", "commit": "{COMMIT}", "folders": {{']
    for i, (folder, entries) in enumerate(out["folders"].items()):
        rows = ",\n".join(f"  {json.dumps(e)}" for e in entries)
        lines.append(f'"{folder}": [\n{rows}\n]{"," if i < len(folders) - 1 else ""}')
    lines.append("}}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    from robotforge.design.library import CATALOG

    write_manifest(Path(sys.argv[1]), [robot.folder for robot in CATALOG.values()])
    print(f"wrote {MANIFEST}")
