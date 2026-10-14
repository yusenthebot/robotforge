"""Downloading robot folders from MuJoCo Menagerie: git route, plain HTTPS route, and their failures.

Offline: the git route runs against a local git repository made from tests/fixtures/menagerie,
and the HTTPS route against a fake urlopen. ROBOTFORGE_NETWORK_TESTS=1 also downloads the real
Go2 by both routes into a temporary cache and prints how long each took.
"""

import io
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

from robotforge.design import library, menagerie

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"
NETWORK = os.environ.get("ROBOTFORGE_NETWORK_TESTS") == "1"
HAS_GIT = shutil.which("git") is not None


def _entries(folder: str) -> list[menagerie.FileEntry]:
    root = FIXTURES / folder
    return [
        menagerie.FileEntry(p.relative_to(root).as_posix(), p.stat().st_size, menagerie._git_hash(p.read_bytes()))
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty cache, no offline checkout, and a file list made from the fixtures."""
    monkeypatch.delenv("ROBOTFORGE_MENAGERIE", raising=False)
    monkeypatch.setenv("ROBOTFORGE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path / "no_full_clone"))
    manifest = {folder: _entries(folder) for folder in ("unitree_go2", "trs_so_arm100")}
    monkeypatch.setattr(menagerie, "_manifest", lambda: manifest)
    monkeypatch.setattr(menagerie.time, "sleep", lambda _s: None)
    real_usable_git = menagerie.usable_git  # tests may swap it out; clear the real one's memo either way
    real_usable_git.cache_clear()
    yield tmp_path / "cache"
    real_usable_git.cache_clear()


def _fake_remote(tmp_path: Path) -> tuple[str, str]:
    """A local git repository shaped like Menagerie; returns (url, commit)."""
    repo = tmp_path / "remote"
    shutil.copytree(FIXTURES, repo)
    git = ["git", "-C", str(repo)]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for key, value in [
        ("user.email", "test@example.com"),
        ("user.name", "test"),
        ("uploadpack.allowFilter", "true"),
        ("uploadpack.allowAnySHA1InWant", "true"),
    ]:
        subprocess.run([*git, "config", key, value], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "models"], check=True)
    commit = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    return repo.as_uri(), commit


# ---- route 1: git ---------------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GIT or menagerie.usable_git() is None, reason="needs git >= 2.35")
def test_git_route_fetches_only_the_requested_folder(cache, tmp_path, monkeypatch):
    url, commit = _fake_remote(tmp_path)
    monkeypatch.setattr(menagerie, "GIT_URL", url)
    monkeypatch.setattr(menagerie, "COMMIT", commit)
    seen = {}
    move = menagerie._move_into_place

    def inspect_then_move(source, target):
        seen["checkout"] = sorted(p.relative_to(source.parent).as_posix() for p in source.parent.rglob("*"))
        move(source, target)

    monkeypatch.setattr(menagerie, "_move_into_place", inspect_then_move)
    steps = []
    folder = menagerie.fetch("unitree_go2", lambda f, m: steps.append(f))

    assert (folder / "go2.xml").is_file()
    assert folder == cache / f"menagerie-{commit[:12]}" / "unitree_go2"
    assert not any(p.startswith("trs_so_arm100") for p in seen["checkout"])  # the other robot never came down
    assert sorted(p.name for p in folder.parent.iterdir()) == ["unitree_go2"]  # no leftovers
    assert steps[-1] == 1.0
    assert library.is_downloaded("go2")

    monkeypatch.setattr(menagerie, "_git", lambda *_a, **_k: pytest.fail("a cached robot is not downloaded again"))
    assert menagerie.fetch("unitree_go2") == folder


def test_git_runs_without_prompts_and_gives_up_on_dead_links(monkeypatch):
    calls = []
    monkeypatch.setattr(
        menagerie.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)) or subprocess.CompletedProcess(cmd, 0)
    )
    menagerie._git("git", ["fetch"], network=True)
    cmd, kw = calls[0]
    assert kw["stdin"] is subprocess.DEVNULL  # never reads the MCP server's stdin
    assert kw["env"]["GIT_TERMINAL_PROMPT"] == "0" and kw["env"]["GIT_ASKPASS"] == ""
    assert kw["env"]["LC_ALL"] == "C" and kw["errors"] == "replace"
    assert "http.lowSpeedTime=30" in cmd and "core.autocrlf=false" in cmd


def test_dropped_connections_are_retried_but_missing_commits_are_not(monkeypatch):
    monkeypatch.setattr(menagerie.time, "sleep", lambda _s: None)

    def replies(*results):
        it = iter(results)
        calls = []

        def run(cmd, **_kw):
            calls.append(cmd)
            code, err = next(it)
            return subprocess.CompletedProcess(cmd, code, "", err)

        monkeypatch.setattr(menagerie.subprocess, "run", run)
        return calls

    ssl_drop = (128, "fatal: unable to access 'https://github.com/x/': SSL_ERROR_SYSCALL")
    replies(ssl_drop, (0, ""))
    menagerie._git("git", ["fetch"], network=True)  # second attempt succeeds

    calls = replies(ssl_drop, ssl_drop, ssl_drop)
    with pytest.raises(menagerie.MenagerieError, match="internet connection or proxy.*SSL_ERROR_SYSCALL"):
        menagerie._git("git", ["fetch"], network=True)
    assert len(calls) == menagerie.ATTEMPTS

    calls = replies((128, "fatal: remote error: upload-pack: not our ref deadbeef"))
    with pytest.raises(menagerie.MenagerieError, match="does not have this robot"):
        menagerie._git("git", ["fetch"], network=True)
    assert len(calls) == 1

    calls = replies((1, "error: unknown option `cone'"))
    with pytest.raises(menagerie.MenagerieError, match="git failed"):
        menagerie._git("git", ["sparse-checkout"])
    assert len(calls) == 1


def test_a_git_timeout_becomes_a_plain_message(monkeypatch):
    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(menagerie.subprocess, "run", hang)
    with pytest.raises(menagerie.MenagerieError, match="timed out; check your internet connection"):
        menagerie._git("git", ["fetch"], network=True)


@pytest.mark.parametrize(
    ("platform", "version", "xcode_ok", "usable"),
    [
        ("linux", "git version 2.43.0", True, True),
        ("linux", "git version 2.20.1", True, False),  # too old for sparse-checkout set --cone
        ("darwin", "git version 2.39.5 (Apple Git-154)", True, True),
        ("darwin", "", False, False),  # Apple's placeholder without command line tools: never run it
    ],
)
def test_only_a_capable_git_is_used(monkeypatch, platform, version, xcode_ok, usable):
    ran = []

    def run(cmd, **_kw):
        ran.append(cmd[0])
        if cmd[0] == "xcode-select":
            return subprocess.CompletedProcess(cmd, 0 if xcode_ok else 2, b"", b"")
        return subprocess.CompletedProcess(cmd, 0, version, "")

    monkeypatch.setattr(menagerie.sys, "platform", platform)
    monkeypatch.setattr(menagerie.shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(menagerie.subprocess, "run", run)
    menagerie.usable_git.cache_clear()
    try:
        assert (menagerie.usable_git() is not None) == usable
    finally:
        menagerie.usable_git.cache_clear()
    if platform == "darwin" and not xcode_ok:
        assert "/usr/bin/git" not in ran  # would pop up Apple's installer dialog


# ---- route 2: plain HTTPS -----------------------------------------------------------------------


class _FakeGitHub:
    """Serves fixture files for raw.githubusercontent.com URLs; can drop or damage some requests."""

    def __init__(self, drop_first: str | None = None, damage: str | None = None):
        self.drop_first, self.damage = drop_first, damage
        self.urls: list[str] = []

    def __call__(self, request, timeout):
        url = request.full_url
        self.urls.append(url)
        folder, rel = url.split(f"/{menagerie.COMMIT}/", 1)[1].split("/", 1)
        rel = urllib.parse.unquote(rel)
        if rel == self.drop_first:
            self.drop_first = None
            raise urllib.error.URLError(ConnectionResetError("connection reset by peer"))
        data = (FIXTURES / folder / rel).read_bytes()
        if rel == self.damage:
            data = data[: len(data) // 2]  # a proxy cut the response short
        return io.BytesIO(data)


def test_without_git_files_come_one_by_one_and_are_checked(cache, monkeypatch):
    monkeypatch.setattr(menagerie, "usable_git", lambda: None)
    github = _FakeGitHub(drop_first="go2.xml")
    monkeypatch.setattr(menagerie.urllib.request, "urlopen", github)
    fractions = []
    folder = menagerie.fetch("unitree_go2", lambda f, m: fractions.append(f))
    assert (folder / "go2.xml").read_bytes() == (FIXTURES / "unitree_go2" / "go2.xml").read_bytes()
    assert all(f"/{menagerie.COMMIT}/unitree_go2/" in url for url in github.urls)  # only this robot, pinned
    assert sum(url.endswith("/go2.xml") for url in github.urls) == 2  # the dropped request was retried
    assert fractions == sorted(fractions) and fractions[-1] == 1.0


def test_damaged_downloads_are_rejected(cache, monkeypatch):
    monkeypatch.setattr(menagerie, "usable_git", lambda: None)
    monkeypatch.setattr(menagerie.urllib.request, "urlopen", _FakeGitHub(damage="go2.xml"))
    with pytest.raises(menagerie.MenagerieError, match="go2.xml arrived damaged.*Installing git"):
        menagerie.fetch("unitree_go2")
    assert not (cache / f"menagerie-{menagerie.COMMIT[:12]}" / "unitree_go2").exists()


def test_a_failing_git_falls_back_to_https(cache, monkeypatch):
    monkeypatch.setattr(menagerie, "usable_git", lambda: "git")

    def git_fails(*_args):
        raise menagerie.MenagerieError("could not reach GitHub (MuJoCo Menagerie)")

    monkeypatch.setattr(menagerie, "_git_fetch", git_fails)
    monkeypatch.setattr(menagerie.urllib.request, "urlopen", _FakeGitHub())
    assert (menagerie.fetch("unitree_go2") / "go2.xml").is_file()


def test_github_blocking_downloads_is_explained(cache, monkeypatch):
    def forbidden(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(menagerie, "usable_git", lambda: None)
    monkeypatch.setattr(menagerie.urllib.request, "urlopen", forbidden)
    with pytest.raises(menagerie.MenagerieError, match="proxy or firewall"):
        menagerie.fetch("unitree_go2")


# ---- moving into the cache --------------------------------------------------------------------------


def test_a_busy_folder_is_retried_then_explained(tmp_path, monkeypatch):
    source, target = tmp_path / "new", tmp_path / "cache" / "unitree_go2"
    source.mkdir()
    target.parent.mkdir()
    tries = []

    def busy(self, _target):
        tries.append(1)
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(menagerie.time, "sleep", lambda _s: None)
    monkeypatch.setattr(Path, "rename", busy)
    with pytest.raises(menagerie.MenagerieError, match="antivirus"):
        menagerie._move_into_place(source, target)
    assert len(tries) == menagerie.RENAME_ATTEMPTS


def test_losing_the_race_to_another_download_is_fine(tmp_path, monkeypatch):
    source, target = tmp_path / "new", tmp_path / "cache" / "unitree_go2"
    source.mkdir()

    def someone_else_won(self, _target):
        target.mkdir(parents=True)  # the other process moved its complete copy in first
        raise OSError(66, "Directory not empty")

    monkeypatch.setattr(Path, "rename", someone_else_won)
    menagerie._move_into_place(source, target)  # no error
    assert target.is_dir()


def test_leftovers_of_interrupted_downloads_are_swept(tmp_path):
    old, fresh = tmp_path / ".fetch-old", tmp_path / ".fetch-fresh"
    for d in (old, fresh):
        (d / ".git").mkdir(parents=True)
        (d / ".git" / "pack").write_bytes(b"x")
        (d / ".git" / "pack").chmod(0o444)  # git's pack files are read-only
    hour_ago = time.time() - menagerie.STALE_FETCH - 60
    os.utime(old, (hour_ago, hour_ago))
    menagerie._sweep_stale(tmp_path)
    assert not old.exists() and fresh.exists()


def test_a_full_robot_descriptions_clone_is_reused(cache, tmp_path, monkeypatch):
    full = tmp_path / "rd" / "mujoco_menagerie"
    shutil.copytree(FIXTURES / "unitree_go2", full / "unitree_go2")
    (full / ".git").mkdir()
    (full / ".git" / "HEAD").write_text(menagerie.COMMIT + "\n")
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path / "rd"))
    monkeypatch.setattr(menagerie, "_download", lambda *_a: pytest.fail("must not download"))
    assert menagerie.fetch("unitree_go2") == full / "unitree_go2"


# ---- the real thing (opt-in) -------------------------------------------------------------------------


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download from GitHub")
@pytest.mark.parametrize("route", ["git", "https"])
def test_real_download_of_go2(route, tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOTFORGE_MENAGERIE", raising=False)
    monkeypatch.setenv("ROBOTFORGE_CACHE", str(tmp_path))
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path / "none"))
    if route == "git":
        if menagerie.usable_git() is None:
            pytest.skip("needs git >= 2.35")
    else:
        monkeypatch.setattr(menagerie, "usable_git", lambda: None)
    started = time.perf_counter()
    folder = menagerie.fetch("unitree_go2")
    seconds = time.perf_counter() - started
    print(f"\nGo2 by {route}: {menagerie.download_size('unitree_go2') / 1e6:.1f} MB in {seconds:.1f} s")
    menagerie._verify("unitree_go2", folder)  # every file matches its git hash at the pin
