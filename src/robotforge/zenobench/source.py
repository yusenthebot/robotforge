"""Where the ZenoBench checkout lives, and how to fetch it (git plus Git LFS for the USD scenes).

ZenoBench has no licence of its own, so robotforge ships none of its code or assets: the user's machine
fetches the upstream repository at a pinned commit, and robotforge runs it from there.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO = "https://github.com/CMU-Agentic-Library/ZenoBench.git"
COMMIT = "d573467e50f50545596dcd40ffd44e93a535c7af"  # 2026-10-26, verified on MuJoCo


class SourceError(RuntimeError):
    pass


def root() -> Path:
    """$ZENOBENCH_ROOT, else the fetched checkout in robotforge's cache."""
    if os.environ.get("ZENOBENCH_ROOT"):
        return Path(os.environ["ZENOBENCH_ROOT"]).expanduser().resolve()
    from robotforge.design.menagerie import cache_root

    return cache_root() / "zenobench" / f"ZenoBench-{COMMIT[:12]}"


def check(path: Path | None = None) -> Path:
    """The checkout, verified to hold real USD files (not Git LFS pointers)."""
    path = path or root()
    marker = path / "zeno_skills" / "skills.py"
    if not marker.exists():
        raise SourceError(f"no ZenoBench checkout at {path}; run `robotforge zenobench fetch` or set ZENOBENCH_ROOT")
    usd = path / "usd" / "house_wideopen.usd"
    if usd.exists() and usd.read_bytes()[:7] == b"version":
        raise SourceError(f"{path} holds Git LFS pointers, not the scenes; run `git lfs pull` there")
    return path


def fetch(commit: str = COMMIT, target: Path | None = None) -> Path:
    """Clone ZenoBench at `commit` with its LFS assets (about 1 GB), or reuse a finished checkout."""
    target = target or root()
    try:
        return check(target)
    except SourceError:
        pass
    git = shutil.which("git")
    if git is None:
        raise SourceError("git is not installed")
    if subprocess.run([git, "lfs", "version"], capture_output=True).returncode != 0:
        raise SourceError("Git LFS is not installed (macOS: brew install git-lfs)")
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        _run([git, "clone", "--no-checkout", REPO, str(target)])
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    _run([git, "-C", str(target), "checkout", "-q", commit], env=env)
    _run([git, "-C", str(target), "lfs", "install", "--local"])
    _run([git, "-C", str(target), "config", "lfs.concurrenttransfers", "3"])
    for _ in range(4):  # GitHub's LFS endpoint drops large batches now and then
        if subprocess.run([git, "-C", str(target), "lfs", "pull"]).returncode == 0:
            break
    return check(target)


def _run(argv: list[str], env: dict | None = None) -> None:
    result = subprocess.run(argv, env=env)
    if result.returncode != 0:
        raise SourceError(f"failed: {' '.join(argv)}")
