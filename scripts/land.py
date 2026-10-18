"""Land a lane's branch on main: merge main in, check, fast-forward main, push.

Parallel lanes work in their own worktrees and land after every verified step, so no
branch drifts far from main and nobody has to merge piles of branches by hand.
One lane lands at a time: a lock directory in the repository's git dir serialises them,
and the checks run under the lock, so every commit main gains has passed them.

Run it from the lane's worktree, with the main checkout's copy of this script:

    uv run python <main checkout>/scripts/land.py
    uv run python <main checkout>/scripts/land.py --playtest   # lessons could be affected

Exit codes: 0 landed (or nothing to land), 1 not ready (dirty tree, on main),
2 main conflicts with the branch, 3 a check failed, 4 the lock was not free in time.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STALE_AFTER_S = 50 * 60  # a lock older than this belonged to a lane that died while landing
POLL_S = 15
# iCloud's conflict copies ("manager 2.py") are untracked noise, never part of a lane's work
ICLOUD_COPY = re.compile(r'^\?\? "?.* [0-9]+(\.[^/" ]*)?"?$')
PUSH_TRIES = 3


def git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=check, text=True, capture_output=True, encoding="utf-8")


def main_checkout(cwd: Path) -> Path:
    """The worktree that has main checked out."""
    listing = git("worktree", "list", "--porcelain", cwd=cwd).stdout
    path = None
    for line in listing.splitlines():
        if line.startswith("worktree "):
            path = Path(line.removeprefix("worktree "))
        elif line == "branch refs/heads/main" and path is not None:
            return path
    raise SystemExit("land: no worktree has main checked out")


def set_aside_icloud_refs(git_dir: Path) -> None:
    """iCloud copies ref files ("main 3"); git then fails on the bad name, so move them out of refs."""
    aside = git_dir / "icloud-copies"
    for top in ("refs", "logs"):
        for path in (git_dir / top).rglob("*") if (git_dir / top).exists() else ():
            if path.is_file() and re.search(r" [0-9]+$", path.name):
                target = aside / path.relative_to(git_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                path.replace(target)
                print(f"land: set aside the iCloud copy {path.relative_to(git_dir)}", flush=True)


class Lock:
    def __init__(self, git_dir: Path, owner: str, wait_s: float) -> None:
        self.path = git_dir / "land.lock"
        self.owner = owner
        self.wait_s = wait_s

    def __enter__(self) -> Lock:
        deadline = time.monotonic() + self.wait_s
        while True:
            try:
                self.path.mkdir()
            except FileExistsError:
                age = time.time() - self.path.stat().st_mtime
                if age > STALE_AFTER_S:
                    print(f"land: breaking a stale lock ({age / 60:.0f} min old)", flush=True)
                    shutil.rmtree(self.path, ignore_errors=True)
                    continue
                if time.monotonic() > deadline:
                    raise SystemExit(4) from None
                holder = (self.path / "owner").read_text(encoding="utf-8") if (self.path / "owner").exists() else "?"
                print(f"land: waiting for {holder.strip()} to land", flush=True)
                time.sleep(POLL_S)
                continue
            (self.path / "owner").write_text(f"{self.owner} pid {os.getpid()}\n", encoding="utf-8")
            return self

    def __exit__(self, *exc: object) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def run_checks(cwd: Path, playtest: bool) -> None:
    checks = [
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "ruff", "format", "--check", "."],
        ["uv", "run", "pytest", "-q", "-x"],
        ["uv", "run", "python", "scripts/smoke.py"],
    ]
    if playtest:
        checks.append(["uv", "run", "python", "scripts/playtest.py"])
    for command in checks:
        print("land: " + " ".join(command), flush=True)
        started = time.monotonic()
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, encoding="utf-8")
        print(f"land:   {time.monotonic() - started:.0f} s, exit {result.returncode}", flush=True)
        if result.returncode != 0:
            print((result.stdout + result.stderr)[-6000:])
            print("land: a check failed; main is unchanged. Fix it on the branch, commit, land again.")
            raise SystemExit(3)


def push(main: Path) -> None:
    for attempt in range(1, PUSH_TRIES + 1):
        result = git("push", "origin", "main", cwd=main, check=False)
        if result.returncode == 0:
            return
        print(f"land: push failed (try {attempt}): {result.stderr.strip()[-300:]}", flush=True)
        time.sleep(10 * attempt)
    print("land: main moved locally but is not pushed; the next landing pushes it")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--playtest", action="store_true", help="also play every lesson (scripts/playtest.py)")
    parser.add_argument("--wait-min", type=float, default=120, help="how long to wait for the lock")
    args = parser.parse_args()

    here = Path(git("rev-parse", "--show-toplevel", cwd=Path.cwd()).stdout.strip())
    branch = git("branch", "--show-current", cwd=here).stdout.strip()
    if branch in ("", "main"):
        raise SystemExit("land: run this from a lane's worktree, on its branch, not on main")
    status = git("status", "--porcelain", cwd=here).stdout.splitlines()
    dirty = [line for line in status if not ICLOUD_COPY.search(line)]
    if dirty:
        print("land: the worktree has uncommitted changes; commit them first:\n" + "\n".join(dirty[:20]))
        raise SystemExit(1)
    main_tree = main_checkout(here)
    if git("status", "--porcelain", "--untracked-files=no", cwd=main_tree).stdout.strip():
        raise SystemExit(f"land: the main checkout {main_tree} has uncommitted changes; land refuses to move it")
    git_dir = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=here).stdout.strip())

    with Lock(git_dir, branch, args.wait_min * 60):
        set_aside_icloud_refs(git_dir)
        merged = git("merge", "main", "--no-edit", cwd=here, check=False)
        if merged.returncode != 0:
            conflicts = git("diff", "--name-only", "--diff-filter=U", cwd=here).stdout.split()
            git("merge", "--abort", cwd=here, check=False)
            print("land: main conflicts with this branch in: " + ", ".join(conflicts or ["(see git merge main)"]))
            print("land: run `git merge main`, resolve by reading both sides, commit, then land again.")
            raise SystemExit(2)
        ahead = git("rev-list", "--count", f"main..{branch}", cwd=here).stdout.strip()
        if ahead == "0":
            print("land: nothing to land; the branch is already on main")
            return
        run_checks(here, args.playtest)
        git("merge", "--ff-only", branch, cwd=main_tree)
        push(main_tree)
        head = git("log", "-1", "--format=%h %s", "main", cwd=main_tree).stdout.strip()
        print(f"land: landed {ahead} commits from {branch}; main is at {head}")


if __name__ == "__main__":
    sys.exit(main())
