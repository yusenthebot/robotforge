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
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STALE_AFTER_S = 50 * 60  # a lock older than this belonged to a lane that died while landing
POLL_S = 5
# The suite runs in this many processes at once; MuJoCo tests are CPU bound and other lanes share the machine.
# A test that took longer than this runs only when the branch touches its file or a module it imports.
HEAVY_S = 90
SHARDS = max(2, min(6, (os.cpu_count() or 4) // 3))
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


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def holder_pid(text: str) -> int | None:
    match = re.search(r"pid (\d+)", text)
    return int(match.group(1)) if match else None


class Lock:
    """One lane lands at a time, in the order the lanes asked (a ticket queue), so none starves."""

    def __init__(self, git_dir: Path, owner: str, wait_s: float) -> None:
        self.path = git_dir / "land.lock"
        self.queue = git_dir / "land.queue"
        self.owner = owner
        self.wait_s = wait_s
        self.ticket = self.queue / f"{time.time_ns()}-{os.getpid()}"

    def _first_in_line(self) -> bool:
        tickets = []
        for ticket in sorted(self.queue.iterdir()):
            pid = int(ticket.name.rsplit("-", 1)[-1]) if ticket.name.rsplit("-", 1)[-1].isdigit() else None
            if pid is None or not pid_alive(pid):
                ticket.unlink(missing_ok=True)  # a lane that died while waiting
                continue
            tickets.append(ticket)
        return bool(tickets) and tickets[0] == self.ticket

    def _break_if_dead(self) -> bool:
        owner_file = self.path / "owner"
        text = owner_file.read_text(encoding="utf-8") if owner_file.exists() else ""
        pid = holder_pid(text)
        age = time.time() - self.path.stat().st_mtime
        if (pid is not None and not pid_alive(pid)) or age > STALE_AFTER_S:
            print(f"land: breaking a lock its holder left ({text.strip() or '?'}, {age / 60:.0f} min old)", flush=True)
            shutil.rmtree(self.path, ignore_errors=True)
            return True
        return False

    def __enter__(self) -> Lock:
        self.queue.mkdir(exist_ok=True)
        self.ticket.write_text(f"{self.owner}\n", encoding="utf-8")
        deadline = time.monotonic() + self.wait_s
        said = ""
        try:
            while True:
                if self._first_in_line():
                    try:
                        self.path.mkdir()
                    except FileExistsError:
                        if self._break_if_dead():
                            continue
                    else:
                        (self.path / "owner").write_text(f"{self.owner} pid {os.getpid()}\n", encoding="utf-8")
                        return self
                if time.monotonic() > deadline:
                    raise SystemExit(4)
                ahead = sum(1 for t in self.queue.iterdir() if t.name < self.ticket.name)
                holder = (
                    (self.path / "owner").read_text(encoding="utf-8").strip() if (self.path / "owner").exists() else "?"
                )
                news = f"land: waiting, {ahead} ahead in line; {holder} is landing"
                if news != said:
                    print(news, flush=True)
                    said = news
                time.sleep(POLL_S)
        finally:
            self.ticket.unlink(missing_ok=True)

    def __exit__(self, *exc: object) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def collect(cwd: Path) -> list[str]:
    """Every test id the suite would run (the default markers apply)."""
    files = sorted(str(p.relative_to(cwd)) for p in (cwd / "tests").glob("test_*.py") if " " not in p.name)
    out = subprocess.run(
        ["uv", "run", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *files],
        cwd=cwd,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )
    ids = [line.strip() for line in out.stdout.splitlines() if "::" in line and line.startswith("tests/")]
    if out.returncode not in (0, 5) or not ids:
        print(out.stdout[-4000:] + out.stderr[-2000:])
        print("land: the tests do not even collect; main is unchanged. Fix it on the branch, commit, land again.")
        raise SystemExit(3)
    return ids


def shards(ids: list[str], seconds: dict[str, float], count: int) -> list[list[str]]:
    """Longest first onto the least loaded shard, so the shards end at about the same time."""
    known = sorted(v for v in seconds.values() if v > 0)
    guess = known[len(known) // 2] if known else 0.5
    loads = [[0.0, []] for _ in range(count)]
    for test in sorted(ids, key=lambda t: -seconds.get(t, guess)):
        lightest = min(loads, key=lambda load: load[0])
        lightest[0] += seconds.get(test, guess)
        lightest[1].append(test)
    # keep each shard in file order, so module-level fixtures are set up once per file
    return [sorted(tests) for _, tests in loads if tests]


DURATION = re.compile(r"^\s*([0-9.]+)s (?:setup|call|teardown)\s+(tests/\S+)$", re.M)
FAILED = re.compile(r"^(?:FAILED|ERROR) (tests/[^:\s]+\.py)", re.M)


def touched_by(cwd: Path, changed: set[str], test_file: str) -> bool:
    """Whether the branch changed this test file or a robotforge module it imports."""
    if test_file in changed:
        return True
    try:
        text = (cwd / test_file).read_text(encoding="utf-8")
    except OSError:
        return True
    modules = set(re.findall(r"^\s*import\s+(robotforge\.[\w.]+)", text, re.M))
    for package, names in re.findall(r"^\s*from\s+(robotforge(?:\.\w+)*)\s+import\s+(\([^)]*\)|[\w \t,]+)", text, re.M):
        modules.add(package)
        modules.update(f"{package}.{name.strip()}" for name in names.strip("()").split(",") if name.strip())
    for module in modules - {"robotforge"}:
        base = Path("src", *module.split("."))
        for path in (base.with_suffix(".py"), base / "__init__.py"):
            if str(path) in changed:
                return True
        if any(c.startswith(str(base) + "/") for c in changed):
            return True
    return False


def run_suite(cwd: Path, git_dir: Path) -> None:
    """The whole suite in parallel shards; a file that fails only beside others is run again alone."""
    record = git_dir / "land.durations.json"
    try:
        seconds = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        seconds = {}
    started = time.monotonic()
    ids = collect(cwd)
    changed = set(git("diff", "--name-only", "main...HEAD", cwd=cwd).stdout.split())
    heavy = [t for t in ids if seconds.get(t, 0.0) > HEAVY_S and not touched_by(cwd, changed, t.split("::")[0])]
    if heavy:
        print(
            f"land:   leaving out {len(heavy)} heavy tests the branch does not touch (over {HEAVY_S} s each)",
            flush=True,
        )
        ids = [t for t in ids if t not in set(heavy)]
    groups = shards(ids, seconds, SHARDS)
    command = ["uv", "run", "pytest", "-q", "-p", "no:cacheprovider", "--durations=0", "--durations-min=0.01"]
    running = []
    for number, group in enumerate(groups):
        listing = git_dir / f"land.shard{number}.txt"
        listing.write_text("\n".join(group) + "\n", encoding="utf-8")
        running.append(
            subprocess.Popen(
                [*command, "@" + str(listing)],
                cwd=cwd,
                text=True,
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        )
    outputs = [proc.communicate()[0] for proc in running]
    print(f"land:   {len(groups)} shards in {time.monotonic() - started:.0f} s", flush=True)
    measured: dict[str, float] = {}
    for out in outputs:
        for value, test in DURATION.findall(out):
            measured[test] = measured.get(test, 0.0) + float(value)
    failing = sorted(
        {
            name
            for proc, out in zip(running, outputs, strict=True)
            if proc.returncode != 0
            for name in FAILED.findall(out)
        }
    )
    broken = [out for proc, out in zip(running, outputs, strict=True) if proc.returncode not in (0, 1)]
    if broken or (any(proc.returncode != 0 for proc in running) and not failing):
        print("\n".join(out[-4000:] for out in (broken or outputs)))
        print("land: the suite failed; main is unchanged. Fix it on the branch, commit, land again.")
        raise SystemExit(3)
    if failing:
        print("land:   failed beside other shards: " + ", ".join(failing) + "; running them alone", flush=True)
        alone = subprocess.run(
            ["uv", "run", "pytest", "-q", "-x", "-p", "no:cacheprovider", *failing],
            cwd=cwd,
            text=True,
            capture_output=True,
            encoding="utf-8",
        )
        if alone.returncode != 0:
            print((alone.stdout + alone.stderr)[-6000:])
            print("land: a test failed; main is unchanged. Fix it on the branch, commit, land again.")
            raise SystemExit(3)
        print("land:   they pass alone (the shards disturbed each other); landing", flush=True)
    seconds.update(measured)
    with contextlib.suppress(OSError):
        record.write_text(json.dumps(seconds, indent=0, sort_keys=True), encoding="utf-8")


def run_checks(cwd: Path, git_dir: Path, playtest: bool) -> None:
    for command in (["uv", "run", "ruff", "check", "."], ["uv", "run", "ruff", "format", "--check", "."]):
        quick(cwd, command)
    print("land: the suite, in parallel shards", flush=True)
    run_suite(cwd, git_dir)
    quick(cwd, ["uv", "run", "python", "scripts/smoke.py"])
    if playtest:
        quick(cwd, ["uv", "run", "python", "scripts/playtest.py"])


def quick(cwd: Path, command: list[str]) -> None:
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
        run_checks(here, git_dir, args.playtest)
        git("merge", "--ff-only", branch, cwd=main_tree)
        push(main_tree)
        head = git("log", "-1", "--format=%h %s", "main", cwd=main_tree).stdout.strip()
        print(f"land: landed {ahead} commits from {branch}; main is at {head}")


if __name__ == "__main__":
    sys.exit(main())
