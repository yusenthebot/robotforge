"""Run ZenoBench scripts the way Isaac Lab's python would, on MuJoCo.

    python -m robotforge.zenobench.python benchmark/run_episode.py --task collect_fruits --planner oracle
    python -m robotforge.zenobench.python -m env_service.worker ...

The ZenoBench checkout goes on `sys.path`, the Isaac stand-ins are installed, then the script or module
runs as `__main__`. `wrapper()` writes a one-line executable for places that want a path to Isaac
Lab's python, such as ZenoBench's env_service (`$ISAACLAB_PYTHON`).
"""

from __future__ import annotations

import logging
import os
import runpy
import sys
from pathlib import Path

from robotforge.zenobench import shim, source
from robotforge.zenobench.world import SUBSTEPS


def prepare() -> Path:
    if not logging.getLogger("robotforge").handlers:  # ZenoBench's scripts do not configure logging
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logging.getLogger("robotforge").addHandler(handler)
        logging.getLogger("robotforge").setLevel(logging.INFO)
    zb = source.check()
    if str(zb) not in sys.path:
        sys.path.insert(0, str(zb))
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    # ZenoBench tools that start workers (verify_skills --jobs, env_service) run $ISAACLAB_PYTHON
    os.environ.setdefault("ISAACLAB_PYTHON", str(wrapper()))
    shim.install(shim.zeno_options(), substeps=int(os.environ.get("ZENOBENCH_SUBSTEPS", SUBSTEPS)))
    return zb


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    prepare()
    if argv[0] == "-m":
        if len(argv) < 2:
            print("-m needs a module name", file=sys.stderr)
            return 2
        sys.argv = [argv[1], *argv[2:]]
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
        return 0
    if argv[0] == "-c":
        sys.argv = ["-c", *argv[2:]]
        exec(compile(argv[1], "<string>", "exec"), {"__name__": "__main__"})  # noqa: S102 - like python -c
        return 0
    script = Path(argv[0]).resolve()
    sys.argv = [str(script), *argv[1:]]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")
    return 0


def wrapper() -> Path:
    """An executable that behaves like `$ISAACLAB_PYTHON`, written next to the cache."""
    from robotforge.design.menagerie import cache_root

    path = cache_root() / "zenobench" / "bin" / "isaaclab-python"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = f'#!/bin/sh\nexec "{sys.executable}" -m robotforge.zenobench.python "$@"\n'
    if not path.exists() or path.read_text(encoding="utf-8") != body:
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
