"""Command line: `python -m robotforge.training job <folder>` runs one training job (what train_policy starts).

python -m robotforge.training algorithms      # the algorithms and their settings
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "job":
        from robotforge.training.runs import run_job

        return run_job(Path(args[1]))
    if args == ["algorithms"]:
        from robotforge.training import algorithms

        print(json.dumps(algorithms.catalog(), indent=1))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
