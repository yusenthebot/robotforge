"""See what the open catalogue's downloads take up, and free some of it (milestone G12a).

    uv run python scripts/catalogue_cache.py                  # sizes, biggest first
    uv run python scripts/catalogue_cache.py --remove NAME    # delete one downloaded repository (it comes back on use)
    uv run python scripts/catalogue_cache.py --older-than 60  # delete repositories not used for 60 days

Every robot is fetched once, as just its folder, into the cache; nothing deletes it by itself, and a few robots
(Stretch, TIAGo, R1 Pro) carry tens of MB of meshes. A removed repository is downloaded again the next time one of
its robots is added. Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from robotforge.design import catalogue_fetch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remove", nargs="*", default=[], metavar="NAME", help="folder names to delete")
    parser.add_argument("--older-than", type=float, default=None, metavar="DAYS", help="delete the unused ones")
    args = parser.parse_args()
    report = catalogue_fetch.cache_report(limit=1000)
    root = Path(str(report["folder"]))
    print(f"{report['mb']} MB in {root}")
    now = time.time()
    for row in report["largest"]:  # type: ignore[union-attr]
        folder = root / row["repo"]
        age = (now - folder.stat().st_mtime) / 86400
        gone = row["repo"] in args.remove or (args.older_than is not None and age > args.older_than)
        print(f"  {row['mb']:6d} MB  {age:5.0f} d  {row['repo']}{'   -> removed' if gone else ''}")
        if gone:
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    main()
