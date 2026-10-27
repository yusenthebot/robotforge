"""Create local Malo, Go2 and Panda workspaces without bundling external robot assets.

uv run python scripts/setup_malo.py /path/to/zeno_malo_edu.generated.urdf
Models stay in their original folders; the saved Malo design keeps that reference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from robotforge.design import combine, library
from robotforge.design.store import worlds_home
from robotforge.playground import Playground


def create(model: Path, prefix: str = "zeno", *, library_worlds: bool = True) -> dict:
    model = model.expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    from robotforge.design.world import NAME_RE

    if not NAME_RE.fullmatch(prefix) or len(prefix) > 16:
        raise ValueError("prefix must start with a lowercase letter and use up to 16 letters, digits or underscores")
    design = f"{prefix}_malo"
    names = [
        f"{prefix}_malo_workshop",
        *([f"{prefix}_go2_warehouse", f"{prefix}_panda_workbench"] if library_worlds else []),
    ]
    if any((worlds_home() / name).exists() for name in names) or combine.saved_design(design) is not None:
        raise ValueError("workspace or saved design already exists; choose a different --prefix")
    p = Playground.with_demo()
    p.new_world(names[0], "concrete")
    p.add_robot("malo", file=str(model), params={"respect_effort_limits": True}, pos=[0, 0])
    p.add_object("workbench", shape="table", size=[1.4, 0.7, 0.75], pos=[1.25, 0.2], fixed=True)
    p.add_object("parts", shape="bin", size=[0.25, 0.2, 0.12], pos=[1.25, 0.2, 0.75], fixed=True, color="gray")
    p.add_object("marker", shape="box", size=[0.12, 0.12, 0.12], pos=[1.0, 0.2, 0.81], color="orange")
    # An idealized simulated camera at the URDF's head frame, not a hardware-calibrated stereo replica.
    p.add_sensor(
        "malo",
        {
            "type": "camera",
            "name": "head_rgb",
            "on": "stereo_camera_link",
            "at": [0.03, 0, 0],
            "direction": [1, 0, 0],
            "width": 160,
            "height": 120,
            "fov": 70,
            "rate_hz": 5,
        },
    )
    p.add_sensor("malo", {"type": "lidar", "name": "floor_scan", "on": "lidar_link", "rays": 36, "rate_hz": 5})
    p.session.paused = True
    saved = combine.save(p.store.load().robots[0], design, p.store.dir)
    outputs = [{"world": names[0], "path": str(p.store.dir), "robots": p.behaviors.robots()}]
    if library_worlds:
        # Explicit setup is allowed to fetch these named library models; the core installation stays unchanged.
        p.new_world(names[1])
        p.load_scene("warehouse", starter=False, seed=0)
        p.add_robot("go2", library="go2", pos=[0, -1.5])
        p.add_sensor("go2", {"type": "lidar", "name": "route_scan", "rays": 36, "rate_hz": 5})
        p.session.paused = True
        outputs.append({"world": names[1], "path": str(p.store.dir), "robots": p.behaviors.robots()})
        p.new_world(names[2], "wood")
        p.add_object("workbench", shape="table", size=[1.6, 1.0, 0.75], pos=[0, 0], fixed=True)
        p.add_robot("panda", library="panda", pos=[0, 0, 0.75])
        p.add_object("marker", shape="box", size=[0.08, 0.08, 0.08], pos=[0.4, 0, 0.79], color="orange")
        p.session.paused = True
        outputs.append({"world": names[2], "path": str(p.store.dir), "robots": p.behaviors.robots()})
    return {
        "design": saved,
        "workspaces": outputs,
        "model_assets_bundled": False,
        "camera": "idealized head RGB, 160x120, 70 degree vertical FOV; not hardware-calibrated stereo",
        "library_cached": {name: library.is_downloaded(name) for name in ("go2", "panda")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--prefix", default="zeno")
    parser.add_argument("--home", type=Path, help="optional ROBOTFORGE_HOME for an isolated workspace")
    parser.add_argument("--malo-only", action="store_true", help="skip the Go2 and Panda example worlds")
    args = parser.parse_args()
    if args.home is not None:
        os.environ["ROBOTFORGE_HOME"] = str(args.home.expanduser().resolve())
    print(json.dumps(create(args.model, args.prefix, library_worlds=not args.malo_only), indent=2))


if __name__ == "__main__":
    main()
