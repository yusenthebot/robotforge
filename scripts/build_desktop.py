"""Build the Claude Desktop artifacts locally: the MCP bundle (.mcpb) and the agent skill as a zip.

    uv run python scripts/build_desktop.py          # dist/robotforge-<version>.mcpb and dist/robotforge-skill.zip
    uv run python scripts/build_desktop.py --check  # then unpack the bundle and start it the way Claude Desktop does

The bundle uses the mcpb "uv" server type: it ships the source, pyproject.toml and uv.lock (well under 1 MB), and
uv installs the locked dependencies on the user's machine at first start. Packing and validation use the official
mcpb CLI through npx (a build tool, not a project dependency). Nothing is published or released.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
MCPB_CLI = "@anthropic-ai/mcpb@2.1.2"  # pinned: the manifest schema (0.4) must match the CLI that validates it
BUNDLE_FILES = ["pyproject.toml", "uv.lock", ".python-version"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
FIRST_START_SECONDS = 600  # a cold first start downloads Python packages


def version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


async def server_tools() -> list[dict[str, str]]:
    """Name and first sentence of every tool, read from the real server so the listing never drifts."""
    from mcp import Client

    from robotforge.cli import find_free_port
    from robotforge.mcp_server import build_server
    from robotforge.playground import Playground

    server = build_server(Playground.with_demo(), find_free_port(8900))
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
    return [{"name": t.name, "description": first_sentence(t.description or "")} for t in tools]


def first_sentence(text: str) -> str:
    flat = " ".join(text.split())
    match = re.match(r"(.+?[.!?])(\s|$)", flat)
    return match.group(1) if match else flat


def stage(target: Path) -> dict[str, Any]:
    manifest = json.loads((ROOT / "mcpb" / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != version():
        raise SystemExit(f"mcpb/manifest.json version {manifest['version']} != pyproject {version()}")
    manifest["tools"] = asyncio.run(server_tools())
    for name in BUNDLE_FILES:
        shutil.copy2(ROOT / name, target / name)
    shutil.copytree(ROOT / "src", target / "src", ignore=SKIP)
    shutil.copy2(ROOT / "assets" / "icon.png", target / manifest["icon"])
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def mcpb(*args: str) -> None:
    npx = shutil.which("npx")
    if npx is None:
        raise SystemExit("npx not found: install Node.js to pack the bundle")
    subprocess.run([npx, "-y", MCPB_CLI, *args], check=True)


def build_bundle() -> Path:
    out = DIST / f"robotforge-{version()}.mcpb"
    with tempfile.TemporaryDirectory(prefix="robotforge-mcpb-") as tmp:
        staged = Path(tmp) / "robotforge"
        staged.mkdir()
        manifest = stage(staged)
        mcpb("validate", str(staged / "manifest.json"))
        out.unlink(missing_ok=True)
        mcpb("pack", str(staged), str(out))
    print(f"bundle: {out} ({out.stat().st_size / 1e3:.0f} kB, {len(manifest['tools'])} tools)")
    return out


def build_skill_zip() -> Path:
    """The skill as Claude Desktop and claude.ai take it: a zip with one folder holding SKILL.md."""
    out = DIST / "robotforge-skill.zip"
    skill = ROOT / "skills" / "robotforge"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(skill.rglob("*")):
            if path.is_file() and path.name != ".DS_Store":
                zf.write(path, Path("robotforge") / path.relative_to(skill))
    print(f"skill:  {out} ({out.stat().st_size / 1e3:.0f} kB)")
    return out


async def check_bundle(bundle: Path) -> None:
    """Unpack the bundle and start it with its own mcp_config, as Claude Desktop does, then drive a rover."""
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    with tempfile.TemporaryDirectory(prefix="robotforge-mcpb-check-") as tmp:
        unpacked, home = Path(tmp) / "bundle", Path(tmp) / "home"
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(unpacked)
        config = json.loads((unpacked / "manifest.json").read_text(encoding="utf-8"))["server"]["mcp_config"]
        args = [a.replace("${__dirname}", str(unpacked)) for a in config["args"]]
        env = {k: v for k, v in os.environ.items() if k not in {"UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"}}
        env.update(ROBOTFORGE_NO_BROWSER="1", ROBOTFORGE_HOME=str(home))
        params = StdioServerParameters(command=shutil.which(config["command"]) or config["command"], args=args, env=env)
        print(f"check: starting {config['command']} {' '.join(args)}")
        async with Client(params, read_timeout_seconds=FIRST_START_SECONDS) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            prompts = {p.name for p in (await client.list_prompts()).prompts}
            print(f"check: {len(tools)} tools, {len(prompts)} prompts")
            for tool, args_ in [
                ("new_world", {"name": "bundle_check"}),
                ("add_robot", {"name": "rover", "template": "rover"}),
                ("move", {"robot": "rover", "speed": 0.3, "seconds": 2}),
            ]:
                result = await client.call_tool(tool, args_)
                text = "".join(c.text for c in result.content if getattr(c, "type", "") == "text")
                if result.is_error:
                    raise SystemExit(f"check failed: {tool}: {text}")
            moved = json.loads(text)["distance_m"]
            if moved < 0.2:
                raise SystemExit(f"check failed: the rover drove only {moved} m")
            print(f"check: the bundle's server drove a rover {moved} m")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="start the packed bundle and drive a robot")
    args = parser.parse_args()
    DIST.mkdir(exist_ok=True)
    bundle = build_bundle()
    build_skill_zip()
    if args.check:
        asyncio.run(check_bundle(bundle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
