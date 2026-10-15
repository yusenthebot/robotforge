"""Plugin packaging for Claude Code, Codex and Claude Desktop: manifests agree with each other and with the code.

The clients' own validators (claude plugin validate, mcpb validate) check the schemas; these tests catch drift
between the files: versions, the server launch command, and tool names the skill tells agents to call.
"""

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.mcp_server import build_server
from robotforge.playground import Playground

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "robotforge"
TOOL_CALL = re.compile(r"`([a-z_]+)\(")
# Tools that other optional features add; the skill mentions them only as "when the tool list has it".
OPTIONAL_TOOLS = {"train_policy", "training_status"}


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_every_manifest_carries_the_package_version():
    version = _version()
    assert _json(".claude-plugin/plugin.json")["version"] == version
    assert _json(".codex-plugin/plugin.json")["version"] == version
    assert _json("mcpb/manifest.json")["version"] == version


def test_marketplace_lists_the_plugin_at_the_repo_root():
    market = _json(".claude-plugin/marketplace.json")
    names = {_json(".claude-plugin/plugin.json")["name"], _json(".codex-plugin/plugin.json")["name"]}
    assert names == {"robotforge"}
    [entry] = market["plugins"]
    assert entry["name"] == "robotforge" and entry["source"] == "./"


def _launch(args: list[str]) -> list[str]:
    """The arguments after `uv run`, with the project path left out."""
    assert args[0] == "run"
    i = args.index("--project")
    return args[1:i] + args[i + 2 :]


def test_each_client_starts_the_same_locked_server():
    claude = _json(".claude-plugin/plugin.json")["mcpServers"]["robotforge"]
    codex_manifest = _json(".codex-plugin/plugin.json")
    codex = _json(codex_manifest["mcpServers"].removeprefix("./"))["mcpServers"]["robotforge"]
    desktop = _json("mcpb/manifest.json")["server"]["mcp_config"]

    for server in (claude, codex, desktop):
        assert server["command"] == "uv"
    expected = ["--frozen", "--no-dev", "robotforge", "mcp"]
    assert _launch(claude["args"]) == expected
    assert _launch(codex["args"]) == expected
    # The Desktop bundle runs from its unpacked folder; --directory also makes it the working directory.
    assert desktop["args"][:3] == ["run", "--directory", "${__dirname}"] and desktop["args"][3:] == expected

    assert claude["args"][claude["args"].index("--project") + 1] == "${CLAUDE_PLUGIN_ROOT}"
    assert claude["env"]["UV_PROJECT_ENVIRONMENT"].startswith("${CLAUDE_PLUGIN_DATA}")
    # Without this, non-interactive Codex runs refuse every robotforge tool.
    assert codex["default_tools_approval_mode"] == "approve"
    assert {"ROBOTFORGE_HOME", "ROBOTFORGE_NO_BROWSER"} <= set(codex["env_vars"])


def test_codex_manifest_paths_exist():
    manifest = _json(".codex-plugin/plugin.json")
    paths = [manifest["skills"], manifest["mcpServers"], manifest["interface"]["logo"]]
    for path in paths:
        assert path.startswith("./") and (ROOT / path).exists(), path


def test_skill_follows_the_agent_skills_format():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front)
    assert meta["name"] == SKILL.name
    assert re.fullmatch(r"[a-z0-9-]{1,64}", meta["name"])
    assert 50 < len(meta["description"]) <= 250  # Codex cuts skill descriptions after about 280 characters
    assert len(body.splitlines()) < 150, "keep SKILL.md short; details go in references/"
    for link in re.findall(r"\]\(([^)#]+)\)", body):
        if not link.startswith("http"):
            assert (SKILL / link).exists(), f"broken link {link}"


def test_skill_texts_have_no_em_dashes():
    for path in SKILL.rglob("*.md"):
        assert "—" not in path.read_text(encoding="utf-8"), path


@pytest.mark.anyio
async def test_skill_only_names_tools_the_server_has():
    server = build_server(Playground.with_demo(), find_free_port(8790))
    async with Client(server) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
    named = {name for path in SKILL.rglob("*.md") for name in TOOL_CALL.findall(path.read_text(encoding="utf-8"))}
    # Script API calls (sim.move(...)) appear inside code blocks without backticks, so they are not counted here.
    unknown = named - tools - OPTIONAL_TOOLS
    assert not unknown, f"the skill names tools the server does not have: {sorted(unknown)}"
