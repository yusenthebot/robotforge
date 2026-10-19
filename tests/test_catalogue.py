"""The open robot catalogue: the list, the search, adding a robot by name from a folder already on disk (no network),
and what the structure of a loaded robot says it is."""

from __future__ import annotations

import io
import json
import re
import tarfile
from pathlib import Path

import pytest
import yaml
from importer_fixtures import box_link, critter_urdf, revolute, robot, rover_urdf, write
from mcp import Client
from test_mjcf_doctor import _legs as _dog_mjcf

from robotforge.design import catalogue, catalogue_fetch, library
from robotforge.design.world import RobotEntry
from robotforge.mcp_server import build_server
from robotforge.playground import Playground, PlaygroundError

# ---- the list -------------------------------------------------------------------------------------------------


def test_the_list_is_clean():
    entries = catalogue.entries()
    assert len(entries) > 180
    for e in entries.values():
        assert re.fullmatch(r"[a-z][a-z0-9_]*", e.id), e.id
        assert re.fullmatch(r"[0-9a-f]{40}|[\w.-]+", e.commit), e.id  # a commit, or a tag
        assert e.format in {"mjcf", "urdf", "xacro"} and e.model.lower().endswith(
            (".xml", ".urdf", ".xacro", ".mjcf")
        ), e.id
        assert e.kind in catalogue.KINDS
    assert {"arm", "quadruped", "humanoid", "wheeled", "hand", "aircraft"} <= set(catalogue.counts())


def test_tags_become_kinds():
    assert catalogue.get("hyq").kind == "quadruped"
    assert catalogue.get("open_manipulator_x").kind == "arm"
    # a biped that also has wheels is a legged robot first; a mobile manipulator is wheeled
    tagged = {e.id: e for e in catalogue.entries().values()}
    assert tagged["tiago"].kind == "wheeled"
    assert all(e.kind == "humanoid" for e in tagged.values() if "biped" in e.tags and "quadruped" not in e.tags)


def test_the_tuned_library_is_not_listed_twice():
    shown = {e.id for e in catalogue.search()}
    assert not shown & set(library.CATALOG)
    assert all(not e.in_library for e in catalogue.search())
    # the Menagerie model of the library's Go2 is the library's, whatever name the catalogue has it under
    assert catalogue.source_of("go2_mj") == ("go2", None)
    assert catalogue.source_of("Unitree Go2") == ("go2", None)


def test_search_by_kind_and_words():
    arms = catalogue.search("arm", "kinova")
    assert arms and all("kinova" in f"{e.id} {e.robot} {e.maker}".lower() and e.kind == "arm" for e in arms)
    assert catalogue.search(None, "no such robot xyz") == []
    with pytest.raises(catalogue.CatalogueError, match="kind must be one of"):
        catalogue.search("tank")
    first = catalogue.search(None, "hyq")[0]
    assert first.id == "hyq"  # an exact word in the name comes before partial matches


def test_a_name_means_one_robot():
    assert catalogue.source_of("HyQ") == (None, "hyq")
    assert catalogue.source_of("open manipulator x") == (None, "open_manipulator_x")
    with pytest.raises(catalogue.CatalogueError, match="Did you mean.*hyq"):
        catalogue.get("hyqq")


def test_the_listing_answers_with_what_to_do_next():
    plain = catalogue.listing()
    assert {r["name"] for r in plain["robots"]} == set(library.CATALOG)
    assert plain["open_catalogue"]["robots"] > 150 and "arm" in plain["open_catalogue"]["by_kind"]
    some = catalogue.listing("quadruped", None, limit=3)
    assert len(some["robots"]) == 3 + sum(r["kind"] == "quadruped" for r in library.catalog())
    assert some["matches"] > 3 and "more" in some
    row = next(r for r in some["robots"] if r.get("source") == "catalogue")
    assert {"name", "title", "kind", "summary", "license", "downloaded"} <= set(row)
    assert catalogue.listing("hand", "zzz")["robots"] == []


# ---- adding a robot by name -----------------------------------------------------------------------------------


def _wrist_arm() -> str:
    """Five joints, about 0.7 m long, the last link bare: an arm with a flange for a gripper."""
    return robot(
        box_link("base_link", (0.2, 0.2, 0.1), 5.0, "0 0 0.05")
        + box_link("upper", (0.06, 0.06, 0.3), 1.0, "0 0 0.15")
        + box_link("fore", (0.05, 0.05, 0.25), 0.6, "0 0 0.125")
        + box_link("hand", (0.04, 0.04, 0.05), 0.3, "0 0 0.025")
        + box_link("flange_link", (0.04, 0.04, 0.05), 0.2, "0 0 0.025")
        + box_link("turret", (0.1, 0.1, 0.05), 1.0, "0 0 0.025")
        + revolute("waist", "base_link", "turret", "0 0 0.1", axis="0 0 1")
        + revolute("shoulder", "turret", "upper", "0 0 0.05")
        + revolute("elbow", "upper", "fore", "0 0 0.3")
        + revolute("wrist_pitch", "fore", "hand", "0 0 0.25")
        + revolute("wrist_roll", "hand", "flange_link", "0 0 0.05", axis="0 0 1"),
        "wrist_arm",
    )


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """A fake repository on disk holding a few robots, listed in a fake catalogue."""
    repos = tmp_path / "repos"
    write(repos / "fake_repo" / "robots" / "arm" / "arm.urdf", _wrist_arm())
    write(repos / "fake_repo" / "robots" / "critter" / "critter.urdf", critter_urdf())
    write(repos / "fake_repo" / "robots" / "rover" / "rover.urdf", rover_urdf())
    write(repos / "fake_repo" / "robots" / "dog" / "dog.xml", _dog_mjcf())
    monkeypatch.setenv("ROBOTFORGE_REPOS", str(repos))
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ROBOTFORGE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path / "none"))

    def entry(name: str, tags: tuple[str, ...]) -> catalogue.Entry:
        suffix = "xml" if name == "dog" else "urdf"
        return catalogue.Entry(
            id=name,
            robot=name.title(),
            maker="Acme",
            dof=3,
            tags=tags,
            license="MIT",
            repo="fake_repo",
            url="https://example.invalid/fake_repo.git",
            commit="0" * 40,
            package=f"robots/{name}",
            format="mjcf" if suffix == "xml" else "urdf",
            model=f"robots/{name}/{name}.{suffix}",
            variants=(),
            xacro_args=(),
        )

    fake = {
        n: entry(n, t)
        for n, t in (("arm", ("arm",)), ("critter", ("quadruped",)), ("rover", ("wheeled",)), ("dog", ("quadruped",)))
    }
    monkeypatch.setattr(catalogue, "entries", lambda: fake)
    return fake


def test_an_arm_by_name_gets_a_tool_point_and_a_description(folders):
    pg = Playground.with_demo()
    pg.new_world("t")
    result = pg.add_robot("a", library="arm")
    about = result["catalogue"]
    assert about["kind"] == "arm" and about["joints"] == 5
    assert 0.5 < about["reach_m"] < 0.95
    assert any("move_arm" in line for line in about["can"]) and any("no gripper" in line for line in about["can"])
    assert any("tool point" in note for note in result["notes"])
    names = {pg.session.model.site(i).name for i in range(pg.session.model.nsite)}
    assert {"a/tcp", "a/flange"} <= names
    assert pg.behaviors.kind("a").value == "arm"
    # the world remembers the name, not where the folder was: it loads again from the catalogue
    saved = yaml.safe_load((pg.store.dir / "world.yaml").read_text(encoding="utf-8"))
    assert saved["robots"][0]["catalogue"] == "arm"


def test_the_arm_reaches_a_point(folders):
    pg = Playground.with_demo()
    pg.new_world("t")
    pg.add_robot("a", library="arm", pos=[0.0, 0.0])
    motion = pg.behaviors.move_arm("a", [0.3, 0.2, 0.3])
    pg.run_fast(lambda: motion.done, 20.0)
    tcp = pg.session.data.site_xpos[pg.session.model.site("a/tcp").id]
    assert abs(tcp[0] - 0.3) < 0.03 and abs(tcp[1] - 0.2) < 0.03 and abs(tcp[2] - 0.3) < 0.03


def test_a_robot_on_wheels_and_one_on_four_legs_are_recognized_by_structure(folders):
    pg = Playground.with_demo()
    pg.new_world("t")
    wheels = pg.add_robot("w", library="rover")["catalogue"]
    assert wheels["kind"] == "wheeled" and any("move" in line for line in wheels["can"])
    legs = pg.add_robot("c", library="critter", pos=[1.0, 0.0])["catalogue"]
    assert legs["kind"] == "quadruped" and legs["legs"] == 4
    # no gait code knows two-joint legs: it says so and how to get a walk, as an imported critter always has
    assert any("learn" in line for line in legs["can"])


def test_a_name_the_catalogue_does_not_know_says_what_is_close(folders):
    pg = Playground.with_demo()
    pg.new_world("t")
    with pytest.raises(PlaygroundError, match="no robot 'rovr'.*Did you mean rover"):
        pg.add_robot("x", library="rovr")


def test_a_world_without_catalogue_robots_saves_as_before():
    from robotforge.design.world import World

    world = World.model_validate({"name": "w", "robots": [{"name": "g", "library": "go2"}]})
    assert "catalogue" not in json.dumps(world.model_dump(mode="json"))


def test_one_source_only():
    with pytest.raises(ValueError, match="exactly one"):
        RobotEntry(name="g", library="go2", catalogue="hyq")


def test_a_folder_that_is_not_on_disk_says_so_instead_of_downloading(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_REPOS", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_CACHE", str(tmp_path / "cache"))  # not the folders this machine downloaded
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path / "none"))
    (tmp_path / "example-robot-data").mkdir()
    with pytest.raises(catalogue.CatalogueError, match="not in ROBOTFORGE_REPOS"):
        catalogue.model_path("hyq")


# ---- the download ---------------------------------------------------------------------------------------------


def _archive(files: dict[str, bytes]) -> io.BytesIO:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return buffer


def test_an_archive_download_keeps_only_the_robots_folder(tmp_path):
    stream = _archive(
        {
            "repo-abc/robots/a/a.urdf": b"<robot/>",
            "repo-abc/robots/a/meshes/m.stl": b"mesh",
            "repo-abc/robots/b/b.urdf": b"other robot",
            "repo-abc/README.md": b"readme",
            "repo-abc/robots/a/../../../escape.txt": b"nope",
        }
    )
    catalogue_fetch._extract(stream, "robots/a/", tmp_path)
    found = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert found == ["robots/a/a.urdf", "robots/a/meshes/m.stl"]


def test_a_downloaded_folder_is_found_again(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_CACHE", str(tmp_path))
    monkeypatch.delenv("ROBOTFORGE_REPOS", raising=False)
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path / "none"))
    commit = "a" * 40
    assert catalogue_fetch.folder_path("r", commit, "x/y") is None
    write(catalogue_fetch.repo_dir("r", commit) / "x" / "y" / "f.urdf", "<robot/>")
    assert catalogue_fetch.folder_path("r", commit, "x/y") == catalogue_fetch.repo_dir("r", commit) / "x" / "y"


# ---- the tools ------------------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_library_searches_and_add_robot_loads_by_name(folders):
    pg = Playground.with_demo()
    async with Client(build_server(pg, 0)) as client:

        async def call(tool: str, args: dict) -> dict:
            return json.loads((await client.call_tool(tool, args)).content[0].text)

        found = await call("list_library", {"kind": "arm", "search": "arm"})
        assert [r["name"] for r in found["robots"] if r.get("source") == "catalogue"] == ["arm"]
        await call("new_world", {"name": "t"})
        added = await call("add_robot", {"name": "a", "library": "arm"})
        assert added["catalogue"]["kind"] == "arm"
        bad = await client.call_tool("add_robot", {"name": "z", "library": "nothing_like_it"})
        assert bad.is_error and "list_library" in bad.content[0].text


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_the_agent_is_told_about_it():
    from robotforge.mcp_server import INSTRUCTIONS

    skill = (Path(__file__).parent.parent / "skills" / "robotforge" / "SKILL.md").read_text(encoding="utf-8")
    assert "open catalogue" in skill and "list_library" in skill
    assert "open robots" in INSTRUCTIONS


def test_a_dog_of_torque_motors_with_no_standing_pose_stands_and_walks(folders):
    """The unitree_mujoco kind of file: motors that take torques, joint limits that exclude the zero pose, no
    keyframe. It is driven as servos and stood up, so it stands, and the walker finds its legs."""
    pg = Playground.with_demo()
    pg.new_world("t")
    result = pg.add_robot("d", library="dog")
    assert any("torque motors driven as position servos" in n for n in result["notes"])
    assert any("standing pose" in n for n in result["notes"])
    start = float(pg.session.data.xpos[pg.behaviors._root_body("d")][2])
    pg.run_fast(lambda: False, 3.0)
    now = pg.behaviors.pose("d")
    assert now["upright"] and start - now["position"][2] < 0.1
    assert result["catalogue"]["kind"] == "quadruped" and result["catalogue"]["legs"] == 4
