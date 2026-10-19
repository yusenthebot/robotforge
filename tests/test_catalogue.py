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


def test_search_matches_words_of_the_name_not_the_file_format():
    # "ur" once matched "urdf" (the format) and listed baxter, edo and iiwa among the UR arms
    found = catalogue.search("arm", "ur")
    assert found
    for e in found:
        assert any(t.startswith("ur") for t in catalogue._tokens(e.id, e.robot, e.maker, " ".join(e.tags))), e.id
    assert not {e.id for e in found} & {"baxter", "edo", "iiwa14"}
    assert not catalogue.search(None, "urdf")  # the file format is not a name


def test_a_kind_can_be_asked_for_by_the_name_a_listing_row_uses():
    assert catalogue.canonical_kind("gripper") == "hand" == catalogue.canonical_kind("Hand")
    assert catalogue.canonical_kind("dog") == "quadruped" == catalogue.canonical_kind("legged")
    assert catalogue.canonical_kind("drone") == "aircraft" and catalogue.canonical_kind(None) is None
    assert {e.id for e in catalogue.search("gripper")} == {e.id for e in catalogue.search("hand")}
    # every kind the tuned library's rows report is a kind list_library accepts
    assert {catalogue.canonical_kind(r["kind"]) for r in library.catalog()} <= set(catalogue.KINDS)
    rows = catalogue.listing("gripper")["robots"]
    assert any(r["kind"] == "gripper" for r in rows)
    with pytest.raises(catalogue.CatalogueError, match="kind must be one of"):
        catalogue.listing("tank")


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


def test_a_humanoid_whose_leg_joint_has_no_motor_gets_an_error_that_says_so(tmp_path, monkeypatch):
    """Oli and Cassie have linkage legs: a walker looking up the motor of a passive joint failed with a bare "5"."""
    from robotforge.behaviors import humanoid

    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))

    def no_motor(*_args, **_kwargs):
        raise KeyError(5)

    monkeypatch.setattr(humanoid, "for_template", no_motor)
    pg = Playground.with_demo()
    pg.new_world("t")
    pg.add_robot("h", template="humanoid")
    with pytest.raises(ValueError, match="no gait fits the body of h.*no motor of its own"):
        pg.behaviors.move("h", 0.2, 0.0)


def test_a_xacro_that_includes_a_package_outside_its_folder_gets_the_whole_repository(tmp_path, monkeypatch):
    """The Toyota HSR's xacro asks for `$(find hsrb_parts_description)`, a folder next to its own: the first download
    is the robot's folder only, so the import finds the package missing, downloads the repository and goes again."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ROBOTFORGE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path / "none"))
    monkeypatch.delenv("ROBOTFORGE_REPOS", raising=False)
    sha = "b" * 40
    entry = catalogue.Entry(
        id="needy",
        robot="Needy",
        maker="Acme",
        dof=1,
        tags=("arm",),
        license="MIT",
        repo="needy_repo",
        url="https://example.invalid/needy_repo",
        commit=sha,
        package="robots/needy",
        format="xacro",
        model="robots/needy/needy.urdf.xacro",
        variants=(),
        xacro_args=(),
    )
    monkeypatch.setattr(catalogue, "entries", lambda: {"needy": entry})
    root = catalogue_fetch.repo_dir("needy_repo", sha)
    xmlns = 'xmlns:xacro="http://www.ros.org/wiki/xacro"'
    write(
        root / "robots" / "needy" / "needy.urdf.xacro",
        f'<robot {xmlns} name="needy"><xacro:include filename="$(find parts)/parts.xacro"/><xacro:parts/></robot>',
    )
    wholes = []

    def download(name, url, commit, folder, progress=None, whole=False):
        wholes.append(whole)
        write(
            root / "robots" / "parts" / "parts.xacro",
            f'<robot {xmlns}><xacro:macro name="parts"><link name="base"><inertial>'
            '<mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>'
            '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link></xacro:macro></robot>',
        )
        (root / catalogue_fetch.WHOLE_MARKER).write_text(commit, encoding="utf-8")
        return root

    monkeypatch.setattr(catalogue_fetch, "fetch", download)
    pg = Playground.with_demo()
    pg.new_world("t")
    pg.add_robot("n", library="needy")
    assert wholes == [True]
    assert catalogue_fetch.is_whole("needy_repo", sha)


# ---- the trial of a legged robot and what the description makes of it ----------------------------------------


def test_a_legged_robot_is_tried_when_it_is_added_and_the_result_is_kept(folders, monkeypatch):
    from robotforge.design import catalogue_probe

    pg = Playground.with_demo()
    pg.new_world("t")
    about = pg.add_robot("d", library="dog")["catalogue"]
    checked = about["checked"]
    assert checked["stands"] is True and checked["walk"] in {"walked", "fell", "stalled"}
    assert checked["summary"].startswith("stood")
    assert catalogue_probe.cached(folders["dog"]) is not None
    # the same robot again reads the kept result instead of running it (a second world, nothing is simulated)
    monkeypatch.setattr(catalogue_probe, "run", lambda *_: pytest.fail("tried twice"))
    pg.new_world("u")
    assert pg.add_robot("d", library="dog")["catalogue"]["checked"] == checked
    # arms, wheels and hands are not tried: the trial is for legs
    assert "checked" not in pg.add_robot("w", library="rover", pos=[2.0, 0.0])["catalogue"]


def test_the_trial_does_not_try_a_robot_inside_its_own_trial(folders, monkeypatch):
    from robotforge.design import catalogue_probe

    monkeypatch.setattr(catalogue_probe, "run", lambda *_: pytest.fail("ran"))
    token = catalogue_probe._probing.set(True)
    try:
        assert catalogue_probe.trial(folders["dog"]) is None
    finally:
        catalogue_probe._probing.reset(token)


def test_a_trial_that_breaks_does_not_break_the_add(folders, monkeypatch):
    from robotforge.design import catalogue_probe

    def broken(*_):
        raise RuntimeError("scratch world failed")

    monkeypatch.setattr(catalogue_probe, "run", broken)
    pg = Playground.with_demo()
    pg.new_world("t")
    about = pg.add_robot("d", library="dog")["catalogue"]
    assert about["kind"] == "quadruped" and "checked" not in about


def _described(kind: str, can: list[str]):
    from robotforge.design.catalogue_describe import Described

    what = {"humanoid": "a walking robot on two legs", "quadruped": "a four-legged robot"}[kind]
    return Described(kind, f"Acme X: {what}, 30.0 kg, 12 motors (URDF, MIT)", can, legs=2)


GAIT = "move starts a gait found from its legs (up to 0.5 m/s); the gait is tuned for the library's robots"


def test_a_robot_that_collapsed_is_not_called_a_walking_robot():
    trial = {"settle_s": 2.0, "stands": False, "height_kept": 0.33, "walk": "not tried"}
    described = _described("humanoid", [GAIT]).with_trial(trial)
    out = described.to_dict()
    assert "walking robot" not in out["about"] and "collapses at home" in out["about"]
    assert len(out["can"]) == 1 and "move starts" not in out["can"][0] and "collapsed" in out["can"][0]
    assert out["checked"]["stands"] is False and "33%" in out["checked"]["summary"]


def test_a_gait_that_fell_is_offered_with_that_said():
    trial = {
        "stands": True,
        "walk": "fell",
        "fell_after_m": 1.7,
        "fell_after_s": 3.4,
        "walk_speed": 0.2,
        "settle_s": 2.0,
    }
    out = _described("quadruped", [GAIT]).with_trial(trial).to_dict()
    assert "move starts a gait" in out["can"][0] and "fell over after 1.7 m" in out["can"][0]
    stalled = {"stands": True, "walk": "stalled", "walked_m": 0.0, "walk_s": 6.0, "walk_speed": 0.2, "settle_s": 2.0}
    assert "got only 0.0 m" in _described("humanoid", [GAIT]).with_trial(stalled).to_dict()["can"][0]
    walked = {"stands": True, "walk": "walked", "walked_m": 1.2, "walk_s": 6.0, "walk_speed": 0.2, "settle_s": 2.0}
    assert "walked 1.2 m upright" in _described("quadruped", [GAIT]).with_trial(walked).to_dict()["can"][0]


def test_run_fast_gives_up_on_a_simulation_that_blew_up(monkeypatch):
    """MuJoCo resets a diverged simulation to time 0; run_fast used to wait for a clock that kept starting over
    (the trial of H1_2 sat at full speed for twenty minutes)."""
    pg = Playground.with_demo()
    pg.session.data.time = 5.0
    steps = []

    def blow_up(n: int = 1) -> None:
        steps.append(n)
        pg.session.data.time = 0.0

    monkeypatch.setattr(pg.session, "step", blow_up)
    assert pg.run_fast(lambda: False, 1.0) == 0.0 and len(steps) == 1


def test_a_trial_walk_that_blows_up_is_a_fall(folders, monkeypatch):
    from robotforge.design import catalogue_probe

    pg = Playground.with_demo()
    pg.new_world("t")
    pg.add_robot("d", library="dog")
    real = pg.run_fast
    calls = []

    def blown_on_the_third(until, seconds):
        calls.append(seconds)
        return 0.0 if len(calls) == 4 else real(until, seconds)  # the settle, then the third look at the walk

    monkeypatch.setattr(pg, "run_fast", blown_on_the_third)
    out = catalogue_probe.watch(pg, "d")
    assert out["stands"] and out["walk"] == "fell"


# ---- fixes found on real robots, each with the file that showed it ---------------------------------------------


def _compiled(xml: str):
    import mujoco

    return mujoco.MjModel.from_xml_string(xml)


def test_a_hand_whose_file_names_a_site_tcp_is_not_an_arm():
    from robotforge.design import catalogue_describe

    finger = (
        '<body name="r/{n}" pos="{x} 0 -0.03"><joint name="r/{n}_j1" axis="0 1 0" range="0 1.5"/>'
        '<geom type="capsule" fromto="0 0 0 0 0 -0.05" size="0.008" mass="0.02"/>'
        '<body name="r/{n}_tip" pos="0 0 -0.05"><joint name="r/{n}_j2" axis="0 1 0" range="0 1.5"/>'
        '<geom type="capsule" fromto="0 0 0 0 0 -0.04" size="0.007" mass="0.01"/></body></body>'
    )
    model = _compiled(
        '<mujoco><worldbody><body name="r/palm" pos="0 0 0.3"><geom type="box" size="0.04 0.03 0.02" mass="0.3"/>'
        '<site name="r/tcp" pos="0 0 -0.03"/>'
        + finger.format(n="a", x=-0.02)
        + finger.format(n="b", x=0.02)
        + "</body></worldbody></mujoco>"
    )
    assert catalogue_describe.classify(model, "r", model.qpos0)[0] == "hand"


def test_two_toe_and_heel_pairs_under_a_tall_body_are_two_feet():
    """ergoCub: four contact places, but a body far taller than the feet are long stands on two."""
    from robotforge.design import catalogue_describe

    def leg(side: float) -> str:
        return (
            f'<body name="r/leg{side}" pos="0 {side} -0.2"><joint name="r/hip{side}" axis="0 1 0" range="-1 1"/>'
            '<geom type="capsule" fromto="0 0 0 0 0 -0.6" size="0.04" mass="3"/>'
            f'<body name="r/toe{side}" pos="0.15 0 -0.84"><geom type="sphere" size="0.02" mass="0.1"/></body>'
            f'<body name="r/heel{side}" pos="-0.15 0 -0.84"><geom type="sphere" size="0.02" mass="0.1"/></body></body>'
        )

    model = _compiled(
        '<mujoco><worldbody><body name="r/trunk" pos="0 0 1.2"><freejoint/>'
        '<geom type="box" size="0.1 0.2 0.3" mass="20"/>' + leg(-0.1) + leg(0.1) + "</body></worldbody></mujoco>"
    )
    assert catalogue_describe.classify(model, "r", model.qpos0) == ("humanoid", 2)


def test_a_catalogue_arm_holds_itself_against_gravity(folders):
    """An iiwa stopped 7 cm short: its file's servos know nothing of the weight they carry."""
    pg = Playground.with_demo()
    pg.new_world("t")
    pg.add_robot("a", library="arm")
    assert pg.behaviors._arm_motion("a")._holds_itself is True


def test_the_plain_listing_says_what_the_downloads_take_up(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_CACHE", str(tmp_path))
    monkeypatch.delenv("ROBOTFORGE_REPOS", raising=False)
    write(catalogue_fetch.repo_dir("r", "a" * 40) / "x" / "big.bin", "x" * 3_000_000)
    write(catalogue_fetch.repo_dir("s", "b" * 40) / "y.bin", "y" * 1_000_000)
    write(tmp_path / "catalogue" / catalogue_fetch.TRIALS / "t.json", "{}")  # kept trial results are not downloads
    cache = catalogue.listing()["open_catalogue"]["cache"]
    assert cache["folder"] == str(tmp_path / "catalogue") and cache["mb"] == 4
    assert [row["repo"] for row in cache["largest"]] == [f"r-{'a' * 12}", f"s-{'b' * 12}"]


def test_the_menagerie_folders_the_tuned_library_fetched_are_not_fetched_again(tmp_path, monkeypatch):
    from robotforge.design import menagerie

    monkeypatch.setenv("ROBOTFORGE_CACHE", str(tmp_path))
    monkeypatch.delenv("ROBOTFORGE_REPOS", raising=False)
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", str(tmp_path / "none"))
    fetched = tmp_path / f"menagerie-{menagerie.COMMIT[:12]}" / "unitree_h1"
    write(fetched / "h1.xml", "<mujoco/>")
    assert catalogue_fetch.folder_path("mujoco_menagerie", menagerie.COMMIT, "unitree_h1") == fetched
    assert catalogue_fetch.folder_path("mujoco_menagerie", menagerie.COMMIT, "unitree_h2") is None
