"""Where worlds live on disk (design/store.py)."""

from robotforge.design.store import WorldStore


def test_a_world_saved_before_names_got_codes_still_opens_by_its_name(tmp_path, monkeypatch):
    """Review of the error audit: "Café" now lives in caf_<code>, but robotforge saved it in caf before."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    old = tmp_path / "worlds" / "caf"
    old.mkdir(parents=True)
    store = WorldStore.create("cafe_tmp")
    (old / "world.yaml").write_text(store.file.read_text(encoding="utf-8").replace("cafe_tmp", "caf"), encoding="utf-8")
    assert WorldStore.open("Café").dir == old
    assert WorldStore.create("Café").dir == old and WorldStore.create("Café").reopened
    assert WorldStore.create("咖啡馆").dir.name.startswith("world_")  # names with no Latin letters keep their code
