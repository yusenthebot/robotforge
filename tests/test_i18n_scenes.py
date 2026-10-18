"""Every scene has its name and a one-line summary in every language of the viewer: without them the maps panel
falls back to the scene's English title and long summary, which showed as a five-line English card among the
one-line Chinese ones."""

import re
from pathlib import Path

from robotforge.design import scenes

I18N = Path(__file__).parents[1] / "src" / "robotforge" / "viewer" / "static" / "i18n.js"


def _locales() -> dict[str, set[str]]:
    text = I18N.read_text(encoding="utf-8")
    body = text[text.index("const STRINGS = {") :]
    starts = [(m.group(1), m.end()) for m in re.finditer(r"^  ([a-z]{2}): \{$", body, re.M)]
    locales = {}
    for (lang, start), nxt in zip(starts, [*starts[1:], (None, len(body))], strict=True):
        block = body[start : nxt[1]]
        locales[lang] = set(re.findall(r"^ {4}([a-z0-9_]+):", block, re.M))
    return locales


def test_every_scene_has_a_title_and_summary_in_every_language():
    locales = _locales()
    assert set(locales) >= {"en", "zh"}
    for lang, keys in locales.items():
        missing = [k for name in scenes.SCENES for k in (f"map_{name}", f"map_{name}_desc") if k not in keys]
        assert not missing, f"{lang} lacks {missing}"
