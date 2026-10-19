"""Packaged goods: cartons with printed labels, books, tins, foam bricks and squeeze bottles (YCB-like)."""

from __future__ import annotations

import numpy as np

from robotforge.design.visuals import mesh as M
from robotforge.objects import kit, labels
from robotforge.objects.kit import Choice, Flag, Gen, Range, mix, shade
from robotforge.objects.registry import generator

WORDS = labels.WORDS
WHITE = [0.97, 0.96, 0.93, 1.0]
INK = [0.1, 0.1, 0.12, 1.0]


def _word(g: Gen, label: str = "word") -> str:
    return WORDS[int(g.rng(label).integers(len(WORDS)))]


def _add(g: Gen, sheet) -> None:
    for rgba, mesh in sheet.patches:
        g.decor(mesh, rgba, "paint")


@generator(
    "package",
    size="[length, width, height]",
    default_size=(0.15, 0.058, 0.2),
    color="#e0a030",
    material="paper",
    kind="package",
    summary="a printed carton (cereal box, tea box or brown parcel): a brand, a picture, a barcode on its back; "
    "take it across its narrow side",
    keywords=("package", "carton", "cereal box", "cereal", "tea box", "parcel", "packet", "box of"),
    params={
        "style": Choice("cereal", "tea", "parcel", doc="what is printed: a cereal box, a tea box or a taped parcel"),
        "density": Range(150.0, 500.0, 300.0, "kg/m^3 of what is inside"),
    },
    palette=("#e0a030", "#c0392b", "#2e86c1", "#27ae60", "#8e44ad"),
    builtin=True,
)
def package(g: Gen) -> None:
    length, width, height = g.size
    hx, hy = length / 2, width / 2
    style = g.p("style")
    base = list(g.rgba)
    if style == "parcel":
        base = [0.72, 0.56, 0.36, 1.0]
    body = M.rounded_box(hx, hy, height / 2, 0.0018, segments=2, corner=0.0018).moved((0, 0, height / 2))
    mass = length * width * height * g.p("density")
    g.part(body, "paper", [kit.box(hx, hy, height / 2, (0, 0, height / 2))], base, mass=mass)
    word = _word(g)
    ink, accent = shade(base, 0.35), mix(base, WHITE, 0.55)
    front = labels.Sheet((0, -hy, 0), (1, 0, 0), (0, 0, 1))
    back = labels.Sheet((0, hy, 0), (-1, 0, 0), (0, 0, 1))
    if style == "cereal":
        front.rect(ink, -hx + 0.004, height * 0.83, hx - 0.004, height * 0.965)
        front.text(WHITE, word, 0.0, height * 0.862, height * 0.07, "center")
        front.disc(WHITE, 0.0, height * 0.52, min(hx, height) * 0.36, layer=0)
        front.disc(accent, 0.0, height * 0.52, min(hx, height) * 0.28, layer=1)
        front.disc(ink, 0.0, height * 0.52, min(hx, height) * 0.14, layer=2)
        front.stripes(WHITE, -hx + 0.006, height * 0.07, hx - 0.006, height * 0.16, 3)
        front.text(WHITE, "NEW", hx - 0.006, height * 0.72, height * 0.04, "right")
    elif style == "tea":
        front.rect(WHITE, -hx + 0.004, height * 0.3, hx - 0.004, height * 0.74)
        front.text(ink, word, 0.0, height * 0.52, height * 0.075, "center")
        front.stripes(ink, -hx + 0.01, height * 0.36, hx - 0.01, height * 0.42, 2)
        front.text(WHITE, "20 BAGS", 0.0, height * 0.12, height * 0.045, "center", layer=0)
    else:
        front.rect([0.85, 0.8, 0.5, 1.0], -hx, height * 0.55, hx, height * 0.62)  # tape
        front.text(INK, "FRAGILE", 0.0, height * 0.3, height * 0.08, "center", layer=0)
        front.rect(INK, -hx * 0.4, height * 0.12, hx * 0.4, height * 0.16)
    back.rect(WHITE, -hx + 0.006, height * 0.1, -hx + 0.006 + hx * 0.7, height * 0.5)
    back.text(INK, "FACTS", -hx + 0.01, height * 0.44, height * 0.04)
    back.stripes(INK, -hx + 0.01, height * 0.2, -hx + hx * 0.64, height * 0.4, 5)
    back.rect(WHITE, hx * 0.15, height * 0.1, hx - 0.006, height * 0.28)
    back.barcode(INK, hx * 0.2, height * 0.12, hx * 0.7, height * 0.14, int(g.rng("bar").integers(1 << 30)))
    for sheet in (front, back):
        _add(g, sheet)
    seam = labels.Sheet((0, -hy, 0), (1, 0, 0), (0, 0, 1))
    seam.rect(shade(base, 0.7), -hx, height * 0.985, hx, height * 0.992)
    _add(g, seam)
    jaw = (0.0, 1.0)
    g.grasp(
        "body", (0.0, 0.0, height * 0.5), jaw, width + 0.004, 1.0, "across the narrow side, jaws on the two big faces"
    )
    g.anchor("top", (0.0, 0.0, height))
    g.fact(kind="package", top_z=height)


@generator(
    "book",
    size="[length, width, thickness]",
    default_size=(0.21, 0.14, 0.028),
    color="#2a5d8f",
    material="paper",
    kind="package",
    summary="a hardback book with a title on its cover; standing on its spine it can be taken across its thickness, "
    "lying flat it is only for stacking things on",
    keywords=("book", "novel", "notebook", "textbook"),
    params={"lying": Flag(True, "lying flat (true) or standing on its bottom edge (false)", vary=False)},
    palette=("#2a5d8f", "#8e2c2c", "#2e6b4f", "#5b3a1f", "#3d3d5c"),
    hints={"standing": {"lying": False}, "upright": {"lying": False}, "flat": {"lying": True}},
    builtin=True,
)
def book(g: Gen) -> None:
    length, width, thick = g.size
    hx, hy = length / 2, width / 2
    lying = g.p("lying")
    cover = g.rgba
    page = [0.95, 0.92, 0.82, 1.0]
    board = 0.0016

    def at(mesh: M.Mesh) -> M.Mesh:  # lying frame: x the length, y the width, z the thickness above the table
        if lying:
            return mesh
        rot = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
        return mesh.moved((-thick / 2 + 0 * hx, 0.0, hx), rot)

    def pt(v: tuple[float, float, float]) -> tuple[float, float, float]:
        if lying:
            return v
        return (v[2] - thick / 2, -v[1], v[0] + hx)

    boards = [
        M.rounded_box(hx, hy, board / 2, 0.0008, 1, 0.0015).moved((0, 0, board / 2)),
        M.rounded_box(hx, hy, board / 2, 0.0008, 1, 0.0015).moved((0, 0, thick - board / 2)),
        M.rounded_box(board / 2 + 0.0006, hy, thick / 2, 0.001, 2, 0.0015).moved((-hx + board / 2, 0, thick / 2)),
    ]
    g_cover = M.merge([at(b) for b in boards])
    pages = at(
        M.rounded_box(hx - 0.004, hy - 0.002, thick / 2 - board, 0.0005, 1, 0.0005).moved((-0.001, 0.0, thick / 2))
    )
    half = (hx, hy, thick / 2) if lying else (thick / 2, hy, hx)
    pos = (0.0, 0.0, thick / 2) if lying else (0.0, 0.0, hx)
    g.part(g_cover, "paper", [kit.box(*half, pos)], cover, mass=length * width * thick * 650.0)
    g.decor(pages, page, "paint")
    word = _word(g)
    top = labels.Sheet((0, 0, thick), (1, 0, 0), (0, 1, 0))
    top.rect(shade(cover, 1.6), -hx + 0.03, -0.012, hx - 0.02, 0.026)
    top.text(WHITE, word, 0.0, -0.002, 0.016, "center")
    top.rect(WHITE, -hx + 0.03, -0.034, hx - 0.02, -0.032)
    top.text(shade(cover, 1.9), "A NOVEL", 0.0, -0.05, 0.007, "center")
    for rgba, mesh in top.patches:
        g.decor(at(mesh), rgba, "paint")
    if lying:
        g.fact(kind="book", lying=True, top_z=thick)
    else:
        g.grasp(
            "edge", (0.0, 0.0, length - 0.014), (1.0, 0.0), thick + 0.004, 1.0, "across the thickness by the top edge"
        )
        g.fact(kind="book", lying=False, top_z=length)
    g.anchor("top", pt((0.0, 0.0, thick)))


@generator(
    "tin",
    size="[length, width, height]",
    default_size=(0.1, 0.052, 0.082),
    color="#3b6fb6",
    material="steel",
    kind="package",
    summary="a rectangular meat tin with a pull-ring lid and a printed label, like the YCB potted meat can",
    keywords=("tin", "spam", "potted meat", "meat can", "ycb"),
    palette=("#3b6fb6", "#c0392b", "#2e8b57"),
    builtin=True,
)
def tin(g: Gen) -> None:
    length, width, height = g.size
    hx, hy = length / 2, width / 2
    lip = 0.004
    body = M.rounded_box(hx, hy, height / 2, 0.003, 2, 0.01).moved((0, 0, height / 2))
    g.part(body, "steel", [kit.box(hx, hy, height / 2, (0, 0, height / 2))], [0.75, 0.77, 0.8, 1.0], mass=0.37)
    label = M.rounded_box(hx + 0.0006, hy + 0.0006, (height - 2 * lip) / 2, 0.002, 2, 0.01).moved((0, 0, height / 2))
    g.decor(label, g.rgba, "paint")
    word = "MEAT" if g.rng("w").random() < 0.5 else "HAM"
    front = labels.Sheet((0, -hy - 0.0006, lip), (1, 0, 0), (0, 0, 1))
    h = height - 2 * lip
    front.rect(WHITE, -hx + 0.008, h * 0.28, hx - 0.008, h * 0.72)
    front.text(g.rgba, word, 0.0, h * 0.4, h * 0.2, "center")
    front.stripes(WHITE, -hx + 0.008, h * 0.06, hx - 0.008, h * 0.18, 2)
    _add(g, front)
    g.decor(
        M.rounded_box(hx - 0.006, hy - 0.006, 0.0006, 0.0004, 1, 0.006).moved((0, 0, height + 0.0006)),
        [0.8, 0.82, 0.84, 1.0],
        "metal",
    )
    g.decor(M.torus(0.007, 0.0012, 20, 6).moved((hx * 0.5, 0, height + 0.003)), [0.85, 0.86, 0.88, 1.0], "metal")
    g.grasp("body", (0.0, 0.0, height * 0.5), (0.0, 1.0), width + 0.004, 1.0, "across the narrow side")
    g.fact(kind="tin", top_z=height)


@generator(
    "brick",
    size="[length, width, height]",
    default_size=(0.075, 0.05, 0.05),
    color="#c8463c",
    material="foam",
    kind="thing",
    summary="a foam brick (light, soft-looking, grippy), like the YCB foam brick",
    keywords=("foam brick", "brick", "sponge", "foam block", "block"),
    palette=("#c8463c", "#3a7bd5", "#e0b030"),
    builtin=True,
)
def brick(g: Gen) -> None:
    length, width, height = g.size
    body = M.noise_displaced(
        M.dense_rounded_box(length / 2, width / 2, height / 2, 0.005, 0.006).moved((0, 0, height / 2)),
        int(g.rng("n").integers(1 << 30)),
        0.0006,
        0.01,
    )
    g.part(body, "foam", [kit.box(length / 2, width / 2, height / 2, (0, 0, height / 2))], g.rgba)
    g.grasp("body", (0.0, 0.0, height * 0.5), (0.0, 1.0), width + 0.004, 1.0, "across the narrow side")
    g.fact(kind="brick", top_z=height)


@generator(
    "squeeze_bottle",
    size="[length, width, height]",
    default_size=(0.095, 0.058, 0.19),
    color="#f2c200",
    material="plastic",
    kind="container",
    summary="a flat squeeze bottle with a flip cap and a label (mustard or ketchup), like the YCB mustard bottle",
    keywords=("mustard", "ketchup", "squeeze bottle", "sauce bottle", "sauce"),
    params={"cap": Choice("red", "white", "black")},
    palette=("#f2c200", "#c0392b"),
    hints={"ketchup": {"cap": "red", "_color": "red"}, "mustard": {"cap": "white", "_color": "yellow"}},
    builtin=True,
)
def squeeze_bottle(g: Gen) -> None:
    length, width, height = g.size
    sx, sy = length / 2, width / 2
    prof = [
        (0, 0),
        (0.88, 0),
        (1.0, 0.03),
        (1.0, 0.6),
        (0.9, 0.72),
        (0.55, 0.82),
        (0.27, 0.86),
        (0.27, 0.9),
        (0.0, 0.9),
    ]
    outer = [(r, z * height) for r, z in prof]
    mesh = M.lathe(outer, 40, sharp=(1,)).moved(scale=(sx, sy, 1.0))
    hull_prof = [(0, 0), (0.9, 0), (1.0, 0.04), (1.0, 0.6), (0.55, 0.82), (0.27, 0.9), (0.0, 0.9)]
    convex = M.lathe([(r, z * height) for r, z in hull_prof], 24, sharp=(1,)).moved(scale=(sx, sy, 1.0))
    g.part(mesh, "plastic", [kit.hull(convex)], g.rgba, mass=0.2)
    cap_color = {"red": [0.75, 0.1, 0.08, 1], "white": [0.93, 0.93, 0.92, 1], "black": [0.08, 0.08, 0.09, 1]}[
        g.p("cap")
    ]
    cap = M.lathe(
        [
            (0, height * 0.88),
            (sx * 0.3, height * 0.88),
            (sx * 0.27, height * 0.95),
            (sx * 0.12, height * 0.99),
            (0, height),
        ],
        24,
    ).moved(scale=(1.0, sy / sx * 1.0, 1.0))
    g.part(cap, "plastic", [kit.cylinder(sx * 0.28, height * 0.06, (0, 0, height * 0.94))], cap_color, mass=0.01)
    word = "MUSTARD" if g.p("cap") != "red" else "KETCHUP"
    front = labels.Sheet((0, -sy - 0.0008, 0), (1, 0, 0), (0, 0, 1))
    front.rect(WHITE, -sx * 0.7, height * 0.2, sx * 0.7, height * 0.5)
    front.text(shade(g.rgba, 0.55), word, 0.0, height * 0.33, height * 0.045, "center")
    front.stripes(shade(g.rgba, 0.55), -sx * 0.62, height * 0.23, sx * 0.62, height * 0.29, 2)
    _add(g, front)
    g.grasp("body", (0.0, 0.0, height * 0.32), (0.0, 1.0), width + 0.004, 1.0, "across the flat sides")
    g.anchor("top", (0.0, 0.0, height))
    g.fact(kind="bottle", closed=True)
