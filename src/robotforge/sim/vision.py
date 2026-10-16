"""Finding things in a camera picture by their color, and working out where they are in the world.

Numpy only, the way a first robot vision program is written: convert the picture to hue, saturation and
value, keep the pixels whose color is in a range (the mask), clean the mask, join neighbouring pixels into
blobs, take each blob's centroid, and turn that pixel back into a ray from the camera. The ray meets the
table at an assumed height, or stops where the depth camera says the surface is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.design import components


@dataclass(frozen=True)
class ColorRange:
    """Pixels whose hue is within `width` degrees of `hue`, at least `sat` saturated (and less than `sat_max`)
    and at least `val` bright."""

    hue: float  # degrees, 0 red, 120 green, 240 blue
    width: float  # degrees either side of `hue`
    sat: float = 0.3  # 0 grey .. 1 pure color (grey floors and white mats stay under 0.2)
    val: float = 0.2  # 0 black .. 1 full brightness
    sat_max: float = math.inf  # brown is a greyer orange: it stops where orange starts

    def describe(self) -> dict[str, float]:
        return {"hue": self.hue, "width": self.width, "sat": self.sat, "val": self.val}


# Windows for the palette's colors (design/components.COLORS), none overlapping, measured on MuJoCo renders of the
# palette's cubes from seven views (tests/test_vision.py) and of every scene's props. They hold for a picture that
# is not overexposed: where the brightest channel clips, hues slide toward the next secondary color (in the world's
# full sun a top-down view put orange at 58 degrees, beside yellow, and blue at 186, beside teal), so the finder
# looks with the exposure a camera would pick (EXPOSURE). There, the palette cubes' middle 90 % of pixels measured:
# purple 259 to 265 degrees and teal 167 to 183, so both windows leave a few degrees past that.
# Orange and brown share their hues and differ in saturation. Orange things measured 0.66 to 0.80 (median over a
# cube, the top seen from above the lowest: the light's shine on it greys it), the palette's brown 0.41 to 0.55,
# and the wood and cardboard of every scene (floors, tables, shelves, crates) at most 0.49 in 99 % of their pixels:
# with orange from 0.3 up, a kitchen's wooden floor swallowed the orange on its table as one orange blob.
SPLIT = 0.6  # saturation between brown (below) and orange (from here up)
NAMED: dict[str, ColorRange] = {
    "red": ColorRange(2.0, 12.0),
    "orange": ColorRange(29.0, 9.0, sat=SPLIT),
    "brown": ColorRange(28.0, 10.0, sat=0.25, sat_max=SPLIT),
    "yellow": ColorRange(47.0, 8.0),
    "green": ColorRange(140.0, 25.0),
    "teal": ColorRange(183.0, 16.0),
    "blue": ColorRange(218.0, 16.0),
    "purple": ColorRange(264.0, 26.0),
    "pink": ColorRange(330.0, 16.0, sat=0.25),  # a pale color: about 0.45 saturated
}
# Share of the world's light the finder's camera lets in, as its exposure: at 0.6 no channel of a sunlit cube
# top clips, and every palette color stayed in its window (at 1.0, the 3D view's brightness, several left it).
EXPOSURE = 0.6
MIN_AREA = 12  # px: smaller blobs are noise (a 4 cm cube 0.7 m from a 640 px camera covers about 800)
MIN_INTERIOR = 9  # px: a blob with fewer pixels off its edge is measured over all of them
TOP_BAND = 0.005  # m: measured points this close to an object's highest count as its top
RING = 4  # px around a blob's box where the depth camera measures what the object stands on
SKY = 0.999  # share of the far clipping distance from which a depth reading means nothing was there
# m: a pixel on a blob's edge more than this nearer or farther than every pixel inside it blends in something in
# front of the object or behind it (measured: an arm's hand 18 cm nearer than a box it hid part of, a park bench
# 1.6 m behind a ball). A cube's own edge pixels were at most 1.1 cm past its inside.
SLACK = 0.03
SUPPORT_BIN = 0.01  # m: heights around a blob are counted in bins this tall; a flat surface fills one of them
HIDDEN_SHARE = 0.05  # of the pixels round a blob: this many nearer than all of it means something covers part of it
SIDE_ON = 0.01  # m: a top seen this shallow along the view is only the top edge of the face the camera sees
MIN_SLANT = 0.1  # sine of the lowest angle (6 degrees) a camera is taken to look down at a surface from
# Share of an object's width, both ways, its top must span for the object to count as box-like (a box, a standing
# can). A ball's top band (TOP_BAND deep) spans 2 * sqrt(2 * TOP_BAND * r) of its 2r: 0.7 or more only for a ball
# under 2 cm in radius, which seen from above ends less than SLACK over the floor, so no ball is taken as raised.
# Coarse pixels make a far ball look boxier (the kitchen's apple from above: 0.79); SIDE_ROWS keeps those out.
BOXY = 0.7
# Pixel rows of an object's side the camera must see to tell where the side ends. A cube on a cube, or in the arm's
# gripper, showed 10 to 17 from every slanting preset view. From straight above, the sides of the scenes' boxes and
# cans showed at most 0.1 and the apple's 0.3 (a big ball shows more of itself, 8 for the park's, but is not boxy).
SIDE_ROWS = 4
RAISED_SHARE = 0.1  # of the measured points round a blob: a flat surface this big under its top may be its support


def color_range(color: str | list[float]) -> ColorRange:
    """A named color's window, or one around any color components.color accepts ("#9c6b3f", [r, g, b])."""
    if isinstance(color, str):
        name = components.COLOR_SPELLINGS.get(color.strip().lower(), color.strip().lower())
        if name in NAMED:
            return NAMED[name]
    rgb = np.array(components.color(color)[:3], dtype=np.float32).reshape(1, 1, 3) * 255
    h, s, v = to_hsv(rgb.astype(np.uint8))[0, 0]
    if s < 0.2:
        raise ValueError(f"{color!r} is too grey to find by its color; find works for colors like red or blue")
    return ColorRange(float(h), 12.0, sat=float(s) * 0.5, val=float(v) * 0.4)


# ---- the mask ---------------------------------------------------------------------------------------------


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    """An HxWx3 uint8 picture as hue (degrees), saturation and value (0..1), float32."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    v = x.max(axis=-1)
    c = v - x.min(axis=-1)
    safe = np.where(c > 0, c, 1.0)
    h = np.where(v == r, ((g - b) / safe) % 6.0, np.where(v == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    h = np.where(c > 0, h * 60.0, 0.0)
    s = np.where(v > 0, c / np.where(v > 0, v, 1.0), 0.0)
    return np.stack([h, s, v], axis=-1).astype(np.float32)


def threshold(hsv: np.ndarray, rng: ColorRange) -> np.ndarray:
    """The mask: True where a pixel's color is inside the range. Hue wraps around (350 is near 10)."""
    off = np.abs((hsv[..., 0] - rng.hue + 180.0) % 360.0 - 180.0)
    sat = hsv[..., 1]
    return (off <= rng.width) & (sat >= rng.sat) & (sat < rng.sat_max) & (hsv[..., 2] >= rng.val)


def _shifts(mask: np.ndarray, fill: bool) -> list[np.ndarray]:
    padded = np.pad(mask, 1, constant_values=fill)
    h, w = mask.shape
    return [padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def erode(mask: np.ndarray) -> np.ndarray:
    """Keep a pixel only if its whole 3x3 neighbourhood is in the mask: specks and thin edges go."""
    return np.logical_and.reduce(_shifts(mask, True))


def dilate(mask: np.ndarray) -> np.ndarray:
    """Add every pixel next to one in the mask: what erosion shaved off a real blob grows back."""
    return np.logical_or.reduce(_shifts(mask, False))


def clean(mask: np.ndarray) -> np.ndarray:
    """Erode, then dilate (an "opening"): lone noisy pixels vanish, solid blobs keep their shape."""
    return dilate(erode(mask))


def blur(rgb: np.ndarray) -> np.ndarray:
    """Each pixel the average of its 3x3 neighbourhood: random noise shrinks to a third, edges soften a pixel."""
    x = np.pad(rgb.astype(np.float32), ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = rgb.shape[:2]
    total = sum(x[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    return np.clip(total / 9.0 + 0.5, 0, 255).astype(np.uint8)


def white_balance(rgb: np.ndarray) -> np.ndarray:
    """Gray world: scale each channel so the picture's average is grey, which takes a colored light's tint
    out the way a camera's automatic white balance does."""
    x = rgb.astype(np.float32)
    means = x.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1.0)
    return np.clip(x * gain, 0, 255).astype(np.uint8)


# ---- blobs ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Blob:
    """Touching mask pixels (8-connected). Pixel (x, y) covers x..x+1 across and y..y+1 down."""

    area: int  # pixels
    u: float  # centroid across, px from the left edge
    v: float  # centroid down, px from the top edge
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive)
    rows: tuple[tuple[int, int, int], ...]  # its runs: (y, x0, x1), for the pixels it covers

    def pixels(self) -> tuple[np.ndarray, np.ndarray]:
        """Row and column indices of every pixel in the blob."""
        ys = np.concatenate([np.full(x1 - x0 + 1, y) for y, x0, x1 in self.rows])
        xs = np.concatenate([np.arange(x0, x1 + 1) for _, x0, x1 in self.rows])
        return ys, xs

    def interior(self) -> tuple[np.ndarray, np.ndarray]:
        """The pixels whose eight neighbours are all in the blob too (all of them for a blob too thin to have
        any). Along its edge a pixel blends the object with what is behind it, and its depth may be either."""
        x0, y0, x1, y1 = self.box
        local = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
        ys, xs = self.pixels()
        local[ys - y0, xs - x0] = True
        inner = np.logical_and.reduce(_shifts(local, False))
        if inner.sum() < MIN_INTERIOR:
            return ys, xs
        iy, ix = np.nonzero(inner)
        return iy + y0, ix + x0


def _runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Every horizontal run of True as (y, x0, x1), row by row, left to right."""
    h, w = mask.shape
    edges = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    out = []
    for y in range(h):
        row = edges[y]
        starts = np.flatnonzero(row == 1)
        ends = np.flatnonzero(row == -1) - 1
        out.extend((y, int(a), int(b)) for a, b in zip(starts, ends, strict=True))
    return out


def blobs(mask: np.ndarray, min_area: int = MIN_AREA) -> list[Blob]:
    """The mask's blobs of at least `min_area` pixels, biggest first.

    Labelled run by run: runs on neighbouring rows that touch (diagonals count) join one blob, kept with
    union-find. A picture has a few hundred runs, so this is quick without looping over pixels.
    """
    runs = _runs(mask)
    parent = list(range(len(runs)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_row: dict[int, list[int]] = {}
    for i, (y, _, _) in enumerate(runs):
        by_row.setdefault(y, []).append(i)
    for y, here in by_row.items():
        above = by_row.get(y - 1, [])
        k = 0
        for i in here:
            _, a0, a1 = runs[i]
            while k < len(above) and runs[above[k]][2] < a0 - 1:  # ends left of this run (and its diagonal)
                k += 1
            j = k
            while j < len(above) and runs[above[j]][1] <= a1 + 1:
                ri, rj = root(i), root(above[j])
                if ri != rj:
                    parent[ri] = rj
                j += 1
    groups: dict[int, list[tuple[int, int, int]]] = {}
    for i, run in enumerate(runs):
        groups.setdefault(root(i), []).append(run)
    found = []
    for rows in groups.values():
        lengths = np.array([x1 - x0 + 1 for _, x0, x1 in rows], dtype=np.float64)
        area = int(lengths.sum())
        if area < min_area:
            continue
        xs = np.array([(x0 + x1) / 2 for _, x0, x1 in rows])  # the mean column of each run
        ys = np.array([y for y, _, _ in rows], dtype=np.float64)
        box = (min(r[1] for r in rows), min(r[0] for r in rows), max(r[2] for r in rows), max(r[0] for r in rows))
        found.append(
            Blob(
                area=area,
                u=float((lengths * xs).sum() / area) + 0.5,
                v=float((lengths * ys).sum() / area) + 0.5,
                box=box,
                rows=tuple(rows),
            )
        )
    return sorted(found, key=lambda b: -b.area)


# ---- from a pixel back into the world ---------------------------------------------------------------------


@dataclass(frozen=True)
class Pinhole:
    """A camera as the picture was taken: where it is, which way it looks, and how wide."""

    pos: np.ndarray  # world position, m
    right: np.ndarray  # unit vectors in the world: across the picture, up the picture, and where it looks
    up: np.ndarray
    forward: np.ndarray
    fx: float  # focal lengths in pixels
    fy: float
    cx: float  # where the optical axis meets the picture, px
    cy: float
    width: int
    height: int
    far: float = np.inf  # m along the view: the depth camera reads this where nothing is (the sky)

    @classmethod
    def from_scene(cls, scene: mujoco.MjvScene, width: int, height: int) -> Pinhole:
        """The camera MuJoCo drew the last picture with (a model camera or a free view alike)."""
        # The scene keeps a left and a right eye half the stereo eye distance (6.8 cm) apart; a plain picture
        # is drawn from between them. Using one eye put every estimate 3.4 cm to the side.
        cam = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
        forward = np.array(cam.forward, dtype=np.float64)
        up = np.array(cam.up, dtype=np.float64)
        right = np.cross(forward, up)
        near = float(cam.frustum_near)
        half_h = (float(cam.frustum_top) - float(cam.frustum_bottom)) / 2
        # MuJoCo stretches the frustum's width to the picture's shape when it renders (mjr_render).
        half_w = float(cam.frustum_width) or half_h * width / height
        fy = height / 2 * near / half_h
        fx = width / 2 * near / half_w
        cx = width / 2 - float(cam.frustum_center) / half_w * width / 2
        cy = height / 2 + (float(cam.frustum_top) + float(cam.frustum_bottom)) / 2 / half_h * height / 2
        far = float(cam.frustum_far)
        return cls(np.array(cam.pos, dtype=np.float64), right, up, forward, fx, fy, cx, cy, width, height, far)

    def ray(self, u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
        """Directions through pixel positions, scaled so each goes one meter along the optical axis: a point
        `d` meters deep (the depth camera's reading) is pos + d * ray."""
        u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        x = (u - self.cx) / self.fx
        y = -(v - self.cy) / self.fy
        return self.forward + x[..., None] * self.right + y[..., None] * self.up

    def on_plane(self, u: float, v: float, z: float) -> np.ndarray | None:
        """Where the ray through pixel (u, v) meets the horizontal plane at height z, or None if it never does."""
        d = self.ray(u, v)
        if d[2] >= -1e-9 or self.pos[2] <= z:
            return None
        return self.pos + (z - self.pos[2]) / d[2] * d

    def project(self, point: np.ndarray) -> tuple[float, float] | None:
        """The pixel position a world point lands on, or None if it is behind the camera."""
        rel = np.asarray(point, dtype=np.float64) - self.pos
        depth = float(rel @ self.forward)
        if depth <= 1e-9:
            return None
        return self.cx + self.fx * float(rel @ self.right) / depth, self.cy - self.fy * float(rel @ self.up) / depth


def depth_points(cam: Pinhole, depth: np.ndarray, blob: Blob, interior: bool = True) -> np.ndarray:
    """The world points the depth camera measured over a blob, one per pixel (Nx3): inside it, or all of it."""
    ys, xs = blob.interior() if interior else blob.pixels()
    return cam.pos + depth[ys, xs][:, None] * cam.ray(xs + 0.5, ys + 0.5)


def locate(cam: Pinhole, depth: np.ndarray, blob: Blob) -> tuple[np.ndarray, np.ndarray]:
    """Where the object under a blob is, by the depth camera: its middle and the size of the box around it.

    A camera above an object sees all of its top but only part of its sides, and none of its bottom, so the
    middle is not the middle of the points it measured (that sat 0.6 cm high and toward the camera on the
    lesson's cubes). Across, it is the middle of the top (the points within TOP_BAND of the highest); up, half
    way between the top and its bottom (_bottom): the surface it stands on, which the depth camera measures
    around it, or where its sides are seen to end when that is well above the surface. The outermost 2 percent
    of points are ignored everywhere.
    """
    center, size, _ = _locate(cam, depth, blob)
    return center, size


def _locate(
    cam: Pinhole, depth: np.ndarray, blob: Blob, tinted: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, bool]:
    """locate, and whether the object may stand on something higher than its bottom was taken to be. `tinted`
    marks the pixels of the blob's hue however dark (tinted): its own shaded faces, which are not around it."""
    points = depth_points(cam, depth, blob)
    top, upper, whole = _top(cam, depth, blob, points)
    across = (np.percentile(upper[:, :2], 2, axis=0) + np.percentile(upper[:, :2], 98, axis=0)) / 2
    low, high = np.percentile(points, 2, axis=0), np.percentile(points, 98, axis=0)
    lowest = min(float(low[2]), float(np.percentile(whole[:, 2], 2)))
    bottom, maybe_higher = _bottom(cam, depth, blob, top, upper, whole, lowest, tinted)
    size = np.array([high[0] - low[0], high[1] - low[1], top - bottom])
    return np.array([across[0], across[1], (top + bottom) / 2]), size, maybe_higher


def _top(cam: Pinhole, depth: np.ndarray, blob: Blob, inside: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """An object's top height (from the points inside its blob), the points of its top, and all its points.
    The top is one of the inside's points, so the top's points are never none."""
    top = float(np.percentile(inside[:, 2], 98, method="lower"))
    whole = own_points(cam, depth, blob)
    # The top's edge pixels count: one that blends in the floor behind has the floor's height and drops out
    # anyway, and leaving them all out shaved the far edge only (the near one runs on into the side), which
    # pulled the middle 3 mm toward the camera. The band is closed above too, for what own_points lets through.
    z = whole[:, 2]
    return top, whole[(z >= top - TOP_BAND) & (z <= top + TOP_BAND)], whole


def own_points(cam: Pinhole, depth: np.ndarray, blob: Blob) -> np.ndarray:
    """The world points measured over a whole blob, less its edge pixels that show something else.

    An edge pixel blends the object's color with what is next to it, and the depth camera gives it one of the
    two depths: a hand in front of a box lent the box's blob four pixels 10 cm above the box's top, enough to
    move its middle 7.7 cm. Pixels more than SLACK nearer or farther than every pixel inside the blob go (the
    inside's own pixels all stay).
    """
    ys, xs = blob.pixels()
    d = depth[ys, xs]
    iy, ix = blob.interior()
    near, far = float(depth[iy, ix].min()), float(depth[iy, ix].max())
    keep = (d >= near - SLACK) & (d <= far + SLACK)
    ys, xs, d = ys[keep], xs[keep], d[keep]
    return cam.pos + d[:, None] * cam.ray(xs + 0.5, ys + 0.5)


def _bottom(
    cam: Pinhole,
    depth: np.ndarray,
    blob: Blob,
    top: float,
    upper: np.ndarray,
    whole: np.ndarray,
    lowest: float,
    tinted: np.ndarray | None = None,
) -> tuple[float, bool]:
    """The height of an object's bottom, and whether it may in fact be higher (find's flag `raised`).

    Usually the bottom is the surface the object stands on (_support). But a box-like thing whose sides are seen
    to end well above that surface stands on something smaller or is held up, and the camera sees past its
    bottom: a red cube on a blue one came out 2.5 cm low, one in the arm's gripper 4 cm low. Its bottom is then
    the lowest point seen on it. Where that cannot be told (no side is seen, from straight above, or it is round
    and its widest is what is seen) it is taken to stand on the surface, and may be higher if a flat surface
    above that one is seen around it.
    """
    around = _ring_points(cam, depth, blob)
    support = _support(around, top, lowest)
    # The lowest point seen, not a percentile: on a small blob the 2nd percentile is a pixel or two up its side.
    # An edge pixel that blends in what is under it sits lower still, which only says "it stands there".
    end = float(whole[:, 2].min())
    if end - support <= SLACK:  # its sides run down to the surface: it stands there
        return support, False
    # A side is what is seen below the top band: a wrist camera close over a cube sees only its top, whose depth
    # varied by 3 mm, and from that steep a pixel row is half a millimetre, so the rows alone took it for a side.
    side = int((whole[:, 2] < top - 2 * TOP_BAND).sum())
    if side >= MIN_INTERIOR and top - end >= SIDE_ROWS * _row_height(cam, depth, blob) and _boxy(upper, whole):
        return end, False
    # Close up, the ring round a cube is half its own shaded side, whose top few cm piled up like a surface
    # just under its top; that side has the cube's hue, so what has it is left out.
    return support, _higher_surface(_ring_points(cam, depth, blob, tinted), top, support)


def _row_height(cam: Pinhole, depth: np.ndarray, blob: Blob) -> float:
    """How much of an upright side one pixel row covers at the object (m): a pixel's width at its distance,
    stretched by how steeply the camera looks down at it (from straight above a side is not seen at all)."""
    iy, ix = blob.interior()
    distance = float(np.median(depth[iy, ix]))
    ray = cam.ray(blob.u, blob.v)
    level = float(np.linalg.norm(ray[:2]) / np.linalg.norm(ray))
    return distance / min(cam.fx, cam.fy) / max(level, MIN_SLANT)


def _boxy(upper: np.ndarray, whole: np.ndarray) -> bool:
    """Its top spans at least BOXY of its width both ways, as a box's or a standing can's does; a ball's top
    is a small cap. Measured along the main directions of what was seen, so a box turned 45 degrees counts."""
    xy = whole[:, :2] - whole[:, :2].mean(axis=0)
    axes = np.linalg.svd(xy, full_matrices=False)[2]

    def span(points: np.ndarray) -> np.ndarray:
        along = points[:, :2] @ axes.T
        return np.percentile(along, 98, axis=0) - np.percentile(along, 2, axis=0)

    return bool(np.all(span(upper) >= BOXY * span(whole)))


def _support(around: np.ndarray, top: float, lowest: float) -> float:
    """The height of the surface the object stands on: the flat surface seen most around it, below its top.

    A flat surface puts many points in one SUPPORT_BIN; a wall or a table's side spreads its points over many,
    so the median of everything around (the old way) sank to 0.6 m for a cereal box on a 0.9 m kitchen island
    with the island's side below it. It cannot be above the lowest point seen on the object itself.
    """
    below = around[around[:, 2] < top - TOP_BAND, 2]
    if below.size < MIN_INTERIOR:
        return lowest
    bins = np.floor((below - below.min()) / SUPPORT_BIN).astype(np.int64)
    fullest = int(np.argmax(np.bincount(bins)))
    return min(float(np.median(below[bins == fullest])), lowest)


def _higher_surface(around: np.ndarray, top: float, support: float) -> bool:
    """A flat surface around the object, more than SLACK above the one it is taken to stand on and below its
    top, covering at least RAISED_SHARE of what is seen around it (in one SUPPORT_BIN): it may stand on that."""
    below = around[around[:, 2] < top - TOP_BAND, 2]
    higher = below[below > support + SLACK]
    if higher.size < max(MIN_INTERIOR, RAISED_SHARE * below.size):
        return False
    bins = np.floor((higher - higher.min()) / SUPPORT_BIN).astype(np.int64)
    return int(np.bincount(bins).max()) >= RAISED_SHARE * below.size


def _ring_points(cam: Pinhole, depth: np.ndarray, blob: Blob, leave_out: np.ndarray | None = None) -> np.ndarray:
    """The measured points just outside a blob's box (RING pixels wide): what the object stands on, mostly.
    Where nothing was measured (the sky) is left out, and so are the pixels `leave_out` marks."""
    h, w = depth.shape
    x0, y0, x1, y1 = blob.box
    xa, ya, xb, yb = max(x0 - RING, 0), max(y0 - RING, 0), min(x1 + RING, w - 1), min(y1 + RING, h - 1)
    ys, xs = np.mgrid[ya : yb + 1, xa : xb + 1]
    outside = (xs < x0) | (xs > x1) | (ys < y0) | (ys > y1)
    ys, xs = ys[outside], xs[outside]
    d = depth[ys, xs]
    seen = d < cam.far * SKY
    if leave_out is not None:
        seen &= ~leave_out[ys, xs]
    return cam.pos + d[seen][:, None] * cam.ray(xs[seen] + 0.5, ys[seen] + 0.5)


# ---- finding colored objects (observe(find=...)) ----------------------------------------------------------


def cut_off(blob: Blob, shape: tuple[int, ...]) -> bool:
    """The blob touches the picture's edge, so part of its object may be out of view and its middle off."""
    x0, y0, x1, y1 = blob.box
    return x0 == 0 or y0 == 0 or x1 == shape[1] - 1 or y1 == shape[0] - 1


def hidden(depth: np.ndarray, blob: Blob) -> bool:
    """Something in front covers part of the object: of the pixels just round the blob and its own edge pixels,
    at least HIDDEN_SHARE (counted against the pixels round it) are more than SLACK nearer than all of its inside.
    What it stands on and what is behind it are never nearer, so they do not count."""
    h, w = depth.shape
    x0, y0, x1, y1 = blob.box
    xa, ya = max(x0 - 1, 0), max(y0 - 1, 0)
    local = np.zeros((min(y1 + 1, h - 1) - ya + 1, min(x1 + 1, w - 1) - xa + 1), dtype=bool)
    ys, xs = blob.pixels()
    local[ys - ya, xs - xa] = True
    ny, nx = np.nonzero(dilate(local) & ~local)
    iy, ix = blob.interior()
    near = float(depth[iy, ix].min()) - SLACK
    nearer = int((depth[ny + ya, nx + xa] < near).sum() + (depth[ys, xs] < near).sum())
    return nearer >= HIDDEN_SHARE * max(ny.size, 1)


def side_on(cam: Pinhole, depth: np.ndarray, blob: Blob) -> bool:
    """The camera sees a face of the object and not its top (from low down, or with a shelf above it): the
    top it measured is a line less than SIDE_ON deep along the view, and the middle found is that face's."""
    ahead = cam.forward[:2]
    if np.linalg.norm(ahead) < 1e-6:  # looking straight down: every top is in view
        return False
    _, upper, _ = _top(cam, depth, blob, depth_points(cam, depth, blob))
    along = upper[:, :2] @ (ahead / np.linalg.norm(ahead))
    return float(np.percentile(along, 98) - np.percentile(along, 2)) < SIDE_ON


def pixel_reach(cam: Pinhole, depth: np.ndarray, blob: Blob) -> float:
    """How much of a level surface one pixel covers at the object, along the view (m): a pixel's width at the
    object's distance, stretched by how slanting the camera looks at it (seen from 10 degrees up, six times)."""
    iy, ix = blob.interior()
    distance = float(np.median(depth[iy, ix]))
    ray = cam.ray(blob.u, blob.v)
    down = max(-float(ray[2]) / float(np.linalg.norm(ray)), MIN_SLANT)
    return distance / min(cam.fx, cam.fy) / down


@dataclass(frozen=True)
class Found:
    """A blob of one color, where its object is by the depth camera, and what may make that less sure."""

    color: str
    blob: Blob
    position: np.ndarray  # the middle of the box around its measured points, world m
    size: np.ndarray  # that box's extent, m
    cut_off: bool = False  # the blob touches the picture's edge
    hidden: bool = False  # something nearer covers part of it
    side_on: bool = False  # only a face of it is seen, not its top: the middle found is the face's
    raised: bool = False  # it may stand on something higher than its bottom was taken to be (_bottom)
    reach: float = 0.0  # m of a level surface one pixel covers at the object, along the view (pixel_reach)


def find(rgb: np.ndarray, depth: np.ndarray, cam: Pinhole, color: str, min_area: int = MIN_AREA) -> list[Found]:
    """Every blob of `color` (a name, a hex code or [r, g, b]) in the picture, biggest first, located in 3D."""
    rng = color_range(color)
    hsv = to_hsv(rgb)
    # Only where the depth camera measured something: a bluish grey sky otherwise came back as a blue thing.
    mask = clean(threshold(hsv, rng) & (depth < cam.far * SKY))
    tinted = threshold(hsv, ColorRange(rng.hue, rng.width, sat=rng.sat, val=0.0))  # its hue in any light or shade
    out = []
    for blob in blobs(mask, min_area):
        center, size, raised = _locate(cam, depth, blob, tinted)
        out.append(
            Found(
                color if isinstance(color, str) else "color",
                blob,
                center,
                size,
                cut_off=cut_off(blob, rgb.shape),
                hidden=hidden(depth, blob),
                side_on=side_on(cam, depth, blob),
                raised=raised,
                reach=pixel_reach(cam, depth, blob),
            )
        )
    return out
