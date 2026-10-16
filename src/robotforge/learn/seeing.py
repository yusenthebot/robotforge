"""Sensing III: a camera finds the red cube by its color, works out where it is, and the arm picks it.

A camera on a post looks down at five cubes on a mat. The lesson runs a real color finder on the real
picture (robotforge.sim.vision): hue, saturation and value thresholds make a mask, the mask's blobs give
centroids, and the centroid's ray meets the table at an assumed height, or the depth camera says how far
the surface is. The arm then picks the red cube where vision says it is, never where the simulator knows
it to be. Then the light turns warm and dim, and the finder has to be fixed. The numbers in the text were
measured with scripts/vision_bench.py.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from robotforge.behaviors.arm import OutOfReach
from robotforge.behaviors.manager import BehaviorError
from robotforge.design import props
from robotforge.design.world import World, compile_world
from robotforge.learn.engine import Bench, Control, Lesson, LessonContext, Metric, Plot, Quiz, Series, Step
from robotforge.sim import vision
from robotforge.sim.render import Layers, RenderUnavailable, SnapshotRenderer, render_unavailable_reason

Text = dict[str, str]


def _text(zh: str, en: str) -> Text:
    return {"zh": zh, "en": en}


CAMERA = "overhead"
EYE = (0.22, -0.34, 0.42)  # m: on a post in front of the mat, looking down at its middle 39 degrees off vertical
AT = (0.22, 0.0, 0.0)
FOVY = 42.0  # degrees across the short side, as the color camera of a RealSense D435
WIDTH, HEIGHT = 320, 240  # px
CUBE = 0.04  # m, edge of each cube
MAT = ((0.22, -0.01), (0.34, 0.36))  # centre and size of the light mat the cubes stand on, m
# The cubes and their start spots. Pink and orange sit next to red on the hue circle; under the warm evening
# light pink moves to the same hue as red (measured 9 degrees against red's 8).
CUBES = (
    ("red", (0.24, -0.07)),
    ("orange", (0.17, -0.03)),
    ("pink", (0.27, 0.05)),
    ("green", (0.17, 0.08)),
    ("blue", (0.3, -0.13)),
)
TARGET = "red_cube"
TRAY = (0.07, 0.19)  # m: where the arm puts the red cube, in the camera's view so it is still seen there
TRAY_SIZE = (0.12, 0.12, 0.03)
PARK = (-0.04, 0.17, 0.26)  # m: the tool waits here, out of the camera's way, between picks
REACH = 0.29  # m from the arm's base: it picks a cube straight down from this far (measured, 12 of 12)
FRAME_DT = 0.2  # s of simulated time between camera frames, five a second like a slow webcam
MIN_AREA = 4  # px: smaller blobs are not counted; "clean" removes the specks noise makes above that
SPOTS = 12  # spots in a test run
SPACING = 0.075  # m between cube centres when they are shuffled, so no two touch
INSET = 0.04  # m: cube centres stay this far inside the mat, so the whole cube is in the picture


@dataclass(frozen=True)
class Light:
    label: Text
    sun: tuple[float, float, float]  # diffuse color of the world's sun
    ambient: tuple[float, float, float]  # the headlight MuJoCo adds, as in compile_world
    head: tuple[float, float, float]
    noise: float  # grey levels (0..255) of camera noise: a camera turns its gain up in dim light


LIGHTS = (
    Light(_text("白天", "Daylight"), (0.8, 0.8, 0.8), (0.25, 0.25, 0.25), (0.35, 0.35, 0.35), 2.0),
    Light(_text("傍晚台灯", "Evening lamp"), (0.55, 0.36, 0.16), (0.12, 0.08, 0.04), (0.14, 0.09, 0.05), 5.0),
)


def camera_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The overhead camera's right, up and viewing direction in the world."""
    forward = np.subtract(AT, EYE)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return right, np.cross(right, forward), forward


def build_world() -> tuple[World, mujoco.MjModel]:
    """The arm, the mat, five cubes and a tray, plus the camera on its post (added to the compiled spec,
    since a world.yaml has no cameras of its own)."""
    objects: list[dict[str, Any]] = [
        {"name": "mat", "shape": "marker", "size": list(MAT[1]), "pos": list(MAT[0]), "color": [0.88, 0.87, 0.84]},
        {"name": "tray", "shape": "bin", "size": list(TRAY_SIZE), "pos": list(TRAY), "color": "teal"},
    ]
    objects += [
        {"name": f"{color}_cube", "shape": "box", "size": [CUBE] * 3, "pos": list(spot), "color": color}
        for color, spot in CUBES
    ]
    world = World.model_validate(
        {
            "name": "seeing",
            "floor": {"style": "concrete"},
            "robots": [{"name": "arm", "template": "arm", "params": {"color": "white"}, "pos": [0.0, 0.0]}],
            "objects": objects,
        }
    )
    spec = compile_world(world).spec
    right, up, forward = camera_axes()
    _camera_post(spec, right, up, forward)
    spec.worldbody.add_camera(name=CAMERA, pos=list(EYE), xyaxes=[*right, *up], fovy=FOVY)
    return world, spec.compile()


def _camera_post(spec: mujoco.MjSpec, right: np.ndarray, up: np.ndarray, forward: np.ndarray) -> None:
    """A post with an arm out to the camera, and the camera's housing; nothing collides with them."""
    G = mujoco.mjtGeom
    dark, metal, glass = [0.18, 0.19, 0.22, 1], [0.62, 0.64, 0.68, 1], [0.08, 0.1, 0.16, 1]
    base = (EYE[0], EYE[1] - 0.12)  # the post stands a little behind the camera
    top = EYE[2] + 0.05
    w = spec.worldbody
    ghost = {"contype": 0, "conaffinity": 0}
    w.add_geom(name="post_foot", type=G.mjGEOM_CYLINDER, size=[0.06, 0.008, 0], pos=[*base, 0.008], rgba=dark, **ghost)
    w.add_geom(name="post", type=G.mjGEOM_CYLINDER, size=[0.012, top / 2, 0], pos=[*base, top / 2], rgba=metal, **ghost)
    w.add_geom(
        name="post_arm",
        type=G.mjGEOM_CAPSULE,
        fromto=[*base, top, EYE[0], EYE[1], top],
        size=[0.009, 0, 0],
        rgba=metal,
        **ghost,
    )
    w.add_geom(
        name="camera_drop",
        type=G.mjGEOM_CYLINDER,
        fromto=[EYE[0], EYE[1], top, EYE[0], EYE[1], EYE[2] + 0.015],
        size=[0.006, 0, 0],
        rgba=metal,
        **ghost,
    )
    # The housing is a box turned to look where the camera looks, the lens just in front of the camera.
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([right, up, -forward]).flatten())
    w.add_geom(
        name="camera_body",
        type=G.mjGEOM_BOX,
        size=[0.045, 0.014, 0.016],
        pos=(np.array(EYE) - forward * 0.022).tolist(),
        quat=quat.tolist(),
        rgba=dark,
        **ghost,
    )
    w.add_geom(
        name="camera_lens",
        type=G.mjGEOM_CYLINDER,
        size=[0.009, 0.004, 0],
        pos=(np.array(EYE) - forward * 0.004).tolist(),
        quat=quat.tolist(),
        rgba=glass,
        **ghost,
    )


def apply_light(model: mujoco.MjModel, light: Light) -> None:
    """Light the world for the camera (the 3D view in the browser keeps its own studio light)."""
    sun = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "sun")
    model.light_diffuse[sun] = light.sun
    model.vis.headlight.ambient[:] = light.ambient
    model.vis.headlight.diffuse[:] = light.head


# ---- one camera frame through the color finder ------------------------------------------------------------


@dataclass(frozen=True)
class Look:
    """What the finder made of one camera frame."""

    picture: np.ndarray  # what it looked at: the camera's picture with its noise, white balanced if asked
    mask: np.ndarray
    blobs: list[vision.Blob]
    on: list[str | None]  # the object under each blob: the simulator's truth, used only to grade and explain
    estimate: np.ndarray | None  # where the biggest blob's object is, world m
    size: np.ndarray | None  # the depth camera's box around it, m (depth mode only)


def noisy(rgb: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """The picture as a real camera delivers it: every value off by a little random noise."""
    if sigma <= 0:
        return rgb
    return np.clip(rgb + rng.normal(0.0, sigma, rgb.shape), 0, 255).astype(np.uint8)


def look(layers: Layers, picture: np.ndarray, values: dict[str, float], owner: np.ndarray) -> Look:
    """Run the finder as the knobs say: mask, blobs, and where the biggest blob's object is.

    `owner` maps a geom id (and -1 for the sky, its last entry) to the index of the cube it belongs to, or -1.
    """
    if values["balance"]:
        picture = vision.white_balance(picture)
    if values["clean"]:  # the usual recipe: smooth the noise out of the picture, then open the mask
        picture = vision.blur(picture)
    rng = vision.ColorRange(values["hue"], values["width"], values["sat"], values["val"])
    mask = vision.threshold(vision.to_hsv(picture), rng)
    if values["clean"]:
        mask = vision.clean(mask)
    found = vision.blobs(mask, MIN_AREA)
    on = [_under(b, layers.geoms, owner) for b in found]
    estimate = size = None
    if found:
        best = found[0]
        if values["depth"]:
            estimate, size = vision.locate(layers.camera, layers.depth, best)
        else:
            estimate = layers.camera.on_plane(best.u, best.v, values["height"] / 100)
    return Look(picture, mask, found, on, estimate, size)


def _under(blob: vision.Blob, geoms: np.ndarray | None, owner: np.ndarray) -> str | None:
    """The cube most of a blob's pixels show, if one does."""
    if geoms is None:
        return None
    ys, xs = blob.pixels()
    counts = np.bincount(owner[geoms[ys, xs]] + 1, minlength=len(CUBES) + 1)
    best = int(np.argmax(counts[1:]))
    return f"{CUBES[best][0]}_cube" if counts[best + 1] * 2 > blob.area else None


def _jpeg(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _png(mask: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---- the steps ----------------------------------------------------------------------------------------------

NAMES = {
    "red_cube": _text("红", "red"),
    "orange_cube": _text("橙", "orange"),
    "pink_cube": _text("粉", "pink"),
    "green_cube": _text("绿", "green"),
    "blue_cube": _text("蓝", "blue"),
}


def _seen(ctx: LessonContext) -> dict[str, Any]:
    return ctx.results.get("seen", {})


def _looked(ctx: LessonContext) -> bool:
    """The camera works and the learner moved a cube under it."""
    return ctx.results.get("frames", 0) > 1 and "dragged" in ctx.events


def _only_red(ctx: LessonContext) -> bool:
    """The mask holds one blob, and it is the red cube."""
    return _seen(ctx).get("on") == [TARGET]


def _close_on_the_plane(ctx: LessonContext) -> bool:
    """The plane estimate of the red cube is within half a centimetre."""
    seen = _seen(ctx)
    err = seen.get("err_cm")
    return seen.get("depth") == 0 and seen.get("on", [None])[:1] == [TARGET] and err is not None and err < 0.5


def _picked(ctx: LessonContext) -> bool:
    return ctx.results.get("picked", 0) >= 1


def _compared(ctx: LessonContext) -> bool:
    """The record holds a test on the plane and one with the depth camera."""
    return {0.0, 1.0} <= {run["params"]["depth"] for run in ctx.runs}


def _fixed_for_the_evening(ctx: LessonContext) -> bool:
    """Under the evening lamp, the mask holds the red cube and nothing else."""
    return _seen(ctx).get("light") == 1 and _only_red(ctx)


STEPS = (
    Step(
        id="look",
        title=_text("相机看到的是一堆数字", "What a camera sees is numbers"),
        body=_text(
            "立柱上的相机斜着往下看垫子上的五个方块，右边“机器人相机”就是它拍到的画面："
            f"{WIDTH} × {HEIGHT} 个像素，每个像素三个数（红、绿、蓝）。\n"
            "机器人并不知道哪些像素是红方块。用鼠标把红方块拖到别处，看它在画面里怎么动。",
            "The camera on the post looks down at five cubes on the mat; 'Robot camera' on the right is its picture: "
            f"{WIDTH} x {HEIGHT} pixels, each three numbers (red, green, blue).\n"
            "The robot does not know which pixels are the red cube. Drag the red cube somewhere else with the mouse "
            "and watch it move in the picture.",
        ),
        done=_text(
            "一张画面就是一张数字表。接下来写一个小程序，从这张表里找出红方块。",
            "A picture is a table of numbers. Next, a small program finds the red cube in it.",
        ),
        check=_looked,
        show=("frustum",),
    ),
    Step(
        id="mask",
        title=_text("按颜色挑像素：色相、饱和度、亮度", "Keep the red pixels: hue, saturation, value"),
        body=_text(
            "把每个像素的 RGB 换成 HSV：色相是什么颜色（0° 红，120° 绿，240° 蓝），饱和度是颜色有多浓（灰色是 0），"
            "亮度是有多亮。\n颜色落在范围里的像素记成白色，其余记成黑色，这张黑白图叫“掩码”。"
            "现在范围太宽，橙色和粉色方块也混了进来。把“色相范围”调窄，直到掩码里只剩红方块一块。",
            "Turn each pixel's RGB into HSV: hue is which color (0° red, 120° green, 240° blue), saturation how "
            "strong it is (grey is 0) and value how bright.\nPixels whose color is in the range become white, the "
            "rest black: this black and white picture is the mask. The range is too wide now, so the orange and "
            "pink cubes get in too. Narrow 'hue range' until only the red cube is left in the mask.",
        ),
        done=_text(
            "在这台相机的画面里，红方块的色相约 2°，橙色约 32°，粉色约 332°（也就是 -28°），"
            "所以范围收到 ±20° 以内就只剩红色。"
            "用 HSV 而不用 RGB，是因为同一种颜色变亮变暗时，色相几乎不变。",
            "In this camera's picture the red cube's hue is about 2°, orange about 32° and pink about 332° (that "
            "is -28°), so a range within ±20° keeps only red. HSV beats RGB here because a color's hue hardly "
            "changes when it gets brighter or darker.",
        ),
        check=_only_red,
        unlock=("hue", "width", "sat", "val"),
        show=("frustum",),
    ),
    Step(
        id="where",
        title=_text("从像素回到桌面", "From a pixel back to the table"),
        body=_text(
            "十字是这一块白像素的质心，也就是它们的平均位置。相机知道自己在哪、镜头多宽，"
            "所以每个像素对应空间里的一条射线（3D 视图里的黄线）。射线和一个水平面的交点，就是方块位置的估计。\n"
            "平面该放多高？先猜，再拖“假设高度”，让估计误差小于 0.5 厘米。",
            "The cross is the centroid of the white blob: the average position of its pixels. The camera knows "
            "where it is and how wide its lens sees, so each pixel is a ray in space (the yellow line in the 3D "
            "view). Where the ray meets a level plane is the estimate of the cube's position.\n"
            "How high should the plane be? Guess first, then move 'assumed height' until the error is under 0.5 cm.",
        ),
        done=_text(
            "方块中心离桌面 2 厘米。平面放在桌面上（0 厘米）时，射线穿过方块还要继续往下才到桌面，"
            "估计在桌面上落到方块后面约 1.7 厘米（算上高度差是 2.7 厘米）；放到 2 厘米，误差只剩 1 毫米左右。",
            "The cube's middle is 2 cm above the table. With the plane on the table (0 cm) the ray goes on past "
            "the cube before it meets the plane, so the estimate lands about 1.7 cm behind the cube (2.7 cm "
            "counting the height it gets wrong); at 2 cm the error is about a millimetre.",
        ),
        check=_close_on_the_plane,
        quiz=Quiz(
            question=_text(
                "假设高度是 0 厘米（桌面）时，估计会落在哪里？",
                "With the assumed height at 0 cm (the table), where does the estimate land?",
            ),
            options=(
                _text("方块后面，离相机更远", "Behind the cube, further from the camera"),
                _text("方块前面，离相机更近", "In front of the cube, nearer the camera"),
                _text("正好在方块中心", "Right at the cube's middle"),
            ),
            answer=0,
            explain=_text(
                "射线先碰到方块的顶面，再往下 2 到 4 厘米才到桌面，这一段沿着射线一直在往远处走。",
                "The ray meets the cube's top first and travels another 2 to 4 cm down to the table, moving away "
                "from the camera all the while.",
            ),
            predict=True,
        ),
        unlock=("height",),
        show=("frustum", "ray", "plane"),
    ),
    Step(
        id="pick",
        title=_text("抓起红色的那块", "Pick the red one"),
        body=_text(
            "按「抓红色的」：机械臂只用视觉算出的位置，伸过去抓起来放进托盘"
            "（方块高 4 厘米它是知道的，所以在 2 厘米高处合爪，往哪儿伸全靠相机）。\n"
            "再按「打乱」，方块换了位置，再抓一次。",
            "Press 'Pick the red one': the arm reaches for the position vision worked out, grasps and drops the "
            "cube in the tray (it knows the cubes are 4 cm tall, so it closes 2 cm above the mat; where to reach "
            "comes only from the camera).\nThen press Shuffle to move the cubes and pick again.",
        ),
        done=_text(
            "它没用方块的名字，也没偷看仿真里的真实位置，只用了相机画面。瞄偏 1.2 厘米以内爪子都夹得住，"
            "偏 2 厘米时 12 次里有 5 次会把方块推歪。",
            "It used no cube names and never peeked at the simulator's true positions: only the picture. The "
            "gripper still closes on a cube aimed up to 1.2 cm off; 2 cm off, it pushed the cube aside in 5 of 12 "
            "tries.",
        ),
        check=_picked,
        unlock=("pick", "shuffle"),
        show=("frustum", "ray"),
    ),
    Step(
        id="depth",
        title=_text("测 12 个位置；深度相机", "Test 12 spots; a depth camera"),
        body=_text(
            "准一次不代表一直准。按「测 12 个位置」：红方块轮流放到垫子上 12 个地方，记下每次的误差。\n"
            "再打开“深度相机”：每个像素还量出了离相机多远，于是这块像素在空间里是一堆点，"
            "围住这些点的盒子的中心就是估计，不用再假设高度。再测一次，比较两次记录。",
            "Right once is not right always. Press 'Test 12 spots': the red cube is put on 12 places on the mat in "
            "turn and the error is recorded each time.\nThen switch on 'depth camera': every pixel also measures "
            "how far away it is, so the blob becomes a cloud of points in space, and the middle of the box around "
            "them is the estimate, with no height to assume. Test again and compare the two records.",
        ),
        done=_text(
            "我们在 60 个位置测过：平面放在 2 厘米时平均误差 0.11 厘米，深度相机 0.26 厘米，"
            "都远小于爪子能容忍的 1.2 厘米；平面放在 0 厘米是 2.7 厘米。"
            "平面法要事先知道物体多高；深度法不用，还量出了方块的大小（看读数）。",
            "Over 60 spots we measured a mean error of 0.11 cm with the plane at 2 cm and 0.26 cm with the depth "
            "camera, well inside the 1.2 cm the gripper forgives; the plane at 0 cm gave 2.7 cm. The plane needs "
            "the object's height known beforehand; depth does not, and it measures the cube's size too (see the "
            "readout).",
        ),
        check=_compared,
        unlock=("test", "depth"),
        show=("frustum", "ray", "plane", "box"),
    ),
    Step(
        id="light",
        title=_text("傍晚的灯光", "Evening light"),
        body=_text(
            "把灯光换成“傍晚台灯”（只有相机画面会变，3D 视图保留自己的灯光好让你看清）。"
            "暖黄的灯给每个像素都染了色：粉方块看起来也成了红色，画面更暗、噪点更多。\n"
            "看看掩码出了什么问题，然后修好它：打开“白平衡”（把整张图的平均颜色拉回灰色，相机的自动白平衡就是这么做的），"
            "再打开“清理掩码”去掉零星的噪点，直到掩码里又只剩红方块。",
            "Switch the light to 'Evening lamp' (only the camera's picture changes; the 3D view keeps its own light so "
            "you can still see). The warm lamp tints every pixel: the pink cube now looks red too, and the picture "
            "is darker and noisier.\nSee what goes wrong in the mask, then fix it: switch on 'white balance' (it "
            "pulls the picture's average color back to grey, as a camera's automatic white balance does) and "
            "'clean the mask' to remove the specks of noise, until only the red cube is left.",
        ),
        done=_text(
            "颜色是物体和灯光一起决定的：台灯下红方块的色相约 8°，粉方块约 9°，光看色相分不开；"
            "白平衡之后又回到 2° 和 334°。真实的机器人会做白平衡、在不同灯光下测试，"
            "还会用大小和形状再确认一遍。",
            "A color comes from the object and the light together: under the lamp the red cube's hue is about 8° and "
            "the pink cube's about 9°, too close to tell apart; white balanced, they are back at 2° and 334°. Real "
            "robots white balance, test under different lights, and check size and shape as well.",
        ),
        check=_fixed_for_the_evening,
        unlock=("light", "balance", "clean"),
        show=("frustum", "ray", "plane", "box"),
    ),
    Step(
        id="quiz",
        title=_text("想一想", "Think about it"),
        body=_text(
            "仓库里的灯光一天里会变好几次。",
            "The light in a warehouse changes several times a day.",
        ),
        done=_text(
            "这就是一个最简单的视觉抓取：颜色挑像素、连成块、算质心、反投影、抓取。换成学出来的检测网络，后面几步也一样。",
            "That is vision-based picking at its simplest: pick pixels by color, join them into blobs, take the "
            "centroid, project it back, grasp. A learned detector replaces the first steps; the rest stays the same.",
        ),
        quiz=Quiz(
            question=_text(
                "怎样让“找红方块”在这种地方更可靠？",
                "What makes 'find the red cube' more reliable there?",
            ),
            options=(
                _text("把色相范围放宽，什么样的红都收", "Widen the hue range so every kind of red gets in"),
                _text(
                    "先做白平衡，再用大小和形状确认是不是方块",
                    "White balance first, then check size and shape to confirm it is the cube",
                ),
                _text("把最低亮度调高", "Raise the minimum value"),
            ),
            answer=1,
            explain=_text(
                "放宽会把橙色和粉色也收进来；调高亮度下限，天一暗就什么都找不到。"
                "先去掉灯光的颜色，再加上别的线索，才可靠。",
                "A wider range lets orange and pink in; a higher minimum value finds nothing once it gets dark. "
                "Taking out the light's color and adding other clues is what makes it reliable.",
            ),
        ),
    ),
)
VIEWS = {"look": "picture", "mask": "mask", "light": "mask"}  # which picture each step opens on; else the camera's


CONTROLS = (
    Control("hue", "slider", _text("色相中心", "hue centre"), 0, 359, 1, 0, "°"),
    Control("width", "slider", _text("色相范围 ±", "hue range ±"), 2, 60, 1, 40, "°"),
    Control("sat", "slider", _text("最低饱和度", "min saturation"), 0, 1, 0.05, 0.3),
    Control("val", "slider", _text("最低亮度", "min value"), 0, 1, 0.05, 0.2),
    Control("height", "slider", _text("假设高度", "assumed height"), 0, 6, 0.5, 0, "cm"),
    Control("pick", "button", _text("抓红色的", "Pick the red one")),
    Control("shuffle", "button", _text("打乱", "Shuffle")),
    Control("test", "button", _text("测 12 个位置", "Test 12 spots")),
    Control("depth", "toggle", _text("深度相机", "depth camera")),
    Control("light", "choice", _text("灯光", "light"), options=tuple(light.label for light in LIGHTS)),
    Control("balance", "toggle", _text("白平衡", "white balance")),
    Control("clean", "toggle", _text("清理掩码", "clean the mask")),
)


class SeeingLesson(Lesson):
    id = "seeing"
    chapter = _text("感知", "Sensing")
    title = _text("感知（三）：用相机找到红方块", "Sensing III: find the red cube with a camera")
    summary = _text(
        "在相机画面里按颜色找出红方块，算出它在桌上的位置，让机械臂去抓；再换上傍晚的灯光，看它怎么认错、怎么修好。",
        "Find the red cube in a camera picture by its color, work out where it is on the table and let the arm "
        "pick it; then switch to evening light, watch it fail and fix it.",
    )
    minutes = 15
    controls = CONTROLS
    plots = (
        Plot(
            "error",
            _text("估计误差", "Estimate error"),
            "cm",
            (Series("err", _text("估计到真实位置", "estimate to the true position"), "amber"),),
            window=10.0,
            span=2.0,
        ),
    )
    steps = STEPS
    bench = Bench(
        title=_text("12 个位置的误差", "Error at 12 spots"),
        x=_text("位置", "spot"),
        x_unit="",
        y=_text("误差", "error"),
        y_unit="cm",
        params=("light", "height", "depth", "balance"),
        short=(_text("灯光", "light"), _text("高度", "height"), _text("深度", "depth"), _text("白平衡", "balance")),
        metrics=(
            Metric("mean", _text("平均误差", "mean error"), "cm"),
            Metric("worst", _text("最大误差", "worst"), "cm"),
            Metric("found", _text("找对", "found"), "", lower_is_better=False),
        ),
    )

    def __init__(self) -> None:
        self.playground = None  # set by the runner: the arm picks through the real behaviors
        self.world: World | None = None  # the behaviors find this lesson's objects here
        self.model: mujoco.MjModel | None = None
        self.renderer = SnapshotRenderer()
        self.unavailable: str | None = None  # why there is no camera picture, on a machine without OpenGL
        self.layers: Layers | None = None
        self.frame_light = 0  # the light the last frame was taken in
        self.raw: np.ndarray | None = None  # the last picture with its noise, before any white balance
        self.seen: Look | None = None
        self.frames = 0
        self.version = 0  # bumped whenever the pictures change (a new frame, a knob turned)
        self.sent = -1  # the version the viewer has
        self.next_frame = 0.0
        self.last_frame = 0.0
        self.renew = False  # take a new frame now (the light changed, the cubes moved)
        self.dirty = False  # run the finder again on the last frame (a knob turned)
        self.noise = np.random.default_rng(0)
        self.shuffles = 0
        self.picking: dict[str, Any] | None = None
        self.step = ""  # the step the learner was on at the last update
        self.samples: list[dict[str, float]] = []
        self.owner = np.full(1, -1)

    # ---- the world -------------------------------------------------------------------------------------

    def build(self) -> mujoco.MjModel:
        self.world, self.model = build_world()
        owner = np.full(self.model.ngeom + 1, -1)  # the last entry stands for -1, the sky
        for k, (color, _) in enumerate(CUBES):
            body = self.model.body(f"{color}_cube").id
            owner[: self.model.ngeom][self.model.geom_bodyid == body] = k
        self.owner = owner
        return self.model

    def attach(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        apply_light(model, LIGHTS[int(ctx.values["light"])])
        arm = self._arm()
        park, _, _ = arm._solve(np.array(PARK))  # out of the camera's way, over the tray
        data.qpos[arm.chain.qpos_adr] = park
        data.qvel[arm.chain.dof_adr] = 0.0
        arm.q = park.copy()
        mujoco.mj_forward(model, data)
        self.picking = None
        self.renew = True
        self.unavailable = render_unavailable_reason()
        return {}

    def _arm(self) -> Any:
        return self.playground.behaviors._arm("arm")

    def close(self) -> None:
        self.renderer.close()

    def camera(self) -> dict[str, list[float]]:
        # From the front and above, so the post, the mat, the arm and the tray all fit between the panels.
        return {"eye": [0.86, -0.48, 1.06], "target": [0.16, -0.08, 0.17]}

    def rig(self, model: mujoco.MjModel) -> dict[str, Any]:
        right, up, forward = camera_axes()
        return {
            "kind": "vision",
            "eye": list(EYE),
            "right": right.round(5).tolist(),
            "up": up.round(5).tolist(),
            "forward": forward.round(5).tolist(),
            "fovy": FOVY,
            "image": [WIDTH, HEIGHT],
            "mat": {"center": list(MAT[0]), "size": list(MAT[1]), "top": props.MARK},
        }

    def describe(self, model: mujoco.MjModel) -> dict[str, Any]:
        self.sent = -1  # a viewer that (re)connects gets the pictures with its next update
        return super().describe(model)

    # ---- learner input ---------------------------------------------------------------------------------

    def on_control(self, control: str, value: float, data: mujoco.MjData, ctx: LessonContext) -> None:
        model = self.model
        assert model is not None
        if control == "light":
            apply_light(model, LIGHTS[int(value)])
            self.renew = True
        elif control == "pick":
            self._pick(data, ctx)
        elif control == "shuffle":
            self._shuffle(model, data, ctx)
        elif control == "test":
            self._test(model, data, ctx)
        else:
            self.dirty = True

    def _pick(self, data: mujoco.MjData, ctx: LessonContext) -> None:
        if self.picking is not None:
            return
        seen = self.seen
        if seen is None or seen.estimate is None:
            ctx.results["summary"] = _text(
                "掩码里没有可抓的东西：先让它找到红方块。",
                "Nothing in the mask to pick: make it find the red cube first.",
            )
            return
        aim = np.array([seen.estimate[0], seen.estimate[1], CUBE / 2])
        truth = data.xpos[self.model.body(TARGET).id].copy()
        arm = self._arm()
        try:
            arm.pick_and_place(aim, CUBE / 2, np.array([TRAY[0], TRAY[1], 0.005]), TRAY_SIZE[2])
        except (OutOfReach, BehaviorError, ValueError):
            ctx.results["summary"] = _text(
                "机械臂够不着估计的那个位置：按「打乱」换个位置再试。",
                "The arm cannot reach the estimated position: press Shuffle and try again.",
            )
            return
        arm.move_to(np.array(PARK))
        self.picking = {"off_cm": float(np.linalg.norm(aim[:2] - truth[:2])) * 100, "settle": None}
        ctx.results["summary"] = _text("正在抓……", "Picking...")

    def _follow_pick(self, data: mujoco.MjData, ctx: LessonContext) -> None:
        run = self.picking
        assert run is not None
        if not self._arm().done:
            return
        if run["settle"] is None:
            run["settle"] = float(data.time) + 0.5
            return
        if data.time < run["settle"]:
            return
        self.picking = None
        inside = [color for color, _ in CUBES if self._in_tray(data, f"{color}_cube")]
        off = run["off_cm"]
        if "red" in inside:
            ctx.results["picked"] = ctx.results.get("picked", 0) + 1
            ctx.results["summary"] = _text(
                f"放进托盘了。瞄准点离红方块中心 {off:.1f} 厘米。",
                f"In the tray. The aim was {off:.1f} cm from the red cube's middle.",
            )
        elif inside:
            got = NAMES[f"{inside[0]}_cube"]
            ctx.results["summary"] = _text(
                f"放进托盘的是{got['zh']}方块：掩码里最大的一块不是红方块。",
                f"The {got['en']} cube went in the tray: the biggest blob in the mask was not the red cube.",
            )
        else:
            ctx.results["summary"] = _text(
                f"没抓住：瞄准点离红方块中心 {off:.1f} 厘米，爪子把它推歪了。",
                f"Missed: the aim was {off:.1f} cm from the red cube's middle and the gripper pushed it aside.",
            )
        self.renew = True

    def _in_tray(self, data: mujoco.MjData, name: str) -> bool:
        pos = data.xpos[self.model.body(name).id]
        return bool(np.all(np.abs(pos[:2] - TRAY) <= np.array(TRAY_SIZE[:2]) / 2) and pos[2] < TRAY_SIZE[2] + CUBE)

    def _shuffle(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> None:
        """Put the cubes down at new spots: the red one where the arm can reach, none touching."""
        if self.picking is not None:
            return
        self.shuffles += 1
        rng = np.random.default_rng(self.shuffles)
        (cx, cy), (sx, sy) = MAT
        spots: list[tuple[float, float]] = []
        for color, _ in CUBES:
            for _ in range(500):
                x = float(rng.uniform(cx - sx / 2 + INSET, cx + sx / 2 - INSET))
                y = float(rng.uniform(cy - sy / 2 + INSET, cy + sy / 2 - INSET))
                if color == "red" and not (x >= 0.12 and math.hypot(x, y) <= REACH):
                    continue
                if math.hypot(x - TRAY[0], y - TRAY[1]) < 0.11 or any(
                    math.hypot(x - a, y - b) < SPACING for a, b in spots
                ):
                    continue
                break
            spots.append((x, y))
        for (color, _), (x, y) in zip(CUBES, spots, strict=True):
            body = model.body(f"{color}_cube")
            adr, dof = model.jnt_qposadr[body.jntadr[0]], model.jnt_dofadr[body.jntadr[0]]
            data.qpos[adr : adr + 7] = [x, y, CUBE / 2, 1.0, 0.0, 0.0, 0.0]
            data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(model, data)
        ctx.results.pop("summary", None)
        self.renew = True

    def test_spots(self, data: mujoco.MjData) -> list[tuple[float, float]]:
        """The test's spots: a grid over the mat, in a fixed mixed-up order, leaving out those where another
        cube or the tray is in the way."""
        model = self.model
        (cx, cy), (sx, sy) = MAT
        grid = [
            (float(x), float(y))
            for x in np.linspace(cx - sx / 2 + INSET, cx + sx / 2 - INSET, 6)
            for y in np.linspace(cy - sy / 2 + INSET, cy + sy / 2 - INSET, 6)
        ]
        others = [data.xpos[model.body(f"{c}_cube").id][:2] for c, _ in CUBES if f"{c}_cube" != TARGET]
        clear = [
            p
            for p in (grid[k] for k in np.random.default_rng(7).permutation(len(grid)))
            if math.hypot(p[0] - TRAY[0], p[1] - TRAY[1]) >= 0.11
            and all(math.hypot(p[0] - o[0], p[1] - o[1]) >= SPACING for o in others)
        ]
        return clear[:SPOTS]

    def _test(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> None:
        """Put the red cube on each test spot in a copy of the world, look, and record how far off each estimate is."""
        if self.unavailable is not None:
            return
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        body = model.body(TARGET)
        adr = model.jnt_qposadr[body.jntadr[0]]
        light = LIGHTS[int(ctx.values["light"])]
        rng = np.random.default_rng(1000 + ctx.run_count)
        xs: list[float] = []
        errors: list[float] = []
        spots = self.test_spots(data)
        for k, (x, y) in enumerate(spots, start=1):
            scratch.qpos[adr : adr + 7] = [x, y, CUBE / 2, 1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(model, scratch)
            try:
                layers = self.renderer.layers_of(model, scratch, CAMERA, WIDTH, HEIGHT, 0, geoms=True)
            except RenderUnavailable as exc:
                self.unavailable = str(exc)
                return
            seen = look(layers, noisy(layers.rgb, light.noise, rng), ctx.values, self.owner)
            if seen.estimate is None or seen.on[0] != TARGET:
                continue
            xs.append(k)
            errors.append(float(np.linalg.norm(seen.estimate - [x, y, CUBE / 2])) * 100)
        metrics = {
            "mean": round(float(np.mean(errors)), 2) if errors else None,
            "worst": round(float(np.max(errors)), 2) if errors else None,
            "found": len(errors),
        }
        params = {p: ctx.values[p] for p in ("light", "height", "depth", "balance")}
        ctx.record_run(params, metrics, xs, errors)
        if errors:
            ctx.results["summary"] = _text(
                f"{len(spots)} 个位置找对 {len(errors)} 个，"
                f"平均误差 {metrics['mean']:.2f} 厘米，最大 {metrics['worst']:.2f} 厘米。",
                f"Found it at {len(errors)} of {len(spots)} spots; mean error {metrics['mean']:.2f} cm, worst "
                f"{metrics['worst']:.2f} cm.",
            )
        else:
            ctx.results["summary"] = _text(
                f"{len(spots)} 个位置一次也没找对：先让掩码只剩红方块。",
                f"It found the red cube at none of the {len(spots)} spots: make the mask keep only the red cube first.",
            )

    # ---- the camera, five times a second ---------------------------------------------------------------

    def live(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        if ctx.step != self.step and self.picking is None:  # a new step does not open on the last one's result
            self.step = ctx.step
            ctx.results.pop("summary", None)
        if self.picking is not None:
            self._follow_pick(data, ctx)
        if self.unavailable is None:
            due = data.time >= self.next_frame or data.time < self.last_frame  # (a reset turns the clock back)
            if due or self.renew or self.raw is None:
                self._frame(model, data, ctx)
            elif self.dirty:
                self._process(data, ctx)
        if self.unavailable is not None:
            note = _text(
                f"这台电脑不能离屏渲染，所以没有相机画面，这节课做不下去（{self.unavailable}）。"
                "需要能用 OpenGL 的电脑。",
                f"This computer cannot render offscreen, so there is no camera picture and the lesson cannot go on "
                f"({self.unavailable}). It needs a machine with OpenGL.",
            )
            ctx.results["summary"] = note
            return {"readout": note, "vision": {"note": note}}
        return {"readout": self._readout(ctx), "vision": self._picture_state(data, ctx)}

    def _frame(self, model: mujoco.MjModel, data: mujoco.MjData, ctx: LessonContext) -> None:
        try:
            self.layers = self.renderer.layers_of(model, data, CAMERA, WIDTH, HEIGHT, 0, geoms=True)
        except RenderUnavailable as exc:
            self.unavailable = str(exc)
            return
        self.frame_light = int(ctx.values["light"])
        self.raw = noisy(self.layers.rgb, LIGHTS[self.frame_light].noise, self.noise)
        self.frames += 1
        self.last_frame = float(data.time)
        self.next_frame = float(data.time) + FRAME_DT
        self.renew = False
        ctx.results["frames"] = self.frames
        self._process(data, ctx)
        err = ctx.results["seen"]["err_cm"]
        if err is not None:
            self.samples.append({"t": float(data.time), "err": err})

    def _process(self, data: mujoco.MjData, ctx: LessonContext) -> None:
        assert self.layers is not None and self.raw is not None
        self.seen = look(self.layers, self.raw, ctx.values, self.owner)
        self.dirty = False
        self.version += 1
        truth = data.xpos[self.model.body(TARGET).id]
        est = self.seen.estimate
        err = round(float(np.linalg.norm(est - truth)) * 100, 2) if est is not None else None
        # With the settings it was seen with: a check runs before the next frame, and must not pass a light
        # or a method the finder has not looked with yet.
        ctx.results["seen"] = {
            "blobs": len(self.seen.blobs),
            "on": self.seen.on,
            "err_cm": err,
            "light": self.frame_light,
            "depth": int(ctx.values["depth"]),
        }

    def _picture_state(self, data: mujoco.MjData, ctx: LessonContext) -> dict[str, Any]:
        seen = self.seen
        out: dict[str, Any] = {"view": VIEWS.get(ctx.step, "picture"), "size": [WIDTH, HEIGHT]}
        if seen is None:
            return out
        if self.version != self.sent:  # the pictures go out only when they changed
            self.sent = self.version
            out["pictures"] = {"picture": _jpeg(seen.picture), "mask": _png(seen.mask)}
        out["blobs"] = [
            {"box": list(b.box), "u": round(b.u, 1), "v": round(b.v, 1), "area": b.area} for b in seen.blobs[:12]
        ]
        out["truth"] = data.xpos[self.model.body(TARGET).id].round(4).tolist()
        if seen.estimate is not None:
            out["estimate"] = seen.estimate.round(4).tolist()
        if ctx.values["depth"]:
            if seen.size is not None:
                out["box"] = seen.size.round(4).tolist()
        else:
            out["plane"] = round(ctx.values["height"] / 100, 4)
        return out

    def _readout(self, ctx: LessonContext) -> Text:
        seen = self.seen
        if seen is None:
            return _text("相机还在开……", "The camera is starting...")
        if not seen.blobs:
            return _text(
                "掩码里一块也没有：放宽范围或调低下限。", "Nothing in the mask: widen the range or lower the minimums."
            )
        # Two lines (the panel keeps line breaks): what the mask holds, then where the biggest blob is.
        names = [NAMES.get(on or "", _text("别的", "other")) for on in seen.on[:4]]
        more = len(seen.on) > len(names)
        zh = "、".join(n["zh"] for n in names) + ("……" if more else "")
        en = ", ".join(n["en"] for n in names) + (", ..." if more else "")
        best = seen.blobs[0]
        plural = "s" if len(seen.blobs) != 1 else ""
        where = {"zh": f"最大一块的质心 ({best.u:.0f}, {best.v:.0f})", "en": f"biggest at ({best.u:.0f}, {best.v:.0f})"}
        err = ctx.results.get("seen", {}).get("err_cm")
        if err is not None:
            where = {"zh": f"{where['zh']} · 误差 {err:.1f} cm", "en": f"{where['en']} · error {err:.1f} cm"}
        if seen.size is not None and ctx.values["depth"]:
            dims = " × ".join(f"{s * 100:.1f}" for s in seen.size)
            where = {"zh": f"{where['zh']} · 盒子 {dims} cm", "en": f"{where['en']} · box {dims} cm"}
        return _text(
            f"掩码里 {len(seen.blobs)} 块：{zh}\n{where['zh']}",
            f"Mask: {len(seen.blobs)} blob{plural} ({en})\n{where['en']}",
        )

    def drain_samples(self) -> list[dict[str, float]]:
        out, self.samples = self.samples, []
        return out
