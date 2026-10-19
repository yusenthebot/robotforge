"""Cut the README promo video from the demo recordings of scripts/demos.py.

    uv run python scripts/promo.py --clips <folder with the demos' MP4s> --out docs/media

Real footage only: each cut is a few seconds of a recording of the real viewer, with a one-line caption; a title card
and an end card (their backdrops are footage too, blurred) frame them. Writes promo.mp4 (1920x1080, H.264), a looping
preview (promo_preview.webp, 960 px wide) and a poster (promo_poster.jpg). ffmpeg here has no drawtext, so captions
and cards are drawn with Pillow and laid over the footage.

Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
FADE = 0.4  # s of cross-fade between cuts
SANS = ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/HelveticaNeue.ttc")
MONO = "/System/Library/Fonts/Menlo.ttc"


@dataclass(frozen=True)
class Cut:
    clip: str  # recording name (demos.py Demo.name)
    start: float  # s into the recording
    seconds: float
    caption: str = ""
    corner: str = "bl"  # where the caption sits: bl, tl
    zoom: tuple[int, int, int, int] | None = None  # crop (x, y, w, h) of the 1920x1080 frame, scaled back up
    card: str | None = None  # "title" or "end": a card over blurred footage instead of a captioned cut


TITLE = ("robotforge", "Any robot, from a sentence, in a lightweight simulator")
END = ("Try it", "uvx --from git+https://github.com/yusenthebot/robotforge robotforge demo")

CUTS: list[Cut] = [
    Cut("generated_three", 3.0, 3.4, card="title"),
    Cut("generated_three", 3.0, 6.0, "Robots built from one sentence each: add_robot(describe=...)"),
    Cut("generated_home_bring", 16.0, 6.0, "A home and its task from one sentence: load_scene(description=...)"),
    Cut("splice_walkers", 2.0, 6.0, "Real robots spliced together: Go2 + PiPER, ANYmal C + UR5e, Spot + Z1"),
    Cut("splice_go2_pick", 29.0, 7.0, "A spliced Go2 carries the cube to the bin", zoom=(320, 40, 1280, 720)),
    Cut("fetch_kitchen_apple", 24.0, 7.0, "Mobile pick and place: the apple goes to the dining table"),
    Cut("sensors_go2_aisle", 3.0, 6.0, "Sensors: 3D lidar rays and a depth camera on a Go2"),
    Cut("lesson_learn_to_walk", 14.0, 5.0, "Learning to walk by trial and error"),
    Cut("kitchen_drawer", 2.0, 6.0, "Mechanisms: an arm opens a drawer"),
    Cut("factory_sorting", 0.0, 6.0, "Mechanisms: a conveyor and a sorting arm"),
    Cut("water_lake", 3.0, 6.0, "Water: a boat crosses the lake to the dock"),
    Cut("water_lake", 20.0, 4.2, card="end"),
]


def _font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        return ImageFont.truetype(MONO, size)
    return ImageFont.truetype(SANS[0], size, index=1 if bold else 0)


def caption_png(text: str, corner: str, out: Path) -> None:
    image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _font(38)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    pad_x, pad_y = 26, 16
    width, height = right - left + 2 * pad_x, bottom - top + 2 * pad_y
    x = 56
    y = 60 if corner == "tl" else H - 150 - height
    draw.rounded_rectangle((x, y, x + width, y + height), radius=height // 2, fill=(14, 17, 22, 205))
    draw.text((x + pad_x - left, y + pad_y - top), text, font=font, fill=(255, 255, 255, 255))
    image.save(out)


def card_png(kind: str, out: Path) -> None:
    image = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    title, line = TITLE if kind == "title" else END
    if kind == "title":
        big, small = _font(168, bold=True), _font(50)
    else:
        big, small = _font(110, bold=True), _font(40, mono=True)
    for text, font, y, fill in (
        (title, big, H // 2 - 150, (255, 255, 255, 255)),
        (line, small, H // 2 + 60, (214, 224, 236, 255)),
    ):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.text(((W - (right - left)) / 2 - left, y - top), text, font=font, fill=fill)
    if kind == "end":
        note = "github.com/yusenthebot/robotforge"
        font = _font(40)
        left, top, right, bottom = draw.textbbox((0, 0), note, font=font)
        draw.text(((W - (right - left)) / 2 - left, H // 2 + 190 - top), note, font=font, fill=(255, 190, 70, 255))
    image.save(out)


def build(clips: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rf-promo-") as tmp:
        tmp_dir = Path(tmp)
        inputs: list[str] = []
        graph: list[str] = []
        for i, cut in enumerate(CUTS):
            source = clips / f"{cut.clip}.mp4"
            if not source.exists():
                raise SystemExit(f"missing recording: {source}")
            overlay = tmp_dir / f"o{i}.png"
            if cut.card:
                card_png(cut.card, overlay)
            elif cut.caption:
                caption_png(cut.caption, cut.corner, overlay)
            else:
                Image.new("RGBA", (W, H), (0, 0, 0, 0)).save(overlay)
            length = cut.seconds + (FADE if i < len(CUTS) - 1 else 0.0)
            inputs += ["-ss", f"{cut.start}", "-t", f"{length}", "-i", str(source)]
            inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{length}", "-i", str(overlay)]
            v, o = 2 * i, 2 * i + 1
            crop = f"crop={cut.zoom[2]}:{cut.zoom[3]}:{cut.zoom[0]}:{cut.zoom[1]}," if cut.zoom else ""
            blur = "gblur=sigma=14,eq=brightness=-0.32:saturation=0.8," if cut.card else ""
            graph.append(
                f"[{v}:v]setpts=PTS-STARTPTS,fps={FPS},{crop}scale={W}:{H}:flags=lanczos,{blur}setsar=1,format=yuva420p[v{i}]"
            )
            fade_in = "fade=t=in:st=0.3:d=0.4:alpha=1"
            fade_out = f",fade=t=out:st={max(0.0, cut.seconds - 0.5):.2f}:d=0.35:alpha=1" if not cut.card else ""
            graph.append(f"[{o}:v]format=rgba,{fade_in}{fade_out}[c{i}]")
            graph.append(f"[v{i}][c{i}]overlay=0:0:format=auto,format=yuv420p[w{i}]")
        # xfade chain: transition k starts FADE before the joined clip's end
        joined, total = "w0", CUTS[0].seconds
        for i in range(1, len(CUTS)):
            # each cut but the last carries a FADE tail, so a transition starts after the sum of the cuts before it
            graph.append(f"[{joined}][w{i}]xfade=transition=fade:duration={FADE}:offset={total:.3f}[x{i}]")
            joined, total = f"x{i}", total + CUTS[i].seconds
        out = out_dir / "promo.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", *inputs,
                "-filter_complex", ";".join(graph), "-map", f"[{joined}]",
                "-c:v", "libx264", "-preset", "slow", "-crf", "27", "-maxrate", "1500k", "-bufsize", "4000k",
                "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart", str(out),
            ],
            check=True,
        )  # fmt: skip
    return out


POSTER_AT = 18.4  # s into the video: the spliced robots, captioned

# Bits of the finished video for the looping preview: (start s, seconds), each inside a captioned cut.
PREVIEW = [(4.6, 2.0), (11.0, 2.0), (16.6, 2.0), (22.6, 2.0), (30.0, 2.0), (36.4, 2.0), (47.6, 1.6), (54.0, 2.0)]


def preview(video: Path, out_dir: Path) -> None:
    """The looping preview (animated WebP, 960 px wide) cut from bits of the video, and the poster (JPEG)."""
    inputs: list[str] = []
    parts: list[str] = []
    for i, (start, seconds) in enumerate(PREVIEW):
        inputs += ["-ss", f"{start}", "-t", f"{seconds}", "-i", str(video)]
        parts.append(f"[{i}:v]setpts=PTS-STARTPTS,fps=15,scale=960:-1:flags=lanczos[p{i}]")
    joined = "".join(f"[p{i}]" for i in range(len(PREVIEW)))
    graph = ";".join([*parts, f"{joined}concat=n={len(PREVIEW)}:v=1:a=0[v]"])
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph, "-map", "[v]",
            "-c:v", "libwebp", "-quality", "60", "-compression_level", "6", "-loop", "0",
            str(out_dir / "promo_preview.webp"),
        ],
        check=True,
    )  # fmt: skip
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{POSTER_AT}", "-i", str(video), "-frames:v", "1",
            "-vf", "scale=1280:-1:flags=lanczos", "-q:v", "4", str(out_dir / "promo_poster.jpg"),
        ],
        check=True,
    )  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/media"))
    parser.add_argument("--no-preview", action="store_true", help="only the video")
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on PATH")
    video = build(args.clips, args.out)
    print(f"{video} ({video.stat().st_size / 1e6:.1f} MB)")
    if not args.no_preview:
        preview(video, args.out)


if __name__ == "__main__":
    main()
