"""Lossless encoding keeps the previous material channels and bounded lazy loaders."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import re
from importlib.resources import files
from pathlib import Path

import pytest
from PIL import Image


@pytest.mark.parametrize(
    "style,encoded_sha,rgba_sha,loader_sha,body_limit",
    [
        (
            "wood",
            "fd7e9db4b72ffe13cdcc7b80a247599de63b8528c48c76f9e6df54bb4891bc73",
            "48039c51a4638ac3a7e213fbd61125e3da47b5de91471c7d8aa95d7f6bc54dfa",
            "1f4b6f3ab189c3ed490e4ab0fd73b6f6fa9c4ffb5bc85a3341b101e85c215251",
            98234,
        ),
        (
            "grass",
            "71e9e218db6a39f483d8cced9a85274ce6b741928b669fdb81549179fe2d4389",
            "40f0ec20a41e4b4f80e1ce699dee876753dde9e60db330daafd413737ec84442",
            "b18d224179d10d5a0884c98eacfc57d316143b393f5b0a727577207996bf7f37",
            80705,
        ),
    ],
)
def test_compact_detail_preserves_original_rgba_and_loader(style, encoded_sha, rgba_sha, loader_sha, body_limit):
    module = files("robotforge.viewer").joinpath(f"static/ground-{style}.js").read_text(encoding="utf-8")
    assert len(module.encode("utf-8")) <= body_limit
    images = re.findall(r"data:image/(jpeg|webp);base64,([^\"]+)", module)
    assert [kind for kind, _ in images] == ["jpeg", "webp"]
    detail = base64.b64decode(images[1][1], validate=True)
    assert hashlib.sha256(detail).hexdigest() == encoded_sha
    with Image.open(io.BytesIO(detail)) as image:
        assert image.size == (256, 256) and image.mode == "RGB"
        # These are frozen decoded hashes of the former PNGs, including height,
        # roughness, unused blue and opaque alpha; no tolerance or quantization.
        assert hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest() == rgba_sha
    loader = re.sub(r"data:image/[^;]+;base64,[^\"]+", "<IMAGE>", module)
    assert hashlib.sha256(loader.encode("utf-8")).hexdigest() == loader_sha


@pytest.mark.parametrize("existing", [False, True])
def test_offline_wood_rejects_changed_original_before_writing(tmp_path, existing):
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_ground_wood.py"
    spec = importlib.util.spec_from_file_location("prepare_ground_wood", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "oak_wood_planks_diff_1k.jpg").write_bytes(b"unverified source")
    output = tmp_path / "output.js"
    if existing:
        output.write_text("preserve existing output", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected source checksum"):
        module.generate(tmp_path, output)
    if existing:
        assert output.read_text(encoding="utf-8") == "preserve existing output"
    else:
        assert not output.exists()
