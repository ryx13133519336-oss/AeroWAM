"""ROS Image conversion and deterministic 256x256 JPEG encoding."""

from __future__ import annotations

from io import BytesIO
from typing import Any


def ros_image_to_jpeg(message: Any, *, size: tuple[int, int] = (256, 256), quality: int = 90) -> bytes:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for model-ready RGB recording") from exc
    width, height = int(message.width), int(message.height)
    encoding = str(message.encoding).lower()
    raw = bytes(message.data)
    modes = {
        "rgb8": ("RGB", "RGB", 3), "bgr8": ("RGB", "BGR", 3),
        "rgba8": ("RGBA", "RGBA", 4), "bgra8": ("RGBA", "BGRA", 4),
        "mono8": ("L", "L", 1),
    }
    if encoding not in modes:
        raise ValueError(f"unsupported camera encoding: {encoding}")
    mode, raw_mode, channels = modes[encoding]
    step = int(getattr(message, "step", width * channels))
    expected = step * height
    if len(raw) < expected:
        raise ValueError(f"image buffer is short: {len(raw)} < {expected}")
    image = Image.frombytes(mode, (width, height), raw[:expected], "raw", raw_mode, step, 1)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(size, Image.Resampling.BILINEAR)
    output = BytesIO()
    image.save(output, format="JPEG", quality=int(quality), optimize=False, progressive=False, subsampling=2)
    return output.getvalue()


def is_jpeg(payload: bytes) -> bool:
    return len(payload) >= 4 and payload[:2] == b"\xff\xd8" and payload[-2:] == b"\xff\xd9"
