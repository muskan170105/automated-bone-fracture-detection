"""Image validation and preprocessing utilities."""
from __future__ import annotations

import io
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps

ALLOWED_FORMATS = {"JPEG", "PNG"}


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a usable image."""


def validate_image(data: bytes, max_mb: float = 10.0, min_side: int = 64) -> Image.Image:
    """Validate raw bytes and return an 8-bit RGB PIL image."""
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise ImageValidationError(f"File is larger than {max_mb:.0f} MB.")
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise ImageValidationError("The file is not a valid image.") from exc
    if img.format not in ALLOWED_FORMATS:
        raise ImageValidationError("Unsupported format. Please upload a JPG, JPEG or PNG image.")
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I", "F"):  # 16-bit radiographs -> 8-bit
        arr = np.asarray(img).astype("float64")
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        img = Image.fromarray(arr.astype("uint8"))
    img = img.convert("RGB")
    if min(img.size) < min_side:
        raise ImageValidationError(f"Image is too small (minimum side {min_side}px).")
    return img


def enhance_contrast(img: Image.Image, clip_limit: float = 2.0, tile: int = 8) -> Image.Image:
    """Optional CLAHE contrast enhancement (off by default: the model is trained on raw images)."""
    import cv2

    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]))


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def tile_boxes(width: int, height: int, tile: int = 640, overlap: float = 0.25) -> List[Tuple[int, int, int, int]]:
    """Return overlapping (x0, y0, x1, y1) tiles covering the image."""
    if tile <= 0 or not 0 <= overlap < 1:
        raise ValueError("tile must be > 0 and 0 <= overlap < 1")
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(width - tile, 0) + 1, step))
    ys = list(range(0, max(height - tile, 0) + 1, step))
    if xs[-1] + tile < width:
        xs.append(width - tile)
    if ys[-1] + tile < height:
        ys.append(height - tile)
    return [(x, y, min(x + tile, width), min(y + tile, height)) for y in ys for x in xs]
