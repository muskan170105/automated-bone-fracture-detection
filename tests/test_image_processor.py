import io
import numpy as np
import pytest
from PIL import Image

from src.image_processor import ImageValidationError, tile_boxes, validate_image


def _png(size=(128, 128), mode="RGB"):
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(size[1], size[0]) * 255).astype("uint8")).convert(mode).save(buf, format="PNG")
    return buf.getvalue()


def test_valid_image_returns_rgb():
    assert validate_image(_png()).mode == "RGB"


def test_grayscale_converted():
    assert validate_image(_png(mode="L")).mode == "RGB"


def test_16bit_png_supported():
    arr = (np.random.rand(100, 100) * 65535).astype("uint16")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    assert validate_image(buf.getvalue()).mode == "RGB"


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_invalid_bytes(data):
    with pytest.raises(ImageValidationError):
        validate_image(data)


def test_too_small_and_too_large():
    with pytest.raises(ImageValidationError):
        validate_image(_png((20, 20)))
    with pytest.raises(ImageValidationError):
        validate_image(_png((300, 300)), max_mb=0.0001)


def test_unsupported_format():
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, format="BMP")
    with pytest.raises(ImageValidationError):
        validate_image(buf.getvalue())


def test_tiles_cover_image():
    tiles = tile_boxes(1500, 1000, 640, 0.25)
    assert max(t[2] for t in tiles) == 1500 and max(t[3] for t in tiles) == 1000
