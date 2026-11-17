import pytest
from PIL import Image

from src.detector import Detection, FractureDetector, ModelNotAvailableError, draw_detections


def test_missing_weights_not_available(tmp_path):
    det = FractureDetector(tmp_path / "missing.pt")
    assert det.available is False
    with pytest.raises(ModelNotAvailableError):
        det.load()
    with pytest.raises(ModelNotAvailableError):
        det.predict(Image.new("RGB", (100, 100)))


def test_draw_detections_keeps_size():
    img = Image.new("RGB", (200, 150), "black")
    out = draw_detections(img, [Detection("fracture", 0.8, (10, 10, 80, 90))])
    assert out.size == img.size and out.getpixel((10, 10)) != (0, 0, 0)


def test_nms_removes_duplicates():
    raw = [(0, 0, 100, 100, 0.9, 0), (2, 2, 100, 100, 0.8, 0), (300, 300, 380, 380, 0.7, 0)]
    assert len(FractureDetector._nms(raw, 0.1, 0.5)) == 2
