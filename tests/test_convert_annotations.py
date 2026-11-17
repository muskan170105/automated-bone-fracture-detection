import json

import yaml
from PIL import Image

from training.convert_annotations import build_yolo_dataset, group_key, parse_voc, xyxy_to_yolo
from training.validate_dataset import summarize_yolo_dataset

VOC = """<annotation><size><width>100</width><height>100</height></size>
<object><name>fracture</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox></object></annotation>"""


def test_xyxy_to_yolo():
    assert xyxy_to_yolo(10, 20, 50, 60, 100, 100) == (0.3, 0.4, 0.4, 0.4)
    assert xyxy_to_yolo(10, 10, 10, 50, 100, 100) is None


def test_group_key():
    assert group_key("IMG1_jpg.rf.abcdef") == group_key("IMG1_jpg.rf.zzzzzz") == "IMG1"
    assert group_key("patient00012_study1_image2") == "patient00012"


def test_build_dataset_voc(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(12):
        Image.new("RGB", (100, 100), "white").save(src / f"img{i}_jpg.rf.h{i}.jpg")
        if i % 2 == 0:
            (src / f"img{i}_jpg.rf.h{i}.xml").write_text(VOC)
    (src / "bad.jpg").write_bytes(b"corrupt")
    rep = build_yolo_dataset(src, tmp_path / "out", unlabeled_as_negative=True)
    assert rep["label_sources"]["voc"] == 6 and len(rep["corrupted"]) == 1 and rep["images_used"] == 12
    cfg = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())
    assert cfg["names"][0] == "fracture"
    assert summarize_yolo_dataset(tmp_path / "out")["ok"]


def test_parse_voc(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text(VOC)
    w, h, objs = parse_voc(p)
    assert (w, h) == (100, 100) and objs[0][0] == "fracture"
