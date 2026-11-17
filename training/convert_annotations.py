"""Annotation conversion (YOLO txt / Pascal VOC xml / COCO json -> YOLO) and leakage-aware dataset building."""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(root) -> List[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file())


def validate_image_file(path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:  # noqa: BLE001
        return False


def xyxy_to_yolo(x1, y1, x2, y2, w, h) -> Optional[Tuple[float, float, float, float]]:
    """Clip to the image and return normalised (xc, yc, bw, bh); None if degenerate."""
    if w <= 0 or h <= 0:
        return None
    x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
    y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1e-6 or bh <= 1e-6:
        return None
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, bw / w, bh / h)


def parse_voc(xml_path) -> Tuple[int, int, List[Tuple[str, float, float, float, float]]]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    objs = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        if bb is None:
            continue
        objs.append((o.findtext("name", "object").strip(),
                     float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                     float(bb.findtext("xmax")), float(bb.findtext("ymax"))))
    return w, h, objs


def parse_coco(json_path) -> Dict[str, Tuple[int, int, list]]:
    """Return {image basename: (w, h, [(class_name, x1, y1, x2, y2)])}; {} if not a COCO file."""
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        return {}
    names = {c["id"]: c["name"] for c in data.get("categories", [])}
    imgs = {im["id"]: im for im in data["images"]}
    out = {Path(im["file_name"]).name: (int(im["width"]), int(im["height"]), []) for im in data["images"]}
    for a in data["annotations"]:
        im = imgs.get(a["image_id"])
        if im is None or "bbox" not in a:
            continue
        x, y, bw, bh = a["bbox"]
        out[Path(im["file_name"]).name][2].append(
            (names.get(a["category_id"], str(a["category_id"])), x, y, x + bw, y + bh))
    return out


def read_yolo_txt(path) -> List[Tuple[int, float, float, float, float]]:
    """Read YOLO detection labels; polygon (segmentation) rows are converted to their bounding box."""
    out = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        vals = list(map(float, p[1:]))
        if len(vals) == 4:
            xc, yc, bw, bh = vals
        else:
            xs, ys = vals[0::2], vals[1::2]
            xc, yc, bw, bh = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
        box = xyxy_to_yolo((xc - bw / 2), (yc - bh / 2), (xc + bw / 2), (yc + bh / 2), 1.0, 1.0)
        if box:
            out.append((cls, *box))
    return out


def load_class_names_from_yaml(root) -> Optional[List[str]]:
    for yp in sorted(Path(root).rglob("*.yaml")) + sorted(Path(root).rglob("*.yml")):
        try:
            d = yaml.safe_load(yp.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and "names" in d:
            n = d["names"]
            if isinstance(n, dict):
                return [str(n[k]) for k in sorted(n, key=lambda x: int(x))]
            if isinstance(n, list):
                return [str(x) for x in n]
    return None


def group_key(stem: str) -> str:
    """Key used to keep related images together (prevents leakage).
    Uses a patient id when the filename has one (e.g. MURA 'patient00012'); otherwise strips
    Roboflow-style augmentation suffixes ('name_jpg.rf.<hash>') so augmented copies stay together."""
    m = re.search(r"(patient\d+)", stem, flags=re.I)
    if m:
        return m.group(1).lower()
    base = re.split(r"[._-]rf[._-]", stem)[0]
    return re.sub(r"_(jpg|jpeg|png)$", "", base, flags=re.I)


def build_yolo_dataset(src_root, out_root, ratios=(0.70, 0.15, 0.15), seed: int = 42,
                       unlabeled_as_negative: bool = False, max_side: Optional[int] = None,
                       class_names: Optional[List[str]] = None) -> dict:
    """Convert any supported annotation format to a YOLO dataset with group-aware train/val/test splits."""
    src_root, out_root = Path(src_root), Path(out_root)
    images = find_images(src_root)
    yolo_txt, voc_xml, coco = {}, {}, {}
    for p in src_root.rglob("*"):
        if p.suffix.lower() == ".txt":
            yolo_txt.setdefault(p.stem, p)
        elif p.suffix.lower() == ".xml":
            voc_xml.setdefault(p.stem, p)
    for jp in src_root.rglob("*.json"):
        coco.update(parse_coco(jp))

    names: List[str] = list(class_names or load_class_names_from_yaml(src_root) or [])

    def cid(name: str) -> int:
        if name not in names:
            names.append(name)
        return names.index(name)

    report = {"images_found": len(images), "corrupted": [], "skipped_no_label": 0, "invalid_boxes_dropped": 0,
              "label_sources": {"yolo": 0, "voc": 0, "coco": 0}}
    records = []
    for img in images:
        if not validate_image_file(img):
            report["corrupted"].append(str(img))
            continue
        with Image.open(img) as im:
            iw, ih = im.size
        boxes, src = None, None
        if img.stem in yolo_txt:
            boxes, src = read_yolo_txt(yolo_txt[img.stem]), "yolo"
        elif img.stem in voc_xml:
            w, h, objs = parse_voc(voc_xml[img.stem])
            w, h = (w or iw), (h or ih)
            boxes = []
            for (n, x1, y1, x2, y2) in objs:
                b = xyxy_to_yolo(x1, y1, x2, y2, w, h)
                if b:
                    boxes.append((cid(n), *b))
                else:
                    report["invalid_boxes_dropped"] += 1
            src = "voc"
        elif img.name in coco:
            w, h, objs = coco[img.name]
            w, h = (w or iw), (h or ih)
            boxes = []
            for (n, x1, y1, x2, y2) in objs:
                b = xyxy_to_yolo(x1, y1, x2, y2, w, h)
                if b:
                    boxes.append((cid(n), *b))
                else:
                    report["invalid_boxes_dropped"] += 1
            src = "coco"
        if boxes is None:
            if unlabeled_as_negative:
                boxes = []
            else:
                report["skipped_no_label"] += 1
                continue
        if src:
            report["label_sources"][src] += 1
        records.append({"path": img, "stem": img.stem, "boxes": boxes})

    max_id = max((b[0] for r in records for b in r["boxes"]), default=-1)
    while len(names) <= max_id:
        names.append(f"class{len(names)}")
    if not names:
        names = ["fracture"]

    groups: Dict[str, list] = {}
    for r in records:
        groups.setdefault(group_key(r["stem"]), []).append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    total = len(records)
    t_train, t_val = ratios[0] * total, (ratios[0] + ratios[1]) * total
    split_of, running = {}, 0
    for k in keys:
        split = "train" if running < t_train else ("val" if running < t_val else "test")
        for r in groups[k]:
            split_of[id(r)] = split
        running += len(groups[k])

    if out_root.exists():
        shutil.rmtree(out_root)
    for s in ("train", "val", "test"):
        (out_root / "images" / s).mkdir(parents=True)
        (out_root / "labels" / s).mkdir(parents=True)
    used, stats = set(), {s: {"images": 0, "positive": 0, "negative": 0, "boxes": 0} for s in ("train", "val", "test")}
    for r in records:
        s = split_of[id(r)]
        name, n = r["stem"], 1
        while name in used:
            name, n = f"{r['stem']}_{n}", n + 1
        used.add(name)
        ext = r["path"].suffix.lower()
        dst = out_root / "images" / s / f"{name}{ext}"
        img = None
        if max_side:
            import cv2
            img = cv2.imread(str(r["path"]))
        if img is not None and max(img.shape[:2]) > max_side:
            f = max_side / max(img.shape[:2])
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), img)
        else:
            shutil.copy2(r["path"], dst)
        lines = [f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for (c, xc, yc, bw, bh) in r["boxes"]]
        (out_root / "labels" / s / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        st = stats[s]
        st["images"] += 1
        st["boxes"] += len(lines)
        st["positive" if lines else "negative"] += 1

    data = {"path": str(out_root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {i: n for i, n in enumerate(names)}}
    (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    report.update({"classes": names, "splits": stats, "images_used": total, "group_count": len(groups),
                   "split_method": "grouped (patient id / augmentation-source) shuffle, seed=%d" % seed})
    return report


def main():
    ap = argparse.ArgumentParser(description="Convert annotations to a YOLO dataset.")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unlabeled-as-negative", action="store_true")
    ap.add_argument("--max-side", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(build_yolo_dataset(a.src, a.out, seed=a.seed, unlabeled_as_negative=a.unlabeled_as_negative,
                                        max_side=a.max_side), indent=2))


if __name__ == "__main__":
    main()
