"""Validate a YOLO-format dataset (images, labels, value ranges, class distribution)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def summarize_yolo_dataset(root) -> dict:
    root = Path(root)
    out = {"splits": {}, "problems": []}
    for split in ("train", "val", "test"):
        imgs = sorted(p for p in (root / "images" / split).glob("*") if p.suffix.lower() in IMG_EXTS)
        cls_count, pos, neg = Counter(), 0, 0
        for p in imgs:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:  # noqa: BLE001
                out["problems"].append(f"corrupted image: {p}")
                continue
            lab = root / "labels" / split / f"{p.stem}.txt"
            if not lab.exists():
                out["problems"].append(f"missing label file: {lab}")
                continue
            rows = [l.split() for l in lab.read_text().splitlines() if l.strip()]
            for r in rows:
                try:
                    vals = list(map(float, r[1:5]))
                    if len(r) != 5 or any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                        out["problems"].append(f"bad label row in {lab}: {' '.join(r)}")
                    cls_count[int(float(r[0]))] += 1
                except (ValueError, IndexError):
                    out["problems"].append(f"unparseable label row in {lab}")
            pos, neg = (pos + 1, neg) if rows else (pos, neg + 1)
        out["splits"][split] = {"images": len(imgs), "positive": pos, "negative": neg,
                                "boxes_per_class": dict(cls_count)}
    out["ok"] = not out["problems"] and out["splits"]["train"]["images"] > 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    print(json.dumps(summarize_yolo_dataset(ap.parse_args().root), indent=2))


if __name__ == "__main__":
    main()
