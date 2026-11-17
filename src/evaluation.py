"""Evaluation helpers: IoU, box matching, image-level sensitivity/specificity, threshold tuning, latency."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image


def safe_div(a: float, b: float) -> Optional[float]:
    """a / b, or None when undefined (never silently 0)."""
    return None if b == 0 else a / b


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_boxes(gt_boxes, pred_boxes, pred_scores, iou_thr: float = 0.5):
    """Greedy, score-ordered, class-agnostic matching. Returns (tp, fp, fn, matched_ious)."""
    order = sorted(range(len(pred_boxes)), key=lambda i: -pred_scores[i])
    used, ious, tp = set(), [], 0
    for i in order:
        best, best_iou = -1, 0.0
        for j, g in enumerate(gt_boxes):
            if j in used:
                continue
            v = box_iou(pred_boxes[i], g)
            if v > best_iou:
                best, best_iou = j, v
        if best >= 0 and best_iou >= iou_thr:
            used.add(best)
            tp += 1
            ious.append(best_iou)
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp, ious


def image_level_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, Optional[float]]:
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    prec = safe_div(tp, tp + fp)
    f1 = safe_div(2 * prec * sens, prec + sens) if (prec is not None and sens is not None) else None
    acc = safe_div(tp + tn, tp + fp + tn + fn)
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def read_yolo_labels(label_path, w: int, h: int):
    """Return [(cls, x1, y1, x2, y2)] in pixels."""
    p = Path(label_path)
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        out.append((cls, (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return out


def load_ground_truth(image_paths, labels_dir) -> Dict[str, List[List[float]]]:
    """Map image path -> list of GT boxes [x1,y1,x2,y2]. Image-level truth: positive iff >= 1 GT box."""
    gts = {}
    for p in image_paths:
        with Image.open(p) as im:
            w, h = im.size
        lab = Path(labels_dir) / (Path(p).stem + ".txt")
        gts[str(p)] = [list(b[1:]) for b in read_yolo_labels(lab, w, h)]
    return gts


def collect_predictions(model, image_paths, conf: float = 0.001, iou: float = 0.6, imgsz: int = 640,
                        device=None, batch: int = 16):
    out = []
    paths = [str(p) for p in image_paths]
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        results = model.predict(chunk, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
        for p, r in zip(chunk, results):
            h, w = r.orig_shape
            if r.boxes is not None and len(r.boxes):
                b = r.boxes.xyxy.cpu().numpy().tolist()
                s = r.boxes.conf.cpu().numpy().tolist()
            else:
                b, s = [], []
            out.append({"path": p, "w": w, "h": h, "boxes": b, "scores": s})
    return out


def evaluate_at(preds, gts, conf: float, iou_thr: float = 0.5) -> dict:
    """Image-level confusion + box-level TP/FP/FN at a confidence threshold."""
    itp = ifp = itn = ifn = 0
    btp = bfp = bfn = 0
    ious: List[float] = []
    for p in preds:
        keep = [i for i, s in enumerate(p["scores"]) if s >= conf]
        pb = [p["boxes"][i] for i in keep]
        ps = [p["scores"][i] for i in keep]
        g = gts[p["path"]]
        has_gt, has_pred = len(g) > 0, len(pb) > 0
        if has_gt and has_pred:
            itp += 1
        elif has_gt:
            ifn += 1
        elif has_pred:
            ifp += 1
        else:
            itn += 1
        tp, fp, fn, m = match_boxes(g, pb, ps, iou_thr)
        btp, bfp, bfn = btp + tp, bfp + fp, bfn + fn
        ious += m
    img = image_level_metrics(itp, ifp, itn, ifn)
    bp, br = safe_div(btp, btp + bfp), safe_div(btp, btp + bfn)
    bf1 = safe_div(2 * bp * br, bp + br) if (bp is not None and br is not None) else None
    return {"image": img,
            "box": {"precision": bp, "recall": br, "f1": bf1, "tp": btp, "fp": bfp, "fn": bfn,
                    "mean_matched_iou": float(np.mean(ious)) if ious else None, "iou_threshold": iou_thr}}


def tune_thresholds(model, image_paths, gts, conf_grid, iou_grid, imgsz=640, device=None):
    """Grid-search NMS-IoU and confidence on VALIDATION data (maximise image-level F1, tie -> specificity)."""
    rows = []
    for iou in iou_grid:
        preds = collect_predictions(model, image_paths, conf=min(conf_grid) * 0.5, iou=iou, imgsz=imgsz, device=device)
        for c in conf_grid:
            r = evaluate_at(preds, gts, c)
            im = r["image"]
            rows.append({"conf": float(c), "iou": float(iou), "f1": im["f1"] or 0.0,
                         "sensitivity": im["sensitivity"], "specificity": im["specificity"],
                         "box_f1": r["box"]["f1"] or 0.0})
    df = pd.DataFrame(rows)
    ranked = df.assign(_spec=df["specificity"].fillna(0.0)).sort_values(["f1", "_spec", "box_f1"], ascending=False)
    best = ranked.iloc[0]
    return df, {"conf": float(best["conf"]), "iou": float(best["iou"]), "val_image_f1": float(best["f1"])}


def hardware_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:  # noqa: BLE001
        info["device"] = "CPU"
    return info


def measure_latency(model, image_path, imgsz=640, conf=0.25, iou=0.45, warmup=10, runs=50, device=None) -> dict:
    """End-to-end predict() latency (pre-process + inference + NMS) after warm-up, in ms."""
    import cv2

    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        cuda = False
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read {image_path}")

    def sync():
        if cuda:
            import torch
            torch.cuda.synchronize()

    for _ in range(warmup):
        model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
    times = []
    for _ in range(runs):
        sync()
        t = time.perf_counter()
        model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
        sync()
        times.append((time.perf_counter() - t) * 1000.0)
    arr = np.array(times)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95)),
            "std": float(arr.std()), "warmup_runs": warmup, "timed_runs": runs, "imgsz": imgsz,
            "batch_size": 1, "conf": conf, "iou": iou, **hardware_info()}


def save_results(out_dir, metrics: dict, tune_df: Optional[pd.DataFrame] = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    flat = []
    for section, vals in metrics.items():
        if isinstance(vals, dict):
            for k, v in vals.items():
                flat.append({"section": section, "metric": k, "value": v})
    pd.DataFrame(flat).to_csv(out / "metrics.csv", index=False)
    if tune_df is not None:
        tune_df.to_csv(out / "threshold_tuning_val.csv", index=False)
