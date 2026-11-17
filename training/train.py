"""Fine-tune YOLOv8-Nano on the prepared fracture dataset."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def train_model(data_yaml, project_dir, run_name="fracture_yolov8n", weights="yolov8n.pt", imgsz=640,
                epochs=50, batch=16, lr0=0.01, patience=20, seed=42, device=None, workers=2,
                augment_overrides=None, export_to=None) -> dict:
    """Train and return {'save_dir', 'best'}; optionally copy best.pt to `export_to`."""
    import torch
    from ultralytics import YOLO

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    aug = dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, degrees=5.0, translate=0.1, scale=0.3, fliplr=0.5,
               flipud=0.0, mosaic=1.0, mixup=0.0, close_mosaic=10)  # grayscale X-rays: no colour jitter
    aug.update(augment_overrides or {})
    model = YOLO(weights)  # pretrained COCO weights are downloaded automatically
    model.train(data=str(data_yaml), imgsz=imgsz, epochs=epochs, batch=batch, lr0=lr0, patience=patience,
                seed=seed, deterministic=True, device=device, workers=workers, project=str(project_dir),
                name=run_name, exist_ok=True, pretrained=True, plots=True, **aug)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found in {save_dir}/weights")
    if export_to:
        Path(export_to).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, export_to)
    return {"save_dir": str(save_dir), "best": str(best)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--export", default="models/best.pt")
    a = ap.parse_args()
    print(train_model(a.data, a.project, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, lr0=a.lr0,
                      export_to=a.export))


if __name__ == "__main__":
    main()
