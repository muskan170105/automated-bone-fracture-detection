
"""Modular fracture detection pipeline around a trained YOLOv8 model."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .image_processor import bgr_to_pil, pil_to_bgr, tile_boxes


class ModelNotAvailableError(RuntimeError):
    """Raised when the trained weights file does not exist yet."""


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[float, float, float, float]


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    annotated: Optional[Image.Image] = None
    inference_ms: float = 0.0
    image_size: Tuple[int, int] = (0, 0)


def draw_detections(img: Image.Image, detections: List[Detection]) -> Image.Image:
    """Draw boxes and labels on a copy of the image."""
    bgr = pil_to_bgr(img)
    h, w = bgr.shape[:2]

    thick = max(2, int(round(min(h, w) / 300)))
    scale = max(0.5, min(h, w) / 900)

    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d.box]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        color = (0, 200, 255)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thick)

        text = f"{d.label} {d.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thick - 1)
        )

        ty = max(y1, th + base + 2)
        cv2.rectangle(
            bgr, (x1, ty - th - base - 2),
            (x1 + tw + 4, ty), color, -1
        )
        cv2.putText(
            bgr, text, (x1 + 2, ty - base - 1),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20),
            max(1, thick - 1), cv2.LINE_AA
        )

    return bgr_to_pil(bgr)


class FractureDetector:
    def __init__(
        self,
        weights_path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device=None
    ):
        self.weights_path = Path(weights_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self._model = None

    @property
    def available(self) -> bool:
        return (
            self.weights_path.is_file()
            and self.weights_path.stat().st_size > 0
        )

    def load(self):
        if self._model is None:
            if not self.available:
                raise ModelNotAvailableError(
                    f"Model weights not found at {self.weights_path}"
                )

            from ultralytics import YOLO
            self._model = YOLO(str(self.weights_path))

        return self._model

    def _run(self, bgr, conf, iou, imgsz):
        model = self.load()

        result = model.predict(
            source=bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )[0]

        names = result.names
        output = []

        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls_id in zip(xyxy, scores, classes):
                output.append((
                    float(box[0]), float(box[1]),
                    float(box[2]), float(box[3]),
                    float(score), int(cls_id)
                ))

        return output, names

    def predict(
        self,
        image: Image.Image,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        tiled: bool = False,
        tile_size: int = 640,
        overlap: float = 0.25
    ) -> DetectionResult:

        conf = self.conf if conf is None else conf
        iou = self.iou if iou is None else iou

        if not 0 <= conf <= 1:
            raise ValueError("Confidence must be between 0 and 1.")

        if not 0 <= iou <= 1:
            raise ValueError("IoU must be between 0 and 1.")

        bgr = pil_to_bgr(image)
        h, w = bgr.shape[:2]

        t0 = time.perf_counter()

        raw, names = self._run(bgr, conf, iou, self.imgsz)

        if tiled and max(h, w) > tile_size:
            for x0, y0, x1, y1 in tile_boxes(
                w, h, tile_size, overlap
            ):
                part, names = self._run(
                    bgr[y0:y1, x0:x1],
                    conf,
                    iou,
                    tile_size
                )

                raw.extend([
                    (
                        a + x0, b + y0,
                        c + x0, d + y0,
                        score, cls_id
                    )
                    for a, b, c, d, score, cls_id in part
                ])

            raw = self._nms(raw, conf, iou)

        elapsed = (time.perf_counter() - t0) * 1000.0

        detections = [
            Detection(
                label=str(names.get(cls_id, cls_id)),
                confidence=score,
                box=(x1, y1, x2, y2)
            )
            for x1, y1, x2, y2, score, cls_id in raw
        ]

        detections.sort(
            key=lambda detection: -detection.confidence
        )

        return DetectionResult(
            detections=detections,
            annotated=draw_detections(image, detections),
            inference_ms=elapsed,
            image_size=(w, h)
        )

    @staticmethod
    def _nms(raw, conf, iou):
        """Apply class-aware NMS to tiled predictions."""
        if not raw:
            return []

        kept = []

        class_ids = sorted(set(row[5] for row in raw))

        for cls_id in class_ids:
            class_rows = [
                row for row in raw if row[5] == cls_id
            ]

            boxes = [
                [a, b, c - a, d - b]
                for a, b, c, d, _, _ in class_rows
            ]
            scores = [row[4] for row in class_rows]

            indices = cv2.dnn.NMSBoxes(
                boxes, scores, conf, iou
            )

            if indices is None or len(indices) == 0:
                continue

            for index in np.array(indices).reshape(-1):
                kept.append(class_rows[int(index)])

        return kept
