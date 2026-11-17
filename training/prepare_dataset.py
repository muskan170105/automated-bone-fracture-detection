
from pathlib import Path
import shutil
from collections import Counter

SOURCE = Path("raw_dataset/BoneFractureYolo8")
OUTPUT = Path("processed_dataset")

SPLITS = ["train", "valid", "test"]

# Original class ID -> new class ID
CLASS_MAPPING = {
    0: 0,
    1: 1,
    2: 2,
    4: 3,
    5: 4,
    6: 5,
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

if not SOURCE.exists():
    raise FileNotFoundError(f"Source dataset not found: {SOURCE}")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Rename or inspect it before running again."
    )


def polygon_to_bbox(coords):
    """Convert normalized polygon coordinates to YOLO xywh."""

    xs = coords[0::2]
    ys = coords[1::2]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


total_images = 0
total_boxes = Counter()
excluded_boxes = 0

for split in SPLITS:
    source_images = SOURCE / split / "images"
    source_labels = SOURCE / split / "labels"

    output_images = OUTPUT / "images" / split
    output_labels = OUTPUT / "labels" / split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    for image_path in source_images.iterdir():

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing annotation file for {image_path.name}"
            )

        # Preserve original image
        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        converted_annotations = []

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:
            if not line.strip():
                continue

            parts = line.split()

            original_id = int(parts[0])

            if original_id == 3:
                excluded_boxes += 1
                continue

            if original_id not in CLASS_MAPPING:
                raise ValueError(
                    f"Unexpected class ID {original_id} in {label_path}"
                )

            coords = list(map(float, parts[1:]))

            if (
                len(coords) < 6
                or len(coords) % 2 != 0
                or not all(0 <= v <= 1 for v in coords)
            ):
                raise ValueError(
                    f"Invalid polygon in {label_path}: {line}"
                )

            x, y, w, h = polygon_to_bbox(coords)

            new_id = CLASS_MAPPING[original_id]

            converted_annotations.append(
                f"{new_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}"
            )

            total_boxes[new_id] += 1

        # Empty files are retained for images without annotations
        output_label = output_labels / f"{image_path.stem}.txt"

        output_label.write_text(
            "\n".join(converted_annotations),
            encoding="utf-8"
        )

        total_images += 1

    print(f"{split.upper()} prepared.")

# Generate new YOLO dataset configuration
data_yaml = """path: ../processed_dataset

train: images/train
val: images/valid
test: images/test

names:
  0: elbow positive
  1: fingers positive
  2: forearm fracture
  3: humerus
  4: shoulder fracture
  5: wrist positive
"""

(OUTPUT / "data.yaml").write_text(
    data_yaml,
    encoding="utf-8"
)

print("\nDATASET PREPARATION COMPLETE")
print("Total images:", total_images)
print("Excluded class-3 annotations:", excluded_boxes)
print("Converted bounding boxes:", dict(sorted(total_boxes.items())))
print("Output:", OUTPUT.resolve())
