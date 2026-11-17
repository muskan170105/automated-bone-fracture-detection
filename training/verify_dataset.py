
from pathlib import Path
from collections import Counter

DATASET = Path("raw_dataset/BoneFractureYolo8")

for split in ["train", "valid", "test"]:
    label_dir = DATASET / split / "labels"

    class_counts = Counter()
    valid_polygons = 0
    invalid_lines = 0
    empty_files = 0

    for file in label_dir.glob("*.txt"):
        lines = file.read_text(encoding="utf-8").splitlines()

        if not any(line.strip() for line in lines):
            empty_files += 1

        for line in lines:
            if not line.strip():
                continue

            parts = line.split()

            try:
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))

                if (
                    class_id not in range(7)
                    or len(coords) < 6
                    or len(coords) % 2 != 0
                    or not all(0 <= v <= 1 for v in coords)
                ):
                    invalid_lines += 1
                    continue

                valid_polygons += 1
                class_counts[class_id] += 1

            except (ValueError, IndexError):
                invalid_lines += 1

    print(f"\n--- {split.upper()} ---")
    print("Valid polygons:", valid_polygons)
    print("Class distribution:", dict(sorted(class_counts.items())))
    print("Invalid lines:", invalid_lines)
    print("Empty label files:", empty_files)
