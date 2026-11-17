
from pathlib import Path
import hashlib

root = Path("raw_dataset")

folders = [
    root / "bone fracture detection.v4-v4.yolov8",
    root / "BoneFractureYolo8"
]

for folder in folders:
    print("\nDATASET:", folder.name)

    for split in ["train", "valid", "test"]:
        label_dir = folder / split / "labels"
        files = list(label_dir.glob("*.txt")) if label_dir.exists() else []

        class_counts = {}
        for file in files:
            for line in file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts:
                    class_id = parts[0]
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1

        print(split, "files:", len(files))
        print("Class counts:", class_counts)

# Compare the two training label collections
for folder in folders:
    label_dir = folder / "train" / "labels"
    hashes = {
        hashlib.sha256(f.read_bytes()).hexdigest()
        for f in label_dir.glob("*.txt")
    }
    print("\nUnique label contents:", folder.name, len(hashes))
