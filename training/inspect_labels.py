
from pathlib import Path

label_dir = Path("raw_dataset/BoneFractureYolo8/train/labels")

files = list(label_dir.glob("*.txt"))

print("Total label files:", len(files))

shown = 0

for file in files:
    content = file.read_text(encoding="utf-8").strip()

    if content:
        print("\nFile:", file.name)

        for line in content.splitlines()[:3]:
            print("Annotation:", line)

        shown += 1

    if shown == 3:
        break
