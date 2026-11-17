
from pathlib import Path

label_dir = Path("raw_dataset/BoneFractureYolo8/train/labels")

for file in list(label_dir.glob("*.txt"))[:10]:
    print("\nFILE:", file.name)

    lines = file.read_text(encoding="utf-8").splitlines()

    for line in lines[:2]:
        values = line.split()

        print("Raw:", repr(line))
        print("Number of values:", len(values))
        print("Values:", values)
