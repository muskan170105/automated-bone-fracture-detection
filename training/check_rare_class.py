from pathlib import Path

root = Path("raw_dataset")

print("Folders inside raw_dataset:\n")

for folder in root.iterdir():
    if folder.is_dir():
        print("\nFolder:", repr(folder.name))

        labels = folder / "train" / "labels"

        print("Train labels exists:", labels.exists())

        if labels.exists():
            print("Label files:", len(list(labels.glob("*.txt"))))