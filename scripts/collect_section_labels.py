import glob
from pathlib import Path
import json


def collect_labels():
    files = glob.glob("dataset/sections/*.txt")
    labels = set()
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":")
                if len(parts) >= 3:
                    labels.add(parts[2].strip())

    sorted_labels = sorted(list(labels))
    print(f"Found {len(sorted_labels)} unique labels:")
    for l in sorted_labels:
        print(l)

    # Map labels to indices (0-based)
    label_map = {i: l for i, l in enumerate(sorted_labels)}

    output_path = "data/music_structures.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)
    print(f"Saved label map to {output_path}")


if __name__ == "__main__":
    collect_labels()
