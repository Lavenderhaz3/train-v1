#!/usr/bin/env python3
"""
02_prepare_dataset.py — MakeSense.ai YOLO export → train/val split for YOLOv8

Usage:
    python scripts/02_prepare_dataset.py [--ratio 0.3] [--seed 42]

Input:
    data/yolo/labels/  ← unzip MakeSense annotations.zip here
    data/raw/          ← original FLIR JPEGs (same filenames as labels)

Output:
    data/yolo/images/train/  70% images
    data/yolo/images/val/    30% images
    data/yolo/labels/train/  70% labels
    data/yolo/labels/val/    30% labels
    data/yolo/dataset.yaml   class paths
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "data" / "yolo" / "labels"
RAW_DIR = ROOT / "data" / "raw"
IMAGES_TRAIN = ROOT / "data" / "yolo" / "images" / "train"
IMAGES_VAL = ROOT / "data" / "yolo" / "images" / "val"
LABELS_TRAIN = ROOT / "data" / "yolo" / "labels" / "train"
LABELS_VAL = ROOT / "data" / "yolo" / "labels" / "val"

CLASS_NAMES = [
    "transformer",
    "switchgear",
    "cable",
    "busbar",
    "insulator",
]


def validate_label_file(path: Path) -> bool:
    """Check that a YOLO label file has valid format.
    Each line: class_id x_center y_center width height (all normalized 0~1)
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"  [WARN] {path.name}: invalid line '{line}' — expected 5 values")
                return False
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                print(f"  [WARN] {path.name}: non-numeric value in '{line}'")
                return False
            # Check normalized range
            if vals[1] < 0 or vals[1] > 1 or vals[2] < 0 or vals[2] > 1:
                print(f"  [WARN] {path.name}: coordinates out of [0,1] in '{line}'")
    return True


def find_label_files() -> list[Path]:
    """Find all .txt label files in data/yolo/labels/ (flat, from unzipped MakeSense export)."""
    txts = sorted(LABELS_DIR.glob("*.txt"))
    if not txts:
        # Labels might be in subdirectories (MakeSense sometimes nests them)
        txts = sorted(LABELS_DIR.rglob("*.txt"))
    return txts


def find_image(label_stem: str) -> Path | None:
    """Find the matching image for a label file.
    Tries: .jpg, .jpeg, .png, .JPG, .JPEG
    """
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    for ext in exts:
        path = RAW_DIR / f"{label_stem}{ext}"
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from MakeSense annotations")
    parser.add_argument("--ratio", type=float, default=0.3,
                        help="Validation split ratio (default: 0.3 = 30%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split reproducibility")
    args = parser.parse_args()

    # Step 1: Find labels
    label_files = find_label_files()
    if not label_files:
        print(f"ERROR: No .txt label files found in {LABELS_DIR}")
        print("Unzip your MakeSense annotations.zip into:", LABELS_DIR)
        sys.exit(1)

    print(f"Found {len(label_files)} label files")

    # Step 2: Validate and pair with images
    pairs = []
    bad_labels = 0
    missing_images = 0

    for lf in label_files:
        if not validate_label_file(lf):
            bad_labels += 1
            continue
        stem = lf.stem
        img = find_image(stem)
        if img is None:
            print(f"  [SKIP] {stem}: label exists but no matching image in {RAW_DIR}")
            missing_images += 1
            continue
        pairs.append((lf, img))

    print(f"Valid pairs: {len(pairs)}  (bad labels: {bad_labels}, missing images: {missing_images})")

    if len(pairs) < 5:
        print("ERROR: Need at least 5 image-label pairs. Add more images to data/raw/ and re-annotate.")
        sys.exit(1)

    # Step 3: Shuffle and split
    random.seed(args.seed)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - args.ratio))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"\nSplit: train={len(train_pairs)}, val={len(val_pairs)}")

    # Step 4: Copy files
    for dir_path in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        dir_path.mkdir(parents=True, exist_ok=True)

    for lf, img in train_pairs:
        shutil.copy2(lf, LABELS_TRAIN / lf.name)
        shutil.copy2(img, IMAGES_TRAIN / img.name)

    for lf, img in val_pairs:
        shutil.copy2(lf, LABELS_VAL / lf.name)
        shutil.copy2(img, IMAGES_VAL / img.name)

    print("Files copied to data/yolo/images/{train,val}/ and data/yolo/labels/{train,val}/")

    # Step 5: Generate dataset.yaml
    dataset_yaml = {
        "path": str(ROOT / "data" / "yolo"),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    yaml_path = ROOT / "data" / "yolo" / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nDataset YAML: {yaml_path}")
    print("Contents:")
    print(yaml.dump(dataset_yaml, default_flow_style=False, allow_unicode=True, sort_keys=False))
    print("\nReady for training: python scripts/03_train.py")


if __name__ == "__main__":
    main()
