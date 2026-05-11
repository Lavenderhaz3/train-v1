#!/usr/bin/env python3
"""
02_prepare_dataset.py — MakeSense.ai YOLO export → train/val split for YOLOv8

Two annotation workflows:
    --source visible   (recommended)  Annotate visible-light images → train on FLIR display images
    --source display   (fallback)     Annotate FLIR Iron-colormap display → train on same

Usage:
    python scripts/02_prepare_dataset.py --source visible
    python scripts/02_prepare_dataset.py --source display [--ratio 0.3] [--seed 42]

Workflow A (visible, recommended):
    1. python scripts/00_extract_visible.py          ← extract visible images from FLIR
    2. Upload data/visible/ to MakeSense, annotate
    3. Export → YOLO → unzip to data/yolo/labels/
    4. python scripts/02_prepare_dataset.py --source visible

Workflow B (display, direct):
    1. Upload data/raw/ to MakeSense, annotate
    2. Export → YOLO → unzip to data/yolo/labels/
    3. python scripts/02_prepare_dataset.py --source display

Output (both modes):
    data/yolo/images/train/   FLIR display JPEGs (70%)
    data/yolo/images/val/     FLIR display JPEGs (30%)
    data/yolo/labels/train/   YOLO .txt labels (70%)
    data/yolo/labels/val/     YOLO .txt labels (30%)
    data/yolo/dataset.yaml    Training config
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
LABELS_SRC = ROOT / "data" / "yolo" / "labels"       # unzipped MakeSense annotations
RAW_DIR = ROOT / "data" / "raw"                       # FLIR display JPEGs
VISIBLE_DIR = ROOT / "data" / "visible"               # extracted visible images
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
    """Check YOLO format: class_id cx cy w h (all normalized 0~1)."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"  [WARN] {path.name}: invalid — expected 5 values, got '{line}'")
                return False
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                print(f"  [WARN] {path.name}: non-numeric in '{line}'")
                return False
            if any(v < 0 for v in vals[1:]):
                print(f"  [WARN] {path.name}: negative coords in '{line}'")
    return True


def find_label_files() -> list[Path]:
    """Find all .txt label files (flat or nested from MakeSense export)."""
    txts = sorted(LABELS_SRC.glob("*.txt"))
    if not txts:
        txts = sorted(LABELS_SRC.rglob("*.txt"))
    return txts


def find_flir_image(label_stem: str) -> Path | None:
    """Match a label to its FLIR display JPEG in data/raw/."""
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        path = RAW_DIR / f"{label_stem}{ext}"
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare YOLO dataset from MakeSense annotations"
    )
    parser.add_argument(
        "--source", choices=["visible", "display"], default="visible",
        help="Annotation source: 'visible' (recommended) or 'display' (direct)"
    )
    parser.add_argument("--ratio", type=float, default=0.3,
                        help="Validation split ratio (default: 0.3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # ── Step 1: Validation checks ──────────────────────────────────
    using_visible = args.source == "visible"

    if using_visible and not VISIBLE_DIR.exists():
        print(f"ERROR: {VISIBLE_DIR} not found.")
        print("Run first: python scripts/00_extract_visible.py")
        sys.exit(1)

    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.jpg")):
        print(f"ERROR: No FLIR JPEGs in {RAW_DIR}")
        print("Put your FLIR images in data/raw/ first.")
        sys.exit(1)

    # ── Step 2: Find labels ────────────────────────────────────────
    label_files = find_label_files()
    if not label_files:
        print(f"ERROR: No .txt label files in {LABELS_SRC}")
        print("Unzip your MakeSense annotations.zip into:", LABELS_SRC)
        sys.exit(1)

    print(f"Mode: --source {args.source}")
    print(f"Found {len(label_files)} label files\n")

    # ── Step 3: Validate and pair labels → FLIR display images ─────
    pairs = []
    bad = 0
    missing = 0

    for lf in label_files:
        if not validate_label_file(lf):
            bad += 1
            continue

        stem = lf.stem
        flir_img = find_flir_image(stem)
        if flir_img is None:
            print(f"  [SKIP] {stem}: label exists but no FLIR JPEG in {RAW_DIR}")
            missing += 1
            continue

        # — Coordinate remapping for visible-source annotations —
        # YOLO uses normalized coords (0-1). Since visible and FLIR display
        # share the same field of view (MSX alignment), normalized coords
        # are identical. No numerical remapping needed.
        # If FOV mismatch is discovered, add per-image scale/offset from
        # data/visible/mapping.json here.

        pairs.append((lf, flir_img))

    print(f"Valid pairs: {len(pairs)}  (bad: {bad}, missing FLIR image: {missing})")

    if len(pairs) < 5:
        print("ERROR: Need at least 5 image-label pairs.")
        sys.exit(1)

    # ── Step 4: Shuffle and split train/val ────────────────────────
    random.seed(args.seed)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - args.ratio))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"\nSplit: train={len(train_pairs)}, val={len(val_pairs)}")

    # ── Step 5: Copy FLIR display images + labels ──────────────────
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)

    for lf, img in train_pairs:
        shutil.copy2(img, IMAGES_TRAIN / img.name)     # FLIR display JPEG
        shutil.copy2(lf, LABELS_TRAIN / lf.name)       # YOLO label (same norm coords)

    for lf, img in val_pairs:
        shutil.copy2(img, IMAGES_VAL / img.name)
        shutil.copy2(lf, LABELS_VAL / lf.name)

    print("Copied images + labels to data/yolo/{images,labels}/{train,val}/")

    # ── Step 6: Generate dataset.yaml ──────────────────────────────
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

    print(f"\nDataset YAML → {yaml_path}")
    print(yaml.dump(dataset_yaml, default_flow_style=False, allow_unicode=True))
    print("\nReady for training: python scripts/03_train.py")


if __name__ == "__main__":
    main()
