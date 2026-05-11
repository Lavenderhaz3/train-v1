#!/usr/bin/env python3
"""
00_extract_visible.py — Extract visible-light images from FLIR radiometric JPEGs

FLIR cameras (T1040 etc.) embed a visible-light digital photo alongside
the thermal data. This script extracts those visible images for annotation
in MakeSense.ai — much easier to see equipment shapes than Iron colormap.

Usage:
    python scripts/00_extract_visible.py

Input:
    data/raw/  ← FLIR radiometric JPEGs (as from camera)

Output:
    data/visible/             ← visible-light JPEGs (same filenames, 1280×960 RGB)
    data/visible/mapping.json ← coordinate mapping: visible → FLIR display

Then upload data/visible/ images to https://www.makesense.ai for annotation.
"""

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
VISIBLE_DIR = ROOT / "data" / "visible"


def is_flir_radiometric(path: Path) -> bool:
    """Quick check: does this JPEG have FLIR Planck constants?"""
    try:
        result = subprocess.run(
            ["exiftool", "-PlanckR1", "-PlanckR2", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        return "PlanckR1" in result.stdout and "PlanckR2" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_display_dims(path: Path) -> tuple[int, int]:
    """Get the FLIR display JPEG dimensions."""
    with Image.open(path) as img:
        return img.size  # (width, height)


def extract_visible(flir_path: Path, out_path: Path) -> bool:
    """Extract the visible-light EmbeddedImage from a FLIR JPEG."""
    try:
        result = subprocess.run(
            ["exiftool", "-b", "-EmbeddedImage", str(flir_path)],
            capture_output=True, timeout=15,
        )
        if result.returncode != 0 or len(result.stdout) == 0:
            print(f"  [FAIL] {flir_path.name}: no EmbeddedImage found")
            return False

        out_path.write_bytes(result.stdout)

        # Verify it's a valid image
        with Image.open(out_path) as img:
            w, h = img.size
        print(f"  {flir_path.name} → visible {w}×{h}")
        return True

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {flir_path.name}")
        return False
    except Exception as e:
        print(f"  [ERROR] {flir_path.name}: {e}")
        return False


def main():
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} not found.")
        print("Put your FLIR JPEGs in data/raw/ first.")
        sys.exit(1)

    flir_files = sorted(
        p for p in RAW_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg")
    )

    if not flir_files:
        print(f"ERROR: No JPEG files in {RAW_DIR}")
        print("Put your FLIR radiometric JPEGs in data/raw/ first.")
        sys.exit(1)

    print(f"Found {len(flir_files)} JPEG files in data/raw/\n")

    # Create output directory
    VISIBLE_DIR.mkdir(parents=True, exist_ok=True)

    # Extract visible images and build mapping
    mapping = {}
    extracted = 0

    for flir_path in flir_files:
        visible_path = VISIBLE_DIR / flir_path.name
        if visible_path.exists():
            print(f"  [SKIP] {flir_path.name}: already extracted")
            # Still read metadata for mapping
            extracted += 1
        else:
            if not extract_visible(flir_path, visible_path):
                continue
            extracted += 1

        display_w, display_h = get_display_dims(flir_path)

        with Image.open(visible_path) as img:
            visible_w, visible_h = img.size

        mapping[flir_path.name] = {
            "display_w": display_w,
            "display_h": display_h,
            "visible_w": visible_w,
            "visible_h": visible_h,
            "scale_x": display_w / visible_w,
            "scale_y": display_h / visible_h,
        }

    if extracted == 0:
        print("\nERROR: No visible images extracted.")
        print("Make sure data/raw/ contains FLIR radiometric JPEGs (not regular JPEGs).")
        sys.exit(1)

    # Save mapping
    mapping_path = VISIBLE_DIR / "mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Extracted {extracted}/{len(flir_files)} visible images → {VISIBLE_DIR}")
    print(f"   Mapping saved → {mapping_path}")
    print(f"\nNext steps:")
    print(f"   1. Open https://www.makesense.ai")
    print(f"   2. Drag in images from: {VISIBLE_DIR}")
    print(f"   3. Import labels from: scripts/01_labels.txt")
    print(f"   4. Label equipment with bounding boxes")
    print(f"   5. Export → YOLO format → unzip to data/yolo/labels/")
    print(f"   6. Run: python scripts/02_prepare_dataset.py --source visible")


if __name__ == "__main__":
    main()
