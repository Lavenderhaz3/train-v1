#!/usr/bin/env python3
"""
03_train.py — YOLOv8 device detection training (5 models, independent)

Usage:
    python scripts/03_train.py [--model yolov8n] [--epochs 100] [--imgsz 640]

    # Train all 5 classes
    python scripts/03_train.py

    # Train a single class
    python scripts/03_train.py --class transformer --epochs 50
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = ROOT / "data" / "yolo" / "dataset.yaml"
RUNS_DIR = ROOT / "runs"

CLASS_NAMES = [
    "transformer",
    "switchgear",
    "cable",
    "busbar",
    "insulator",
]


def check_dataset() -> bool:
    """Verify dataset.yaml and image/label counts."""
    if not DATASET_YAML.exists():
        print(f"ERROR: {DATASET_YAML} not found.")
        print("Run: python scripts/02_prepare_dataset.py first.")
        return False

    import yaml
    with open(DATASET_YAML) as f:
        ds = yaml.safe_load(f)

    train_dir = ROOT / "data" / "yolo" / ds["train"]
    val_dir = ROOT / "data" / "yolo" / ds["val"]

    train_imgs = len(list(train_dir.glob("*.[jJpP][pPnN][gG]")) + list(train_dir.glob("*.[jJ][pP][eE][gG]")))
    val_imgs = len(list(val_dir.glob("*.[jJpP][pPnN][gG]")) + list(val_dir.glob("*.[jJ][pP][eE][gG]")))

    print(f"Dataset: {train_imgs} train + {val_imgs} val images")
    if train_imgs < 5:
        print("ERROR: Need at least 5 training images per class.")
        return False
    return True


def train_model(model_type: str, base_model: str, epochs: int, imgsz: int) -> bool:
    """Train a single YOLOv8 model on images of one class.

    Strategy: filter dataset.yaml's nc=1 with only the target class.
    This works because each project trains for ONE device type.
    """
    import yaml
    import tempfile

    with open(DATASET_YAML) as f:
        ds = yaml.safe_load(f)

    # Create a class-specific dataset config (nc=1, single class)
    class_idx = CLASS_NAMES.index(model_type)
    class_yaml = {
        "path": ds["path"],
        "train": ds["train"],
        "val": ds["val"],
        "nc": 1,
        "names": [model_type],
    }

    tmp_yaml = ROOT / "data" / "yolo" / f"dataset_{model_type}.yaml"
    with open(tmp_yaml, "w") as f:
        yaml.dump(class_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"Training: {model_type}")
    print(f"Model: {base_model}  Epochs: {epochs}  Image size: {imgsz}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "ultralytics",
        "detect", "train",
        f"data={tmp_yaml}",
        f"model={base_model}.pt",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"project={RUNS_DIR}",
        f"name={model_type}",
        "exist_ok=true",
    ]

    try:
        result = subprocess.run(cmd, cwd=ROOT, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print(f"\nTraining {model_type} interrupted by user.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 device detection models")
    parser.add_argument("--model", default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                        help="YOLOv8 model size (default: yolov8n)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--class", dest="single_class", type=str, default=None,
                        choices=CLASS_NAMES,
                        help="Train only one specific class (default: all 5)")
    args = parser.parse_args()

    if not check_dataset():
        sys.exit(1)

    if args.single_class:
        ok = train_model(args.single_class, args.model, args.epochs, args.imgsz)
        if ok:
            print(f"\n✅ Training complete: {args.single_class}")
        else:
            print(f"\n❌ Training failed: {args.single_class}")
            sys.exit(1)
    else:
        failed = []
        for cls_name in CLASS_NAMES:
            ok = train_model(cls_name, args.model, args.epochs, args.imgsz)
            if not ok:
                failed.append(cls_name)
                print(f"  ⚠️  {cls_name} failed, continuing with next...")

        print(f"\n{'='*60}")
        if failed:
            print(f"❌ Failed: {', '.join(failed)}")
        print(f"✅ Success: {', '.join(c for c in CLASS_NAMES if c not in failed)}")
        print(f"\nBest weights saved under: {RUNS_DIR}/<class>/weights/best.pt")
        print(f"Next: python scripts/04_evaluate.py")


if __name__ == "__main__":
    main()
