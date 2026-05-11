#!/usr/bin/env python3
"""
04_evaluate.py — Evaluate trained YOLOv8 models

Usage:
    python scripts/04_evaluate.py              # Evaluate all 5 classes
    python scripts/04_evaluate.py --class transformer  # Single class
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

CLASS_NAMES = [
    "transformer",
    "switchgear",
    "cable",
    "busbar",
    "insulator",
]


def evaluate_model(model_type: str, imgsz: int = 640):
    """Run YOLOv8 validation on a trained model."""
    best_pt = RUNS_DIR / model_type / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"  [SKIP] {model_type}: no best.pt at {best_pt}")
        return None

    dataset_yaml = ROOT / "data" / "yolo" / f"dataset_{model_type}.yaml"

    print(f"\n--- Evaluating: {model_type} ---")

    cmd = [
        sys.executable, "-m", "ultralytics",
        "detect", "val",
        f"model={best_pt}",
        f"data={dataset_yaml}",
        f"imgsz={imgsz}",
        f"project={RUNS_DIR}",
        f"name={model_type}_eval",
        "exist_ok=true",
    ]

    try:
        result = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

        # Try to find mAP values from output
        for line in result.stdout.split("\n"):
            if "mAP50" in line and "all" in line.lower():
                print(f"  >>> {line.strip()}")

        return result.returncode == 0

    except KeyboardInterrupt:
        print(f"\nEvaluation {model_type} interrupted.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLOv8 models")
    parser.add_argument("--class", dest="single_class", type=str, default=None,
                        choices=CLASS_NAMES, help="Evaluate a single class")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    if args.single_class:
        ok = evaluate_model(args.single_class, args.imgsz)
        if ok is None:
            sys.exit(1)
    else:
        print("=" * 60)
        print("YOLOv8 Model Evaluation Report")
        print("=" * 60)

        results = {}
        for cls_name in CLASS_NAMES:
            results[cls_name] = evaluate_model(cls_name, args.imgsz)

        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for cls_name, ok in results.items():
            status = "✅ PASS" if ok else ("❌ FAIL" if ok is False else "⚠️  NO MODEL")
            print(f"  {cls_name:20} {status}")

        print("\nDeployable models (where best.pt exists):")
        for cls_name in CLASS_NAMES:
            best_pt = RUNS_DIR / cls_name / "weights" / "best.pt"
            if best_pt.exists():
                size_mb = best_pt.stat().st_size / (1024 * 1024)
                print(f"  {cls_name:20} → {best_pt}  ({size_mb:.1f} MB)")

        print("\nResults folder:", RUNS_DIR)
        print("Next: python scripts/05_deploy.py")


if __name__ == "__main__":
    main()
