#!/usr/bin/env python3
"""
05_deploy.py — Deploy trained models for USB transfer to Mac main project

Usage:
    python scripts/05_deploy.py                # Deploy all classes
    python scripts/05_deploy.py --class transformer  # Single class
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"

CLASS_NAMES = [
    "transformer",
    "switchgear",
    "cable",
    "busbar",
    "insulator",
]

TARGET_MAC_PATH = '"/Users/mba/claude code/detect/backend/models/weights/"'


def deploy_model(model_type: str) -> bool:
    """Copy best.pt → models/{model_type}.pt"""
    best_pt = RUNS_DIR / model_type / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"  [SKIP] {model_type}: no best.pt found at {best_pt}")
        return False

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / f"{model_type}.pt"

    shutil.copy2(best_pt, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  ✅ {model_type:15} → {dest}  ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Deploy trained models for USB transfer")
    parser.add_argument("--class", dest="single_class", type=str, default=None,
                        choices=CLASS_NAMES, help="Deploy a single class")
    args = parser.parse_args()

    if args.single_class:
        deploy_model(args.single_class)
    else:
        print("Deploying all models...\n")
        deployed = 0
        for cls_name in CLASS_NAMES:
            if deploy_model(cls_name):
                deployed += 1

        print(f"\n{deployed}/{len(CLASS_NAMES)} models deployed to: {MODELS_DIR}")

    if not any(MODELS_DIR.glob("*.pt")):
        print("\nNo models to deploy. Train first: python scripts/03_train.py")
        sys.exit(1)

    print(f"""
{'='*60}
 USB TRANSFER INSTRUCTIONS
{'='*60}

1. Copy the .pt files to a USB drive:
   {MODELS_DIR}
   └── *.pt  ← copy these 5 files

2. On your Mac, paste them to:
   {TARGET_MAC_PATH}

3. Restart the backend:
   cd {TARGET_MAC_PATH}
   python3 -m uvicorn main:app --port 8000

4. Create a project with the matching model_type
   → Auto-detection will use your trained model!
{'='*60}
""")


if __name__ == "__main__":
    main()
