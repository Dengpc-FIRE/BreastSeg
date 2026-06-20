from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_SCRIPTS = {
    "kpta": "train/train_kpta_mama_nact.py",
    "attention_gated": "ContrastModel/Attention-Gated-Networks-master/train_mama_nact.py",
    "deeplabv3plus": "ContrastModel/DeepLabV3Plus-Pytorch-master/train_mama_nact.py",
    "emcad": "ContrastModel/EMCAD-main/train_mama_nact.py",
    "hcrt": "ContrastModel/HCRT-main/train_mama_nact.py",
    "mobile_uvit": "ContrastModel/Mobile-U-ViT-main/train_mama_nact.py",
    "msdahnet": "ContrastModel/Multi-Scale-Dual-Attention-Hybrid-Convolution-Network-main/train_mama_nact.py",
    "nnunet": "ContrastModel/nnUNet-master/train_mama_nact.py",
    "pdpnet": "ContrastModel/PDPNet-main/train_mama_nact.py",
    "plhn": "ContrastModel/PLHN-main/train_mama_nact.py",
    "pytorch_unet": "ContrastModel/Pytorch-UNet-master/train_mama_nact.py",
    "transunet": "ContrastModel/TransUNet-main/train_mama_nact.py",
    "unetplusplus": "ContrastModel/UNetPlusPlus-master/train_mama_nact.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAMA-NACT training for selected models.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SCRIPTS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extra_args = args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args
    failures: list[tuple[str, int]] = []
    for model in args.models:
        if model not in MODEL_SCRIPTS:
            raise ValueError(f"Unknown model {model!r}; expected one of {sorted(MODEL_SCRIPTS)}")
        cmd = [sys.executable, "-u", str(PROJECT_ROOT / MODEL_SCRIPTS[model]), *extra_args]
        print(" ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
            if result.returncode != 0:
                failures.append((model, result.returncode))
                print(f"[failed] {model} exited with code {result.returncode}; continuing.")
                if args.stop_on_error:
                    break
    if failures:
        print("Failed models:")
        for model, code in failures:
            print(f"  {model}: exit_code={code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

