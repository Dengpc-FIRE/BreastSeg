from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIRS = {
    "attention_gated": "Attention-Gated-Networks-master",
    "deeplabv3plus": "DeepLabV3Plus-Pytorch-master",
    "emcad": "EMCAD-main",
    "hcrt": "HCRT-main",
    "mobile_uvit": "Mobile-U-ViT-main",
    "msdahnet": "Multi-Scale-Dual-Attention-Hybrid-Convolution-Network-main",
    "nnunet": "nnUNet-master",
    "pdpnet": "PDPNet-main",
    "plhn": "PLHN-main",
    "pytorch_unet": "Pytorch-UNet-master",
    "transunet": "TransUNet-main",
    "unetplusplus": "UNetPlusPlus-master",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run or print all MAMA-MIA generalization commands.")
    parser.add_argument("--mama-root", default=".")
    parser.add_argument("--cohorts", nargs="+", default=["DUKE", "ISPY1", "ISPY2", "NACT"])
    parser.add_argument("--checkpoint-name", default="checkpoints/best_model.pth")
    parser.add_argument("--kpta-checkpoint", default=None)
    parser.add_argument("--kpta-config", default="configs/kpta_25d_net.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only print commands.")
    parser.add_argument("--include-nnunet", action="store_true", help="Also print/run nnU-Net wrapper; it requires a separate adapter.")
    return parser.parse_args()


def command_for_contrast(model_key: str, model_dir_name: str, args) -> list[str]:
    model_dir = PROJECT_ROOT / "ContrastModel" / model_dir_name
    cmd = [
        sys.executable,
        str(model_dir / "generalize_mama_mia.py"),
        "--mama-root",
        args.mama_root,
        "--cohorts",
        *args.cohorts,
        "--checkpoint",
        str(model_dir / args.checkpoint_name),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    return cmd


def command_for_kpta(args) -> list[str] | None:
    if not args.kpta_checkpoint:
        return None
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "model" / "generalize_mama_mia.py"),
        "--mama-root",
        args.mama_root,
        "--cohorts",
        *args.cohorts,
        "--config",
        args.kpta_config,
        "--checkpoint",
        args.kpta_checkpoint,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    return cmd


def main():
    args = parse_args()
    commands = []
    for model_key, model_dir_name in MODEL_DIRS.items():
        if not args.include_nnunet and model_key == "nnunet":
            continue
        commands.append(command_for_contrast(model_key, model_dir_name, args))
    kpta_cmd = command_for_kpta(args)
    if kpta_cmd is not None:
        commands.append(kpta_cmd)

    for cmd in commands:
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
