import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_NAMES = [
    "predict_masks.py",
    "slice_attention.py",
    "phase_attention.py",
    "kinetic_maps.py",
    "uncertainty_boundary.py",
]


def main():
    parser = argparse.ArgumentParser(description="Run all visualization scripts.")
    parser.add_argument("--config", default="configs/kpta_25d_net.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--split_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    base_cmd = [
        "--config", args.config,
        "--split", args.split,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--max_samples", str(args.max_samples),
        "--threshold", str(args.threshold),
        "--device", args.device,
    ]
    if args.checkpoint:
        base_cmd.extend(["--checkpoint", args.checkpoint])
    if args.split_path:
        base_cmd.extend(["--split_path", args.split_path])
    if args.output_dir:
        base_cmd.extend(["--output_dir", args.output_dir])

    for script in SCRIPT_NAMES:
        cmd = [sys.executable, str(root / script), *base_cmd]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

