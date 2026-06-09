import argparse
import sys
from pathlib import Path

import yaml


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduce_msdahnet.datasets.breastdm_2d_dataset import (  # noqa: E402
    collect_pairs,
    collect_split_pairs,
    convert_processed_17ch_to_breastdm_dirs,
    convert_processed_17ch_to_fixed_split_dirs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BreastDM 2D pre-contrast data for MSDAHNet reproduction.")
    parser.add_argument("--config", default="reproduce_msdahnet/configs/msdahnet_breastdm_5fold.yaml")
    parser.add_argument("--source", default=None, help="Source processed_17ch_dce directory.")
    parser.add_argument("--output", default=None, help="Output BreastDM directory containing pre_contrast_images and masks.")
    parser.add_argument("--layout", choices=["flat", "fixed"], default="fixed", help="flat for 5-fold CV, fixed for train/val/test.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source_root = resolve_path(args.source or cfg["data"]["source_processed_17ch_dir"])
    default_output = default_output_root(cfg, args.layout)
    output_root = Path(resolve_path(args.output)) if args.output else default_output

    if args.layout == "fixed":
        stats = convert_processed_17ch_to_fixed_split_dirs(source_root=source_root, output_root=str(output_root))
        paired = {}
        for split in ("train", "val", "test"):
            paired[split] = len(collect_split_pairs(str(output_root / split)))
    else:
        stats = convert_processed_17ch_to_breastdm_dirs(source_root=source_root, output_root=str(output_root))
        paired = {"all": len(collect_pairs(str(output_root / "pre_contrast_images"), str(output_root / "masks")))}

    print(f"source_root: {source_root}")
    print(f"output_root: {output_root}")
    print(f"converted_images: {stats['images']}")
    print(f"converted_masks: {stats['masks']}")
    print(f"skipped_non_3d_arrays: {stats['skipped']}")
    for split, count in paired.items():
        print(f"{split}_paired_samples: {count}")
    print("Next step:")
    if args.layout == "fixed":
        print("python reproduce_msdahnet/train_fixed_split.py --config reproduce_msdahnet/configs/msdahnet_breastdm_fixed_split.yaml")
    else:
        print(f"python reproduce_msdahnet/train_5fold.py --config {args.config}")


def default_output_root(cfg, layout: str) -> Path:
    if layout == "fixed":
        return Path(resolve_path(cfg.get("data", {}).get("processed_fixed_root", "./reproduce_msdahnet/BreastDM_fixed")))
    image_dir = Path(resolve_path(cfg["data"]["image_dir"]))
    return image_dir.parent


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


def load_config(path: str):
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
