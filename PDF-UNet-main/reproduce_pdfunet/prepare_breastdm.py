import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduce_pdfunet.datasets.breastdm_2d_dataset import collect_split_pairs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BreastDM data folders for PDF-UNet adaptation.")
    parser.add_argument("--source", default="../processed_17ch_dce", help="Source directory with train/val/test data and GT folders.")
    parser.add_argument("--output", default="./processed_breastdm", help="Output directory used by default configs.")
    parser.add_argument("--input-mode", default="single_channel_pre", choices=["single_channel_pre", "single_channel_sub", "multi_phase"])
    parser.add_argument("--sub-channel-index", type=int, default=9)
    args = parser.parse_args()

    source = Path(resolve_path(args.source))
    output = Path(resolve_path(args.output))
    stats = {"images": 0, "masks": 0, "missing_masks": 0}
    for split in ("train", "val", "test"):
        data_dir = source / split / "data"
        gt_dir = source / split / "GT"
        out_data = output / split / "data"
        out_gt = output / split / "GT"
        out_data.mkdir(parents=True, exist_ok=True)
        out_gt.mkdir(parents=True, exist_ok=True)
        if not data_dir.exists() or not gt_dir.exists():
            continue
        for image_path in sorted(data_dir.iterdir()):
            if image_path.suffix.lower() not in {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                continue
            mask_path = find_by_stem(gt_dir, image_path.stem)
            if args.input_mode == "multi_phase" and image_path.suffix.lower() == ".npy":
                shutil.copyfile(image_path, out_data / image_path.name)
                out_name = image_path.with_suffix(".npy").name
            else:
                image = load_single_channel(image_path, args.input_mode, args.sub_channel_index)
                image = minmax_uint8(image)
                out_name = image_path.with_suffix(".png").name
                cv2.imwrite(str(out_data / out_name), image)
            stats["images"] += 1
            if mask_path is None:
                stats["missing_masks"] += 1
                continue
            mask = load_mask(mask_path)
            if out_name.endswith(".npy"):
                np.save(str(out_gt / out_name), mask.astype(np.float32))
            else:
                cv2.imwrite(str(out_gt / out_name), (mask * 255).astype(np.uint8))
            stats["masks"] += 1

    for split in ("train", "val", "test"):
        split_path = output / split
        if split_path.exists():
            pairs = collect_split_pairs(str(split_path))
            print(f"{split}: paired_samples={len(pairs)}")
    print(f"source: {source}")
    print(f"output: {output}")
    print(f"converted_images: {stats['images']}")
    print(f"converted_masks: {stats['masks']}")
    print(f"missing_masks: {stats['missing_masks']}")


def load_single_channel(path: Path, input_mode: str, sub_channel_index: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
        if arr.ndim == 4:
            arr = arr[arr.shape[0] // 2]
        if arr.ndim == 3 and arr.shape[-1] <= 64:
            idx = 0 if input_mode == "single_channel_pre" else min(sub_channel_index, arr.shape[-1] - 1)
            return np.asarray(arr[:, :, idx], dtype=np.float32)
        if arr.ndim == 3:
            idx = 0 if input_mode == "single_channel_pre" else min(sub_channel_index, arr.shape[0] - 1)
            return np.asarray(arr[idx], dtype=np.float32)
        return np.asarray(arr, dtype=np.float32)
    from PIL import Image

    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def minmax_uint8(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + eps)
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(str(path))
        if mask.ndim == 4:
            mask = mask[mask.shape[0] // 2]
        if mask.ndim == 3:
            mask = mask[:, :, 0] if mask.shape[-1] <= 64 else mask[mask.shape[0] // 2]
    else:
        from PIL import Image

        mask = np.array(Image.open(path).convert("L"))
    return (mask > 0).astype(np.float32)



def find_by_stem(root: Path, stem: str):
    for ext in (".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


if __name__ == "__main__":
    main()
