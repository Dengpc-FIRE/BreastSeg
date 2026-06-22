from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose CC-Tumor-Heterogeneity 3D raw/cache data for HCRT.")
    parser.add_argument("--raw-root", default="processed_cc_tumor_heterogeneity_17ch_3d_raw")
    parser.add_argument("--cache-root", default="ContrastModel/dataset/processed_3d_cc_tumor_heterogeneity_17ch/hcrt")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def image_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(path.glob(f"*{ext}"))
    return sorted(files)


def read_slice(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.squeeze(np.load(path)).astype(np.float32)
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def count_stack(path: Path) -> tuple[int, int, tuple[int, ...] | None]:
    files = image_files(path)
    if not files:
        return 0, 0, None
    if len(files) == 1 and files[0].suffix.lower() == ".npy":
        arr = np.load(files[0])
        arr = np.squeeze(arr)
        return int(arr.shape[0] if arr.ndim >= 3 else 1), int((arr > 0).sum()), tuple(arr.shape)
    voxels = 0
    shape = None
    for file in files:
        arr = read_slice(file)
        voxels += int((arr > 0).sum())
        shape = tuple(arr.shape)
    return len(files), voxels, shape


def find_label_dir(label_patient_dir: Path) -> Path | None:
    for name in ["GT", "label", "labels", "mask", "tumor"]:
        candidate = label_patient_dir / name
        if candidate.exists():
            return candidate
    if label_patient_dir.exists():
        files = image_files(label_patient_dir)
        if files:
            return label_patient_dir
        children = sorted([child for child in label_patient_dir.iterdir() if child.is_dir() or child.suffix.lower() == ".npy"])
        return children[0] if children else None
    return None


def cache_info(cache_path: Path) -> tuple[str, str, str]:
    if not cache_path.exists():
        return "missing", "", ""
    try:
        with np.load(cache_path) as cached:
            image = np.asarray(cached["image"])
            mask = np.asarray(cached["mask"])
        return str(tuple(image.shape)), str(tuple(mask.shape)), str(int((mask > 0).sum()))
    except Exception as exc:
        return "bad", type(exc).__name__, str(exc)


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root)
    cache_root = Path(args.cache_root)
    headers = [
        "split",
        "patient",
        "phases",
        "raw_slices",
        "raw_mask_voxels",
        "raw_shape",
        "cache_image_shape",
        "cache_mask_shape",
        "cache_mask_voxels",
        "status",
    ]
    rows = []
    for split in args.splits:
        images_root = raw_root / split / "images"
        labels_root = raw_root / split / "labels"
        if not images_root.exists():
            rows.append([split, "<missing>", "", "", "", "", "", "", "", f"missing {images_root}"])
            continue
        for patient_dir in sorted([path for path in images_root.iterdir() if path.is_dir()]):
            patient = patient_dir.name
            phases = sorted([path for path in patient_dir.iterdir() if path.is_dir()])
            first_phase = phases[0] if phases else patient_dir
            raw_slices, _, raw_shape = count_stack(first_phase)
            label_dir = find_label_dir(labels_root / patient)
            label_slices, raw_mask_voxels, raw_mask_shape = count_stack(label_dir) if label_dir else (0, 0, None)
            cache_image_shape, cache_mask_shape, cache_mask_voxels = cache_info(cache_root / split / f"{patient}.npz")
            status = "ok"
            if raw_mask_voxels == 0:
                status = "empty_raw_mask"
            elif cache_mask_voxels not in ("", str(raw_mask_voxels)):
                status = "cache_raw_mask_mismatch"
            rows.append(
                [
                    split,
                    patient,
                    str(len(phases)),
                    str(raw_slices or label_slices),
                    str(raw_mask_voxels),
                    str(raw_mask_shape or raw_shape),
                    cache_image_shape,
                    cache_mask_shape,
                    cache_mask_voxels,
                    status,
                ]
            )

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
