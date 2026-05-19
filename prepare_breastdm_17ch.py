import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTENSIONS


def read_gray(path: Path, size: int):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


def read_label(path: Path, size: int):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return image


def find_label(label_root: Path, slice_stem: str):
    candidates = [
        label_root / f"{slice_stem}.png",
        label_root / f"{slice_stem}.jpg",
        label_root / f"{slice_stem}.jpeg",
        label_root / f"{slice_stem}.bmp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not label_root.exists():
        return None
    for path in label_root.iterdir():
        if path.stem in {slice_stem, f"{slice_stem}_mask"} and is_image_file(path):
            return path
    return None


def read_or_zero(path: Path, size: int):
    if not path.exists():
        return np.zeros((size, size), dtype=np.uint8)
    image = read_gray(path, size)
    if image is None:
        return np.zeros((size, size), dtype=np.uint8)
    return image


def process_patient(img_patient_dir: Path, lbl_patient_dir: Path, out_data_dir: Path, out_gt_dir: Path, size: int, label_phase: str):
    patient_id = img_patient_dir.name
    pre_dir = img_patient_dir / "VIBRANT"
    label_dir = lbl_patient_dir / label_phase
    if not pre_dir.exists() or not label_dir.exists():
        return

    post_dirs = [img_patient_dir / f"VIBRANT+C{i}" for i in range(1, 9)]
    sub_dirs = [img_patient_dir / f"SUB{i}" for i in range(1, 9)]
    slices = sorted([p for p in pre_dir.iterdir() if is_image_file(p)])

    for slice_path in slices:
        pre = read_gray(slice_path, size)
        if pre is None:
            continue
        channels = [pre]
        channels.extend(read_or_zero(d / slice_path.name, size) for d in post_dirs)
        channels.extend(read_or_zero(d / slice_path.name, size) for d in sub_dirs)
        if len(channels) != 17:
            continue

        label_path = find_label(label_dir, slice_path.stem)
        if label_path is None:
            continue
        label = read_label(label_path, size)
        if label is None:
            continue

        out_name = f"{patient_id}_{slice_path.stem}"
        np.save(str(out_data_dir / f"{out_name}.npy"), np.stack(channels, axis=-1))
        cv2.imwrite(str(out_gt_dir / f"{out_name}.png"), label)


def main():
    parser = argparse.ArgumentParser(description="Build 17-channel BreastDM DCE samples.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root containing train/val/test folders.")
    parser.add_argument("--output_root", type=str, default="./processed_17ch_dce")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--label_phase", type=str, default="VIBRANT", help="Label folder to use, e.g. VIBRANT or SUB2.")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    for split in ["train", "val", "test"]:
        split_img_dir = root / split / "images"
        split_lbl_dir = root / split / "labels"
        if not split_img_dir.exists() or not split_lbl_dir.exists():
            print(f"[Info] skip {split}: missing images or labels directory")
            continue

        out_data_dir = output_root / split / "data"
        out_gt_dir = output_root / split / "GT"
        out_data_dir.mkdir(parents=True, exist_ok=True)
        out_gt_dir.mkdir(parents=True, exist_ok=True)

        patients = [p for p in split_img_dir.iterdir() if p.is_dir()]
        for patient_dir in tqdm(patients, desc=f"Processing {split}"):
            process_patient(patient_dir, split_lbl_dir / patient_dir.name, out_data_dir, out_gt_dir, args.size, args.label_phase)

    print(f"Saved 17-channel samples to {output_root}")
    print("Channel layout: 0=VIBRANT, 1-8=VIBRANT+C1..C8, 9-16=SUB1..SUB8")


if __name__ == "__main__":
    main()
