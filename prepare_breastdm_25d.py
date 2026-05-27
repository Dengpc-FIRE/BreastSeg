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


def read_or_zero(path: Path, size: int):
    if not path.exists():
        return np.zeros((size, size), dtype=np.uint8)
    image = read_gray(path, size)
    if image is None:
        return np.zeros((size, size), dtype=np.uint8)
    return image


def find_label(label_root: Path, slice_stem: str):
    if not label_root.exists():
        return None
    candidates = [
        label_root / f"{slice_stem}.png",
        label_root / f"{slice_stem}.jpg",
        label_root / f"{slice_stem}.jpeg",
        label_root / f"{slice_stem}.bmp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for path in label_root.iterdir():
        if is_image_file(path) and path.stem in {slice_stem, f"{slice_stem}_mask"}:
            return path
    return None


def read_phase_stack(img_patient_dir: Path, slice_name: str, size: int):
    channels = []
    pre = read_gray(img_patient_dir / "VIBRANT" / slice_name, size)
    if pre is None:
        return None
    channels.append(pre)
    channels.extend(read_or_zero(img_patient_dir / f"VIBRANT+C{i}" / slice_name, size) for i in range(1, 9))
    channels.extend(read_or_zero(img_patient_dir / f"SUB{i}" / slice_name, size) for i in range(1, 9))
    return np.stack(channels, axis=0)  # [T,H,W]


def neighbor_indices(z: int, count: int, num_slices: int):
    half = num_slices // 2
    indices = []
    for offset in range(-half, half + 1):
        indices.append(min(max(z + offset, 0), count - 1))
    return indices


def process_patient(
    img_patient_dir: Path,
    lbl_patient_dir: Path,
    out_data_dir: Path,
    out_gt_dir: Path,
    size: int,
    label_phase: str,
    num_slices: int,
):
    pre_dir = img_patient_dir / "VIBRANT"
    label_dir = lbl_patient_dir / label_phase
    if not pre_dir.exists() or not label_dir.exists():
        return

    slices = sorted([p for p in pre_dir.iterdir() if is_image_file(p)])
    if not slices:
        return

    phase_cache = []
    label_cache = []
    for slice_path in slices:
        phase_stack = read_phase_stack(img_patient_dir, slice_path.name, size)
        label_path = find_label(label_dir, slice_path.stem)
        label = read_label(label_path, size) if label_path is not None else None
        phase_cache.append(phase_stack)
        label_cache.append(label)

    for z, slice_path in enumerate(slices):
        if phase_cache[z] is None or label_cache[z] is None:
            continue
        stacks = []
        for neighbor in neighbor_indices(z, len(slices), num_slices):
            stack = phase_cache[neighbor]
            if stack is None:
                stack = phase_cache[z]
            stacks.append(stack)
        x = np.stack(stacks, axis=0)  # [K,T,H,W]
        out_name = f"{img_patient_dir.name}_{slice_path.stem}"
        np.save(str(out_data_dir / f"{out_name}.npy"), x)
        cv2.imwrite(str(out_gt_dir / f"{out_name}.png"), label_cache[z])


def main():
    parser = argparse.ArgumentParser(description="Build 2.5D BreastDM DCE samples.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root containing train/val/test folders.")
    parser.add_argument("--output_root", type=str, default="./processed_25d_dce")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--label_phase", type=str, default="VIBRANT", help="Label folder to use, e.g. VIBRANT or SUB2.")
    parser.add_argument("--num_slices", type=int, default=3, help="Odd number of neighboring slices, default [z-1,z,z+1].")
    args = parser.parse_args()

    if args.num_slices < 1 or args.num_slices % 2 == 0:
        raise ValueError("--num_slices must be a positive odd number.")

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
            process_patient(
                patient_dir,
                split_lbl_dir / patient_dir.name,
                out_data_dir,
                out_gt_dir,
                args.size,
                args.label_phase,
                args.num_slices,
            )

    print(f"Saved 2.5D samples to {output_root}")
    print("Sample layout: [K,T,H,W], T: 0=VIBRANT, 1-8=VIBRANT+C1..C8, 9-16=SUB1..SUB8")


if __name__ == "__main__":
    main()
