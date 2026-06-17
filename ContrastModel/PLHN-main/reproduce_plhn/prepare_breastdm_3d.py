from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import SimpleITK as sitk


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def natural_key(value: str) -> List[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def resolve_path(path: Union[str, Path], base: Optional[Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if base is not None:
        candidate = base / p
        if candidate.exists() or not p.exists():
            return candidate
    return Path.cwd() / p


def collect_stems(phase_dir: Path) -> Dict[str, Path]:
    if not phase_dir.exists():
        return {}
    stems: Dict[str, Path] = {}
    for path in phase_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            stems[path.stem] = path
    return stems


def read_gray(path: Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.shape != (image_size, image_size):
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return image.astype(np.float32)


def read_mask(path: Path, image_size: int, threshold: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    if mask.shape != (image_size, image_size):
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return (mask > threshold).astype(np.uint8)


def write_volume(path: Path, volume: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(volume)
    sitk.WriteImage(image, str(path))


def convert_patient(
    image_patient_dir: Path,
    label_patient_dir: Path,
    output_patient_dir: Path,
    pre_phase: str,
    post_phase: str,
    label_phase: str,
    image_size: int,
    mask_threshold: int,
) -> Optional[Dict[str, object]]:
    pre_paths = collect_stems(image_patient_dir / pre_phase)
    post_paths = collect_stems(image_patient_dir / post_phase)
    label_paths = collect_stems(label_patient_dir / label_phase)
    common_stems = sorted(
        set(pre_paths).intersection(post_paths).intersection(label_paths),
        key=natural_key,
    )
    if not common_stems:
        return None

    pre_slices = []
    post_slices = []
    label_slices = []
    for stem in common_stems:
        pre_slices.append(read_gray(pre_paths[stem], image_size))
        post_slices.append(read_gray(post_paths[stem], image_size))
        label_slices.append(read_mask(label_paths[stem], image_size, mask_threshold))

    pre_volume = np.stack(pre_slices, axis=0).astype(np.float32)
    post_volume = np.stack(post_slices, axis=0).astype(np.float32)
    label_volume = np.stack(label_slices, axis=0).astype(np.uint8)

    output_patient_dir.mkdir(parents=True, exist_ok=True)
    write_volume(output_patient_dir / "P0.nii.gz", pre_volume)
    write_volume(output_patient_dir / "P1.nii.gz", post_volume)
    write_volume(output_patient_dir / "GT.nii.gz", label_volume)

    return {
        "patient": image_patient_dir.name,
        "num_slices": len(common_stems),
        "first_slice": common_stems[0],
        "last_slice": common_stems[-1],
        "foreground_voxels": int(label_volume.sum()),
    }


def convert_split(
    seg_root: Path,
    output_root: Path,
    split: str,
    fold: int,
    pre_phase: str,
    post_phase: str,
    label_phase: str,
    image_size: int,
    mask_threshold: int,
) -> Tuple[List[str], List[Dict[str, object]]]:
    image_root = seg_root / split / "images"
    label_root = seg_root / split / "labels"
    if not image_root.exists():
        raise FileNotFoundError(f"Missing split image root: {image_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"Missing split label root: {label_root}")

    entries: List[str] = []
    rows: List[Dict[str, object]] = []
    for image_patient_dir in sorted((p for p in image_root.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
        label_patient_dir = label_root / image_patient_dir.name
        output_patient_dir = output_root / split / image_patient_dir.name
        stats = convert_patient(
            image_patient_dir=image_patient_dir,
            label_patient_dir=label_patient_dir,
            output_patient_dir=output_patient_dir,
            pre_phase=pre_phase,
            post_phase=post_phase,
            label_phase=label_phase,
            image_size=image_size,
            mask_threshold=mask_threshold,
        )
        if stats is None:
            rows.append({"split": split, "patient": image_patient_dir.name, "status": "skipped"})
            continue
        entry = f"{split}/{image_patient_dir.name}"
        entries.append(entry)
        rows.append({"split": split, "status": "converted", "entry": entry, **stats})

    data_folder = output_root / "data_folder"
    data_folder.mkdir(parents=True, exist_ok=True)
    list_path = data_folder / f"{split}{fold}.txt"
    list_path.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
    return entries, rows


def convert_dataset(
    seg_root: Union[str, Path],
    output_root: Union[str, Path],
    fold: int = 1,
    pre_phase: str = "VIBRANT",
    post_phase: str = "VIBRANT+C8",
    label_phase: str = "VIBRANT",
    image_size: int = 256,
    mask_threshold: int = 0,
    splits: Sequence[str] = SPLITS,
) -> Dict[str, object]:
    seg_root = Path(seg_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    counts: Dict[str, int] = {}
    for split in splits:
        entries, rows = convert_split(
            seg_root=seg_root,
            output_root=output_root,
            split=split,
            fold=fold,
            pre_phase=pre_phase,
            post_phase=post_phase,
            label_phase=label_phase,
            image_size=image_size,
            mask_threshold=mask_threshold,
        )
        counts[split] = len(entries)
        manifest_rows.extend(rows)

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in manifest_rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "seg_root": str(seg_root),
        "output_root": str(output_root),
        "fold": int(fold),
        "pre_phase": pre_phase,
        "post_phase": post_phase,
        "label_phase": label_phase,
        "image_size": int(image_size),
        "mask_threshold": int(mask_threshold),
        "counts": counts,
        "manifest": str(manifest_path),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert BreastDM seg split into PLHN 3D NIfTI cases.")
    parser.add_argument("--seg_root", default="../seg", help="Root containing train/val/test images and labels.")
    parser.add_argument("--output_root", default="./reproduce_plhn/BreastDM_PLHN_3D")
    parser.add_argument("--fold", type=int, default=1, help="Suffix for data_folder/train{fold}.txt.")
    parser.add_argument("--pre_phase", default="VIBRANT")
    parser.add_argument("--post_phase", default="VIBRANT+C8")
    parser.add_argument("--label_phase", default="VIBRANT")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--mask_threshold", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plhn_root = Path(__file__).resolve().parents[1]
    summary = convert_dataset(
        seg_root=resolve_path(args.seg_root, plhn_root),
        output_root=resolve_path(args.output_root, plhn_root),
        fold=args.fold,
        pre_phase=args.pre_phase,
        post_phase=args.post_phase,
        label_phase=args.label_phase,
        image_size=args.image_size,
        mask_threshold=args.mask_threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
