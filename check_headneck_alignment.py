from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def robust_uint8(plane: np.ndarray) -> np.ndarray:
    plane = np.nan_to_num(plane.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(plane, [1.0, 99.5])
    if hi <= lo:
        lo, hi = float(plane.min()), float(plane.max())
    if hi <= lo:
        return np.zeros_like(plane, dtype=np.uint8)
    return (np.clip((plane - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)


def overlay(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    gray = robust_uint8(image)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    mask_bool = mask.astype(bool)
    rgb[mask_bool] = 0.45 * rgb[mask_bool] + 0.55 * np.array([255, 40, 40], dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def bbox(mask: np.ndarray) -> str:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return ""
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return f"z[{lo[0]},{hi[0]}], y[{lo[1]},{hi[1]}], x[{lo[2]},{hi[2]}]"


def load_case(volume_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(volume_path) as item:
        image = np.asarray(item["image"], dtype=np.float32)
        mask = np.asarray(item["mask"], dtype=np.float32)
    if mask.ndim == 4:
        mask = mask[0]
    return image, mask


def inspect_case(volume_path: Path, output_dir: Path, max_overlays: int) -> dict:
    image, mask = load_case(volume_path)
    positive_slices = np.flatnonzero(mask.reshape(mask.shape[0], -1).sum(axis=1) > 0)
    row = {
        "case_id": volume_path.stem,
        "shape": "x".join(str(v) for v in image.shape),
        "mask_voxels": int(mask.sum()),
        "mask_fraction": float(mask.mean()),
        "positive_slices": ";".join(str(int(v)) for v in positive_slices),
        "bbox": bbox(mask),
    }
    if max_overlays <= 0 or positive_slices.size == 0:
        return row
    case_dir = output_dir / volume_path.stem
    case_dir.mkdir(parents=True, exist_ok=True)
    scores = [(int(z), int(mask[int(z)].sum())) for z in positive_slices]
    for z, _ in sorted(scores, key=lambda item: item[1], reverse=True)[:max_overlays]:
        base = image[0, z]
        late = image[-1, z]
        diff = late - base
        panels = [overlay(base, mask[z]), overlay(late, mask[z]), overlay(diff, mask[z])]
        canvas = Image.new("RGB", (panels[0].width * len(panels), panels[0].height))
        for idx, panel in enumerate(panels):
            canvas.paste(panel, (idx * panel.width, 0))
        canvas.save(case_dir / f"z{z:03d}_c0_late_diff_overlay.png")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check processed HeadNeckDCE image/mask alignment.")
    parser.add_argument("--processed-3d", default="processed_headneck_dce_56ch_3d")
    parser.add_argument("--output-dir", default="headneck_alignment_qc")
    parser.add_argument("--max-overlays", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.processed_3d)
    output_dir = Path(args.output_dir)
    rows = []
    for split in ("train", "val", "test"):
        for path in sorted((root / split / "volumes").glob("*.npz")):
            row = inspect_case(path, output_dir / split, args.max_overlays)
            row["split"] = split
            rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["case_id", "split", "shape", "mask_voxels", "mask_fraction", "positive_slices", "bbox"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"Checked {len(rows)} cases. Summary: {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

