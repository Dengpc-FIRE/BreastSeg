from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ContrastModel.dataset.config import PHASE_NAMES  # noqa: E402
from experiments.generalization.mama_mia import (  # noqa: E402
    MamaMiaCase,
    discover_cases,
    load_case_17ch,
    neighbor_indices,
    selected_cohorts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the MAMA-MIA NACT cohort for training 2D, KPTA-2.5D, and 3D models."
    )
    parser.add_argument("--mama-root", default=".", help="Downloaded MAMA-MIA root directory.")
    parser.add_argument("--cohorts", nargs="+", default=["NACT"], help="Usually NACT for this benchmark.")
    parser.add_argument("--output-2d", default="processed_mama_nact_17ch")
    parser.add_argument("--output-25d", default="processed_mama_nact_25d")
    parser.add_argument("--output-3d", default="processed_mama_nact_3d")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-slices", type=int, default=3)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--slice-policy",
        choices=["positive", "positive_margin", "all"],
        default="positive",
        help="2D/2.5D sample selection. 3D output always keeps full volumes.",
    )
    parser.add_argument("--slice-margin", type=int, default=1)
    parser.add_argument("--source-scale", default="per_slice_shared_uint8")
    parser.add_argument("--subtraction-mode", default="raw_positive_per_slice_uint8")
    parser.add_argument("--depth-axis", type=int, default=2)
    parser.add_argument("--skip-2d", action="store_true")
    parser.add_argument("--skip-25d", action="store_true")
    parser.add_argument("--skip-3d", action="store_true")
    return parser.parse_args()


def _resolve_out(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _split_cases(cases: Sequence[MamaMiaCase], train_ratio: float, val_ratio: float, seed: int):
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = min(max(n_train, 1), max(n_total - 2, 1)) if n_total >= 3 else max(n_total - 1, 1)
    n_val = min(max(n_val, 1), max(n_total - n_train - 1, 0)) if n_total >= 3 else 0
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _selected_slices(mask: np.ndarray, policy: str, margin: int) -> list[int]:
    depth = mask.shape[0]
    if policy == "all":
        return list(range(depth))
    positive = np.flatnonzero(mask.reshape(depth, -1).any(axis=1))
    if positive.size == 0:
        return []
    if policy == "positive":
        return [int(z) for z in positive]
    start = max(0, int(positive.min()) - int(margin))
    end = min(depth - 1, int(positive.max()) + int(margin))
    return list(range(start, end + 1))


def _save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)


def _write_2d_case(image: np.ndarray, mask: np.ndarray, split_root: Path, case_id: str, slices: Iterable[int]) -> int:
    count = 0
    data_dir = split_root / "data"
    gt_dir = split_root / "GT"
    data_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    for z in slices:
        stem = f"{case_id}_z{int(z):04d}"
        sample = np.transpose(image[:, int(z)], (1, 2, 0)).astype(np.float32)
        np.save(data_dir / f"{stem}.npy", sample)
        _save_mask_png(mask[int(z)], gt_dir / f"{stem}.png")
        count += 1
    return count


def _write_25d_case(
    image: np.ndarray,
    mask: np.ndarray,
    split_root: Path,
    case_id: str,
    slices: Iterable[int],
    num_slices: int,
) -> int:
    count = 0
    data_dir = split_root / "data"
    gt_dir = split_root / "GT"
    data_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    depth = image.shape[1]
    for z in slices:
        stem = f"{case_id}_z{int(z):04d}"
        idxs = neighbor_indices(int(z), depth, num_slices)
        sample = np.stack([image[:, idx] for idx in idxs], axis=0).astype(np.float32)
        np.save(data_dir / f"{stem}.npy", sample)
        _save_mask_png(mask[int(z)], gt_dir / f"{stem}.png")
        count += 1
    return count


def _write_3d_case(image: np.ndarray, mask: np.ndarray, split_root: Path, case_id: str) -> None:
    image_case_dir = split_root / "images" / case_id
    label_dir = split_root / "labels" / case_id / "GT"
    label_dir.mkdir(parents=True, exist_ok=True)
    for channel, phase_name in enumerate(PHASE_NAMES):
        phase_dir = image_case_dir / phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        for z in range(image.shape[1]):
            np.save(phase_dir / f"z{z:04d}.npy", image[channel, z].astype(np.float32))
    for z in range(mask.shape[0]):
        np.save(label_dir / f"z{z:04d}.npy", (mask[z] > 0).astype(np.float32))


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "case_id", "cohort", "depth", "positive_slices", "exported_slices"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    cohorts = selected_cohorts(args.cohorts)
    cases = discover_cases(args.mama_root, cohorts, mask_source="expert", require_mask=True)
    cases = [case for case in cases if case.cohort == "NACT"] if "NACT" in cohorts else cases
    if not cases:
        raise FileNotFoundError(f"No expert-labeled MAMA-MIA cases found for cohorts={cohorts} under {args.mama_root}")

    splits = _split_cases(cases, args.train_ratio, args.val_ratio, args.seed)
    out_2d = _resolve_out(args.output_2d)
    out_25d = _resolve_out(args.output_25d)
    out_3d = _resolve_out(args.output_3d)
    manifest_rows: list[dict] = []

    split_json = {
        split: [case.case_id for case in split_cases]
        for split, split_cases in splits.items()
    }
    for out_root in [out for out, skip in [(out_2d, args.skip_2d), (out_25d, args.skip_25d), (out_3d, args.skip_3d)] if not skip]:
        out_root.mkdir(parents=True, exist_ok=True)
        with (out_root / "splits.json").open("w", encoding="utf-8") as handle:
            json.dump(split_json, handle, indent=2)
        with (out_root / "preprocess_args.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2)

    for split, split_cases in splits.items():
        for case in tqdm(split_cases, desc=f"Preparing {split}"):
            image, mask = load_case_17ch(
                case,
                image_size=(args.image_size, args.image_size),
                normalize="none",
                source_scale=args.source_scale,
                subtraction_mode=args.subtraction_mode,
                depth_axis=args.depth_axis,
            )
            slices = _selected_slices(mask, args.slice_policy, args.slice_margin)
            positive_slices = int(mask.reshape(mask.shape[0], -1).any(axis=1).sum())
            exported = 0
            if not args.skip_2d:
                exported = _write_2d_case(image, mask, out_2d / split, case.case_id, slices)
            if not args.skip_25d:
                exported = _write_25d_case(image, mask, out_25d / split, case.case_id, slices, args.num_slices)
            if not args.skip_3d:
                _write_3d_case(image, mask, out_3d / split, case.case_id)
            manifest_rows.append(
                {
                    "split": split,
                    "case_id": case.case_id,
                    "cohort": case.cohort,
                    "depth": int(mask.shape[0]),
                    "positive_slices": positive_slices,
                    "exported_slices": exported,
                }
            )

    for out_root, skip in [(out_2d, args.skip_2d), (out_25d, args.skip_25d), (out_3d, args.skip_3d)]:
        if not skip:
            _write_manifest(out_root / "manifest.csv", manifest_rows)

    print("Prepared MAMA-MIA NACT training data.")
    print(f"2D:  {out_2d if not args.skip_2d else 'skipped'}")
    print(f"25D: {out_25d if not args.skip_25d else 'skipped'}")
    print(f"3D:  {out_3d if not args.skip_3d else 'skipped'}")
    print(json.dumps({k: len(v) for k, v in split_json.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
