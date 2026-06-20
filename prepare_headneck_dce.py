from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


ROI_PATTERNS = ("gtv", "gtvp", "primary gtv", "primary tumor", "primary tumour")
EXCLUDE_ROI_PATTERNS = ("parotid", "submandibular", "sublingual", "gland", "salivary")


@dataclass
class DCEVolume:
    case_id: str
    image: np.ndarray
    mask: np.ndarray
    spacing: tuple[float, float, float]
    affine: list[list[float]]
    temporal_positions: list[int]
    slice_positions: list[float]
    ordering: str
    roi_names: list[str]


def natural_key(path: Path) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name))


def require_pydicom():
    try:
        import pydicom
    except Exception as exc:
        raise RuntimeError(
            "HeadNeckDCE preprocessing requires pydicom. Install with: "
            "py -m pip install -r requirements-headneck-dce.txt"
        ) from exc
    return pydicom


def robust_window(values: np.ndarray, lower: float = 1.0, upper: float = 99.5) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    nonzero = values[np.abs(values) > 1e-6]
    stats = nonzero if nonzero.size >= 1024 else values
    lo, hi = np.percentile(stats, [lower, upper]).astype(np.float32)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(stats.min()), float(stats.max())
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def scale_to_uint8_like(volume: np.ndarray) -> np.ndarray:
    lo, hi = robust_window(volume.reshape(-1))
    scaled = (np.clip(volume.astype(np.float32), lo, hi) - lo) / (hi - lo + 1e-6)
    return (scaled * 255.0).astype(np.float32)


def resize_plane(plane: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    mode = Image.NEAREST if nearest else Image.BILINEAR
    return np.asarray(Image.fromarray(plane.astype(np.float32)).resize((size, size), mode), dtype=np.float32)


def resize_image_4d(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[-2:] == (size, size):
        return image.astype(np.float32, copy=False)
    out = np.empty((image.shape[0], image.shape[1], size, size), dtype=np.float32)
    for t in range(image.shape[0]):
        for z in range(image.shape[1]):
            out[t, z] = resize_plane(image[t, z], size, nearest=False)
    return out


def resize_mask_3d(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape[-2:] == (size, size):
        return mask.astype(np.float32, copy=False)
    out = np.empty((mask.shape[0], size, size), dtype=np.float32)
    for z in range(mask.shape[0]):
        out[z] = resize_plane(mask[z], size, nearest=True)
    return (out > 0).astype(np.float32)


def select_channels(image: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
    """Return [C,D,H,W] channels from source [T,D,H,W]."""
    if mode == "full56":
        return image.astype(np.float32), {"channel_mode": mode, "channels": int(image.shape[0])}
    if mode != "breastdm17":
        raise ValueError("--channel-mode must be full56 or breastdm17")
    t = image.shape[0]
    baseline = 0
    post_indices = np.linspace(1, max(t - 1, 1), num=8).round().astype(int).tolist()
    pre = image[baseline]
    posts = [image[idx] for idx in post_indices]
    subs = [np.clip(post - pre, 0.0, None) for post in posts]
    return np.stack([pre, *posts, *subs], axis=0).astype(np.float32), {
        "channel_mode": mode,
        "baseline_index": baseline,
        "post_indices": post_indices,
        "channels": 17,
    }


def affine_from_dicom(first_ds, slice_spacing: float) -> list[list[float]]:
    orientation = np.asarray([float(v) for v in first_ds.ImageOrientationPatient], dtype=np.float64)
    row_cos = orientation[:3]
    col_cos = orientation[3:]
    normal = np.cross(row_cos, col_cos)
    row_spacing, col_spacing = [float(v) for v in first_ds.PixelSpacing]
    origin = np.asarray([float(v) for v in first_ds.ImagePositionPatient], dtype=np.float64)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = row_cos * col_spacing
    affine[:3, 1] = col_cos * row_spacing
    affine[:3, 2] = normal * slice_spacing
    affine[:3, 3] = origin
    return affine.tolist()


def slice_geometry(first_ds, datasets: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, list[float]]:
    orientation = np.asarray([float(v) for v in first_ds.ImageOrientationPatient], dtype=np.float64)
    row_cos = orientation[:3]
    col_cos = orientation[3:]
    normal = np.cross(row_cos, col_cos)
    row_spacing, col_spacing = [float(v) for v in first_ds.PixelSpacing]
    positions = []
    for ds in datasets:
        ipp = np.asarray([float(v) for v in ds.ImagePositionPatient], dtype=np.float64)
        positions.append(float(np.dot(ipp, normal)))
    unique_positions = sorted(set(round(v, 4) for v in positions), reverse=True)
    if len(unique_positions) > 1:
        diffs = np.diff(sorted(unique_positions))
        slice_spacing = float(np.median(np.abs(diffs)))
    else:
        slice_spacing = float(getattr(first_ds, "SliceThickness", 1.0))
    return row_cos, col_cos, normal, row_spacing, col_spacing, slice_spacing, unique_positions


def infer_order(records: list[dict], depth: int) -> str:
    if all(rec["temporal"] is not None for rec in records):
        return "dicom_temporal_position"
    ordered = sorted(records, key=lambda r: (r["instance"], natural_key(r["path"])))
    if len(ordered) >= depth * 2:
        first = [rec["slice_index"] for rec in ordered[:depth]]
        second = [rec["slice_index"] for rec in ordered[depth : 2 * depth]]
        if len(set(first)) == depth and len(set(second)) == depth:
            return "time_major_by_instance"
    return "slice_major_or_inferred_by_instance"


def read_dce_series(dce_dir: Path) -> tuple[np.ndarray, list, tuple[float, float, float], list[list[float]], list[int], list[float], str]:
    pydicom = require_pydicom()
    dicom_paths = sorted(dce_dir.glob("*.dcm"), key=natural_key)
    if not dicom_paths:
        raise FileNotFoundError(f"No DICOM files found in {dce_dir}")
    datasets = [pydicom.dcmread(str(path), force=True) for path in dicom_paths]
    first = datasets[0]
    row_cos, col_cos, normal, row_spacing, col_spacing, slice_spacing, unique_positions = slice_geometry(first, datasets)
    depth = len(unique_positions)
    if depth <= 0:
        raise ValueError(f"Could not infer slice count for {dce_dir}")
    if len(datasets) % depth != 0:
        raise ValueError(f"DICOM count {len(datasets)} is not divisible by inferred depth {depth} in {dce_dir}")
    time_count = len(datasets) // depth
    position_to_index = {pos: idx for idx, pos in enumerate(unique_positions)}
    records = []
    for path, ds in zip(dicom_paths, datasets):
        ipp = np.asarray([float(v) for v in ds.ImagePositionPatient], dtype=np.float64)
        pos = round(float(np.dot(ipp, normal)), 4)
        nearest_pos = min(unique_positions, key=lambda item: abs(item - pos))
        temporal = getattr(ds, "TemporalPositionIdentifier", None)
        records.append(
            {
                "path": path,
                "ds": ds,
                "instance": int(getattr(ds, "InstanceNumber", 0) or 0),
                "temporal": int(temporal) if temporal not in (None, "") else None,
                "slice_index": position_to_index[nearest_pos],
            }
        )
    ordering = infer_order(records, depth)
    image = np.zeros((time_count, depth, int(first.Rows), int(first.Columns)), dtype=np.float32)
    if ordering == "dicom_temporal_position":
        temporal_values = sorted(set(int(rec["temporal"]) for rec in records))
        temporal_to_index = {value: idx for idx, value in enumerate(temporal_values)}
        for rec in records:
            image[temporal_to_index[int(rec["temporal"])], rec["slice_index"]] = rec["ds"].pixel_array.astype(np.float32)
    else:
        ordered = sorted(records, key=lambda r: (r["instance"], natural_key(r["path"])))
        chunks = [ordered[i : i + depth] for i in range(0, len(ordered), depth)]
        if all(len({rec["slice_index"] for rec in chunk}) == depth for chunk in chunks):
            for t, chunk in enumerate(chunks):
                for rec in chunk:
                    image[t, rec["slice_index"]] = rec["ds"].pixel_array.astype(np.float32)
            ordering = "time_major_by_instance"
        else:
            for idx, rec in enumerate(ordered):
                z = idx // time_count
                t = idx % time_count
                image[t, rec["slice_index" if ordering == "time_major_by_instance" else "slice_index"]] = rec["ds"].pixel_array.astype(np.float32)
            ordering = "fallback_instance_with_position"
    slope = float(getattr(first, "RescaleSlope", 1.0))
    intercept = float(getattr(first, "RescaleIntercept", 0.0))
    image = image * slope + intercept
    spacing = (float(row_spacing), float(col_spacing), float(slice_spacing))
    affine = affine_from_dicom(first, slice_spacing)
    temporal_positions = list(range(1, time_count + 1))
    return image, datasets, spacing, affine, temporal_positions, unique_positions, ordering


def roi_name_map(rtstruct) -> dict[int, str]:
    names = {}
    for roi in getattr(rtstruct, "StructureSetROISequence", []):
        names[int(roi.ROINumber)] = str(roi.ROIName)
    return names


def is_gtv_name(name: str) -> bool:
    lowered = name.lower().replace("_", " ")
    if any(pattern in lowered for pattern in EXCLUDE_ROI_PATTERNS):
        return False
    return any(pattern in lowered for pattern in ROI_PATTERNS)


def rasterize_rtstruct(rtstruct_path: Path, reference_ds, reference_datasets: Sequence, depth: int) -> tuple[np.ndarray, list[str]]:
    pydicom = require_pydicom()
    rt = pydicom.dcmread(str(rtstruct_path), force=True)
    names = roi_name_map(rt)
    selected_numbers = {number for number, name in names.items() if is_gtv_name(name)}
    selected_names = [names[number] for number in sorted(selected_numbers)]
    mask = np.zeros((depth, int(reference_ds.Rows), int(reference_ds.Columns)), dtype=np.uint8)
    if not selected_numbers:
        return mask.astype(np.float32), []

    row_cos, col_cos, normal, row_spacing, col_spacing, _, unique_positions = slice_geometry(reference_ds, reference_datasets)
    origins_by_slice = {}
    for ds in reference_datasets:
        ipp = np.asarray([float(v) for v in ds.ImagePositionPatient], dtype=np.float64)
        pos = round(float(np.dot(ipp, normal)), 4)
        nearest_pos = min(unique_positions, key=lambda item: abs(item - pos))
        z_index = unique_positions.index(nearest_pos)
        origins_by_slice[z_index] = ipp
    for roi_contour in getattr(rt, "ROIContourSequence", []):
        if int(roi_contour.ReferencedROINumber) not in selected_numbers:
            continue
        for contour in getattr(roi_contour, "ContourSequence", []):
            data = np.asarray([float(v) for v in contour.ContourData], dtype=np.float64).reshape(-1, 3)
            if data.shape[0] < 3:
                continue
            contour_pos = round(float(np.dot(data.mean(axis=0), normal)), 4)
            nearest_pos = min(unique_positions, key=lambda item: abs(item - contour_pos))
            z = unique_positions.index(nearest_pos)
            origin = origins_by_slice.get(z)
            if origin is None:
                continue
            rel = data - origin
            xs = np.dot(rel, row_cos) / col_spacing
            ys = np.dot(rel, col_cos) / row_spacing
            polygon = [(float(x), float(y)) for x, y in zip(xs, ys)]
            canvas = Image.fromarray(mask[z] * 255)
            draw = ImageDraw.Draw(canvas)
            draw.polygon(polygon, outline=255, fill=255)
            mask[z] = (np.asarray(canvas) > 0).astype(np.uint8)
    return mask.astype(np.float32), selected_names


def find_pre_dce_dir(patient_dir: Path) -> Path:
    for candidate in [patient_dir / "Pre" / "DCE", patient_dir / "Pre-treatment" / "DCE", patient_dir / "pre" / "DCE"]:
        if candidate.exists():
            return candidate
    for path in patient_dir.rglob("*"):
        if path.is_dir() and path.name.lower() == "dce" and path.parent.name.lower().startswith("pre"):
            return path
    raise FileNotFoundError(f"Pre-treatment DCE directory not found for {patient_dir}")


def find_rtstruct(patient_dir: Path) -> Path:
    struct_dirs = [patient_dir / "structures", patient_dir / "Structures"]
    for struct_dir in struct_dirs:
        if struct_dir.exists():
            files = sorted(struct_dir.glob("*.dcm"), key=natural_key)
            if files:
                return files[0]
    files = sorted(patient_dir.rglob("RTSTRUCT*.dcm"), key=natural_key)
    if files:
        return files[0]
    files = sorted((patient_dir / "structures").glob("*.dcm"), key=natural_key) if (patient_dir / "structures").exists() else []
    if files:
        return files[0]
    raise FileNotFoundError(f"RTSTRUCT DICOM not found for {patient_dir}")


def process_patient(patient_dir: Path, image_size: int, channel_mode: str, keep_empty_gtv: bool) -> DCEVolume | None:
    case_id = patient_dir.name
    dce_dir = find_pre_dce_dir(patient_dir)
    rtstruct_path = find_rtstruct(patient_dir)
    image, datasets, spacing, affine, temporal_positions, slice_positions, ordering = read_dce_series(dce_dir)
    mask, roi_names = rasterize_rtstruct(rtstruct_path, datasets[0], datasets[: len(slice_positions)], image.shape[1])
    if not roi_names and not keep_empty_gtv:
        print(f"[skip] {case_id}: no primary GTV-like ROI in {rtstruct_path.name}")
        return None
    image = scale_to_uint8_like(image)
    image = resize_image_4d(image, image_size)
    mask = resize_mask_3d(mask, image_size)
    channels, channel_meta = select_channels(image, channel_mode)
    return DCEVolume(
        case_id=case_id,
        image=channels,
        mask=mask,
        spacing=spacing,
        affine=affine,
        temporal_positions=temporal_positions,
        slice_positions=slice_positions,
        ordering=ordering,
        roi_names=roi_names,
    )


def split_cases(case_ids: Sequence[str], val_count: int, test_count: int) -> dict[str, str]:
    ordered = sorted(case_ids, key=lambda name: int(re.search(r"\d+", name).group(0)) if re.search(r"\d+", name) else name)
    test = set(ordered[-test_count:] if test_count else [])
    val = set(ordered[-test_count - val_count : -test_count] if val_count else [])
    return {case_id: "test" if case_id in test else "val" if case_id in val else "train" for case_id in ordered}


def neighbor_indices(center: int, count: int, num_slices: int) -> list[int]:
    half = num_slices // 2
    return [min(max(center + offset, 0), count - 1) for offset in range(-half, half + 1)]


def write_case_outputs(volume: DCEVolume, split: str, output_2d: Path, output_25d: Path, output_3d: Path, num_slices: int) -> None:
    for root in (output_2d, output_25d, output_3d):
        (root / split).mkdir(parents=True, exist_ok=True)
    c, d, h, w = volume.image.shape
    data_2d = output_2d / split / "data"
    gt_2d = output_2d / split / "GT"
    data_25d = output_25d / split / "data"
    gt_25d = output_25d / split / "GT"
    volume_dir = output_3d / split / "volumes"
    for path in (data_2d, gt_2d, data_25d, gt_25d, volume_dir):
        path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(volume_dir / f"{volume.case_id}.npz", image=volume.image.astype(np.float32), mask=volume.mask[None].astype(np.float32))
    for z in range(d):
        stem = f"{volume.case_id}_z{z:03d}"
        np.save(data_2d / f"{stem}.npy", np.transpose(volume.image[:, z], (1, 2, 0)).astype(np.float32))
        Image.fromarray((volume.mask[z] > 0).astype(np.uint8) * 255).save(gt_2d / f"{stem}.png")
        stack = np.stack([volume.image[:, idx] for idx in neighbor_indices(z, d, num_slices)], axis=0)
        np.save(data_25d / f"{stem}.npy", stack.astype(np.float32))
        Image.fromarray((volume.mask[z] > 0).astype(np.uint8) * 255).save(gt_25d / f"{stem}.png")


def write_metadata(rows: list[dict], output_roots: Iterable[Path]) -> None:
    for root in output_roots:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "cases.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        with (root / "splits.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["case_id", "split", "channels", "depth", "height", "width", "ordering", "roi_names"])
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Head and Neck DCE-MRI DICOM/RTSTRUCT for KPTA and contrast models.")
    parser.add_argument("--raw-root", default="Patient DCE and VFA Files")
    parser.add_argument("--output-2d", default="processed_headneck_dce_56ch_2d")
    parser.add_argument("--output-25d", default="processed_headneck_dce_56ch_25d")
    parser.add_argument("--output-3d", default="processed_headneck_dce_56ch_3d")
    parser.add_argument("--channel-mode", default="full56", choices=["full56", "breastdm17"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-slices", type=int, default=3)
    parser.add_argument("--val-count", type=int, default=3)
    parser.add_argument("--test-count", type=int, default=3)
    parser.add_argument("--keep-empty-gtv", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root)
    patients = sorted([p for p in raw_root.iterdir() if p.is_dir() and p.name.lower().startswith("pat")], key=natural_key)
    if args.limit_cases:
        patients = patients[: args.limit_cases]
    splits = split_cases([p.name for p in patients], args.val_count, args.test_count)
    rows = []
    for patient_dir in tqdm(patients, desc="HeadNeckDCE preprocessing"):
        try:
            volume = process_patient(patient_dir, args.image_size, args.channel_mode, args.keep_empty_gtv)
        except Exception as exc:
            print(f"[skip] {patient_dir.name}: {exc}")
            continue
        if volume is None:
            continue
        split = splits[volume.case_id]
        write_case_outputs(volume, split, Path(args.output_2d), Path(args.output_25d), Path(args.output_3d), args.num_slices)
        rows.append(
            {
                "case_id": volume.case_id,
                "split": split,
                "channels": int(volume.image.shape[0]),
                "depth": int(volume.image.shape[1]),
                "height": int(volume.image.shape[2]),
                "width": int(volume.image.shape[3]),
                "spacing": volume.spacing,
                "affine": volume.affine,
                "temporal_positions": volume.temporal_positions,
                "slice_positions": volume.slice_positions,
                "ordering": volume.ordering,
                "roi_names": ";".join(volume.roi_names),
            }
        )
    write_metadata(rows, [Path(args.output_2d), Path(args.output_25d), Path(args.output_3d)])
    print(f"Prepared {len(rows)} labeled cases.")
    print(f"Pat1-like DCE ordering detected as time-major when TemporalPositionIdentifier increments every 30 slices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

