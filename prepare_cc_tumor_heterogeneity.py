from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


PHASE_NAMES = ["PRE", *[f"POST{i}" for i in range(1, 9)], *[f"SUB{i}" for i in range(1, 9)]]
POST_RELATIVE_POSITIONS = [0.03, 0.07, 0.13, 0.22, 0.35, 0.50, 0.70, 0.95]


@dataclass
class DCECase:
    case_id: str
    subject_id: str
    split: str
    image: np.ndarray
    mask: np.ndarray
    metadata: dict


def require_pydicom():
    try:
        import pydicom
    except Exception as exc:
        raise RuntimeError(
            "CC-Tumor-Heterogeneity preprocessing requires pydicom. "
            "Install with: python -m pip install pydicom"
        ) from exc
    return pydicom


def natural_key(value: str | Path) -> tuple:
    name = value.name if isinstance(value, Path) else str(value)
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name))


def read_metadata(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_location(manifest_root: Path, value: str) -> Path:
    return manifest_root / value.replace(".\\", "").replace("\\", "/")


def is_core_dce(row: dict, min_images: int) -> bool:
    if row.get("Modality") != "MR":
        return False
    desc = row.get("Series Description", "")
    if not re.search(r"DCE|Dyn|Dynamic|Wdyn|dynamic", desc, flags=re.IGNORECASE):
        return False
    try:
        return int(row.get("Number of Images", 0)) >= min_images
    except ValueError:
        return False


def split_subjects(subjects: Sequence[str], val_count: int, test_count: int) -> dict[str, str]:
    ordered = sorted(subjects, key=natural_key)
    test = set(ordered[-test_count:] if test_count else [])
    val = set(ordered[-test_count - val_count : -test_count] if val_count else [])
    return {sid: "test" if sid in test else "val" if sid in val else "train" for sid in ordered}


def match_rtstruct(dce_row: dict, rt_rows: Sequence[dict]) -> dict | None:
    subject = dce_row["Subject ID"]
    date = dce_row["Study Date"]
    same_date = [row for row in rt_rows if row["Subject ID"] == subject and row["Study Date"] == date]
    if not same_date:
        return None
    exact = [row for row in same_date if row["Study Description"] == dce_row["Study Description"]]
    return exact[0] if exact else same_date[0]


def robust_window(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    nonzero = values[np.abs(values) > 1e-6]
    stats = nonzero if nonzero.size >= 1024 else values
    lo, hi = np.percentile(stats, [1.0, 99.5]).astype(np.float32)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(stats.min()), float(stats.max())
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def scale_to_uint8_like(volume: np.ndarray) -> np.ndarray:
    lo, hi = robust_window(volume.reshape(-1))
    scaled = (np.clip(volume.astype(np.float32), lo, hi) - lo) / (hi - lo + 1e-6)
    return (scaled * 255.0).astype(np.float32)


def resize_plane(plane: np.ndarray, image_size: int, nearest: bool = False) -> np.ndarray:
    mode = Image.NEAREST if nearest else Image.BILINEAR
    return np.asarray(Image.fromarray(plane.astype(np.float32)).resize((image_size, image_size), mode), dtype=np.float32)


def resize_4d(volume: np.ndarray, image_size: int) -> np.ndarray:
    if volume.shape[-2:] == (image_size, image_size):
        return volume.astype(np.float32, copy=False)
    out = np.empty((volume.shape[0], volume.shape[1], image_size, image_size), dtype=np.float32)
    for t in range(volume.shape[0]):
        for z in range(volume.shape[1]):
            out[t, z] = resize_plane(volume[t, z], image_size)
    return out


def resize_3d(mask: np.ndarray, image_size: int) -> np.ndarray:
    if mask.shape[-2:] == (image_size, image_size):
        return mask.astype(np.float32, copy=False)
    out = np.empty((mask.shape[0], image_size, image_size), dtype=np.float32)
    for z in range(mask.shape[0]):
        out[z] = resize_plane(mask[z], image_size, nearest=True)
    return (out > 0).astype(np.float32)


def parse_time_to_seconds(value: object) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        hour = int(text[0:2])
        minute = int(text[2:4]) if len(text) >= 4 else 0
        second = float(text[4:]) if len(text) > 4 else 0.0
        return hour * 3600.0 + minute * 60.0 + second
    except Exception:
        return None


def dicom_geometry(first_ds, datasets: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, list[float]]:
    orientation = np.asarray([float(v) for v in first_ds.ImageOrientationPatient], dtype=np.float64)
    row_cos = orientation[:3]
    col_cos = orientation[3:]
    normal = np.cross(row_cos, col_cos)
    row_spacing, col_spacing = [float(v) for v in first_ds.PixelSpacing]
    positions = []
    for ds in datasets:
        ipp = np.asarray([float(v) for v in ds.ImagePositionPatient], dtype=np.float64)
        positions.append(round(float(np.dot(ipp, normal)), 4))
    unique_positions = sorted(set(positions), reverse=True)
    if len(unique_positions) > 1:
        slice_spacing = float(np.median(np.abs(np.diff(sorted(unique_positions)))))
    else:
        slice_spacing = float(getattr(first_ds, "SliceThickness", 1.0))
    return row_cos, col_cos, normal, row_spacing, col_spacing, slice_spacing, unique_positions


def infer_dce_order(records: list[dict], depth: int) -> str:
    if all(rec["temporal"] is not None for rec in records):
        return "dicom_temporal_position"
    ordered = sorted(records, key=lambda rec: (rec["instance"], natural_key(rec["path"])))
    if len(ordered) >= depth * 2:
        if len({rec["slice_index"] for rec in ordered[:depth]}) == depth:
            return "time_major_by_instance"
    return "instance_position_fallback"


def reference_datasets(records: list[dict], depth: int, ordering: str) -> list:
    refs = [None] * depth
    if ordering == "dicom_temporal_position":
        first_t = min(int(rec["temporal"]) for rec in records)
        candidates = [rec for rec in records if int(rec["temporal"]) == first_t]
    else:
        candidates = sorted(records, key=lambda rec: (rec["instance"], natural_key(rec["path"])))[:depth]
    for rec in candidates:
        refs[int(rec["slice_index"])] = rec["ds"]
    for rec in records:
        z = int(rec["slice_index"])
        if refs[z] is None:
            refs[z] = rec["ds"]
    if any(ref is None for ref in refs):
        raise ValueError("Could not infer reference DCE slice geometry.")
    return refs


def read_dce(dce_dir: Path) -> tuple[np.ndarray, list, list, tuple[float, float, float], str]:
    pydicom = require_pydicom()
    paths = sorted(dce_dir.glob("*.dcm"), key=natural_key)
    if not paths:
        raise FileNotFoundError(f"No DICOM files found in {dce_dir}")
    datasets = [pydicom.dcmread(str(path), force=True) for path in paths]
    first = datasets[0]
    row_cos, col_cos, normal, row_spacing, col_spacing, slice_spacing, positions = dicom_geometry(first, datasets)
    depth = len(positions)
    if len(datasets) % depth != 0:
        raise ValueError(f"Series image count {len(datasets)} is not divisible by slice count {depth}: {dce_dir}")
    time_count = len(datasets) // depth
    pos_to_index = {pos: idx for idx, pos in enumerate(positions)}
    records = []
    for path, ds in zip(paths, datasets):
        ipp = np.asarray([float(v) for v in ds.ImagePositionPatient], dtype=np.float64)
        pos = round(float(np.dot(ipp, normal)), 4)
        nearest = min(positions, key=lambda item: abs(item - pos))
        temporal = getattr(ds, "TemporalPositionIdentifier", None)
        records.append(
            {
                "path": path,
                "ds": ds,
                "instance": int(getattr(ds, "InstanceNumber", 0) or 0),
                "temporal": int(temporal) if temporal not in (None, "") else None,
                "slice_index": pos_to_index[nearest],
            }
        )
    ordering = infer_dce_order(records, depth)
    image = np.zeros((time_count, depth, int(first.Rows), int(first.Columns)), dtype=np.float32)
    time_records = [None] * time_count
    if ordering == "dicom_temporal_position":
        temporal_values = sorted(set(int(rec["temporal"]) for rec in records))
        if len(temporal_values) != time_count:
            raise ValueError(f"Temporal positions {len(temporal_values)} do not match inferred time count {time_count}: {dce_dir}")
        temporal_to_index = {value: idx for idx, value in enumerate(temporal_values)}
        for rec in records:
            t = temporal_to_index[int(rec["temporal"])]
            z = int(rec["slice_index"])
            image[t, z] = rec["ds"].pixel_array.astype(np.float32)
            if time_records[t] is None:
                time_records[t] = rec["ds"]
    else:
        ordered = sorted(records, key=lambda rec: (rec["instance"], natural_key(rec["path"])))
        for t, chunk_start in enumerate(range(0, len(ordered), depth)):
            for rec in ordered[chunk_start : chunk_start + depth]:
                z = int(rec["slice_index"])
                image[t, z] = rec["ds"].pixel_array.astype(np.float32)
                if time_records[t] is None:
                    time_records[t] = rec["ds"]
        ordering = "time_major_by_instance"
    slope = float(getattr(first, "RescaleSlope", 1.0))
    intercept = float(getattr(first, "RescaleIntercept", 0.0))
    image = image * slope + intercept
    refs = reference_datasets(records, depth, ordering)
    return image, refs, time_records, (float(row_spacing), float(col_spacing), float(slice_spacing)), ordering


def detect_enhancement_start(image: np.ndarray, time_records: Sequence, default_index: int = 1) -> tuple[int, str]:
    bolus_times = [parse_time_to_seconds(getattr(ds, "ContrastBolusStartTime", None)) for ds in time_records if ds is not None]
    bolus_times = [value for value in bolus_times if value is not None]
    acq_times = [parse_time_to_seconds(getattr(ds, "AcquisitionTime", None)) if ds is not None else None for ds in time_records]
    if bolus_times and all(value is not None for value in acq_times):
        bolus = bolus_times[0]
        adjusted = []
        for value in acq_times:
            value = float(value)
            if value + 43200.0 < bolus:
                value += 86400.0
            adjusted.append(value)
        for idx, value in enumerate(adjusted):
            if value >= bolus:
                return max(1, idx), "dicom_contrast_bolus_start_time"

    mean_image = image.mean(axis=0)
    threshold = np.percentile(mean_image[np.isfinite(mean_image)], 35.0)
    body = mean_image > threshold
    if body.sum() < 512:
        body = np.ones(mean_image.shape, dtype=bool)
    curve = np.asarray([float(frame[body].mean()) for frame in image], dtype=np.float32)
    if curve.size < 4:
        return default_index, "fallback_first_frame"
    baseline_count = min(5, max(2, curve.size // 10))
    baseline = curve[:baseline_count]
    base_mean = float(baseline.mean())
    base_std = float(baseline.std())
    threshold_delta = max(3.0 * base_std, 0.04 * max(abs(base_mean), 1.0))
    for idx in range(1, curve.size - 1):
        if curve[idx] - base_mean > threshold_delta and curve[idx + 1] - base_mean > threshold_delta:
            return idx, "signal_curve"
    return default_index, "fallback_first_frame"


def build_17ch(image: np.ndarray, max_pre_frames: int) -> tuple[np.ndarray, dict]:
    start, method = detect_enhancement_start(image)
    if start <= 1:
        pre_count = 1
    else:
        pre_count = min(max_pre_frames, start)
    pre = image[:pre_count].mean(axis=0)
    post_start = min(max(start, 1), image.shape[0] - 1)
    post_frames = image[post_start:]
    if post_frames.shape[0] == 0:
        post_frames = image[-1:]
        post_start = image.shape[0] - 1
    offsets = [int(round(pos * (post_frames.shape[0] - 1))) for pos in POST_RELATIVE_POSITIONS]
    post_indices = [post_start + offset for offset in offsets]
    posts = [image[idx] for idx in post_indices]
    subs = [np.clip(post - pre, 0.0, None) for post in posts]
    return np.stack([pre, *posts, *subs], axis=0).astype(np.float32), {
        "enhancement_start_index": int(start),
        "enhancement_start_method": method,
        "pre_frame_count": int(pre_count),
        "post_start_index": int(post_start),
        "post_offsets": offsets,
        "post_indices": post_indices,
        "post_relative_positions": POST_RELATIVE_POSITIONS,
    }


def roi_name_map(rtstruct) -> dict[int, str]:
    return {int(roi.ROINumber): str(roi.ROIName) for roi in getattr(rtstruct, "StructureSetROISequence", [])}


def selected_roi_numbers(names: dict[int, str], include_regex: str) -> set[int]:
    pattern = re.compile(include_regex, flags=re.IGNORECASE)
    return {number for number, name in names.items() if pattern.search(name)}


def rasterize_rtstruct(rtstruct_dir: Path, reference_datasets: Sequence, include_regex: str) -> tuple[np.ndarray, list[str]]:
    pydicom = require_pydicom()
    files = sorted(rtstruct_dir.glob("*.dcm"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No RTSTRUCT DICOM found in {rtstruct_dir}")
    rt = pydicom.dcmread(str(files[0]), force=True)
    names = roi_name_map(rt)
    selected = selected_roi_numbers(names, include_regex)
    selected_names = [names[number] for number in sorted(selected)]
    first = reference_datasets[0]
    row_cos, col_cos, normal, row_spacing, col_spacing, _, positions = dicom_geometry(first, reference_datasets)
    depth = len(reference_datasets)
    mask = np.zeros((depth, int(first.Rows), int(first.Columns)), dtype=np.uint8)
    if not selected:
        return mask.astype(np.float32), []
    origins = []
    projected_positions = []
    for ds in reference_datasets:
        ipp = np.asarray([float(v) for v in ds.ImagePositionPatient], dtype=np.float64)
        origins.append(ipp)
        projected_positions.append(round(float(np.dot(ipp, normal)), 4))
    for roi_contour in getattr(rt, "ROIContourSequence", []):
        if int(roi_contour.ReferencedROINumber) not in selected:
            continue
        for contour in getattr(roi_contour, "ContourSequence", []):
            data = np.asarray([float(v) for v in contour.ContourData], dtype=np.float64).reshape(-1, 3)
            if data.shape[0] < 3:
                continue
            cpos = round(float(np.dot(data.mean(axis=0), normal)), 4)
            z = int(np.argmin([abs(cpos - pos) for pos in projected_positions]))
            rel = data - origins[z]
            xs = np.dot(rel, row_cos) / col_spacing
            ys = np.dot(rel, col_cos) / row_spacing
            canvas = Image.fromarray(mask[z] * 255)
            ImageDraw.Draw(canvas).polygon([(float(x), float(y)) for x, y in zip(xs, ys)], outline=255, fill=255)
            mask[z] = (np.asarray(canvas) > 0).astype(np.uint8)
    return mask.astype(np.float32), selected_names


def neighbor_indices(center: int, count: int, num_slices: int) -> list[int]:
    half = num_slices // 2
    return [min(max(center + offset, 0), count - 1) for offset in range(-half, half + 1)]


def save_outputs(case: DCECase, output_2d: Path, output_25d: Path, output_3d: Path, num_slices: int) -> None:
    split = case.split
    image = case.image
    mask = case.mask
    data2d = output_2d / split / "data"
    gt2d = output_2d / split / "GT"
    data25d = output_25d / split / "data"
    gt25d = output_25d / split / "GT"
    raw_images = output_3d / split / "images" / case.case_id
    raw_labels = output_3d / split / "labels" / case.case_id / "GT"
    for folder in (data2d, gt2d, data25d, gt25d, raw_images, raw_labels):
        folder.mkdir(parents=True, exist_ok=True)
    channels, depth, _, _ = image.shape
    for c, phase in enumerate(PHASE_NAMES[:channels]):
        phase_dir = raw_images / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        for z in range(depth):
            np.save(phase_dir / f"z{z:03d}.npy", image[c, z].astype(np.float32))
    for z in range(depth):
        stem = f"{case.case_id}_z{z:03d}"
        np.save(data2d / f"{stem}.npy", np.transpose(image[:, z], (1, 2, 0)).astype(np.float32))
        Image.fromarray((mask[z] > 0).astype(np.uint8) * 255).save(gt2d / f"{stem}.png")
        stack = np.stack([image[:, idx] for idx in neighbor_indices(z, depth, num_slices)], axis=0)
        np.save(data25d / f"{stem}.npy", stack.astype(np.float32))
        Image.fromarray((mask[z] > 0).astype(np.uint8) * 255).save(gt25d / f"{stem}.png")
        Image.fromarray((mask[z] > 0).astype(np.uint8) * 255).save(raw_labels / f"z{z:03d}.png")


def process_row(row: dict, rt_row: dict, manifest_root: Path, split: str, image_size: int, max_pre_frames: int, roi_regex: str) -> DCECase | None:
    dce_dir = normalize_location(manifest_root, row["File Location"])
    rt_dir = normalize_location(manifest_root, rt_row["File Location"])
    raw_image, reference_datasets, time_records, spacing, ordering = read_dce(dce_dir)
    channels, channel_meta = build_17ch(raw_image, max_pre_frames=max_pre_frames)
    channels = scale_to_uint8_like(channels)
    channels = resize_4d(channels, image_size)
    mask, roi_names = rasterize_rtstruct(rt_dir, reference_datasets, roi_regex)
    mask = resize_3d(mask, image_size)
    if not roi_names:
        print(f"[skip] {row['Subject ID']} {row['Study Description']} {row['Study Date']}: no ROI matching {roi_regex!r}")
        return None
    study = re.sub(r"[^A-Za-z0-9]+", "", row["Study Description"])
    date = row["Study Date"].replace("-", "")
    case_id = f"{row['Subject ID']}_{study}_{date}"
    metadata = {
        "case_id": case_id,
        "subject_id": row["Subject ID"],
        "split": split,
        "study_description": row["Study Description"],
        "study_date": row["Study Date"],
        "series_description": row["Series Description"],
        "number_of_images": int(row["Number of Images"]),
        "raw_timepoints": int(raw_image.shape[0]),
        "raw_slices": int(raw_image.shape[1]),
        "spacing": spacing,
        "ordering": ordering,
        "roi_names": roi_names,
        **channel_meta,
    }
    return DCECase(case_id=case_id, subject_id=row["Subject ID"], split=split, image=channels, mask=mask, metadata=metadata)


def write_metadata(rows: list[dict], roots: Iterable[Path]) -> None:
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "cases.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        with (root / "splits.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["case_id", "subject_id", "split", "study_description", "study_date", "series_description", "raw_timepoints", "raw_slices", "pre_frame_count", "enhancement_start_index", "post_indices", "roi_names"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CC-Tumor-Heterogeneity DCE/RTSTRUCT for 13-model comparison.")
    parser.add_argument("--manifest-root", default="CC-Tumor-Heterogeneity/manifest-1655581046477")
    parser.add_argument("--output-2d", default="processed_cc_tumor_heterogeneity_17ch_2d")
    parser.add_argument("--output-25d", default="processed_cc_tumor_heterogeneity_17ch_25d")
    parser.add_argument("--output-3d", default="processed_cc_tumor_heterogeneity_17ch_3d_raw")
    parser.add_argument("--min-dce-images", type=int, default=900)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-slices", type=int, default=3)
    parser.add_argument("--max-pre-frames", type=int, default=5)
    parser.add_argument("--roi-regex", default=r"^Ut-MRT2")
    parser.add_argument("--val-subjects", type=int, default=4)
    parser.add_argument("--test-subjects", type=int, default=4)
    parser.add_argument("--limit-cases", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_root = Path(args.manifest_root)
    rows = read_metadata(manifest_root / "metadata.csv")
    dce_rows = [row for row in rows if is_core_dce(row, args.min_dce_images)]
    rt_rows = [row for row in rows if row.get("Modality") == "RTSTRUCT"]
    subjects = sorted({row["Subject ID"] for row in dce_rows}, key=natural_key)
    splits = split_subjects(subjects, args.val_subjects, args.test_subjects)
    cases = []
    metadata_rows = []
    for row in tqdm(dce_rows[: args.limit_cases] if args.limit_cases else dce_rows, desc="CC DCE preprocessing"):
        rt_row = match_rtstruct(row, rt_rows)
        if rt_row is None:
            print(f"[skip] no RTSTRUCT match: {row['Subject ID']} {row['Study Description']} {row['Study Date']}")
            continue
        try:
            case = process_row(row, rt_row, manifest_root, splits[row["Subject ID"]], args.image_size, args.max_pre_frames, args.roi_regex)
        except Exception as exc:
            print(f"[skip] {row['Subject ID']} {row['Study Description']} {row['Study Date']}: {exc}")
            continue
        if case is None:
            continue
        save_outputs(case, Path(args.output_2d), Path(args.output_25d), Path(args.output_3d), args.num_slices)
        metadata_rows.append(case.metadata)
        cases.append(case)
    write_metadata(metadata_rows, [Path(args.output_2d), Path(args.output_25d), Path(args.output_3d)])
    print(f"Prepared {len(cases)} DCE/RTSTRUCT cases from {len(subjects)} subjects.")
    print("Channels: PRE(mean baseline) + 8 POST(non-uniform relative positions) + 8 SUB(POST-PRE).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

