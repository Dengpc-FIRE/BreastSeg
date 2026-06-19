from __future__ import annotations

import argparse
import csv
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ContrastModel.dataset.metrics import METRIC_KEYS, compute_case_metrics, summarize_metrics, write_summary  # noqa: E402
from experiments.generalization.mama_mia import adapt_channel_count, describe_case_input, discover_cases, load_case_17ch, neighbor_indices, selected_cohorts  # noqa: E402
from train.losses import unpack_model_output  # noqa: E402
from train.train_config import build_model_from_config, load_config, resolve_config_path  # noqa: E402

EXTRA_METRIC_KEYS = [
    "case_id",
    "slice_index",
    "metric_mode",
    "threshold",
    "pred_voxels",
    "target_voxels",
    "pred_fraction",
    "target_fraction",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run MAMA-MIA cohort generalization for KPTA/KPTA-25D.")
    parser.add_argument("--config", default="configs/kpta_25d_net.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mama-root", default=".")
    parser.add_argument("--cohorts", nargs="+", default=["ALL"])
    parser.add_argument("--mask-source", default="expert", choices=["expert"])
    parser.add_argument("--output-dir", default="generalization_results/kpta_mama_mia")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--source-scale", default="breastdm_uint8", choices=["none", "breastdm_uint8", "per_phase_uint8"])
    parser.add_argument("--subtraction-mode", default="positive", choices=["positive", "signed", "raw_positive_uint8"])
    parser.add_argument("--depth-axis", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--eval-slices", default="all", choices=["all", "gt_positive", "gt_bbox"])
    parser.add_argument("--eval-slice-margin", type=int, default=2)
    parser.add_argument("--metric-mode", default="slice", choices=["slice", "volume"])
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--diagnose-cases", type=int, default=0)
    parser.add_argument("--threshold-sweep", nargs="+", type=float, default=None)
    parser.add_argument("--postprocess", default="none", choices=["none", "largest_component"])
    parser.add_argument("--min-component-voxels", type=int, default=0)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def load_checkpoint(model, checkpoint: Path, device, strict: bool = False):
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("model_state", state)
    result = model.load_state_dict(state_dict, strict=strict)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        print(
            f"[checkpoint] loaded with missing_keys={len(missing)}, unexpected_keys={len(unexpected)} "
            f"from {checkpoint}"
        )
        if missing:
            print("[checkpoint] first missing keys:", ", ".join(missing[:20]))
        if unexpected:
            print("[checkpoint] first unexpected keys:", ", ".join(unexpected[:20]))
    else:
        print(f"[checkpoint] all keys matched: {checkpoint}")


def predict_kpta_25d_probs(
    model,
    image: np.ndarray,
    num_slices: int,
    device,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    model.eval()
    probs: List[np.ndarray] = []
    use_amp = bool(amp and torch.device(device).type == "cuda")
    with torch.inference_mode():
        batch = []
        for z in range(image.shape[1]):
            stack = np.stack([image[:, idx] for idx in neighbor_indices(z, image.shape[1], num_slices)], axis=0)
            batch.append(stack)
            if len(batch) == batch_size or z == image.shape[1] - 1:
                tensor = torch.from_numpy(np.stack(batch, axis=0)).to(device=device, dtype=torch.float32)
                autocast_ctx = torch.autocast(device_type="cuda") if use_amp else nullcontext()
                with autocast_ctx:
                    logits, _ = unpack_model_output(model(tensor))
                prob = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()
                probs.extend(prob)
                batch = []
    return np.stack(probs, axis=0)


def _add_volume_stats(metric: dict, pred: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    metric["threshold"] = float(threshold)
    metric["pred_voxels"] = int(np.asarray(pred).astype(bool).sum())
    metric["target_voxels"] = int(np.asarray(target).astype(bool).sum())
    metric["pred_fraction"] = float(np.asarray(pred).astype(bool).mean())
    metric["target_fraction"] = float(np.asarray(target).astype(bool).mean())
    return metric


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    try:
        from scipy import ndimage as ndi

        labels, count = ndi.label(mask)
        if count == 0:
            return mask
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        return labels == int(np.argmax(sizes))
    except Exception:
        return _largest_component_numpy(mask)


def _largest_component_numpy(mask: np.ndarray) -> np.ndarray:
    shape = mask.shape
    visited = np.zeros(shape, dtype=bool)
    best_component: List[tuple[int, int, int]] = []
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    starts = np.argwhere(mask)
    for start in starts:
        z, y, x = (int(v) for v in start)
        if visited[z, y, x]:
            continue
        stack = [(z, y, x)]
        visited[z, y, x] = True
        component: List[tuple[int, int, int]] = []
        while stack:
            current = stack.pop()
            component.append(current)
            cz, cy, cx = current
            for dz, dy, dx in offsets:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if (
                    0 <= nz < shape[0]
                    and 0 <= ny < shape[1]
                    and 0 <= nx < shape[2]
                    and mask[nz, ny, nx]
                    and not visited[nz, ny, nx]
                ):
                    visited[nz, ny, nx] = True
                    stack.append((nz, ny, nx))
        if len(component) > len(best_component):
            best_component = component
    out = np.zeros(shape, dtype=bool)
    if best_component:
        zz, yy, xx = zip(*best_component)
        out[np.asarray(zz), np.asarray(yy), np.asarray(xx)] = True
    return out


def postprocess_prediction(pred: np.ndarray, mode: str, min_component_voxels: int = 0) -> np.ndarray:
    pred = np.asarray(pred).astype(bool)
    if mode == "none":
        out = pred
    elif mode == "largest_component":
        out = _largest_component(pred)
    else:
        raise ValueError(f"Unknown postprocess mode: {mode}")
    if min_component_voxels > 0 and int(out.sum()) < int(min_component_voxels):
        return np.zeros_like(out, dtype=bool)
    return out


def select_eval_slices(target: np.ndarray, mode: str, margin: int = 2) -> np.ndarray:
    target = np.asarray(target).astype(bool)
    if mode == "all":
        return np.arange(target.shape[0])
    positive = np.flatnonzero(target.reshape(target.shape[0], -1).any(axis=1))
    if positive.size == 0:
        return np.arange(target.shape[0])
    if mode == "gt_positive":
        return positive
    if mode == "gt_bbox":
        start = max(0, int(positive[0]) - int(margin))
        end = min(target.shape[0], int(positive[-1]) + int(margin) + 1)
        return np.arange(start, end)
    raise ValueError(f"Unknown eval-slices mode: {mode}")


def _rows_for_threshold(rows: list[dict], threshold: float) -> list[dict]:
    return [row for row in rows if abs(float(row["threshold"]) - float(threshold)) < 1e-9]


def _write_threshold_summary(output_dir: Path, rows: list[dict], thresholds: list[float]) -> tuple[float, dict]:
    threshold_summaries = []
    with (output_dir / "summary_by_threshold.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["threshold", "n", *[f"mean_{key}" for key in METRIC_KEYS]]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for threshold in thresholds:
            threshold_rows = _rows_for_threshold(rows, threshold)
            if not threshold_rows:
                continue
            threshold_summary = summarize_metrics(threshold_rows)
            threshold_summaries.append((float(threshold), threshold_summary))
            writer.writerow({"threshold": float(threshold), "n": len(threshold_rows), **threshold_summary})
    best_threshold, best_summary = max(threshold_summaries, key=lambda item: item[1]["mean_dice"])
    write_summary(output_dir / "summary_best_threshold.txt", best_summary)
    return best_threshold, best_summary


def evaluate_prediction(
    pred_full: np.ndarray,
    target: np.ndarray,
    eval_indices: np.ndarray,
    threshold: float,
    case_id: str,
    cohort: str,
    metric_mode: str,
) -> list[dict]:
    if metric_mode == "volume":
        pred = pred_full[eval_indices]
        target_eval = target[eval_indices]
        metric = compute_case_metrics(pred, target_eval)
        _add_volume_stats(metric, pred, target_eval, threshold)
        metric.update(
            {
                "id": case_id,
                "case_id": case_id,
                "slice_index": "",
                "cohort": cohort,
                "metric_mode": metric_mode,
            }
        )
        return [metric]
    if metric_mode != "slice":
        raise ValueError(f"Unknown metric mode: {metric_mode}")

    rows = []
    for slice_index in eval_indices:
        z = int(slice_index)
        pred = pred_full[z]
        target_slice = target[z]
        metric = compute_case_metrics(pred, target_slice)
        _add_volume_stats(metric, pred, target_slice, threshold)
        metric.update(
            {
                "id": f"{case_id}_z{z:04d}",
                "case_id": case_id,
                "slice_index": z,
                "cohort": cohort,
                "metric_mode": metric_mode,
            }
        )
        rows.append(metric)
    return rows


def run(args):
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    if config["model"]["name"] != "kpta_25d_net":
        raise ValueError("This generalization script currently supports model.name='kpta_25d_net'.")
    img_size = dataset_cfg.get("img_size", 256)
    if isinstance(img_size, (list, tuple)):
        image_size = (int(img_size[0]), int(img_size[1]))
    else:
        image_size = (int(img_size), int(img_size))
    num_slices = int(model_cfg.get("num_slices", 3))
    input_phases = int(model_cfg.get("in_phases", 17))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(config).to(device)
    load_checkpoint(model, Path(args.checkpoint), device, strict=args.strict_load)

    cohorts = selected_cohorts(args.cohorts)
    cases = discover_cases(args.mama_root, cohorts, mask_source=args.mask_source)
    if args.limit_cases:
        cases = cases[: args.limit_cases]
    if not cases:
        raise FileNotFoundError(f"No MAMA-MIA cases found for cohorts={cohorts} under {args.mama_root}")

    rows = []
    thresholds = args.threshold_sweep or [args.threshold]
    thresholds = [float(threshold) for threshold in thresholds]
    for idx, case in enumerate(tqdm(cases, desc="kpta_25d MAMA-MIA")):
        image, mask = load_case_17ch(
            case,
            image_size,
            normalize="none",
            source_scale=args.source_scale,
            subtraction_mode=args.subtraction_mode,
            depth_axis=args.depth_axis,
        )
        if args.diagnose_cases and idx < args.diagnose_cases:
            print(f"[diagnose] {case.case_id}: {describe_case_input(image, mask)}")
        image = adapt_channel_count(image, input_phases)
        prob = predict_kpta_25d_probs(model, image, num_slices, device, args.batch_size, args.amp)
        target = mask.astype(bool)
        eval_indices = select_eval_slices(target, args.eval_slices, margin=args.eval_slice_margin)
        if args.diagnose_cases and idx < args.diagnose_cases and args.eval_slices != "all":
            print(
                f"[diagnose] {case.case_id} eval_slices={args.eval_slices}: "
                f"{len(eval_indices)}/{target.shape[0]} slices"
            )
        for threshold in thresholds:
            pred_full = postprocess_prediction(
                prob >= float(threshold),
                mode=args.postprocess,
                min_component_voxels=args.min_component_voxels,
            )
            metric_rows = evaluate_prediction(
                pred_full,
                target,
                eval_indices,
                float(threshold),
                case.case_id,
                case.cohort,
                args.metric_mode,
            )
            rows.extend(metric_rows)
            if args.diagnose_cases and idx < args.diagnose_cases:
                metric = summarize_metrics(metric_rows)
                pred_fraction = float(np.mean([row["pred_fraction"] for row in metric_rows]))
                target_fraction = float(np.mean([row["target_fraction"] for row in metric_rows]))
                print(
                    f"[diagnose] {case.case_id} threshold={float(threshold):.3f}: "
                    f"dice={metric['mean_dice']:.6f}, precision={metric['mean_precision']:.6f}, "
                    f"sensitivity={metric['mean_sensitivity']:.6f}, pred_fraction={pred_fraction:.6f}, "
                    f"target_fraction={target_fraction:.6f}, metric_mode={args.metric_mode}"
                )

    primary_rows = _rows_for_threshold(rows, float(args.threshold))
    summary = summarize_metrics(primary_rows or rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_with_cohort(output_dir / "metrics.csv", rows)
    write_summary(output_dir / "summary.txt", summary)
    with (output_dir / "summary_by_cohort.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["cohort", "n", *summary.keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cohort in cohorts:
            cohort_rows = [row for row in primary_rows if row["cohort"] == cohort]
            if cohort_rows:
                writer.writerow({"cohort": cohort, "n": len(cohort_rows), **summarize_metrics(cohort_rows)})
    if len(thresholds) > 1:
        best_threshold, best_summary = _write_threshold_summary(output_dir, rows, thresholds)
    print(f"Saved metrics to {output_dir}")
    if len(thresholds) > 1:
        print("Threshold sweep summary:")
        for threshold in thresholds:
            threshold_summary = summarize_metrics(_rows_for_threshold(rows, threshold))
            print(
                f"threshold={threshold:.3f} "
                f"mean_dice={threshold_summary['mean_dice']:.6f} "
                f"mean_precision={threshold_summary['mean_precision']:.6f} "
                f"mean_sensitivity={threshold_summary['mean_sensitivity']:.6f} "
                f"mean_hd95={threshold_summary['mean_hd95']:.6f}"
            )
        print(f"Best threshold by mean_dice: {best_threshold:.3f}")
        for key, value in best_summary.items():
            print(f"{key}: {value:.6f}")
    else:
        for key, value in summary.items():
            print(f"{key}: {value:.6f}")


def _write_metrics_with_cohort(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "cohort", *METRIC_KEYS, *EXTRA_METRIC_KEYS])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


if __name__ == "__main__":
    run(parse_args())
