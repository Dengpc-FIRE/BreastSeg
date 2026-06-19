from __future__ import annotations

import argparse
import csv
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRAST_ROOT = PROJECT_ROOT / "ContrastModel"
for path in (PROJECT_ROOT, CONTRAST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ContrastModel.dataset.config import load_config, select_device  # noqa: E402
from ContrastModel.dataset.metrics import METRIC_KEYS, compute_case_metrics, summarize_metrics, write_summary  # noqa: E402
from ContrastModel.dataset.models import align_logits, build_model, forward_model  # noqa: E402
from ContrastModel.dataset.training import sliding_window_predict_3d  # noqa: E402
from experiments.generalization.mama_mia import adapt_channel_count, describe_case_input, discover_cases, load_case_17ch, selected_cohorts  # noqa: E402

EXTRA_METRIC_KEYS = ["threshold", "pred_voxels", "target_voxels", "pred_fraction", "target_fraction"]


MODEL_DIRS = {
    "attention_gated": "Attention-Gated-Networks-master",
    "deeplabv3plus": "DeepLabV3Plus-Pytorch-master",
    "emcad": "EMCAD-main",
    "hcrt": "HCRT-main",
    "mobile_uvit": "Mobile-U-ViT-main",
    "msdahnet": "Multi-Scale-Dual-Attention-Hybrid-Convolution-Network-main",
    "nnunet": "nnUNet-master",
    "pdpnet": "PDPNet-main",
    "plhn": "PLHN-main",
    "pytorch_unet": "Pytorch-UNet-master",
    "transunet": "TransUNet-main",
    "unetplusplus": "UNetPlusPlus-master",
}


def parse_args(default_model_key: str | None = None, default_model_dir: str | None = None):
    parser = argparse.ArgumentParser(description="Run MAMA-MIA cohort generalization for a contrast model.")
    parser.add_argument("--model-key", default=default_model_key, choices=sorted(MODEL_DIRS))
    parser.add_argument("--model-dir", default=default_model_dir)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mama-root", default=".")
    parser.add_argument("--cohorts", nargs="+", default=["ALL"])
    parser.add_argument("--mask-source", default="expert", choices=["expert"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--source-scale", default="breastdm_uint8", choices=["none", "breastdm_uint8", "per_phase_uint8"])
    parser.add_argument("--subtraction-mode", default="positive", choices=["positive", "signed"])
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--diagnose-cases", type=int, default=0)
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


def predict_2d_case(model, image: np.ndarray, device, threshold: float, batch_size: int, amp: bool) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    use_amp = bool(amp and torch.device(device).type == "cuda")
    with torch.inference_mode():
        for start in range(0, image.shape[1], batch_size):
            batch = torch.from_numpy(image[:, start : start + batch_size]).permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            autocast_ctx = torch.autocast(device_type="cuda") if use_amp else nullcontext()
            with autocast_ctx:
                logits, _ = forward_model(model, batch, None)
            target_shape = torch.zeros((batch.shape[0], 1, batch.shape[2], batch.shape[3]), device=device)
            logits = align_logits(logits, target_shape)
            pred = (torch.sigmoid(logits)[:, 0].detach().cpu().numpy() >= threshold)
            preds.extend(pred)
    return np.stack(preds, axis=0)


def predict_3d_case(model, cfg: Dict[str, Any], image: np.ndarray, device, threshold: float, amp: bool) -> np.ndarray:
    tensor = torch.from_numpy(image[None]).to(device=device, dtype=torch.float32)
    use_amp = bool(amp and torch.device(device).type == "cuda")
    autocast_ctx = torch.autocast(device_type="cuda") if use_amp else nullcontext()
    with autocast_ctx:
        logits = sliding_window_predict_3d(model, tensor, cfg, device)
    return (torch.sigmoid(logits)[0, 0].detach().cpu().numpy() >= threshold)


def _add_volume_stats(metric: dict, pred: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    metric["threshold"] = float(threshold)
    metric["pred_voxels"] = int(np.asarray(pred).astype(bool).sum())
    metric["target_voxels"] = int(np.asarray(target).astype(bool).sum())
    metric["pred_fraction"] = float(np.asarray(pred).astype(bool).mean())
    metric["target_fraction"] = float(np.asarray(target).astype(bool).mean())
    return metric


def run(args) -> Dict[str, float]:
    if not args.model_key:
        raise ValueError("--model-key is required when no wrapper default is provided.")
    if args.model_key == "nnunet":
        raise NotImplementedError(
            "nnUNet is not a direct torch.forward model in this repository. "
            "Use ContrastModel/nnUNet-master/test_17ch.py with an nnU-Net raw "
            "inference folder, or add a temporary nnU-Net export adapter. The "
            "dynamic in-memory MAMA-MIA runner supports the other contrast "
            "models and KPTA."
        )
    model_dir = Path(args.model_dir or (CONTRAST_ROOT / MODEL_DIRS[args.model_key])).resolve()
    config_path = Path(args.config or (model_dir / "configs" / "breastdm_17ch.yaml")).resolve()
    cfg = load_config(config_path, model_dir=model_dir, model_key=args.model_key)
    if args.threshold is not None:
        cfg["eval"]["threshold"] = float(args.threshold)
    threshold = float(cfg["eval"].get("threshold", 0.5))
    device = select_device(args.device)
    model = build_model(args.model_key, cfg, model_dir).to(device)
    checkpoint = Path(args.checkpoint or (Path(cfg["output"]["checkpoint_dir"]) / "best_model.pth")).resolve()
    load_checkpoint(model, checkpoint, device, strict=args.strict_load)

    cohorts = selected_cohorts(args.cohorts)
    cases = discover_cases(args.mama_root, cohorts, mask_source=args.mask_source)
    if args.limit_cases:
        cases = cases[: args.limit_cases]
    if not cases:
        raise FileNotFoundError(f"No MAMA-MIA cases found for cohorts={cohorts} under {args.mama_root}")

    image_size = tuple(int(v) for v in cfg["data"].get("image_size", [256, 256]))
    normalize = str(cfg["data"].get("normalize", "zscore"))
    input_channels = int(cfg["model"].get("input_channels", 17))
    mode = str(cfg["data"].get("mode", "2d")).lower()
    output_dir = Path(args.output_dir or (model_dir / "generalization_results" / "mama_mia")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, case in enumerate(tqdm(cases, desc=f"{args.model_key} MAMA-MIA")):
        image, mask = load_case_17ch(
            case,
            image_size,
            normalize=normalize,
            source_scale=args.source_scale,
            subtraction_mode=args.subtraction_mode,
        )
        if args.diagnose_cases and idx < args.diagnose_cases:
            print(f"[diagnose] {case.case_id}: {describe_case_input(image, mask)}")
        image = adapt_channel_count(image, input_channels)
        if mode == "3d":
            pred = predict_3d_case(model, cfg, image, device, threshold, args.amp)
        else:
            pred = predict_2d_case(model, image, device, threshold, args.batch_size, args.amp)
        target = mask.astype(bool)
        metric = compute_case_metrics(pred, target)
        _add_volume_stats(metric, pred, target, threshold)
        metric.update({"id": case.case_id, "cohort": case.cohort})
        rows.append(metric)
        if args.diagnose_cases and idx < args.diagnose_cases:
            print(
                f"[diagnose] {case.case_id} threshold={threshold:.3f}: "
                f"dice={metric['dice']:.6f}, precision={metric['precision']:.6f}, "
                f"sensitivity={metric['sensitivity']:.6f}, pred_fraction={metric['pred_fraction']:.6f}, "
                f"target_fraction={metric['target_fraction']:.6f}"
            )

    summary = summarize_metrics(rows)
    _write_metrics_with_cohort(output_dir / "metrics.csv", rows)
    write_summary(output_dir / "summary.txt", summary)
    by_cohort = output_dir / "summary_by_cohort.csv"
    with by_cohort.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["cohort", "n", *summary.keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cohort in cohorts:
            cohort_rows = [row for row in rows if row["cohort"] == cohort]
            if not cohort_rows:
                continue
            cohort_summary = summarize_metrics(cohort_rows)
            writer.writerow({"cohort": cohort, "n": len(cohort_rows), **cohort_summary})
    print(f"Saved metrics to {output_dir}")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")
    return summary


def _write_metrics_with_cohort(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "cohort", *METRIC_KEYS, *EXTRA_METRIC_KEYS])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def main(default_model_key: str | None = None, default_model_dir: str | None = None):
    return run(parse_args(default_model_key, default_model_dir))


if __name__ == "__main__":
    main()
