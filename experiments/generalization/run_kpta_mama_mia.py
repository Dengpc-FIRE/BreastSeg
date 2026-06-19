from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ContrastModel.dataset.metrics import METRIC_KEYS, compute_case_metrics, summarize_metrics, write_summary  # noqa: E402
from experiments.generalization.mama_mia import adapt_channel_count, discover_cases, load_case_17ch, neighbor_indices, selected_cohorts  # noqa: E402
from train.losses import unpack_model_output  # noqa: E402
from train.train_config import build_model_from_config, load_config, resolve_config_path  # noqa: E402


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
    return parser.parse_args()


def load_checkpoint(model, checkpoint: Path, device):
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state", state), strict=False)


def predict_kpta_25d_case(model, image: np.ndarray, num_slices: int, device, threshold: float, batch_size: int) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []
    with torch.inference_mode():
        batch = []
        for z in range(image.shape[1]):
            stack = np.stack([image[:, idx] for idx in neighbor_indices(z, image.shape[1], num_slices)], axis=0)
            batch.append(stack)
            if len(batch) == batch_size or z == image.shape[1] - 1:
                tensor = torch.from_numpy(np.stack(batch, axis=0)).to(device=device, dtype=torch.float32)
                logits, _ = unpack_model_output(model(tensor))
                pred = (torch.sigmoid(logits)[:, 0].detach().cpu().numpy() >= threshold)
                preds.extend(pred)
                batch = []
    return np.stack(preds, axis=0)


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
    load_checkpoint(model, Path(args.checkpoint), device)

    cohorts = selected_cohorts(args.cohorts)
    cases = discover_cases(args.mama_root, cohorts, mask_source=args.mask_source)
    if args.limit_cases:
        cases = cases[: args.limit_cases]
    if not cases:
        raise FileNotFoundError(f"No MAMA-MIA cases found for cohorts={cohorts} under {args.mama_root}")

    rows = []
    for case in tqdm(cases, desc="kpta_25d MAMA-MIA"):
        image, mask = load_case_17ch(case, image_size, normalize="none")
        image = adapt_channel_count(image, input_phases)
        pred = predict_kpta_25d_case(model, image, num_slices, device, args.threshold, args.batch_size)
        metric = compute_case_metrics(pred, mask.astype(bool))
        metric.update({"id": case.case_id, "cohort": case.cohort})
        rows.append(metric)

    summary = summarize_metrics(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_with_cohort(output_dir / "metrics.csv", rows)
    write_summary(output_dir / "summary.txt", summary)
    with (output_dir / "summary_by_cohort.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["cohort", "n", *summary.keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cohort in cohorts:
            cohort_rows = [row for row in rows if row["cohort"] == cohort]
            if cohort_rows:
                writer.writerow({"cohort": cohort, "n": len(cohort_rows), **summarize_metrics(cohort_rows)})
    print(f"Saved metrics to {output_dir}")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")


def _write_metrics_with_cohort(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "cohort", *METRIC_KEYS])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


if __name__ == "__main__":
    run(parse_args())
