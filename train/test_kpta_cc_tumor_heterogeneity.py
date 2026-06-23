from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ContrastModel.dataset.metrics import METRIC_KEYS, compute_case_metrics, summarize_metrics, write_summary  # noqa: E402
from train.losses import unpack_model_output  # noqa: E402
from train.train_config import build_model_from_config, checkpoint_name_from_config, load_config, resolve_config_path  # noqa: E402
from train.train_kpta import build_dataset  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "kpta_25d_net_cc_tumor_heterogeneity_17ch-v2.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test KPTA on CC-Tumor-Heterogeneity.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", *METRIC_KEYS])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def main() -> int:
    args = parse_args()
    config_path = Path(resolve_config_path(args.config))
    cfg = load_config(str(config_path))
    train_cfg = cfg["train"]
    dataset_cfg = cfg.get("dataset", {})
    dataset = build_dataset(
        train_cfg["test_path"],
        dataset_cfg.get("type", "breastdm_25d"),
        int(dataset_cfg.get("img_size", 256)),
        input_phase_indices=dataset_cfg.get("input_phase_indices"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(train_cfg.get("eval_batch_size", 4)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(cfg).to(device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(train_cfg["output_path"]) / checkpoint_name_from_config(str(config_path))
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state", state), strict=False)
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, masks, names in tqdm(loader, desc="KPTA CC-Tumor-Heterogeneity test"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits, _ = unpack_model_output(model(images))
            preds = torch.sigmoid(logits)[:, 0].detach().cpu().numpy() >= args.threshold
            targets = masks[:, 0].detach().cpu().numpy() > 0
            for pred, target, name in zip(preds, targets, names):
                metric = compute_case_metrics(pred, target)
                metric["id"] = str(name).replace(".npy", "")
                rows.append(metric)
    summary = summarize_metrics(rows)
    output_dir = Path(args.output_dir or (Path(train_cfg["output_path"]) / "test_cc_tumor_heterogeneity"))
    write_metrics(output_dir / "metrics.csv", rows)
    write_summary(output_dir / "summary.txt", summary)
    for key in ["mean_dice", "mean_iou", "mean_hd95", "mean_sensitivity", "mean_precision", "mean_accuracy"]:
        print(f"{key}: {summary[key]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

