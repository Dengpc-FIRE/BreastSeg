import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(THIS_DIR.parent))

from reproduce_msdahnet.datasets.breastdm_2d_dataset import (  # noqa: E402
    BreastDM2DDataset,
    collect_pairs,
    convert_processed_17ch_to_breastdm_dirs,
)
from reproduce_msdahnet.losses.dice_bce_loss import DiceBCELoss  # noqa: E402
from reproduce_msdahnet.metrics.segmentation_metrics import (  # noqa: E402
    METRIC_KEYS,
    compute_sample_metrics,
    global_pixel_metrics,
    mean_std,
    summarize_patient_metrics,
    summarize_slice_metrics,
)
from reproduce_msdahnet.models.msdahnet import build_msdahnet  # noqa: E402
from reproduce_msdahnet.utils.logger import CSVLogger, save_json  # noqa: E402
from reproduce_msdahnet.utils.seed import seed_everything  # noqa: E402
from reproduce_msdahnet.utils.split import make_folds, save_fold_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(THIS_DIR / "configs" / "msdahnet_breastdm_5fold.yaml"))
    parser.add_argument("--fold", type=int, default=None, help="Run a single fold for debugging.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for smoke tests.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)

    output_dir = Path(resolve_path(cfg["output"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(output_dir / "resolved_config.json"), cfg)
    write_reproduction_notes(output_dir, cfg)

    maybe_convert_data(cfg)
    pairs = collect_pairs(resolve_path(cfg["data"]["image_dir"]), resolve_path(cfg["data"]["mask_dir"]))
    image_paths = [pair[0] for pair in pairs]
    folds, patient_level_guaranteed = make_folds(
        image_paths,
        n_splits=int(cfg["cross_validation"]["n_splits"]),
        split_level=cfg["cross_validation"]["split_level"],
        seed=int(cfg["cross_validation"]["seed"]),
        shuffle=bool(cfg["cross_validation"]["shuffle"]),
    )
    if cfg["cross_validation"].get("save_splits", True):
        save_fold_files(str(output_dir / "splits"), folds, image_paths)

    if not patient_level_guaranteed:
        append_note(
            output_dir,
            "Patient-level split cannot be guaranteed because patient IDs are not available in the processed files.",
        )

    fold_indices = [args.fold] if args.fold is not None else list(range(len(folds)))
    fold_summaries = []
    for fold_idx in fold_indices:
        train_indices, val_indices = folds[fold_idx]
        fold_summary = run_fold(fold_idx, train_indices, val_indices, pairs, cfg, output_dir)
        fold_summaries.append(fold_summary)

    write_final_summary(output_dir, fold_summaries, cfg)


def run_fold(fold_idx: int, train_indices, val_indices, pairs, cfg, output_dir: Path) -> Dict:
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu")

    train_pairs = [pairs[i] for i in train_indices]
    val_pairs = [pairs[i] for i in val_indices]
    gray_to_rgb = bool(cfg["data"].get("gray_to_rgb", False))
    in_channels = int(cfg["model"]["in_channels"])
    if gray_to_rgb:
        append_note(
            output_dir,
            "Because the official code expects 3 input channels, grayscale BreaDM images are repeated into 3 channels.",
        )
    assert (gray_to_rgb and in_channels == 3) or ((not gray_to_rgb) and in_channels == 1), (
        "Use in_channels=1 with gray_to_rgb=false for scheme A, or "
        "in_channels=3 with gray_to_rgb=true for scheme B."
    )
    print(
        f"[Fold {fold_idx}] start | train={len(train_pairs)} val={len(val_pairs)} "
        f"device={device} in_channels={in_channels}",
        flush=True,
    )

    train_ds = BreastDM2DDataset(
        train_pairs,
        image_size=int(cfg["data"]["image_size"]),
        gray_to_rgb=gray_to_rgb,
        mask_threshold=float(cfg["data"]["mask_threshold"]),
    )
    val_ds = BreastDM2DDataset(
        val_pairs,
        image_size=int(cfg["data"]["image_size"]),
        gray_to_rgb=gray_to_rgb,
        mask_threshold=float(cfg["data"]["mask_threshold"]),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    print(
        f"[Fold {fold_idx}] data | train_samples={len(train_ds)} train_batches={len(train_loader)} "
        f"val_samples={len(val_ds)} val_batches={len(val_loader)} batch_size={int(cfg['train']['batch_size'])}",
        flush=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_msdahnet(in_channels=in_channels, num_classes=int(cfg["model"]["num_classes"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg["train"]["scheduler_mode"],
        patience=int(cfg["train"]["scheduler_patience"]),
    )
    loss_fn = DiceBCELoss(
        dice_weight=float(cfg["loss"]["dice_weight"]),
        bce_weight=float(cfg["loss"]["bce_weight"]),
    )

    logger = CSVLogger(
        str(fold_dir / f"training_log_fold{fold_idx}.csv"),
        [
            "epoch",
            "train_loss",
            "val_loss",
            "val_dice",
            "val_iou",
            "val_recall",
            "val_precision",
            "val_accuracy",
            "val_hd",
            "learning_rate",
        ],
    )
    best_dice = -1.0
    best_epoch = -1
    best_metrics = None

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, fold_idx, epoch)
        val_loss, val_eval = evaluate(model, val_loader, loss_fn, device, cfg, fold_idx, epoch)
        val_metrics = val_eval["slice_level"]
        scheduler.step(val_metrics["dice"])
        lr = optimizer.param_groups[0]["lr"]
        logger.write(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_recall": val_metrics["recall"],
                "val_precision": val_metrics["precision"],
                "val_accuracy": val_metrics["accuracy"],
                "val_hd": val_metrics["hd"],
                "learning_rate": lr,
            }
        )
        torch.save(model.state_dict(), fold_dir / f"last_model_fold{fold_idx}.pth")
        is_best = val_metrics["dice"] > best_dice
        if is_best:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            best_metrics = val_eval
            torch.save(model.state_dict(), fold_dir / f"best_model_fold{fold_idx}.pth")
        print(
            f"[Fold {fold_idx}] epoch {epoch:03d}/{int(cfg['train']['epochs'])} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} "
            f"recall={val_metrics['recall']:.4f} precision={val_metrics['precision']:.4f} "
            f"acc={val_metrics['accuracy']:.6f} hd={val_metrics['hd']:.4f} "
            f"lr={lr:.6g}{' best' if is_best else ''}",
            flush=True,
        )

    summary = {"fold": fold_idx, "best_epoch": best_epoch, **best_metrics["slice_level"]}
    save_json(str(fold_dir / f"fold_{fold_idx}_summary.json"), {"summary": summary, "all_metrics": best_metrics})
    print(
        f"[Fold {fold_idx}] done | best_epoch={best_epoch} "
        f"dice={summary['dice']:.4f} iou={summary['iou']:.4f} "
        f"recall={summary['recall']:.4f} precision={summary['precision']:.4f} "
        f"acc={summary['accuracy']:.6f} hd={summary['hd']:.4f}",
        flush=True,
    )
    return summary


def train_one_epoch(model, loader, optimizer, loss_fn, device, fold_idx: int = None, epoch: int = None) -> float:
    model.train()
    losses = []
    desc = "train" if fold_idx is None else f"fold{fold_idx} epoch{epoch:03d} train"
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, dtype=torch.float32)
        masks = batch["mask"].to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(model, loader, loss_fn, device, cfg, fold_idx: int = None, epoch: int = None) -> Dict:
    model.eval()
    losses, rows, preds, gts, paths = [], [], [], [], []
    threshold = float(cfg["eval"]["threshold"])
    hd_empty_value = float(cfg["eval"].get("hd_empty_value", cfg["data"]["image_size"]))
    desc = "eval" if fold_idx is None else f"fold{fold_idx} epoch{epoch:03d} val"
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].to(device, dtype=torch.float32)
            logits = model(images)
            loss = loss_fn(logits, masks)
            losses.append(float(loss.item()))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            mask_np = masks.detach().cpu().numpy()
            for idx in range(probs.shape[0]):
                pred = (probs[idx, 0] >= threshold).astype(np.uint8)
                gt = (mask_np[idx, 0] > 0.5).astype(np.uint8)
                rows.append(compute_sample_metrics(pred, gt, hd_empty_value=hd_empty_value))
                preds.append(pred)
                gts.append(gt)
                paths.append(batch["image_path"][idx])
    result = {
        "slice_level": summarize_slice_metrics(rows),
        "global_pixel_level": global_pixel_metrics(preds, gts, hd_empty_value=hd_empty_value),
        "patient_level": summarize_patient_metrics(rows, paths),
    }
    return float(np.mean(losses)) if losses else 0.0, result


def write_final_summary(output_dir: Path, fold_summaries: List[Dict], cfg) -> None:
    summary = {
        "folds": fold_summaries,
        "mean_std": mean_std(fold_summaries),
        "paper_reported": cfg.get("paper_reported", {}),
        "comparison_to_paper": {},
    }
    for key, value in cfg.get("paper_reported", {}).items():
        if key in summary["mean_std"]:
            summary["comparison_to_paper"][key] = {
                "paper": float(value),
                "reproduction_mean": summary["mean_std"][key]["mean"],
                "difference": summary["mean_std"][key]["mean"] - float(value),
            }
    save_json(str(output_dir / "summary_5fold.json"), summary)

    csv_path = output_dir / "summary_5fold.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "paper", "difference"])
        writer.writeheader()
        for key in METRIC_KEYS:
            writer.writerow(
                {
                    "metric": key,
                    "mean": summary["mean_std"][key]["mean"],
                    "std": summary["mean_std"][key]["std"],
                    "paper": cfg.get("paper_reported", {}).get(key, ""),
                    "difference": summary["comparison_to_paper"].get(key, {}).get("difference", ""),
                }
            )


def maybe_convert_data(cfg) -> None:
    if not cfg["data"].get("convert_from_processed_17ch", False):
        return
    output_root = resolve_path("./reproduce_msdahnet/BreastDM")
    convert_processed_17ch_to_breastdm_dirs(
        source_root=resolve_path(cfg["data"]["source_processed_17ch_dir"]),
        output_root=str(output_root),
    )


def write_reproduction_notes(output_dir: Path, cfg) -> None:
    notes = [
        "The reproduction is based on the official released implementation.",
        "Primary result is slice-level mean metrics, because the paper reports 2D image experiments.",
        "Scheme A is default: grayscale Mode=L, in_channels=1.",
        "Scheme B is supported by setting data.gray_to_rgb=true and model.in_channels=3.",
        "The original official train.py is not used for CV because it does not implement 5-fold patient-level validation.",
    ]
    (output_dir / "reproduction_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


def append_note(output_dir: Path, note: str) -> None:
    with (output_dir / "reproduction_notes.txt").open("a", encoding="utf-8") as f:
        f.write(note + "\n")


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
