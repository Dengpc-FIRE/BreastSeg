import argparse
import csv
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduce_msdahnet.datasets.breastdm_2d_dataset import (  # noqa: E402
    BreastDM2DDataset,
    collect_split_pairs,
    convert_processed_17ch_to_fixed_split_dirs,
)
from reproduce_msdahnet.metrics.segmentation_metrics import (  # noqa: E402
    compute_sample_metrics,
    global_pixel_metrics,
    summarize_patient_metrics,
    summarize_slice_metrics,
)
from reproduce_msdahnet.models.msdahnet import build_msdahnet  # noqa: E402
from reproduce_msdahnet.train_5fold import build_loss, describe_loss  # noqa: E402
from reproduce_msdahnet.utils.logger import CSVLogger, save_json  # noqa: E402
from reproduce_msdahnet.utils.seed import seed_everything  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MSDAHNet on fixed BreastDM train/val/test split.")
    parser.add_argument("--config", default="reproduce_msdahnet/configs/msdahnet_breastdm_fixed_split.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)

    output_dir = Path(resolve_path(cfg["output"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(output_dir / "resolved_config.json"), cfg)
    write_notes(output_dir, cfg)
    maybe_convert_data(cfg)

    train_pairs = collect_split_pairs(resolve_path(cfg["data"]["train_path"]))
    val_pairs = collect_split_pairs(resolve_path(cfg["data"]["val_path"]))
    test_pairs = collect_split_pairs(resolve_path(cfg["data"]["test_path"]))
    save_split_files(output_dir / "splits", train_pairs, val_pairs, test_pairs)
    summary = run_fixed_split(train_pairs, val_pairs, test_pairs, cfg, output_dir)
    save_json(str(output_dir / "metrics_test.json"), summary)


def run_fixed_split(train_pairs, val_pairs, test_pairs, cfg, output_dir: Path) -> Dict:
    device = torch.device(cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    gray_to_rgb = bool(cfg["data"].get("gray_to_rgb", False))
    in_channels = int(cfg["model"]["in_channels"])
    assert (gray_to_rgb and in_channels == 3) or ((not gray_to_rgb) and in_channels == 1), (
        "Use in_channels=1 with gray_to_rgb=false, or in_channels=3 with gray_to_rgb=true."
    )
    print(
        f"[Fixed] start | train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)} "
        f"device={device} in_channels={in_channels}",
        flush=True,
    )

    train_loader = make_loader(train_pairs, cfg, shuffle=True)
    val_loader = make_loader(val_pairs, cfg, shuffle=False)
    test_loader = make_loader(test_pairs, cfg, shuffle=False)
    print(
        f"[Fixed] data | train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"test_batches={len(test_loader)} batch_size={int(cfg['train']['batch_size'])}",
        flush=True,
    )

    model = build_msdahnet(in_channels=in_channels, num_classes=int(cfg["model"]["num_classes"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg["train"]["scheduler_mode"],
        patience=int(cfg["train"]["scheduler_patience"]),
    )
    loss_fn = build_loss(cfg)
    print(f"[Fixed] loss | {describe_loss(cfg)}", flush=True)

    logger = CSVLogger(
        str(output_dir / "training_log.csv"),
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
            "val_gt_positive_ratio",
            "val_pred_positive_ratio",
            "val_prob_mean",
            "learning_rate",
        ],
    )
    best_dice = -1.0
    best_epoch = -1
    best_val = None
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        val_loss, val_eval = evaluate(model, val_loader, loss_fn, device, cfg, epoch=epoch)
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
                "val_gt_positive_ratio": val_eval["debug"]["gt_positive_ratio"],
                "val_pred_positive_ratio": val_eval["debug"]["pred_positive_ratio"],
                "val_prob_mean": val_eval["debug"]["prob_mean"],
                "learning_rate": lr,
            }
        )
        torch.save(model.state_dict(), output_dir / "last_model.pth")
        is_best = val_metrics["dice"] > best_dice
        if is_best:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            best_val = val_eval
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        print(
            f"[Fixed] epoch {epoch:03d}/{int(cfg['train']['epochs'])} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} "
            f"recall={val_metrics['recall']:.4f} precision={val_metrics['precision']:.4f} "
            f"acc={val_metrics['accuracy']:.6f} hd={val_metrics['hd']:.4f} "
            f"gt_pos={val_eval['debug']['gt_positive_ratio']:.5f} "
            f"pred_pos={val_eval['debug']['pred_positive_ratio']:.5f} "
            f"prob_mean={val_eval['debug']['prob_mean']:.5f} "
            f"lr={lr:.6g}{' best' if is_best else ''}",
            flush=True,
        )

    model.load_state_dict(torch.load(output_dir / "best_model.pth", map_location=device))
    best_threshold, threshold_table = find_best_threshold(model, val_loader, device, cfg)
    cfg["eval"]["threshold"] = float(best_threshold)
    save_json(str(output_dir / "threshold_selection.json"), {"best_threshold": best_threshold, "val_thresholds": threshold_table})
    print(f"[Fixed] threshold | selected={best_threshold:.2f} by validation Dice", flush=True)
    pred_dir = output_dir / "predicted_masks" if bool(cfg["output"].get("save_predictions", True)) else None
    test_loss, test_eval = evaluate(model, test_loader, loss_fn, device, cfg, save_predictions_dir=pred_dir)
    test_metrics = test_eval["slice_level"]
    print(
        f"[Fixed] test | best_epoch={best_epoch} test_loss={test_loss:.6f} "
        f"dice={test_metrics['dice']:.4f} iou={test_metrics['iou']:.4f} "
        f"recall={test_metrics['recall']:.4f} precision={test_metrics['precision']:.4f} "
        f"acc={test_metrics['accuracy']:.6f} hd={test_metrics['hd']:.4f}",
        flush=True,
    )
    return {
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "threshold_selection": threshold_table,
        "best_validation": best_val,
        "test_loss": test_loss,
        "test": test_eval,
    }


def make_loader(pairs, cfg, shuffle: bool) -> DataLoader:
    ds = BreastDM2DDataset(
        pairs,
        image_size=int(cfg["data"]["image_size"]),
        gray_to_rgb=bool(cfg["data"].get("gray_to_rgb", False)),
        mask_threshold=float(cfg["data"]["mask_threshold"]),
        input_mode=cfg["data"].get("input_mode", "single_channel_pre"),
        channel_index=cfg["data"].get("channel_index", None),
    )
    return DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch: int) -> float:
    model.train()
    losses = []
    for batch in tqdm(loader, desc=f"fixed epoch{epoch:03d} train", leave=False):
        images = batch["image"].to(device, dtype=torch.float32)
        masks = batch["mask"].to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(model, loader, loss_fn, device, cfg, epoch: int = None, save_predictions_dir: Path = None):
    model.eval()
    losses, rows, preds, gts, paths, prob_means = [], [], [], [], [], []
    threshold = float(cfg["eval"]["threshold"])
    hd_empty_value = float(cfg["eval"].get("hd_empty_value", cfg["data"]["image_size"]))
    desc = "fixed test" if epoch is None else f"fixed epoch{epoch:03d} val"
    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].to(device, dtype=torch.float32)
            logits = model(images)
            losses.append(float(loss_fn(logits, masks).item()))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            mask_np = masks.detach().cpu().numpy()
            prob_means.append(float(probs.mean()))
            for idx in range(probs.shape[0]):
                pred = (probs[idx, 0] >= threshold).astype(np.uint8)
                gt = (mask_np[idx, 0] > 0.5).astype(np.uint8)
                rows.append(compute_sample_metrics(pred, gt, hd_empty_value=hd_empty_value))
                preds.append(pred)
                gts.append(gt)
                image_path = batch["image_path"][idx]
                paths.append(image_path)
                if save_predictions_dir is not None:
                    cv2.imwrite(str(save_predictions_dir / f"{Path(image_path).stem}.png"), (pred * 255).astype(np.uint8))
    return float(np.mean(losses)) if losses else 0.0, {
        "slice_level": summarize_slice_metrics(rows),
        "global_pixel_level": global_pixel_metrics(preds, gts, hd_empty_value=hd_empty_value),
        "patient_level": summarize_patient_metrics(rows, paths),
        "debug": {
            "gt_positive_ratio": float(np.mean([gt.mean() for gt in gts])) if gts else 0.0,
            "pred_positive_ratio": float(np.mean([pred.mean() for pred in preds])) if preds else 0.0,
            "prob_mean": float(np.mean(prob_means)) if prob_means else 0.0,
        },
    }


def find_best_threshold(model, loader, device, cfg):
    thresholds = cfg["eval"].get("threshold_search", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    threshold_values = [float(t) for t in thresholds]
    probs, masks = collect_probs_and_masks(model, loader, device)
    table = []
    best_threshold = threshold_values[0]
    best_dice = -1.0
    hd_empty_value = float(cfg["eval"].get("hd_empty_value", cfg["data"]["image_size"]))
    for threshold in threshold_values:
        rows = []
        pred_pos = []
        for prob, mask in zip(probs, masks):
            pred = (prob >= threshold).astype(np.uint8)
            gt = (mask > 0.5).astype(np.uint8)
            rows.append(compute_sample_metrics(pred, gt, hd_empty_value=hd_empty_value))
            pred_pos.append(float(pred.mean()))
        metrics = summarize_slice_metrics(rows)
        item = {
            "threshold": threshold,
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "pred_positive_ratio": float(np.mean(pred_pos)),
        }
        table.append(item)
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_threshold = threshold
    return best_threshold, table


def collect_probs_and_masks(model, loader, device):
    model.eval()
    probs, masks = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="fixed val threshold", leave=False):
            images = batch["image"].to(device, dtype=torch.float32)
            batch_probs = torch.sigmoid(model(images)).cpu().numpy()
            batch_masks = batch["mask"].cpu().numpy()
            for idx in range(batch_probs.shape[0]):
                probs.append(batch_probs[idx, 0])
                masks.append(batch_masks[idx, 0])
    return probs, masks


def maybe_convert_data(cfg) -> None:
    if not cfg["data"].get("convert_from_processed_17ch", False):
        return
    convert_processed_17ch_to_fixed_split_dirs(
        source_root=resolve_path(cfg["data"]["source_processed_17ch_dir"]),
        output_root=resolve_path(cfg["data"]["processed_fixed_root"]),
        output_format=cfg["data"].get("processed_output_format", "npy"),
    )


def save_split_files(output_dir: Path, train_pairs, val_pairs, test_pairs) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, pairs in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
        with (output_dir / f"{name}.txt").open("w", encoding="utf-8") as f:
            for image_path, _ in pairs:
                f.write(image_path + "\n")


def write_notes(output_dir: Path, cfg) -> None:
    notes = [
        "The reproduction is based on the official released implementation.",
        "This run uses the fixed BreastDM train/val/test split instead of 5-fold cross-validation.",
        "Validation is used only for best checkpoint selection; test is evaluated once with the best validation checkpoint.",
        f"Loss implementation: {describe_loss(cfg)}.",
    ]
    (output_dir / "reproduction_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


def resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


def load_config(path: str):
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
