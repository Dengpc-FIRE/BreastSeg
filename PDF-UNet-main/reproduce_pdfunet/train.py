import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

from reproduce_pdfunet.datasets.breastdm_2d_dataset import BreastDM2DDataset, collect_split_pairs  # noqa: E402
from reproduce_pdfunet.losses.segmentation_losses import build_loss  # noqa: E402
from reproduce_pdfunet.metrics.segmentation_metrics import (  # noqa: E402
    METRIC_KEYS,
    compute_sample_metrics,
    global_pixel_metrics,
    mean_std,
    summarize_patient_metrics,
    summarize_slice_metrics,
)
from reproduce_pdfunet.models.pdfunet_loader import build_pdfunet  # noqa: E402
from reproduce_pdfunet.utils.logger import CSVLogger, save_json  # noqa: E402
from reproduce_pdfunet.utils.seed import set_seed  # noqa: E402
from reproduce_pdfunet.utils.split import make_repeated_patient_splits, save_split_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF-UNet BreastDM adaptation baseline")
    parser.add_argument("--config", default="reproduce_pdfunet/configs/pdfunet_breastdm_focaltversky.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    set_seed(int(cfg["experiment"]["seed"]))

    output_dir = Path(resolve_path(cfg["output"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(resolve_path(args.config), output_dir / "config.yaml")
    write_notes(output_dir, cfg)

    if bool(cfg.get("cross_validation", {}).get("enabled", False)):
        summaries = run_cross_validation(cfg, output_dir)
        summary = {"folds": summaries, "mean_std": mean_std(summaries)}
        save_json(str(output_dir / "summary_cv.json"), summary)
        write_summary_csv(output_dir / "summary_cv.csv", summaries)
    else:
        summary = run_fixed_split(cfg, output_dir)
        save_json(str(output_dir / "metrics_test.json"), summary)


def run_fixed_split(cfg, output_dir: Path) -> Dict:
    train_pairs = collect_split_pairs(resolve_path(cfg["data"]["train_path"]))
    val_pairs = collect_split_pairs(resolve_path(cfg["data"]["val_path"]))
    test_pairs = collect_split_pairs(resolve_path(cfg["data"]["test_path"]))
    save_fixed_split_files(output_dir / "splits", train_pairs, val_pairs, test_pairs)
    run_dir = output_dir / "fixed_split"
    run_dir.mkdir(parents=True, exist_ok=True)
    return train_one_run(cfg, run_dir, train_pairs, val_pairs, test_pairs, run_name="fixed")


def run_cross_validation(cfg, output_dir: Path) -> List[Dict]:
    all_pairs = []
    for key in ("train_path", "val_path", "test_path"):
        split_path = resolve_path(cfg["data"][key])
        if Path(split_path).exists():
            all_pairs.extend(collect_split_pairs(split_path))
    image_paths = [p[0] for p in all_pairs]
    cv = cfg["cross_validation"]
    splits, patient_level = make_repeated_patient_splits(
        image_paths,
        n_splits=int(cv["n_splits"]),
        train_ratio=float(cv["train_ratio"]),
        val_ratio=float(cv["val_ratio"]),
        test_ratio=float(cv["test_ratio"]),
        seed=int(cv["seed"]),
    )
    if bool(cv.get("save_splits", True)):
        save_split_files(str(output_dir / "splits"), splits, image_paths)
    if not patient_level:
        append_note(output_dir, "Patient-level split cannot be guaranteed because patient IDs are not available in the processed files.")

    summaries = []
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        run_dir = output_dir / f"fold_{fold_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = train_one_run(
            cfg,
            run_dir,
            [all_pairs[i] for i in train_idx],
            [all_pairs[i] for i in val_idx],
            [all_pairs[i] for i in test_idx],
            run_name=f"fold_{fold_idx}",
        )
        summary["fold"] = fold_idx
        summaries.append(summary["test"]["slice_level"])
    return summaries


def train_one_run(cfg, run_dir: Path, train_pairs, val_pairs, test_pairs, run_name: str) -> Dict:
    device = torch.device(cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[{run_name}] start | train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)} device={device}", flush=True)
    train_loader = make_loader(train_pairs, cfg, shuffle=True)
    val_loader = make_loader(val_pairs, cfg, shuffle=False)
    test_loader = make_loader(test_pairs, cfg, shuffle=False)

    model = build_pdfunet(cfg).to(device)
    loss_fn = build_loss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg["train"]["scheduler_mode"],
        patience=int(cfg["train"]["scheduler_patience"]),
    )
    logger = CSVLogger(str(run_dir / "training_log.csv"), [
        "epoch", "train_loss", "val_loss", "val_dice", "val_iou",
        "val_recall", "val_precision", "val_accuracy", "val_hd95", "learning_rate",
    ])
    best_dice = -1.0
    best_epoch = -1
    best_val = None
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_eval = evaluate(model, val_loader, loss_fn, device, cfg)
        val_metrics = val_eval["slice_level"]
        scheduler.step(val_metrics["dice"])
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
            "val_accuracy": val_metrics["accuracy"],
            "val_hd95": val_metrics["hd95"],
            "learning_rate": lr,
        }
        logger.write(row)
        torch.save(model.state_dict(), run_dir / "last_model.pth")
        is_best = val_metrics["dice"] > best_dice
        if is_best:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            best_val = val_eval
            torch.save(model.state_dict(), run_dir / "best_model.pth")
        print(
            f"[{run_name}] epoch {epoch:03d}/{int(cfg['train']['epochs'])} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} "
            f"recall={val_metrics['recall']:.4f} precision={val_metrics['precision']:.4f} "
            f"acc={val_metrics['accuracy']:.6f} hd95={val_metrics['hd95']:.4f} "
            f"lr={lr:.6g}{' best' if is_best else ''}",
            flush=True,
        )

    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))
    test_loss, test_eval = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        cfg,
        save_predictions_dir=(run_dir / "predicted_masks") if bool(cfg["output"].get("save_predictions", True)) else None,
    )
    summary = {
        "run": run_name,
        "best_epoch": best_epoch,
        "best_validation": best_val,
        "test_loss": test_loss,
        "test": test_eval,
    }
    save_json(str(run_dir / "metrics_test.json"), summary)
    test_slice = test_eval["slice_level"]
    print(
        f"[{run_name}] test | best_epoch={best_epoch} dice={test_slice['dice']:.4f} "
        f"iou={test_slice['iou']:.4f} recall={test_slice['recall']:.4f} "
        f"precision={test_slice['precision']:.4f} acc={test_slice['accuracy']:.6f} "
        f"hd95={test_slice['hd95']:.4f}",
        flush=True,
    )
    return summary


def make_loader(pairs, cfg, shuffle: bool) -> DataLoader:
    data = cfg["data"]
    dataset = BreastDM2DDataset(
        pairs,
        image_size=int(data["image_size"]),
        input_mode=data["input_mode"],
        in_channels=int(data["in_channels"]),
        mask_threshold=float(data["mask_threshold"]),
        use_center_slice_only=bool(data.get("use_center_slice_only", True)),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )


def train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, dtype=torch.float32)
        masks = batch["mask"].to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(model, loader, loss_fn, device, cfg, save_predictions_dir: Path = None) -> Tuple[float, Dict]:
    model.eval()
    losses, rows, preds, gts, paths = [], [], [], [], []
    threshold = float(cfg["eval"]["threshold"])
    hd95_empty_value = float(cfg["eval"].get("hd95_empty_value", cfg["data"]["image_size"]))
    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].to(device, dtype=torch.float32)
            logits = model(images)
            loss = loss_fn(logits, masks)
            losses.append(float(loss.item()))
            probs = torch.sigmoid(logits).cpu().numpy()
            gt_np = masks.cpu().numpy()
            for idx in range(probs.shape[0]):
                pred = (probs[idx, 0] >= threshold).astype(np.uint8)
                gt = (gt_np[idx, 0] > 0).astype(np.uint8)
                rows.append(compute_sample_metrics(pred, gt, hd95_empty_value=hd95_empty_value))
                preds.append(pred)
                gts.append(gt)
                path = batch["image_path"][idx]
                paths.append(path)
                if save_predictions_dir is not None:
                    out_name = Path(path).stem + ".png"
                    cv2.imwrite(str(save_predictions_dir / out_name), (pred * 255).astype(np.uint8))
    result = {
        "slice_level": summarize_slice_metrics(rows),
        "patient_level": summarize_patient_metrics(rows, paths),
        "global_pixel_level": global_pixel_metrics(preds, gts),
    }
    return float(np.mean(losses)) if losses else 0.0, result


def save_fixed_split_files(output_dir: Path, train_pairs, val_pairs, test_pairs) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, pairs in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs)):
        with (output_dir / f"{name}.txt").open("w", encoding="utf-8") as f:
            for image_path, _ in pairs:
                f.write(image_path + "\n")


def write_summary_csv(path: Path, summaries: List[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold"] + METRIC_KEYS)
        writer.writeheader()
        for idx, row in enumerate(summaries):
            writer.writerow({"fold": idx, **row})


def write_notes(output_dir: Path, cfg) -> None:
    notes = [
        "Original PDF-UNet setting: breast ultrasound datasets.",
        "Our adaptation setting: BreastDM DCE-MRI baseline.",
        "The official PDF-UNet pyramid-dilated architecture is preserved.",
        "Model forward returns logits; sigmoid is applied only in loss/metric code.",
        f"Input mode: {cfg['data']['input_mode']}; image_size: {cfg['data']['image_size']}.",
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
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()

