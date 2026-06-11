"""Configuration-driven training entry point for KPTA segmentation models."""

import argparse
import os
import sys
from pathlib import Path

ABLATION_FLAGS = (
    "disable_kinetic_maps",
    "disable_pseudo_kinetic_maps",
    "disable_pixelwise_phase_attention",
    "disable_kinetic_raw_fusion",
    "disable_uncertainty_refinement",
    "disable_boundary_head",
    "disable_uncertainty_head",
    "disable_attention_smooth_loss",
    "disable_slice_context",
    "disable_transformer_bottleneck",
)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Train KPTANet or KPTA25DNet from a YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="KPTA YAML config, for example configs/kpta_net.yaml.",
    )
    for flag in ABLATION_FLAGS:
        parser.add_argument(f"--{flag}", action="store_true")
    return parser


EARLY_ARGS = (
    build_argument_parser().parse_args()
    if __name__ == "__main__"
    else None
)

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.losses import unpack_model_output  # noqa: E402
from train.train_config import (  # noqa: E402
    build_loss_from_config,
    build_model_from_config,
    load_config,
    resolve_config_path,
)


class BreastDM2DDataset(Dataset):
    """Load KPTA 2D samples stored as [H,W,C] NumPy arrays."""

    def __init__(self, data_dir: str, gt_dir: str, img_size: int = 256) -> None:
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        self.ids = (
            sorted(f for f in os.listdir(data_dir) if f.endswith(".npy"))
            if os.path.isdir(data_dir)
            else []
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        file_name = self.ids[index]
        data = np.load(os.path.join(self.data_dir, file_name))
        if data.ndim != 3:
            raise ValueError(
                f"Expected 2D KPTA input [H,W,C], got {data.shape} for {file_name}"
            )
        image = torch.from_numpy(
            (data.astype(np.float32) / 255.0).transpose(2, 0, 1)
        )
        mask = _load_mask(
            os.path.join(self.gt_dir, file_name.replace(".npy", ".png")),
            (self.img_size, self.img_size),
        )
        return image, mask, file_name


class BreastDM25DDataset(Dataset):
    """Load KPTA 2.5D samples stored as [K,T,H,W] NumPy arrays."""

    def __init__(self, data_dir: str, gt_dir: str) -> None:
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.ids = (
            sorted(f for f in os.listdir(data_dir) if f.endswith(".npy"))
            if os.path.isdir(data_dir)
            else []
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        file_name = self.ids[index]
        data = np.load(os.path.join(self.data_dir, file_name))
        if data.ndim != 4:
            raise ValueError(
                f"Expected 2.5D KPTA input [K,T,H,W], got {data.shape} "
                f"for {file_name}"
            )
        image = torch.from_numpy(data.astype(np.float32))
        mask = _load_mask(
            os.path.join(self.gt_dir, file_name.replace(".npy", ".png")),
            data.shape[-2:],
        )
        return image, mask, file_name


def _load_mask(path: str, shape) -> torch.Tensor:
    height, width = int(shape[0]), int(shape[1])
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        mask = np.zeros((height, width), dtype=np.uint8)
    elif mask.shape != (height, width):
        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    binary = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0)


def build_dataset(
    split_path: str,
    dataset_type: str = "breastdm_2d",
    img_size: int = 256,
) -> Dataset:
    data_dir = os.path.join(split_path, "data")
    gt_dir = os.path.join(split_path, "GT")
    if dataset_type in {"breastdm_25d", "25d", "kpta_25d"}:
        return BreastDM25DDataset(data_dir, gt_dir)
    if dataset_type in {"breastdm_2d", "2d", "kpta_2d"}:
        return BreastDM2DDataset(data_dir, gt_dir, img_size=img_size)
    raise ValueError(f"Unsupported dataset.type: {dataset_type!r}")


def evaluate(model, loader, device, desc: str):
    model.eval()
    totals = {"dice": [], "iou": [], "sensitivity": [], "precision": []}
    with torch.inference_mode():
        for images, masks, _ in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits, _ = unpack_model_output(model(images))
            predictions = (torch.sigmoid(logits) > 0.5).long()
            tp, fp, fn, tn = smp.metrics.get_stats(
                predictions,
                masks.long(),
                mode="binary",
            )
            totals["dice"].append(
                smp.metrics.f1_score(
                    tp,
                    fp,
                    fn,
                    tn,
                    reduction="micro-imagewise",
                ).item()
            )
            totals["iou"].append(
                smp.metrics.iou_score(
                    tp,
                    fp,
                    fn,
                    tn,
                    reduction="micro-imagewise",
                ).item()
            )
            totals["sensitivity"].append(
                (tp.float() / (tp.float() + fn.float() + 1e-7)).mean().item()
            )
            totals["precision"].append(
                (tp.float() / (tp.float() + fp.float() + 1e-7)).mean().item()
            )
    if not totals["dice"]:
        raise ValueError(f"{desc} dataset is empty.")
    return tuple(float(np.mean(totals[key])) for key in totals)


def build_scheduler(optimizer, config, epochs: int):
    scheduler_cfg = config.get("scheduler", {})
    if not bool(scheduler_cfg.get("enabled", True)):
        return None, False, "disabled"
    name = str(scheduler_cfg.get("name", "cosine")).lower()
    if name in {"plateau", "reduce_on_plateau", "reducelronplateau"}:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 10)),
            threshold=float(scheduler_cfg.get("threshold", 1e-4)),
            threshold_mode=str(scheduler_cfg.get("threshold_mode", "abs")),
            cooldown=int(scheduler_cfg.get("cooldown", 0)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        )
        return scheduler, True, name
    if name in {"cosine", "cosine_annealing"}:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("t_max", epochs)),
            eta_min=float(scheduler_cfg.get("min_lr", 0.0)),
        )
        return scheduler, False, name
    raise ValueError(f"Unsupported scheduler.name: {name!r}")


def parse_args():
    return build_argument_parser().parse_args()


def main() -> int:
    args = EARLY_ARGS if EARLY_ARGS is not None else parse_args()
    config_path = resolve_config_path(args.config)
    if config_path is None or not Path(config_path).is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    config = load_config(config_path)
    config.setdefault("ablation", {})
    for key, value in vars(args).items():
        if key.startswith("disable_") and value:
            config["ablation"][key] = True

    train_cfg = config.get("train", {})
    required_train_keys = (
        "train_path",
        "val_path",
        "test_path",
        "output_path",
        "epochs",
        "batch_size",
        "lr",
    )
    missing = [key for key in required_train_keys if key not in train_cfg]
    if missing:
        raise ValueError(f"Missing train config keys: {', '.join(missing)}")

    dataset_cfg = config.get("dataset", {})
    dataset_type = dataset_cfg.get("type", "breastdm_2d")
    img_size = int(dataset_cfg.get("img_size", 256))
    train_dataset = build_dataset(
        train_cfg["train_path"],
        dataset_type,
        img_size,
    )
    val_dataset = build_dataset(train_cfg["val_path"], dataset_type, img_size)
    test_dataset = build_dataset(train_cfg["test_path"], dataset_type, img_size)
    if not train_dataset:
        raise ValueError(f"No training samples found in {train_cfg['train_path']}")

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 4))
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("eval_batch_size", 4)),
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(train_cfg.get("eval_batch_size", 4)),
        shuffle=False,
        **loader_options,
    )

    output_path = Path(train_cfg["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(config).to(device)
    loss_fn = build_loss_from_config(config)
    epochs = int(train_cfg["epochs"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler, scheduler_uses_metric, scheduler_name = build_scheduler(
        optimizer,
        config,
        epochs,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    print(
        f"Config: {config_path}\n"
        f"Model: {config['model']['name']}\n"
        f"Device: {device}\n"
        f"Samples: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={len(test_dataset)}\n"
        f"Scheduler: {scheduler_name}"
    )

    best_dice = -1.0
    best_model_path = output_path / "best_model.pth"
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, masks, _ in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                output = model(images)
                logits, _ = unpack_model_output(output)
                loss = loss_fn(output, masks, images=images)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite KPTA loss: "
                    f"loss={loss.detach().float().item()}, "
                    f"logit_range=({logits.detach().float().amin().item():.4g}, "
                    f"{logits.detach().float().amax().item():.4g})"
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device, "Validation")
        if scheduler is not None:
            if scheduler_uses_metric:
                scheduler.step(val_metrics[0])
            else:
                scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch} | loss={running_loss / len(train_loader):.4f} | "
            f"val_dice={val_metrics[0]:.4f} | val_iou={val_metrics[1]:.4f} | "
            f"val_sens={val_metrics[2]:.4f} | val_prec={val_metrics[3]:.4f} | "
            f"lr={current_lr:.6g}"
        )
        if val_metrics[0] > best_dice:
            best_dice = val_metrics[0]
            torch.save(model.state_dict(), best_model_path)
            test_metrics = evaluate(model, test_loader, device, "Testing best")
            print(
                f"[test_dice] epoch={epoch} val_dice={best_dice:.4f} "
                f"test_dice={test_metrics[0]:.4f} "
                f"test_iou={test_metrics[1]:.4f} "
                f"test_sens={test_metrics[2]:.4f} "
                f"test_prec={test_metrics[3]:.4f}"
            )

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_metrics = evaluate(model, test_loader, device, "Final testing")
    print(
        "Final test | "
        f"dice={final_metrics[0]:.4f} | iou={final_metrics[1]:.4f} | "
        f"sensitivity={final_metrics[2]:.4f} | "
        f"precision={final_metrics[3]:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
