import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_config import build_model_from_config, load_config, resolve_config_path  # noqa: E402


MODES = [
    "full_phases",
    "missing_early_post",
    "missing_late_post",
    "missing_subtraction",
    "only_pre_post",
    "only_pre_subtraction",
    "random_missing_phase",
]


class NpySegDataset(Dataset):
    def __init__(self, split_path: str):
        self.data_dir = Path(split_path) / "data"
        self.gt_dir = Path(split_path) / "GT"
        self.files = sorted([p.name for p in self.data_dir.glob("*.npy")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        arr = np.load(str(self.data_dir / name)).astype(np.float32) / 255.0
        image = torch.from_numpy(arr.transpose(2, 0, 1))
        mask_path = self.gt_dir / name.replace(".npy", ".png")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(arr.shape[:2], dtype=np.uint8)
        mask = cv2.resize(mask, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return image, mask, name


def get_phase_indices(config: Dict):
    phase_cfg = config.get("model", {}).get("phase_indices", {})
    return {
        "pre": int(phase_cfg.get("pre", 0)),
        "post": list(phase_cfg.get("post", [])),
        "subtraction": list(phase_cfg.get("subtraction", [])),
    }


def apply_missing_mode(images: torch.Tensor, mode: str, phase_indices: Dict):
    images = images.clone()
    post = [idx for idx in phase_indices["post"] if 0 <= idx < images.shape[1]]
    sub = [idx for idx in phase_indices["subtraction"] if 0 <= idx < images.shape[1]]
    t = max(len(post), len(sub), 1)
    phase_mask = images.new_ones((images.shape[0], t))

    if mode == "full_phases":
        return images, phase_mask
    if mode == "missing_subtraction":
        zero_channels(images, sub)
        return images, phase_mask
    if mode == "only_pre_post":
        zero_channels(images, sub)
        return images, phase_mask
    if mode == "only_pre_subtraction":
        zero_channels(images, post)
        return images, phase_mask
    if mode == "missing_early_post":
        drop_phase(images, phase_mask, post, sub, 0)
        return images, phase_mask
    if mode == "missing_late_post":
        drop_phase(images, phase_mask, post, sub, t - 1)
        return images, phase_mask
    if mode == "random_missing_phase":
        for b in range(images.shape[0]):
            drop = torch.rand(t, device=images.device) < 0.3
            if (~drop).sum() < 1:
                drop[torch.randint(0, t, (1,), device=images.device)] = False
            for phase_idx in torch.nonzero(drop, as_tuple=False).flatten().tolist():
                drop_phase(images[b : b + 1], phase_mask[b : b + 1], post, sub, phase_idx)
        return images, phase_mask
    raise ValueError(f"Unknown missing phase mode: {mode}")


def zero_channels(images: torch.Tensor, channels: Iterable[int]) -> None:
    for channel in channels:
        images[:, channel : channel + 1] = 0


def drop_phase(images: torch.Tensor, phase_mask: torch.Tensor, post: List[int], sub: List[int], phase_idx: int) -> None:
    phase_mask[:, phase_idx] = 0
    if post:
        ch = post[min(phase_idx, len(post) - 1)]
        images[:, ch : ch + 1] = 0
    if sub:
        ch = sub[min(phase_idx, len(sub) - 1)]
        images[:, ch : ch + 1] = 0


def batch_metrics(logits: torch.Tensor, target: torch.Tensor):
    pred = (torch.sigmoid(logits) > 0.5).float()
    target = target.float()
    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
    iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    sensitivity = (tp + 1e-7) / (tp + fn + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    return dice, iou, sensitivity, precision


def evaluate_mode(model, loader, device, mode: str, phase_indices: Dict):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc=mode, leave=False):
            images = images.to(device)
            masks = masks.to(device)
            images, phase_mask = apply_missing_mode(images, mode, phase_indices)
            output = model(images, phase_mask=phase_mask, return_dict=True)
            logits = output["seg_logits"] if isinstance(output, dict) else output
            all_metrics.append(torch.stack(batch_metrics(logits, masks), dim=1).cpu())
    if not all_metrics:
        return {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "precision": 0.0}
    values = torch.cat(all_metrics, dim=0).mean(dim=0)
    return {
        "dice": float(values[0]),
        "iou": float(values[1]),
        "sensitivity": float(values[2]),
        "precision": float(values[3]),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate KPR-Net missing-phase robustness.")
    parser.add_argument("--config", type=str, default="configs/kpr_net.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--missing_phase_mode", type=str, default="all", choices=["all"] + MODES)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(resolve_config_path(args.config))
    split_path = args.split_path or config.get("train", {}).get("test_path")
    loader = DataLoader(NpySegDataset(split_path), batch_size=args.batch_size, shuffle=False)
    model = build_model_from_config(config).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    phase_indices = get_phase_indices(config)

    modes = MODES if args.missing_phase_mode == "all" else [args.missing_phase_mode]
    results = {mode: evaluate_mode(model, loader, args.device, mode, phase_indices) for mode in modes}
    if "full_phases" in results:
        full_dice = results["full_phases"]["dice"]
        for mode, metrics in results.items():
            metrics["dice_drop"] = full_dice - metrics["dice"]
    if len(results) > 1:
        dice_values = [metrics["dice"] for metrics in results.values()]
        results["summary"] = {
            "mean_dice": float(np.mean(dice_values)),
            "worst_case_dice": float(np.min(dice_values)),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
