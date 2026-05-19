import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_config import build_model_from_config, load_config, resolve_config_path  # noqa: E402


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


def norm_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    denom = x.max()
    if denom > 0:
        x = x / denom
    return (x * 255).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Save KPTA-Net visualizations.")
    parser.add_argument("--config", type=str, default="configs/kpta_net.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(resolve_config_path(args.config))
    split_path = args.split_path or config.get("train", {}).get("val_path")
    output_dir = Path(args.output_dir or f"outputs/visualizations/{Path(args.config).stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(config).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    loader = DataLoader(NpySegDataset(split_path), batch_size=1, shuffle=False)
    with torch.no_grad():
        for idx, (image, mask, name) in enumerate(loader):
            if idx >= args.num_images:
                break
            image = image.to(args.device)
            output = model(image, return_dict=True)
            pred = (torch.sigmoid(output["seg_logits"])[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            gt = (mask[0, 0].numpy() > 0.5).astype(np.uint8) * 255
            base = norm_u8(image[0, 0].cpu().numpy())

            panels = [cv2.cvtColor(base, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR), cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)]
            if "kinetic_maps" in output:
                panels.append(cv2.cvtColor(norm_u8(output["kinetic_maps"][0, 0].cpu().numpy()), cv2.COLOR_GRAY2BGR))
            if output.get("attention_maps"):
                attn = output["attention_maps"][0][0, :, 0].mean(dim=0).cpu().numpy()
                panels.append(cv2.cvtColor(norm_u8(attn), cv2.COLOR_GRAY2BGR))
            if "uncertainty_logits" in output:
                uncertainty = torch.sigmoid(output["uncertainty_logits"])[0, 0].cpu().numpy()
                panels.append(cv2.cvtColor(norm_u8(uncertainty), cv2.COLOR_GRAY2BGR))
            if "boundary_logits" in output:
                boundary = torch.sigmoid(output["boundary_logits"])[0, 0].cpu().numpy()
                panels.append(cv2.cvtColor(norm_u8(boundary), cv2.COLOR_GRAY2BGR))

            cv2.imwrite(str(output_dir / f"{Path(name[0]).stem}.png"), np.hstack(panels))

    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
