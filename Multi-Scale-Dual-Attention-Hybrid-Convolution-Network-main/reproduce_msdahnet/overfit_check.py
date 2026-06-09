import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduce_msdahnet.datasets.breastdm_2d_dataset import BreastDM2DDataset, collect_pairs  # noqa: E402
from reproduce_msdahnet.metrics.segmentation_metrics import compute_sample_metrics  # noqa: E402
from reproduce_msdahnet.models.msdahnet import build_msdahnet  # noqa: E402
from reproduce_msdahnet.train_5fold import build_loss  # noqa: E402
from reproduce_msdahnet.utils.seed import seed_everything  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit a tiny subset to verify MSDAHNet data/model/loss plumbing.")
    parser.add_argument("--config", default="reproduce_msdahnet/configs/msdahnet_breastdm_5fold.yaml")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["experiment"]["seed"]))
    pairs = collect_pairs(resolve_path(cfg["data"]["image_dir"]), resolve_path(cfg["data"]["mask_dir"]))
    pairs = pairs[: min(args.samples, len(pairs))]
    dataset = BreastDM2DDataset(
        pairs,
        image_size=int(cfg["data"]["image_size"]),
        gray_to_rgb=bool(cfg["data"].get("gray_to_rgb", False)),
        mask_threshold=float(cfg["data"]["mask_threshold"]),
    )
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True, num_workers=0)
    eval_loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False, num_workers=0)

    device = torch.device(cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    model = build_msdahnet(
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
    ).to(device)
    loss_fn = build_loss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr or cfg["train"]["lr"]))
    print(f"overfit_samples={len(dataset)} device={device} loss={cfg['loss'].get('implementation', 'official')}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            dice, pred_pos, prob_mean = evaluate_same_set(model, eval_loader, device)
            print(
                f"epoch={epoch:03d} loss={float(np.mean(losses)):.6f} "
                f"dice={dice:.4f} pred_pos={pred_pos:.5f} prob_mean={prob_mean:.5f}",
                flush=True,
            )


def evaluate_same_set(model, loader, device):
    model.eval()
    rows, pred_pos, prob_means = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32)
            masks = batch["mask"].cpu().numpy()
            probs = torch.sigmoid(model(images)).cpu().numpy()
            prob_means.append(float(probs.mean()))
            for idx in range(probs.shape[0]):
                pred = (probs[idx, 0] >= 0.5).astype(np.uint8)
                gt = (masks[idx, 0] > 0.5).astype(np.uint8)
                rows.append(compute_sample_metrics(pred, gt))
                pred_pos.append(float(pred.mean()))
    return float(np.mean([row["dice"] for row in rows])), float(np.mean(pred_pos)), float(np.mean(prob_means))


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
