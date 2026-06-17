import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(THIS_DIR.parent))

from reproduce_msdahnet.datasets.breastdm_2d_dataset import BreastDM2DDataset, collect_pairs  # noqa: E402
from reproduce_msdahnet.losses.dice_bce_loss import DiceBCELoss  # noqa: E402
from reproduce_msdahnet.models.msdahnet import build_msdahnet  # noqa: E402
from reproduce_msdahnet.train_5fold import evaluate, resolve_path  # noqa: E402
from reproduce_msdahnet.utils.logger import save_json  # noqa: E402
from reproduce_msdahnet.utils.split import make_folds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(THIS_DIR / "configs" / "msdahnet_breastdm_5fold.yaml"))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_dir = Path(resolve_path(cfg["output"]["output_dir"]))
    checkpoint = args.checkpoint or str(output_dir / f"fold_{args.fold}" / f"best_model_fold{args.fold}.pth")
    pairs = collect_pairs(resolve_path(cfg["data"]["image_dir"]), resolve_path(cfg["data"]["mask_dir"]))
    folds, _ = make_folds(
        [pair[0] for pair in pairs],
        n_splits=int(cfg["cross_validation"]["n_splits"]),
        split_level=cfg["cross_validation"]["split_level"],
        seed=int(cfg["cross_validation"]["seed"]),
        shuffle=bool(cfg["cross_validation"]["shuffle"]),
    )
    _, val_indices = folds[args.fold]
    val_pairs = [pairs[i] for i in val_indices]
    dataset = BreastDM2DDataset(
        val_pairs,
        image_size=int(cfg["data"]["image_size"]),
        gray_to_rgb=bool(cfg["data"].get("gray_to_rgb", False)),
        mask_threshold=float(cfg["data"]["mask_threshold"]),
    )
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=0)
    device = torch.device(cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    model = build_msdahnet(int(cfg["model"]["in_channels"]), int(cfg["model"]["num_classes"])).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    loss_fn = DiceBCELoss(float(cfg["loss"]["dice_weight"]), float(cfg["loss"]["bce_weight"]))
    loss, metrics = evaluate(model, loader, loss_fn, device, cfg)
    result = {"fold": args.fold, "checkpoint": checkpoint, "loss": loss, "metrics": metrics}
    output_path = args.output or str(output_dir / f"fold_{args.fold}" / f"evaluation_fold{args.fold}.json")
    save_json(output_path, result)
    print(result)


if __name__ == "__main__":
    main()
