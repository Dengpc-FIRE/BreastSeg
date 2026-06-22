from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRAST_ROOT = PROJECT_ROOT / "ContrastModel"
for path in (PROJECT_ROOT, CONTRAST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ContrastModel.dataset.config import load_config, select_device  # noqa: E402
from ContrastModel.dataset.training import test_model, train_model  # noqa: E402


def parse_args(default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.amp is not None:
        cfg["train"]["amp"] = args.amp


def run_train_cli(model_dir: str | Path, model_key: str) -> None:
    model_dir = Path(model_dir).resolve()
    args = parse_args(model_dir / "configs" / "cc_tumor_heterogeneity_17ch.yaml")
    cfg = load_config(args.config, model_dir=model_dir, model_key=model_key)
    apply_overrides(cfg, args)
    train_model(model_dir, model_key, cfg, select_device(args.device))


def run_test_cli(model_dir: str | Path, model_key: str) -> None:
    model_dir = Path(model_dir).resolve()
    args = parse_args(model_dir / "configs" / "cc_tumor_heterogeneity_17ch.yaml")
    cfg = load_config(args.config, model_dir=model_dir, model_key=model_key)
    apply_overrides(cfg, args)
    test_model(model_dir, model_key, cfg, select_device(args.device), checkpoint=args.checkpoint)
