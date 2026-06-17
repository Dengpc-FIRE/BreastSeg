from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping

import yaml


PHASE_NAMES = [
    "VIBRANT",
    "VIBRANT+C1",
    "VIBRANT+C2",
    "VIBRANT+C3",
    "VIBRANT+C4",
    "VIBRANT+C5",
    "VIBRANT+C6",
    "VIBRANT+C7",
    "VIBRANT+C8",
    "SUB1",
    "SUB2",
    "SUB3",
    "SUB4",
    "SUB5",
    "SUB6",
    "SUB7",
    "SUB8",
]


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name": "",
        "input_channels": 17,
        "out_channels": 1,
        "pretrained": False,
    },
    "data": {
        "mode": "2d",
        "train_path": "../../../processed_17ch_dce/train",
        "val_path": "../../../processed_17ch_dce/val",
        "test_path": "../../../processed_17ch_dce/test",
        "raw_dataset_root": "../../../seg",
        "cache_root": "../../dataset/processed_3d_17ch",
        "phase_names": PHASE_NAMES,
        "label_phase": None,
        "image_size": [256, 256],
        "patch_size": [48, 128, 128],
        "samples_per_volume": 4,
        "positive_crop_prob": 0.7,
        "normalize": "zscore",
        "allow_missing_phases": False,
    },
    "train": {
        "epochs": 100,
        "batch_size": 4,
        "lr": 1.0e-4,
        "weight_decay": 1.0e-5,
        "num_workers": 4,
        "early_stopping": 30,
        "scheduler": "reduce_on_plateau",
        "amp": True,
        "seed": 2026,
        "extra_loss_weight": 1.0,
        "max_grad_norm": 0.0,
    },
    "eval": {
        "threshold": 0.5,
        "save_predictions": False,
        "sliding_window_overlap": 0.5,
    },
    "output": {
        "checkpoint_dir": "checkpoints",
        "test_dir": "test_results",
        "log_file": "training_log.csv",
    },
}


def deep_update(base: MutableMapping[str, Any], override: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if isinstance(value, MutableMapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_path(value: Any, base_dir: Path) -> Any:
    if value in (None, ""):
        return value
    if isinstance(value, (str, Path)):
        path = Path(value)
        return str(path if path.is_absolute() else (base_dir / path).resolve())
    return value


def resolve_many(section: Dict[str, Any], keys: Iterable[str], base_dir: Path) -> None:
    for key in keys:
        if key in section:
            section[key] = resolve_path(section[key], base_dir)


def load_config(config_path: str | Path, model_dir: str | Path | None = None, model_key: str | None = None) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    deep_update(cfg, user_cfg)

    model_dir_path = Path(model_dir).resolve() if model_dir is not None else config_path.parent.parent.resolve()
    cfg["runtime"] = {
        "config_path": str(config_path),
        "config_dir": str(config_path.parent),
        "model_dir": str(model_dir_path),
        "contrast_root": str(model_dir_path.parent),
    }

    if model_key:
        cfg["model"]["name"] = model_key

    resolve_many(
        cfg["data"],
        ["train_path", "val_path", "test_path", "raw_dataset_root", "cache_root"],
        config_path.parent,
    )
    resolve_many(cfg["output"], ["checkpoint_dir", "test_dir", "log_file"], model_dir_path)

    if cfg["data"].get("phase_names") is None:
        cfg["data"]["phase_names"] = PHASE_NAMES
    cfg["model"]["input_channels"] = int(cfg["model"].get("input_channels", 17))
    cfg["model"]["out_channels"] = int(cfg["model"].get("out_channels", 1))
    cfg["train"]["epochs"] = int(cfg["train"].get("epochs", 100))
    cfg["train"]["batch_size"] = int(cfg["train"].get("batch_size", 4))
    cfg["train"]["num_workers"] = int(cfg["train"].get("num_workers", 4))
    cfg["train"]["seed"] = int(cfg["train"].get("seed", 2026))
    return cfg


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def select_device(requested: str | None = None):
    import torch

    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_tuple(value: Any, length: int) -> tuple[int, ...]:
    if isinstance(value, int):
        return tuple([value] * length)
    if isinstance(value, (list, tuple)) and len(value) == length:
        return tuple(int(v) for v in value)
    raise ValueError(f"Expected int or length-{length} sequence, got {value!r}")
