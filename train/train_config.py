from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def apply_config_to_args(args, config: Dict[str, Any]):
    train_cfg = config.get("train", {})
    for key, value in train_cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def build_model_from_config(config: Dict[str, Any]):
    model_cfg = deepcopy(config.get("model", {}))
    ablation = deepcopy(config.get("ablation", {}))
    name = model_cfg.pop("name", "sg_ktfnet")
    model_cfg["ablation"] = ablation

    if name in {"sg_ktfnet", "sgktfnet", "SGKTFNet"}:
        from model.sg_ktfnet import SGKTFNet

        return SGKTFNet(**model_cfg)

    module = import_module(f"model.{name}")
    cls = getattr(module, "SwinHR")
    return cls(**model_cfg)


def build_loss_from_config(config: Dict[str, Any]):
    loss_cfg = deepcopy(config.get("loss", {}))
    if not loss_cfg:
        return None
    ablation = deepcopy(config.get("ablation", {}))
    name = loss_cfg.pop("name", "sg_ktfnet_loss")
    loss_cfg["ablation"] = ablation

    if name in {"sg_ktfnet_loss", "sgktfnet_loss", "SGKTFNetLoss"}:
        from train.losses import SGKTFNetLoss

        return SGKTFNetLoss(**loss_cfg)
    raise ValueError(f"Unknown configured loss: {name}")


def resolve_config_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return str(p)
    project_path = Path(__file__).resolve().parents[1] / path
    if project_path.exists():
        return str(project_path)
    return path
