from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Dict, Optional


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        raise ValueError("A KPTA YAML config path is required.")
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def build_model_from_config(config: Dict[str, Any]):
    model_cfg = deepcopy(config.get("model", {}))
    ablation = deepcopy(config.get("ablation", {}))
    name = model_cfg.pop("name", None)
    model_cfg["ablation"] = ablation

    if name == "kpta_net":
        from model.kpta_net import KPTANet

        return KPTANet(**model_cfg)
    if name == "kpta_25d_net":
        from model.kpta_25d_net import KPTA25DNet

        return KPTA25DNet(**model_cfg)
    raise ValueError(
        "Unsupported model.name. Expected 'kpta_net' or 'kpta_25d_net', "
        f"got {name!r}."
    )


def build_loss_from_config(config: Dict[str, Any]):
    loss_cfg = deepcopy(config.get("loss", {}))
    if not loss_cfg:
        raise ValueError("The KPTA config must define a loss section.")
    ablation = deepcopy(config.get("ablation", {}))
    name = loss_cfg.pop("name", None)
    loss_cfg["ablation"] = ablation

    if name == "kpta_net_loss":
        from train.losses import KPTANetLoss

        return KPTANetLoss(**loss_cfg)
    if name == "kpta_25d_loss":
        from train.losses import KPTA25DNetLoss

        return KPTA25DNetLoss(**loss_cfg)
    raise ValueError(
        "Unsupported loss.name. Expected 'kpta_net_loss' or 'kpta_25d_loss', "
        f"got {name!r}."
    )


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


def checkpoint_name_from_config(
    config_path: str,
    prefix: str = "best_model",
) -> str:
    """Build a collision-free checkpoint name from the YAML file name.

    Example:
        configs/kpta_25d_net_lr2e-4.yaml
        -> best_model_kpta_25d_net_lr2e-4.pth
    """
    config_stem = Path(config_path).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", config_stem).strip("._-")
    if not safe_stem:
        safe_stem = "config"
    return f"{prefix}_{safe_stem}.pth"
