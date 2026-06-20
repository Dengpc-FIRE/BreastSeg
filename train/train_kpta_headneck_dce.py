from __future__ import annotations

import runpy
import sys
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "kpta_25d_net_headneck_dce_56ch.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(DEFAULT_CONFIG)])
    runpy.run_module("train.train_kpta", run_name="__main__")
