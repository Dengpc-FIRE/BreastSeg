from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "kpta_25d_net_cc_tumor_heterogeneity_17ch-v8.yaml"


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(DEFAULT_CONFIG)])
    runpy.run_module("train.train_kpta", run_name="__main__")
