from __future__ import annotations

import runpy
import sys
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "kpta_25d_net_cc_tumor_heterogeneity_17ch.yaml"


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(DEFAULT_CONFIG)])
    runpy.run_module("train.train_kpta", run_name="__main__")

