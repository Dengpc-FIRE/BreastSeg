from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mama_nact" / "kpta_25d.yaml"


def _has_config_arg(argv: list[str]) -> bool:
    return any(arg == "--config" or arg.startswith("--config=") for arg in argv)


def main() -> int:
    if not _has_config_arg(sys.argv[1:]):
        sys.argv.extend(["--config", str(DEFAULT_CONFIG)])
    from train.train_kpta import main as train_main

    return int(train_main())


if __name__ == "__main__":
    raise SystemExit(main())

