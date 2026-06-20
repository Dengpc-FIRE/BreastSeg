from __future__ import annotations

import sys
from pathlib import Path


def _has_config_arg(argv: list[str]) -> bool:
    return any(arg == "--config" or arg.startswith("--config=") for arg in argv)


def run_mama_nact_train(model_dir: str | Path, model_key: str) -> None:
    model_dir = Path(model_dir).resolve()
    if not _has_config_arg(sys.argv[1:]):
        sys.argv.extend(["--config", str(model_dir / "configs" / "mama_nact.yaml")])
    from dataset.training import run_train_cli

    run_train_cli(model_dir, model_key)


def run_mama_nact_test(model_dir: str | Path, model_key: str) -> None:
    model_dir = Path(model_dir).resolve()
    if not _has_config_arg(sys.argv[1:]):
        sys.argv.extend(["--config", str(model_dir / "configs" / "mama_nact.yaml")])
    from dataset.training import run_test_cli

    run_test_cli(model_dir, model_key)

