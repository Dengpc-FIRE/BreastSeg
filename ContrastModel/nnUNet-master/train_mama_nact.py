from pathlib import Path
import sys

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MODEL_DIR / "configs" / "mama_nact.yaml"


def _has_config_arg(argv: list[str]) -> bool:
    return any(arg == "--config" or arg.startswith("--config=") for arg in argv)


if __name__ == "__main__":
    if not _has_config_arg(sys.argv[1:]):
        sys.argv.extend(["--config", str(DEFAULT_CONFIG)])
    from train_17ch import main

    main()

