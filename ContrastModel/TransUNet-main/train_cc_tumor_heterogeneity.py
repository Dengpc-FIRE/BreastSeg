from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cc_tumor_heterogeneity.contrast_cli import run_train_cli


if __name__ == "__main__":
    run_train_cli(Path(__file__).resolve().parent, "transunet")

