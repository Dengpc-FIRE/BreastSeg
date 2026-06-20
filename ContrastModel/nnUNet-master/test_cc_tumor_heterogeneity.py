from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.cc_tumor_heterogeneity.nnunet_cli import run_test_cli


if __name__ == "__main__":
    run_test_cli(Path(__file__).resolve().parent)

