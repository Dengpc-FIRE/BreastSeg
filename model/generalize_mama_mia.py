from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.generalization.run_kpta_mama_mia import run, parse_args


if __name__ == "__main__":
    run(parse_args())
