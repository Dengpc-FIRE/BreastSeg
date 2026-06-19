from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.generalization.run_contrast_mama_mia import main


if __name__ == "__main__":
    main("plhn", str(Path(__file__).resolve().parent))
