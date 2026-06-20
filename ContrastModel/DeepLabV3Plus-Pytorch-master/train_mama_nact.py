from pathlib import Path
import sys

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR.parent))
from dataset.mama_nact_cli import run_mama_nact_train

if __name__ == "__main__":
    run_mama_nact_train(MODEL_DIR, "deeplabv3plus")

