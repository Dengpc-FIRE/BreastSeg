from pathlib import Path
import sys

CONTRAST_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CONTRAST_ROOT))

from dataset.training import run_train_cli


if __name__ == "__main__":
    run_train_cli(Path(__file__).resolve().parent, "emcad", default_config_name="breastdm_preonly.yaml")
