from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parent
CONTRAST_ROOT = MODEL_DIR.parent
sys.path.insert(0, str(CONTRAST_ROOT))
sys.path.insert(0, str(MODEL_DIR))

from dataset.config import load_config, select_device
from prepare_17ch_nnunet import prepare_dataset


def _resolve_model_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (MODEL_DIR / path).resolve())


def _configure_env(cfg) -> None:
    nn_cfg = cfg["nnunet"]
    os.environ["nnUNet_raw"] = _resolve_model_path(nn_cfg.get("raw_dir", "raw"))
    os.environ["nnUNet_preprocessed"] = _resolve_model_path(nn_cfg.get("preprocessed_dir", "preprocessed"))
    os.environ["nnUNet_results"] = _resolve_model_path(nn_cfg.get("results_dir", "results"))
    os.environ["BREASTDM17_NNUNET_EPOCHS"] = str(cfg["train"].get("epochs", 100))
    os.environ["BREASTDM17_NNUNET_LR"] = str(cfg["train"].get("lr", 0.0001))
    os.environ["BREASTDM17_NNUNET_WEIGHT_DECAY"] = str(cfg["train"].get("weight_decay", 0.00001))
    os.environ["BREASTDM17_NNUNET_SAVE_EVERY"] = str(max(1, min(50, int(cfg["train"].get("epochs", 100)))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(MODEL_DIR / "configs" / "breastdm_17ch.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-plan", action="store_true")
    parser.add_argument("--fold", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, model_dir=MODEL_DIR, model_key="nnunet")
    _configure_env(cfg)
    if not args.skip_prepare:
        prepare_dataset(cfg, MODEL_DIR)

    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
    from nnunetv2.run.run_training import run_training

    nn_cfg = cfg["nnunet"]
    dataset_id = int(nn_cfg.get("dataset_id", 501))
    configuration = nn_cfg.get("configuration", "3d_fullres")
    plans = nn_cfg.get("plans", "nnUNetPlans")
    trainer = nn_cfg.get("trainer", "nnUNetTrainerBreastDM17")
    fold = str(args.fold if args.fold is not None else nn_cfg.get("fold", 0))
    num_processes = int(nn_cfg.get("num_processes", 4))

    if not args.skip_plan:
        extract_fingerprints(
            [dataset_id],
            "DatasetFingerprintExtractor",
            num_processes,
            bool(nn_cfg.get("verify_dataset_integrity", False)),
            bool(nn_cfg.get("clean", False)),
            False,
            show_progress_bar=True,
        )
        plan_experiments([dataset_id], "ExperimentPlanner", None, "DefaultPreprocessor", None, None)
        preprocess([dataset_id], plans, configurations=[configuration], num_processes=[num_processes], verbose=False)

    device = select_device(args.device)
    run_training(
        str(dataset_id),
        configuration,
        fold,
        trainer_class_name=trainer,
        plans_identifier=plans,
        num_gpus=1,
        device=torch.device(device),
    )


if __name__ == "__main__":
    main()

