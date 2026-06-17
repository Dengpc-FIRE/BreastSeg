from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parent
CONTRAST_ROOT = MODEL_DIR.parent
sys.path.insert(0, str(CONTRAST_ROOT))
sys.path.insert(0, str(MODEL_DIR))

from dataset.config import load_config, select_device
from dataset.metrics import compute_case_metrics, summarize_metrics, write_metrics_csv, write_summary
from prepare_17ch_nnunet import prepare_dataset


try:
    import SimpleITK as sitk
except Exception:
    sitk = None

try:
    import nibabel as nib
except Exception:
    nib = None


def _resolve_model_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (MODEL_DIR / path).resolve())


def _configure_env(cfg) -> None:
    nn_cfg = cfg["nnunet"]
    os.environ["nnUNet_raw"] = _resolve_model_path(nn_cfg.get("raw_dir", "raw"))
    os.environ["nnUNet_preprocessed"] = _resolve_model_path(nn_cfg.get("preprocessed_dir", "preprocessed"))
    os.environ["nnUNet_results"] = _resolve_model_path(nn_cfg.get("results_dir", "results"))


def _read_nifti(path: Path) -> np.ndarray:
    if sitk is not None:
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if nib is not None:
        return np.asarray(nib.load(str(path)).get_fdata())
    raise ImportError("Testing nnU-Net predictions requires SimpleITK or nibabel to read .nii.gz files.")


def _dataset_folder(cfg) -> Path:
    nn_cfg = cfg["nnunet"]
    dataset_id = int(nn_cfg.get("dataset_id", 501))
    dataset_name = str(nn_cfg.get("dataset_name", "BreastDM17"))
    return Path(os.environ["nnUNet_raw"]) / f"Dataset{dataset_id:03d}_{dataset_name}"


def _compute_metrics(pred_dir: Path, labels_dir: Path, out_dir: Path) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for label_path in sorted(labels_dir.glob("*.nii.gz")):
        pred_path = pred_dir / label_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing nnU-Net prediction for {label_path.name}: {pred_path}")
        pred = _read_nifti(pred_path) > 0
        target = _read_nifti(label_path) > 0
        metric = compute_case_metrics(pred, target)
        metric["id"] = label_path.name.replace(".nii.gz", "")
        rows.append(metric)
    summary = summarize_metrics(rows)
    write_metrics_csv(out_dir / "metrics.csv", rows)
    write_summary(out_dir / "summary.txt", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(MODEL_DIR / "configs" / "breastdm_17ch.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--fold", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, model_dir=MODEL_DIR, model_key="nnunet")
    _configure_env(cfg)
    if not args.skip_prepare:
        prepare_dataset(cfg, MODEL_DIR)

    nn_cfg = cfg["nnunet"]
    dataset_id = int(nn_cfg.get("dataset_id", 501))
    configuration = nn_cfg.get("configuration", "3d_fullres")
    trainer = nn_cfg.get("trainer", "nnUNetTrainerBreastDM17")
    plans = nn_cfg.get("plans", "nnUNetPlans")
    fold = str(args.fold if args.fold is not None else nn_cfg.get("fold", 0))
    checkpoint_name = args.checkpoint_name or nn_cfg.get("checkpoint_name", "checkpoint_best.pth")
    out_dir = Path(cfg["output"]["test_dir"])
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_predict:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.file_path_utilities import get_output_folder

        model_folder = get_output_folder(dataset_id, trainer, plans, configuration)
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=(select_device(args.device).type == "cuda"),
            device=torch.device(select_device(args.device)),
            verbose=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(model_folder, use_folds=(fold,), checkpoint_name=checkpoint_name)
        predictor.predict_from_files(
            str(_dataset_folder(cfg) / "imagesTs"),
            str(pred_dir),
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=int(nn_cfg.get("num_processes", 4)),
            num_processes_segmentation_export=int(nn_cfg.get("num_processes", 4)),
        )

    summary = _compute_metrics(pred_dir, _dataset_folder(cfg) / "labelsTs", out_dir)
    for key in ["mean_dice", "mean_iou", "mean_hd95", "mean_sensitivity", "mean_precision", "mean_accuracy"]:
        print(f"{key}: {summary[key]:.6f}")


if __name__ == "__main__":
    main()

