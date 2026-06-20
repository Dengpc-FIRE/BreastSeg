from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRAST_ROOT = PROJECT_ROOT / "ContrastModel"
for _path in (PROJECT_ROOT, CONTRAST_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ContrastModel.dataset.config import load_config, select_device  # noqa: E402
from ContrastModel.dataset.metrics import compute_case_metrics, summarize_metrics, write_metrics_csv, write_summary  # noqa: E402


def _require_nibabel():
    try:
        import nibabel as nib
    except Exception as exc:
        raise RuntimeError("nnU-Net HeadNeckDCE adapter requires nibabel.") from exc
    return nib


def _save_nifti(path: Path, data: np.ndarray) -> None:
    nib = _require_nibabel()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Source arrays are [D,H,W]. NIfTI files are written as [W,H,D].
    nii_data = np.transpose(data, (2, 1, 0)).astype(np.float32)
    nib.save(nib.Nifti1Image(nii_data, np.eye(4)), str(path))


def _read_nifti(path: Path) -> np.ndarray:
    nib = _require_nibabel()
    return np.asarray(nib.load(str(path)).get_fdata()) > 0


def _resolve_model_path(model_dir: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (model_dir / path).resolve())


def configure_env(cfg: Dict, model_dir: Path) -> None:
    nn_cfg = cfg["nnunet"]
    os.environ["nnUNet_raw"] = _resolve_model_path(model_dir, nn_cfg.get("raw_dir", "raw_headneck_dce"))
    os.environ["nnUNet_preprocessed"] = _resolve_model_path(model_dir, nn_cfg.get("preprocessed_dir", "preprocessed_headneck_dce"))
    os.environ["nnUNet_results"] = _resolve_model_path(model_dir, nn_cfg.get("results_dir", "results_headneck_dce"))
    os.environ["BREASTDM17_NNUNET_EPOCHS"] = str(cfg["train"].get("epochs", 100))
    os.environ["BREASTDM17_NNUNET_LR"] = str(cfg["train"].get("lr", 0.0001))
    os.environ["BREASTDM17_NNUNET_WEIGHT_DECAY"] = str(cfg["train"].get("weight_decay", 0.00001))


def dataset_folder(cfg: Dict) -> Path:
    nn_cfg = cfg["nnunet"]
    return Path(os.environ["nnUNet_raw"]) / f"Dataset{int(nn_cfg.get('dataset_id', 556)):03d}_{nn_cfg.get('dataset_name', 'HeadNeckDCE56')}"


def _iter_volumes(split_path: str | Path) -> List[Path]:
    volume_dir = Path(split_path) / "volumes"
    if not volume_dir.exists():
        raise FileNotFoundError(f"HeadNeckDCE volume directory not found: {volume_dir}")
    paths = sorted(volume_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz volumes found in {volume_dir}")
    return paths


def prepare_nnunet_dataset(cfg: Dict) -> Path:
    folder = dataset_folder(cfg)
    for child in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (folder / child).mkdir(parents=True, exist_ok=True)
    channel_count = int(cfg["model"].get("input_channels", 56))
    training_cases = []
    for split in ("train", "val"):
        for path in _iter_volumes(cfg["data"][f"{split}_path"]):
            case_id = path.stem
            with np.load(path) as item:
                image = np.asarray(item["image"], dtype=np.float32)
                mask = np.asarray(item["mask"], dtype=np.float32)
            if image.shape[0] != channel_count:
                raise ValueError(f"Expected {channel_count} channels in {path}, got {image.shape}")
            for channel in range(channel_count):
                _save_nifti(folder / "imagesTr" / f"{case_id}_{channel:04d}.nii.gz", image[channel])
            _save_nifti(folder / "labelsTr" / f"{case_id}.nii.gz", (mask[0] > 0).astype(np.uint8))
            training_cases.append(case_id)
    test_cases = []
    for path in _iter_volumes(cfg["data"]["test_path"]):
        case_id = path.stem
        with np.load(path) as item:
            image = np.asarray(item["image"], dtype=np.float32)
            mask = np.asarray(item["mask"], dtype=np.float32)
        for channel in range(channel_count):
            _save_nifti(folder / "imagesTs" / f"{case_id}_{channel:04d}.nii.gz", image[channel])
        _save_nifti(folder / "labelsTs" / f"{case_id}.nii.gz", (mask[0] > 0).astype(np.uint8))
        test_cases.append(case_id)

    dataset_json = {
        "channel_names": {str(i): f"dce_t{i:02d}" for i in range(channel_count)},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(training_cases),
        "file_ending": ".nii.gz",
        "name": cfg["nnunet"].get("dataset_name", "HeadNeckDCE56"),
        "description": "Head and Neck DCE-MRI pre-treatment primary GTV segmentation.",
    }
    with (folder / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_json, handle, indent=2)
    return folder


def run_train_cli(model_dir: str | Path) -> None:
    model_dir = Path(model_dir).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(model_dir / "configs" / "headneck_dce_56ch.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-plan", action="store_true")
    parser.add_argument("--fold", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, model_dir=model_dir, model_key="nnunet")
    configure_env(cfg, model_dir)
    if not args.skip_prepare:
        prepare_nnunet_dataset(cfg)

    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
    from nnunetv2.run.run_training import run_training

    nn_cfg = cfg["nnunet"]
    dataset_id = int(nn_cfg.get("dataset_id", 556))
    configuration = nn_cfg.get("configuration", "3d_fullres")
    plans = nn_cfg.get("plans", "nnUNetPlans")
    trainer = nn_cfg.get("trainer", "nnUNetTrainer")
    fold = str(args.fold if args.fold is not None else nn_cfg.get("fold", 0))
    num_processes = int(nn_cfg.get("num_processes", 4))
    if not args.skip_plan:
        extract_fingerprints([dataset_id], "DatasetFingerprintExtractor", num_processes, False, False, False, show_progress_bar=True)
        plan_experiments([dataset_id], "ExperimentPlanner", None, "DefaultPreprocessor", None, None)
        preprocess([dataset_id], plans, configurations=[configuration], num_processes=[num_processes], verbose=False)
    import torch

    run_training(str(dataset_id), configuration, fold, trainer_class_name=trainer, plans_identifier=plans, num_gpus=1, device=torch.device(select_device(args.device)))


def run_test_cli(model_dir: str | Path) -> None:
    model_dir = Path(model_dir).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(model_dir / "configs" / "headneck_dce_56ch.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--fold", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, model_dir=model_dir, model_key="nnunet")
    configure_env(cfg, model_dir)
    if not args.skip_prepare:
        prepare_nnunet_dataset(cfg)
    nn_cfg = cfg["nnunet"]
    out_dir = Path(cfg["output"]["test_dir"])
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_predict:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.file_path_utilities import get_output_folder

        dataset_id = int(nn_cfg.get("dataset_id", 556))
        configuration = nn_cfg.get("configuration", "3d_fullres")
        trainer = nn_cfg.get("trainer", "nnUNetTrainer")
        plans = nn_cfg.get("plans", "nnUNetPlans")
        fold = str(args.fold if args.fold is not None else nn_cfg.get("fold", 0))
        checkpoint_name = args.checkpoint_name or nn_cfg.get("checkpoint_name", "checkpoint_best.pth")
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=(select_device(args.device).type == "cuda"), device=torch.device(select_device(args.device)), verbose=False, allow_tqdm=True)
        predictor.initialize_from_trained_model_folder(get_output_folder(dataset_id, trainer, plans, configuration), use_folds=(fold,), checkpoint_name=checkpoint_name)
        predictor.predict_from_files(str(dataset_folder(cfg) / "imagesTs"), str(pred_dir), save_probabilities=False, overwrite=True, num_processes_preprocessing=int(nn_cfg.get("num_processes", 4)), num_processes_segmentation_export=int(nn_cfg.get("num_processes", 4)))

    rows = []
    for label_path in sorted((dataset_folder(cfg) / "labelsTs").glob("*.nii.gz")):
        pred_path = pred_dir / label_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing nnU-Net prediction: {pred_path}")
        metric = compute_case_metrics(_read_nifti(pred_path), _read_nifti(label_path))
        metric["id"] = label_path.name.replace(".nii.gz", "")
        rows.append(metric)
    summary = summarize_metrics(rows)
    write_metrics_csv(out_dir / "metrics.csv", rows)
    write_summary(out_dir / "summary.txt", summary)
    for key in ["mean_dice", "mean_iou", "mean_hd95", "mean_sensitivity", "mean_precision", "mean_accuracy"]:
        print(f"{key}: {summary[key]:.6f}")

