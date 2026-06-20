from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRAST_ROOT = PROJECT_ROOT / "ContrastModel"
for path in (PROJECT_ROOT, CONTRAST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ContrastModel.dataset.config import load_config, select_device  # noqa: E402
from ContrastModel.dataset.metrics import compute_case_metrics, summarize_metrics, write_metrics_csv, write_summary  # noqa: E402


def resolve_model_path(model_dir: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (model_dir / path).resolve())


def configure_env(cfg: dict, model_dir: Path) -> None:
    nn_cfg = cfg["nnunet"]
    os.environ["nnUNet_raw"] = resolve_model_path(model_dir, nn_cfg.get("raw_dir", "raw_cc_tumor_heterogeneity_17ch"))
    os.environ["nnUNet_preprocessed"] = resolve_model_path(model_dir, nn_cfg.get("preprocessed_dir", "preprocessed_cc_tumor_heterogeneity_17ch"))
    os.environ["nnUNet_results"] = resolve_model_path(model_dir, nn_cfg.get("results_dir", "results_cc_tumor_heterogeneity_17ch"))
    os.environ["BREASTDM17_NNUNET_EPOCHS"] = str(cfg["train"].get("epochs", 100))
    os.environ["BREASTDM17_NNUNET_LR"] = str(cfg["train"].get("lr", 0.0001))
    os.environ["BREASTDM17_NNUNET_WEIGHT_DECAY"] = str(cfg["train"].get("weight_decay", 0.00001))


def import_prepare_dataset(model_dir: Path):
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from prepare_17ch_nnunet import prepare_dataset

    return prepare_dataset


def parse_args(default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-plan", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--fold", default=None)
    parser.add_argument("--checkpoint-name", default=None)
    return parser.parse_args()


def run_train_cli(model_dir: str | Path) -> None:
    model_dir = Path(model_dir).resolve()
    args = parse_args(model_dir / "configs" / "cc_tumor_heterogeneity_17ch.yaml")
    cfg = load_config(args.config, model_dir=model_dir, model_key="nnunet")
    configure_env(cfg, model_dir)
    if not args.skip_prepare:
        import_prepare_dataset(model_dir)(cfg, model_dir)

    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
    from nnunetv2.run.run_training import run_training
    import torch

    nn_cfg = cfg["nnunet"]
    dataset_id = int(nn_cfg.get("dataset_id", 557))
    configuration = nn_cfg.get("configuration", "3d_fullres")
    plans = nn_cfg.get("plans", "nnUNetPlans")
    trainer = nn_cfg.get("trainer", "nnUNetTrainer")
    fold = str(args.fold if args.fold is not None else nn_cfg.get("fold", 0))
    num_processes = int(nn_cfg.get("num_processes", 4))
    if not args.skip_plan:
        extract_fingerprints([dataset_id], "DatasetFingerprintExtractor", num_processes, bool(nn_cfg.get("verify_dataset_integrity", False)), bool(nn_cfg.get("clean", False)), False, show_progress_bar=True)
        plan_experiments([dataset_id], "ExperimentPlanner", None, "DefaultPreprocessor", None, None)
        preprocess([dataset_id], plans, configurations=[configuration], num_processes=[num_processes], verbose=False)
    run_training(str(dataset_id), configuration, fold, trainer_class_name=trainer, plans_identifier=plans, num_gpus=1, device=torch.device(select_device(args.device)))


def run_test_cli(model_dir: str | Path) -> None:
    model_dir = Path(model_dir).resolve()
    args = parse_args(model_dir / "configs" / "cc_tumor_heterogeneity_17ch.yaml")
    cfg = load_config(args.config, model_dir=model_dir, model_key="nnunet")
    configure_env(cfg, model_dir)
    if not args.skip_prepare:
        import_prepare_dataset(model_dir)(cfg, model_dir)
    nn_cfg = cfg["nnunet"]
    dataset_id = int(nn_cfg.get("dataset_id", 557))
    dataset_name = str(nn_cfg.get("dataset_name", "CCTumorHeterogeneity17"))
    dataset_folder = Path(os.environ["nnUNet_raw"]) / f"Dataset{dataset_id:03d}_{dataset_name}"
    out_dir = Path(cfg["output"]["test_dir"])
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_predict:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.file_path_utilities import get_output_folder

        configuration = nn_cfg.get("configuration", "3d_fullres")
        trainer = nn_cfg.get("trainer", "nnUNetTrainer")
        plans = nn_cfg.get("plans", "nnUNetPlans")
        fold = str(args.fold if args.fold is not None else nn_cfg.get("fold", 0))
        checkpoint_name = args.checkpoint_name or nn_cfg.get("checkpoint_name", "checkpoint_best.pth")
        device = select_device(args.device)
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=(device.type == "cuda"),
            device=torch.device(device),
            verbose=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            get_output_folder(dataset_id, trainer, plans, configuration),
            use_folds=(fold,),
            checkpoint_name=checkpoint_name,
        )
        predictor.predict_from_files(
            str(dataset_folder / "imagesTs"),
            str(pred_dir),
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=int(nn_cfg.get("num_processes", 4)),
            num_processes_segmentation_export=int(nn_cfg.get("num_processes", 4)),
        )

    rows = []
    reader = _nifti_reader()
    for label_path in sorted((dataset_folder / "labelsTs").glob("*.nii.gz")):
        pred_path = pred_dir / label_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing nnU-Net prediction for {label_path.name}: {pred_path}")
        metric = compute_case_metrics(reader(pred_path) > 0, reader(label_path) > 0)
        metric["id"] = label_path.name.replace(".nii.gz", "")
        rows.append(metric)
    summary = summarize_metrics(rows)
    write_metrics_csv(out_dir / "metrics.csv", rows)
    write_summary(out_dir / "summary.txt", summary)
    for key in ["mean_dice", "mean_iou", "mean_hd95", "mean_sensitivity", "mean_precision", "mean_accuracy"]:
        print(f"{key}: {summary[key]:.6f}")


def _nifti_reader():
    try:
        import SimpleITK as sitk

        return lambda path: sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    except Exception:
        pass
    try:
        import nibabel as nib
        import numpy as np

        return lambda path: np.asarray(nib.load(str(path)).get_fdata())
    except Exception as exc:
        raise ImportError("Reading nnU-Net predictions requires SimpleITK or nibabel.") from exc
