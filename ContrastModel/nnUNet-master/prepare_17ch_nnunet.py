from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

CONTRAST_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CONTRAST_ROOT))

from dataset.breastdm_3d import build_or_load_volume
from dataset.config import load_config


try:
    import SimpleITK as sitk
except Exception:
    sitk = None

try:
    import nibabel as nib
except Exception:
    nib = None


def _resolve_model_path(model_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (model_dir / path).resolve()


def _write_nifti(array: np.ndarray, path: Path, is_label: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sitk is not None:
        img = sitk.GetImageFromArray(array.astype(np.uint8 if is_label else np.float32))
        sitk.WriteImage(img, str(path))
        return
    if nib is not None:
        img = nib.Nifti1Image(array.astype(np.uint8 if is_label else np.float32), np.eye(4))
        nib.save(img, str(path))
        return
    raise ImportError("Preparing nnU-Net data requires SimpleITK or nibabel to write .nii.gz files.")


def _patient_ids(raw_root: Path, split: str) -> List[str]:
    images_root = raw_root / split / "images"
    if not images_root.exists():
        raise FileNotFoundError(f"Raw split images directory not found: {images_root}")
    return sorted([p.name for p in images_root.iterdir() if p.is_dir()])


def _write_case(item: Dict[str, Any], case_id: str, image_dir: Path, label_dir: Path | None) -> None:
    image = item["image"]
    mask = item["mask"]
    for c in range(image.shape[0]):
        _write_nifti(image[c], image_dir / f"{case_id}_{c:04d}.nii.gz", is_label=False)
    if label_dir is not None:
        _write_nifti(mask[0].astype(np.uint8), label_dir / f"{case_id}.nii.gz", is_label=True)


def prepare_dataset(cfg: Dict[str, Any], model_dir: str | Path) -> Path:
    model_dir = Path(model_dir).resolve()
    nn_cfg = cfg["nnunet"]
    dataset_id = int(nn_cfg.get("dataset_id", 501))
    dataset_name = str(nn_cfg.get("dataset_name", "BreastDM17"))
    dataset_folder = _resolve_model_path(model_dir, nn_cfg.get("raw_dir", "raw")) / f"Dataset{dataset_id:03d}_{dataset_name}"

    images_tr = dataset_folder / "imagesTr"
    labels_tr = dataset_folder / "labelsTr"
    images_ts = dataset_folder / "imagesTs"
    labels_ts = dataset_folder / "labelsTs"
    for folder in (images_tr, labels_tr, images_ts, labels_ts):
        folder.mkdir(parents=True, exist_ok=True)

    raw_root = Path(cfg["data"]["raw_dataset_root"])
    cache_root = Path(cfg["data"]["cache_root"])
    phase_names = cfg["data"]["phase_names"]
    label_phase = cfg["data"].get("label_phase")
    normalize = cfg["data"].get("normalize", "zscore")
    allow_missing = bool(cfg["data"].get("allow_missing_phases", False))

    training_cases: List[str] = []
    test_cases: List[str] = []
    for split in ("train", "val"):
        for patient_id in _patient_ids(raw_root, split):
            case_id = f"{split}_{patient_id}"
            item = build_or_load_volume(raw_root, split, patient_id, cache_root, phase_names, label_phase, normalize, allow_missing)
            _write_case(item, case_id, images_tr, labels_tr)
            training_cases.append(case_id)

    for patient_id in _patient_ids(raw_root, "test"):
        case_id = f"test_{patient_id}"
        item = build_or_load_volume(raw_root, "test", patient_id, cache_root, phase_names, label_phase, normalize, allow_missing)
        _write_case(item, case_id, images_ts, labels_ts)
        test_cases.append(case_id)

    dataset_json = {
        "channel_names": {str(i): name for i, name in enumerate(phase_names)},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(training_cases),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "17-channel BreastDM DCE dataset for nnU-Net v2",
    }
    with (dataset_folder / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_json, handle, indent=2)
    return dataset_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs" / "breastdm_17ch.yaml"))
    args = parser.parse_args()
    model_dir = Path(__file__).resolve().parent
    cfg = load_config(args.config, model_dir=model_dir, model_key="nnunet")
    folder = prepare_dataset(cfg, model_dir)
    print(f"Prepared nnU-Net dataset: {folder}")


if __name__ == "__main__":
    main()

