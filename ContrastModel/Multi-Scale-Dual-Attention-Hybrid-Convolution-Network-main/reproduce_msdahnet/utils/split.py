import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from sklearn.model_selection import KFold


def infer_patient_id(path: str) -> str:
    """Infer patient id from common BreastDM processed filenames.

    Expected names often look like BreaDM-Ma-2139_p-036.npy/png. If that
    pattern is unavailable, fall back to the prefix before the first "_".
    """
    stem = Path(path).stem
    match = re.search(r"(BreaDM-[A-Za-z]+-\d+|BreastDM-[A-Za-z]+-\d+)", stem)
    if match:
        return match.group(1)
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def make_folds(
    image_paths: Sequence[str],
    n_splits: int,
    split_level: str,
    seed: int,
    shuffle: bool = True,
) -> Tuple[List[Tuple[List[int], List[int]]], bool]:
    if split_level != "patient":
        indices = list(range(len(image_paths)))
        kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
        return [(list(train), list(val)) for train, val in kfold.split(indices)], False

    patient_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, path in enumerate(image_paths):
        patient_to_indices[infer_patient_id(path)].append(idx)

    patient_ids = sorted(patient_to_indices)
    patient_level_guaranteed = len(patient_ids) < len(image_paths)
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    folds = []
    for train_patient_idx, val_patient_idx in kfold.split(patient_ids):
        train_indices, val_indices = [], []
        for patient_idx in train_patient_idx:
            train_indices.extend(patient_to_indices[patient_ids[patient_idx]])
        for patient_idx in val_patient_idx:
            val_indices.extend(patient_to_indices[patient_ids[patient_idx]])
        folds.append((sorted(train_indices), sorted(val_indices)))
    return folds, patient_level_guaranteed


def save_fold_files(output_dir: str, folds, image_paths: Sequence[str]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        train_path = out / f"fold_{fold_idx}_train.txt"
        val_path = out / f"fold_{fold_idx}_val.txt"
        train_path.write_text("\n".join(str(image_paths[i]) for i in train_indices), encoding="utf-8")
        val_path.write_text("\n".join(str(image_paths[i]) for i in val_indices), encoding="utf-8")
