import re
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def infer_patient_id(path: str) -> str:
    stem = Path(path).stem
    match = re.search(r"(BreaDM-[A-Za-z]+-\d+|BreastDM-[A-Za-z]+-\d+)", stem)
    if match:
        return match.group(1)
    parts = re.split(r"[_\-]", stem)
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def save_split_files(output_dir: str, splits: Sequence[Tuple[List[int], List[int], List[int]]], image_paths: Sequence[str]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fold, (train_idx, val_idx, test_idx) in enumerate(splits):
        for name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            with (out / f"fold_{fold}_{name}.txt").open("w", encoding="utf-8") as f:
                for idx in indices:
                    f.write(str(image_paths[idx]) + "\n")


def make_repeated_patient_splits(
    image_paths: Sequence[str],
    n_splits: int = 5,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[List[int], List[int], List[int]]], bool]:
    groups = defaultdict(list)
    for idx, path in enumerate(image_paths):
        groups[infer_patient_id(path)].append(idx)
    patients = list(groups.keys())
    patient_level = len(patients) < len(image_paths)
    splits = []
    for fold in range(n_splits):
        rng = np.random.default_rng(seed + fold)
        shuffled = patients.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        train_patients = shuffled[:n_train]
        val_patients = shuffled[n_train:n_train + n_val]
        test_patients = shuffled[n_train + n_val:]
        if not test_patients:
            test_patients = val_patients[-1:]
            val_patients = val_patients[:-1]
        splits.append((
            _indices_for(groups, train_patients),
            _indices_for(groups, val_patients),
            _indices_for(groups, test_patients),
        ))
    return splits, patient_level


def _indices_for(groups, patients) -> List[int]:
    indices = []
    for patient in patients:
        indices.extend(groups[patient])
    return sorted(indices)

