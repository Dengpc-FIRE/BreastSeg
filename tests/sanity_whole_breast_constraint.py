"""CPU sanity checks for the inference-only whole-breast constraint.

These tests use a fake volume predictor and therefore do not require nnU-Net
weights or CUDA. They verify data reconstruction, center-slice mapping, mask
application, and the disabled configuration path.
"""

from pathlib import Path

import numpy as np
import torch

from inference.whole_breast_constraint import (
    WholeBreastConstraint,
    build_whole_breast_constraint,
    split_case_and_slice,
)


class _Dataset:
    def __init__(self, data_dir: Path, names):
        self.data_dir = str(data_dir)
        self.ids = list(names)


class _FakeWholeBreastConstraint(WholeBreastConstraint):
    def _predict_volume(self, volume: np.ndarray) -> np.ndarray:
        mask = np.zeros(volume.shape, dtype=np.uint8)
        mask[:, 2:-2, 2:-2] = 1
        return mask


def _write_case(data_dir: Path):
    names = []
    for index in range(3):
        name = f"case_a_p-{index:03d}.npy"
        sample = np.zeros((3, 17, 16, 16), dtype=np.float32)
        sample[1, 0] = float(index + 1)
        np.save(data_dir / name, sample)
        names.append(name)
    return names


def test_case_name_parser():
    assert split_case_and_slice("BreaDM-Ma-1809_p-038.npy") == (
        "BreaDM-Ma-1809",
        "p-038",
    )


def test_disabled_config_returns_none():
    assert (
        build_whole_breast_constraint(
            {"whole_breast": {"enabled": False}},
            torch.device("cpu"),
        )
        is None
    )


def test_case_volume_cache_and_probability_constraint(tmp_path: Path):
    data_dir = tmp_path / "val" / "data"
    data_dir.mkdir(parents=True)
    names = _write_case(data_dir)
    dataset = _Dataset(data_dir, names)
    constraint = _FakeWholeBreastConstraint(
        {
            "enabled": True,
            "raw_data_root": None,
            "dilation_pixels": 0,
            "dilation_slices": 0,
            "min_breast_fraction": 0.0,
            "max_breast_fraction": 1.0,
            "cache": {"enabled": False},
            "runtime_failure_policy": "error",
            "verbose": False,
        },
        device=torch.device("cpu"),
    )
    probabilities = torch.ones((2, 1, 16, 16), dtype=torch.float32)
    constrained, breast_masks = constraint.constrain_probabilities(
        probabilities,
        names[:2],
        dataset,
    )
    assert breast_masks.shape == (2, 1, 16, 16)
    assert constrained.shape == probabilities.shape
    assert torch.all(constrained[:, :, :2] == 0)
    assert torch.all(constrained[:, :, 2:-2, 2:-2] == 1)
    assert len(constraint._slice_cache) == 3


if __name__ == "__main__":
    test_case_name_parser()
    test_disabled_config_returns_none()
    print("whole-breast constraint sanity checks passed")
