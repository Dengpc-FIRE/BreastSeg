"""Preflight the nnU-Net whole-breast adapter without a tumor model."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from inference.whole_breast_constraint import build_whole_breast_constraint
from train.train_config import load_config, resolve_config_path
from train.train_kpta import BreastDM25DDataset


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image.astype(np.float32))
    low, high = np.percentile(image, [1.0, 99.0])
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen whole-breast model on a few BreastDM slices and "
            "save overlays before starting tumor-model evaluation."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/kpta_25d_net_whole_breast.yaml",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
    )
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    train_cfg = config.get("train", {})
    split_path = Path(train_cfg[f"{args.split}_path"])
    dataset = BreastDM25DDataset(
        str(split_path / "data"),
        str(split_path / "GT"),
    )
    if not dataset:
        raise ValueError(f"No samples found in {split_path}")

    output_dir = Path(
        args.output_dir
        or (
            Path(train_cfg.get("output_path", "./results_kpta_25d_net"))
            / "whole_breast_preflight"
            / args.split
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    constraint = build_whole_breast_constraint(
        config,
        device=device,
        output_path=Path(train_cfg.get("output_path", ".")),
    )
    if constraint is None:
        raise ValueError(
            "whole_breast.enabled is false. Use the whole-breast config."
        )

    sample_count = min(max(args.num_samples, 1), len(dataset))
    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=sample_count,
        dtype=int,
    )
    fractions = []
    for index in indices:
        image, _, name = dataset[int(index)]
        height, width = image.shape[-2:]
        breast_mask = constraint.get_center_masks(
            [name],
            dataset,
            spatial_shape=(height, width),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )[0, 0].numpy()
        center_slice = image.shape[0] // 2
        pre_image = image[center_slice, constraint.pre_phase_index].numpy()
        base = normalize_for_display(pre_image)
        overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        active = breast_mask > 0.5
        color = np.zeros_like(overlay)
        color[..., 1] = 255
        overlay[active] = (
            0.55 * overlay[active].astype(np.float32)
            + 0.45 * color[active].astype(np.float32)
        ).astype(np.uint8)
        contour_result = cv2.findContours(
            active.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = contour_result[-2]
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        combined = np.hstack(
            [
                cv2.cvtColor(base, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(
                    (active.astype(np.uint8) * 255),
                    cv2.COLOR_GRAY2BGR,
                ),
                overlay,
            ]
        )
        fraction = float(active.mean())
        fractions.append(fraction)
        cv2.imwrite(
            str(output_dir / f"{Path(name).stem}__fraction-{fraction:.3f}.png"),
            combined,
        )

    print(f"Config: {config_path}")
    print(f"Split: {split_path}")
    print(f"Saved overlays: {output_dir}")
    print(
        "Center-slice breast fraction: "
        f"min={min(fractions):.4f}, "
        f"mean={float(np.mean(fractions)):.4f}, "
        f"max={max(fractions):.4f}"
    )
    print("Panels: pre-contrast | breast mask | overlay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
