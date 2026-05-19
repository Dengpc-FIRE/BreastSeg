import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.dce_kinetic_utils import boundary_target_2d
from model.sg_ktfnet import SGKTFNet
from train.losses import SGKTFNetLoss


def assert_finite(tensor, name):
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf"


def run_case(x, phase_indices):
    model = SGKTFNet(
        in_channels=x.shape[1],
        base_channels=8,
        phase_indices=phase_indices,
        return_dict=True,
    )
    output = model(x)
    assert "seg_logits" in output
    assert "boundary_logits" in output
    assert "kinetic_maps" in output
    assert output["seg_logits"].shape == (x.shape[0], 1, x.shape[2], x.shape[3])
    assert output["boundary_logits"].shape == (x.shape[0], 1, x.shape[2], x.shape[3])
    assert_finite(output["kinetic_maps"], "kinetic_maps")

    mask = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])
    mask[:, :, 32:64, 32:64] = 1
    loss_fn = SGKTFNetLoss(lambda_boundary=0.2, lambda_kinetic=0.1)
    loss = loss_fn(output, mask, images=x)
    assert_finite(loss, "loss")
    loss.backward()

    empty_loss = loss_fn(model(x.detach()), torch.zeros_like(mask), images=x)
    assert_finite(empty_loss, "empty_loss")


def test_single_post_forward_and_loss():
    x = torch.randn(2, 3, 128, 128)
    run_case(x, {"pre": 0, "post": [1], "subtraction": [2]})


def test_multiphase_forward_and_loss():
    x = torch.randn(2, 17, 128, 128)
    run_case(x, {"pre": 0, "post": list(range(1, 9)), "subtraction": list(range(9, 17))})


def test_boundary_target_empty_mask():
    mask = torch.zeros(2, 1, 128, 128)
    boundary = boundary_target_2d(mask, thickness=3)
    assert boundary.shape == mask.shape
    assert boundary.sum() == 0


def test_ablation_flags():
    x = torch.randn(2, 17, 128, 128)
    model = SGKTFNet(
        in_channels=17,
        base_channels=8,
        phase_indices={"pre": 0, "post": list(range(1, 9)), "subtraction": list(range(9, 17))},
        ablation={
            "disable_kinetic_maps": True,
            "disable_subtraction_guided_fusion": True,
            "disable_boundary_head": True,
        },
    )
    output = model(x)
    assert output["kinetic_maps"].shape[1] == 1
    assert output["seg_logits"].shape == (2, 1, 128, 128)


if __name__ == "__main__":
    test_single_post_forward_and_loss()
    test_multiphase_forward_and_loss()
    test_boundary_target_empty_mask()
    test_ablation_flags()
    print("SG-KTFNet sanity checks passed.")
