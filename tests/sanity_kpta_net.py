import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.kpta_net import KPTANet
from train.losses import KPTANetLoss, attention_smoothness_loss


def assert_finite(tensor, name):
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf"


def run_case(x, phase_indices):
    model = KPTANet(
        in_channels=x.shape[1],
        base_channels=8,
        phase_indices=phase_indices,
        return_dict=True,
        store_attention_maps=True,
    )
    output = model(x)
    assert output["seg_logits"].shape == (x.shape[0], 1, x.shape[2], x.shape[3])
    assert output["boundary_logits"].shape == (x.shape[0], 1, x.shape[2], x.shape[3])
    assert output["uncertainty_logits"].shape == (x.shape[0], 1, x.shape[2], x.shape[3])
    assert_finite(output["kinetic_maps"], "kinetic_maps")
    assert output["attention_maps"]
    attn = output["attention_maps"][0]
    assert torch.allclose(attn.sum(dim=1), torch.ones_like(attn.sum(dim=1)), atol=1e-5)
    assert_finite(attention_smoothness_loss(output["attention_maps"]), "attention_smoothness")

    mask = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])
    mask[:, :, 32:64, 32:64] = 1
    loss_fn = KPTANetLoss(lambda_boundary=0.2, lambda_uncertainty=0.1, lambda_attention_smooth=0.01)
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


def test_ablation_flags():
    x = torch.randn(2, 17, 128, 128)
    model = KPTANet(
        in_channels=17,
        base_channels=8,
        phase_indices={"pre": 0, "post": list(range(1, 9)), "subtraction": list(range(9, 17))},
        ablation={
            "disable_pseudo_kinetic_maps": True,
            "disable_pixelwise_phase_attention": True,
            "disable_uncertainty_refinement": True,
            "disable_boundary_head": True,
            "disable_uncertainty_head": True,
        },
    )
    output = model(x)
    assert output["kinetic_maps"].shape[1] == 1
    assert output["seg_logits"].shape == (2, 1, 128, 128)
    assert output["boundary_logits"].abs().sum() == 0
    assert output["uncertainty_logits"].abs().sum() == 0


if __name__ == "__main__":
    test_single_post_forward_and_loss()
    test_multiphase_forward_and_loss()
    test_ablation_flags()
    print("KPTA-Net sanity checks passed.")
