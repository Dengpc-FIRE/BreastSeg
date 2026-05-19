import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.kpr_net import KPRNet, PhaseDropout
from train.losses import KPRNetLoss


def assert_finite(tensor, name):
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf"


def phase_indices_17():
    return {"pre": 0, "post": list(range(1, 9)), "subtraction": list(range(9, 17))}


def run_case(x, phase_indices):
    model = KPRNet(
        in_channels=x.shape[1],
        base_channels=8,
        phase_indices=phase_indices,
        phase_dropout={"enabled": True, "drop_prob": 0.3, "replacement": "zero", "min_available_post_phases": 1},
        return_dict=True,
    )
    model.train()
    mask = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])
    mask[:, :, 32:64, 32:64] = 1
    output = model(x)
    assert output["seg_logits"].shape == (x.shape[0], 1, x.shape[2], x.shape[3])
    assert output["phase_mask"].shape[0] == x.shape[0]
    assert output["phase_mask"].sum(dim=1).min() >= 1
    assert_finite(output["global_kinetic_code"], "global_kinetic_code")
    assert_finite(output["kinetic_maps"], "kinetic_maps")

    loss_fn = KPRNetLoss(lambda_contrastive=0.1, lambda_kinetic=0.05)
    loss = loss_fn(output, mask, images=x)
    assert_finite(loss, "loss")
    loss.backward()

    empty_loss = loss_fn(model(x.detach()), torch.zeros_like(mask), images=x)
    assert_finite(empty_loss, "empty_loss")


def test_full_phase_forward_and_loss():
    x = torch.randn(2, 17, 128, 128)
    run_case(x, phase_indices_17())


def test_single_post_forward_and_loss():
    x = torch.randn(2, 3, 128, 128)
    run_case(x, {"pre": 0, "post": [1], "subtraction": [2]})


def test_missing_phase_masks():
    x = torch.randn(2, 17, 128, 128)
    model = KPRNet(in_channels=17, base_channels=8, phase_indices=phase_indices_17())
    phase_mask = torch.ones(2, 8)
    phase_mask[:, 0] = 0
    output = model(x, phase_mask=phase_mask)
    assert torch.all(output["phase_mask"][:, 0] == 0)
    assert output["seg_logits"].shape == (2, 1, 128, 128)


def test_phase_dropout_constraints():
    x = torch.randn(2, 17, 64, 64)
    dropout = PhaseDropout(phase_indices=phase_indices_17(), enabled=True, drop_prob=1.0, min_available_post_phases=1)
    dropout.train()
    dropped, phase_mask = dropout(x)
    assert torch.allclose(dropped[:, 0], x[:, 0])
    assert phase_mask.sum(dim=1).min() >= 1


def test_ablation_flags():
    x = torch.randn(2, 17, 128, 128)
    model = KPRNet(
        in_channels=17,
        base_channels=8,
        phase_indices=phase_indices_17(),
        ablation={
            "disable_phase_dropout": True,
            "disable_kinetic_prior_encoder": True,
            "disable_kinetic_fusion": True,
        },
    )
    output = model(x)
    assert output["seg_logits"].shape == (2, 1, 128, 128)
    assert output["global_kinetic_code"].abs().sum() == 0


if __name__ == "__main__":
    test_full_phase_forward_and_loss()
    test_single_post_forward_and_loss()
    test_missing_phase_masks()
    test_phase_dropout_constraints()
    test_ablation_flags()
    print("KPR-Net sanity checks passed.")
