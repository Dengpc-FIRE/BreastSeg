import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.kpta_25d_net import KPTA25DNet
from prepare_breastdm_25d import neighbor_indices
from train.losses import KPTA25DNetLoss


def assert_finite(tensor, name):
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf"


def run_forward_case(x, phase_indices):
    model = KPTA25DNet(
        in_phases=x.shape[2],
        num_slices=x.shape[1],
        base_channels=4,
        phase_indices=phase_indices,
        hybrid_encoder={"transformer_depth": 1, "num_heads": 2},
        return_dict=True,
    )
    output = model(x)
    assert output["seg_logits"].shape == (x.shape[0], 1, x.shape[-2], x.shape[-1])
    assert output["boundary_logits"].shape == (x.shape[0], 1, x.shape[-2], x.shape[-1])
    assert output["uncertainty_logits"].shape == (x.shape[0], 1, x.shape[-2], x.shape[-1])
    assert output["kinetic_maps"].ndim == 5
    assert output["kinetic_maps"].shape[:2] == (x.shape[0], x.shape[1])
    assert_finite(output["kinetic_maps"], "kinetic_maps")
    assert output["debug"]["fused_phase_shape"] == (x.shape[0], 4, x.shape[-2], x.shape[-1])
    assert output["debug"]["kinetic_feature_shape"] == (x.shape[0], 4, x.shape[-2], x.shape[-1])
    assert output["debug"]["fused0_shape"] == (x.shape[0], 4, x.shape[-2], x.shape[-1])
    assert_finite(output["debug"]["fused_phase_l2"], "fused_phase_l2")
    assert_finite(output["debug"]["kinetic_feature_l2"], "kinetic_feature_l2")
    assert_finite(output["debug"]["fused0_l2"], "fused0_l2")

    attn = output["phase_attention_maps"][0]
    assert output["phase_attention"].shape == (x.shape[0], x.shape[2], 1, x.shape[-2], x.shape[-1])
    assert torch.allclose(attn.sum(dim=1), torch.ones_like(attn.sum(dim=1)), atol=1e-5)
    assert output["phase_attention_type"] == "pdwa"
    assert output["debug"]["phase_attention_type"] == "pdwa"
    assert output["debug"]["sub_maps_shape"] is not None
    assert "feature_score" in output["pdwa_debug"]
    assert "diff_gate" in output["pdwa_debug"]
    assert_finite(output["pdwa_debug"]["feature_score"], "pdwa_feature_score")
    assert_finite(output["pdwa_debug"]["diff_gate"], "pdwa_diff_gate")

    mask = torch.zeros(x.shape[0], 1, x.shape[-2], x.shape[-1])
    mask[:, :, 32:64, 32:64] = 1
    loss_fn = KPTA25DNetLoss(lambda_boundary=0.2, lambda_uncertainty=0.1, lambda_attention_smooth=0.01)
    loss = loss_fn(output, mask, images=x)
    assert_finite(loss, "loss")
    loss.backward()
    return output


def test_full_phase_forward_and_loss():
    x = torch.randn(2, 3, 17, 128, 128)
    run_forward_case(x, {"pre": 0, "post": list(range(1, 9)), "subtraction": list(range(9, 17))})


def test_single_post_forward_and_loss():
    x = torch.randn(2, 3, 3, 128, 128)
    run_forward_case(x, {"pre": 0, "post": [1], "subtraction": [2]})


def test_csam_slice_attention_forward():
    x = torch.randn(1, 3, 5, 64, 64)
    model = KPTA25DNet(
        in_phases=5,
        num_slices=3,
        base_channels=4,
        phase_indices={"pre": 0, "post": [1, 2], "subtraction": [3, 4]},
        slice_attention_type="csam",
        csam_aggregate="center",
        csam_uncertainty=False,
        hybrid_encoder={"transformer_depth": 1, "num_heads": 2},
        return_dict=True,
    )
    output = model(x)
    assert output["seg_logits"].shape == (1, 1, 64, 64)
    assert output["slice_attention_type"] == "csam"
    assert output["csam_enabled"] is True
    assert output["debug"]["slice_attention_type"] == "csam"
    csam_attn = output["slice_attention_maps"][0]
    assert csam_attn.shape == (1, 3, 5, 1, 1, 1)
    assert_finite(csam_attn, "csam_slice_attention")


def test_pixelwise_phase_attention_forward():
    x = torch.randn(1, 3, 5, 64, 64)
    model = KPTA25DNet(
        in_phases=5,
        num_slices=3,
        base_channels=4,
        phase_indices={"pre": 0, "post": [1, 2], "subtraction": [3, 4]},
        phase_attention_type="pixelwise",
        hybrid_encoder={"transformer_depth": 1, "num_heads": 2},
        return_dict=True,
    )
    output = model(x)
    assert output["seg_logits"].shape == (1, 1, 64, 64)
    assert output["phase_attention_type"] == "pixelwise"
    assert output["pdwa_debug"] == {}


def test_neighbor_padding_indices():
    assert neighbor_indices(0, count=5, num_slices=3) == [0, 0, 1]
    assert neighbor_indices(4, count=5, num_slices=3) == [3, 4, 4]


def test_ablation_flags():
    x = torch.randn(2, 3, 17, 128, 128)
    model = KPTA25DNet(
        in_phases=17,
        num_slices=3,
        base_channels=4,
        phase_indices={"pre": 0, "post": list(range(1, 9)), "subtraction": list(range(9, 17))},
        hybrid_encoder={"transformer_depth": 1, "num_heads": 2},
        ablation={
            "disable_kinetic_maps": True,
            "disable_slice_context": True,
            "disable_pixelwise_phase_attention": True,
            "disable_kinetic_raw_fusion": True,
            "disable_transformer_bottleneck": True,
            "disable_boundary_head": True,
            "disable_uncertainty_head": True,
            "disable_uncertainty_refinement": True,
        },
    )
    output = model(x)
    assert output["kinetic_maps"].shape[2] == 1
    assert output["seg_logits"].shape == (2, 1, 128, 128)
    assert output["debug"]["disable_kinetic_raw_fusion"] is True
    assert output["boundary_logits"].abs().sum() == 0
    assert output["uncertainty_logits"].abs().sum() == 0


if __name__ == "__main__":
    test_full_phase_forward_and_loss()
    test_single_post_forward_and_loss()
    test_csam_slice_attention_forward()
    test_pixelwise_phase_attention_forward()
    test_neighbor_padding_indices()
    test_ablation_flags()
    print("KPTA-2.5D sanity checks passed.")
