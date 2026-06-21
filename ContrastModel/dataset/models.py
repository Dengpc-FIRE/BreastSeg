from __future__ import annotations

import contextlib
import importlib
import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def prepend_sys_path(*paths: Path) -> Iterator[None]:
    inserted = []
    for path in reversed([str(p.resolve()) for p in paths]):
        if path not in sys.path:
            sys.path.insert(0, path)
            inserted.append(path)
    try:
        yield
    finally:
        for path in inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(path)


@contextlib.contextmanager
def isolated_import_prefixes(*prefixes: str) -> Iterator[None]:
    def matches(name: str) -> bool:
        return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)

    saved = {name: module for name, module in sys.modules.items() if matches(name)}
    for name in list(saved):
        del sys.modules[name]
    try:
        yield
    finally:
        for name in [name for name in sys.modules if matches(name)]:
            del sys.modules[name]
        sys.modules.update(saved)


@contextlib.contextmanager
def force_local_package(package_name: str, package_dir: Path) -> Iterator[None]:
    def matches(name: str) -> bool:
        return name == package_name or name.startswith(f"{package_name}.")

    saved = {name: module for name, module in sys.modules.items() if matches(name)}
    for name in list(saved):
        del sys.modules[name]

    package_path = str(package_dir.resolve())
    package = types.ModuleType(package_name)
    package.__path__ = [package_path]
    package.__package__ = package_name
    package.__file__ = str((package_dir / "__init__.py").resolve())
    package.__spec__ = importlib.machinery.ModuleSpec(
        package_name,
        loader=None,
        is_package=True,
    )
    package.__spec__.submodule_search_locations = [package_path]
    sys.modules[package_name] = package

    try:
        yield
    finally:
        for name in [name for name in sys.modules if matches(name)]:
            del sys.modules[name]
        sys.modules.update(saved)


def _logit_from_prob(prob: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(prob / (1.0 - prob))


def extract_logits(output: Any) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, dict):
        extra = output.get("extra_loss")
        for key in ("logits", "out", "seg", "prediction"):
            if key in output:
                return extract_logits(output[key])[0], extra
        raise TypeError(f"Cannot find logits in output dict keys {list(output.keys())}")
    if torch.is_tensor(output):
        return output, None
    if isinstance(output, (list, tuple)):
        extra = None
        if len(output) == 2 and torch.is_tensor(output[1]) and output[1].ndim == 0:
            extra = output[1]
        for item in reversed(output):
            try:
                logits, nested_extra = extract_logits(item)
                return logits, extra if extra is not None else nested_extra
            except TypeError:
                continue
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def forward_model(model: nn.Module, images: torch.Tensor, masks: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if getattr(model, "needs_target", False):
        output = model(images, masks)
    else:
        output = model(images)
    return extract_logits(output)


def align_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[2:] == target.shape[2:]:
        return logits
    mode = "trilinear" if logits.ndim == 5 else "bilinear"
    return F.interpolate(logits, size=target.shape[2:], mode=mode, align_corners=False)


def adapt_first_conv(model: nn.Module, in_channels: int) -> bool:
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.Conv2d, nn.Conv3d)) and child.in_channels != in_channels:
                conv_cls = child.__class__
                new_conv = conv_cls(
                    in_channels,
                    child.out_channels,
                    child.kernel_size,
                    child.stride,
                    child.padding,
                    child.dilation,
                    child.groups if child.groups == 1 else 1,
                    child.bias is not None,
                    child.padding_mode,
                )
                nn.init.kaiming_normal_(new_conv.weight, nonlinearity="relu")
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)
                setattr(module, name, new_conv)
                return True
    return False


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetPlusPlus2D(nn.Module):
    def __init__(self, in_channels: int = 17, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        nb = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = ConvBlock2D(in_channels, nb[0])
        self.conv1_0 = ConvBlock2D(nb[0], nb[1])
        self.conv2_0 = ConvBlock2D(nb[1], nb[2])
        self.conv3_0 = ConvBlock2D(nb[2], nb[3])
        self.conv4_0 = ConvBlock2D(nb[3], nb[4])

        self.conv0_1 = ConvBlock2D(nb[0] + nb[1], nb[0])
        self.conv1_1 = ConvBlock2D(nb[1] + nb[2], nb[1])
        self.conv2_1 = ConvBlock2D(nb[2] + nb[3], nb[2])
        self.conv3_1 = ConvBlock2D(nb[3] + nb[4], nb[3])

        self.conv0_2 = ConvBlock2D(nb[0] * 2 + nb[1], nb[0])
        self.conv1_2 = ConvBlock2D(nb[1] * 2 + nb[2], nb[1])
        self.conv2_2 = ConvBlock2D(nb[2] * 2 + nb[3], nb[2])

        self.conv0_3 = ConvBlock2D(nb[0] * 3 + nb[1], nb[0])
        self.conv1_3 = ConvBlock2D(nb[1] * 3 + nb[2], nb[1])
        self.conv0_4 = ConvBlock2D(nb[0] * 4 + nb[1], nb[0])
        self.final = nn.Conv2d(nb[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        return self.final(x0_4)


class ProbOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module, output_index: int | None = None) -> None:
        super().__init__()
        self.model = model
        self.output_index = output_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if self.output_index is not None and isinstance(output, (list, tuple)):
            output = output[self.output_index]
        prob, _ = extract_logits(output)
        if not torch.isfinite(prob).all():
            prob = torch.nan_to_num(prob, nan=0.5, posinf=1.0 - 1e-4, neginf=1e-4)
        return _logit_from_prob(prob)


class PLHNWrapper(nn.Module):
    needs_target = True

    def __init__(self, net: nn.Module, proto_loss: nn.Module | None = None, output_index: int = -1) -> None:
        super().__init__()
        self.net = net
        self.proto_loss = proto_loss
        self.output_index = output_index

    def forward(self, x: torch.Tensor, target: torch.Tensor | None = None) -> Dict[str, torch.Tensor | None]:
        output = self.net(x, target) if target is not None else self.net(x)
        extra = None
        if target is not None and isinstance(output, (list, tuple)) and output and isinstance(output[0], dict):
            if self.proto_loss is not None:
                extra = self.proto_loss(output[0], target)
        selected = output
        if isinstance(output, (list, tuple)):
            selected = output[self.output_index]
        prob, _ = extract_logits(selected)
        if not torch.isfinite(prob).all():
            prob = torch.nan_to_num(prob, nan=0.5, posinf=1.0 - 1e-4, neginf=1e-4)
        return {"logits": _logit_from_prob(prob), "extra_loss": extra}


class PDPNetWrapper(nn.Module):
    needs_target = True

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor, target: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if target is None:
            if x.shape[0] > 1:
                logits = [self.forward(x[i : i + 1], None)["logits"] for i in range(x.shape[0])]
                return {"logits": torch.cat(logits, dim=0), "extra_loss": None}
            target = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
        logits_list, _, losses = self.net(x, target)
        logits = _logit_from_prob(logits_list[-1])
        extra = None
        if isinstance(losses, (list, tuple)) and losses:
            finite_losses = [loss for loss in losses if torch.is_tensor(loss) and torch.isfinite(loss).all()]
            if finite_losses:
                extra = sum(finite_losses)
        return {"logits": logits, "extra_loss": extra}


def _build_transunet(model_dir: Path, cfg: Dict[str, Any]) -> nn.Module:
    with prepend_sys_path(model_dir):
        vit = importlib.import_module("networks.vit_seg_modeling")
        vit_name = cfg["model"].get("vit_name", "ViT-B_16")
        config_vit = vit.CONFIGS[vit_name]
        config_vit.n_classes = cfg["model"]["out_channels"]
        config_vit.n_skip = int(cfg["model"].get("n_skip", 0))
        if vit_name.startswith("R50") and config_vit.n_skip:
            config_vit.patches.grid = (int(cfg["data"].get("image_size", [256, 256])[0] / 16), int(cfg["data"].get("image_size", [256, 256])[1] / 16))
        return vit.VisionTransformer(
            config_vit,
            img_size=int(cfg["data"].get("image_size", [256, 256])[0]),
            num_classes=cfg["model"]["out_channels"],
            in_channels=cfg["model"]["input_channels"],
        )


def _build_mobile_uvit(model_dir: Path, cfg: Dict[str, Any]) -> nn.Module:
    with prepend_sys_path(model_dir):
        module = importlib.import_module("network.MobileUViT")
        factory = getattr(module, cfg["model"].get("variant", "mobileuvit"))
        return factory(inch=cfg["model"]["input_channels"], out_channel=cfg["model"]["out_channels"])


def _build_emcad(model_dir: Path, cfg: Dict[str, Any]) -> nn.Module:
    with prepend_sys_path(model_dir):
        module = importlib.import_module("lib.networks")
        model = module.EMCADNet(
            num_classes=cfg["model"]["out_channels"],
            encoder=cfg["model"].get("encoder", "resnet18"),
            pretrain=False,
            in_channels=cfg["model"]["input_channels"],
        )
        return model


def _build_deeplab(model_dir: Path, cfg: Dict[str, Any]) -> nn.Module:
    with prepend_sys_path(model_dir):
        module = importlib.import_module("network.modeling")
        arch = cfg["model"].get("arch", "deeplabv3plus_mobilenet")
        model = getattr(module, arch)(
            num_classes=cfg["model"]["out_channels"],
            output_stride=int(cfg["model"].get("output_stride", 16)),
            pretrained_backbone=False,
        )
        adapt_first_conv(model, cfg["model"]["input_channels"])
        return model


def _build_pytorch_unet(model_dir: Path, cfg: Dict[str, Any]) -> nn.Module:
    with prepend_sys_path(model_dir):
        module = importlib.import_module("unet")
        return module.UNet(
            n_channels=cfg["model"]["input_channels"],
            n_classes=cfg["model"]["out_channels"],
            bilinear=bool(cfg["model"].get("bilinear", False)),
        )


def _build_msdahnet(model_dir: Path, cfg: Dict[str, Any]) -> nn.Module:
    with prepend_sys_path(model_dir):
        module = importlib.import_module("resunet")
        return module.DualA_Net(
            in_channels=cfg["model"]["input_channels"],
            num_classes=cfg["model"]["out_channels"],
        )


def _build_attention_gated(model_dir: Path, cfg: Dict[str, Any]) -> nn.Module:
    with prepend_sys_path(model_dir):
        network = cfg["model"].get("network", "unet_nonlocal")
        common_kwargs = {
            "n_classes": cfg["model"]["out_channels"],
            "in_channels": cfg["model"]["input_channels"],
            "feature_scale": int(cfg["model"].get("feature_scale", 4)),
            "is_batchnorm": True,
        }
        if network == "unet_nonlocal":
            module = importlib.import_module("models.networks.unet_nonlocal_2D")
            return module.unet_nonlocal_2D(is_deconv=True, **common_kwargs)
        if network == "unet":
            module = importlib.import_module("models.networks.unet_2D")
            return module.unet_2D(is_deconv=True, **common_kwargs)
        module = importlib.import_module("models.networks")
        return module.get_network(
            network,
            tensor_dim="2D",
            **common_kwargs,
        )


def _build_pdpnet(model_dir: Path, cfg: Dict[str, Any]) -> nn.Module:
    with force_local_package("model", model_dir / "model"), prepend_sys_path(model_dir):
        dpk = importlib.import_module("model.DPKNet")
        seg = dpk.DPKNet(channels=cfg["model"]["input_channels"])
        if not bool(cfg["model"].get("use_location_branch", False)):
            return ProbOutputWrapper(seg)
        dense = importlib.import_module("model.DenseNet5s")
        pdp = importlib.import_module("model.PDPNet")
        loc = dense.densenet121(ch_in=cfg["model"]["input_channels"])
        return PDPNetWrapper(pdp.PDPNet(locmodel=loc, segmodel=seg))


def _build_hcrt(model_dir: Path, cfg: Dict[str, Any]) -> nn.Module:
    with prepend_sys_path(model_dir):
        module = importlib.import_module("Model.HCRT")
        return module.HCRT(
            inch=cfg["model"]["input_channels"],
            outch=cfg["model"]["out_channels"],
            base_channeel=int(cfg["model"].get("base_channels", 32)),
            imgsize=list(cfg["data"].get("patch_size", [48, 128, 128])),
            hidden_size=int(cfg["model"].get("hidden_size", 256)),
        )


def _build_plhn(model_dir: Path, cfg: Dict[str, Any]) -> nn.Module:
    with prepend_sys_path(model_dir):
        module = importlib.import_module("Model.TokenSegV8_prototype_fusion_attentions")
        model = module.TokenSegV8(
            inch=cfg["model"]["input_channels"],
            outch=cfg["model"]["out_channels"],
            base_channeel=int(cfg["model"].get("base_channels", 16)),
            imgsize=list(cfg["data"].get("patch_size", [48, 128, 128])),
            hidden_size=int(cfg["model"].get("hidden_size", 192)),
            TransformerLayerNum=int(cfg["model"].get("transformer_layers", 4)),
        )
        output_index = int(cfg["model"].get("output_index", -1))
        proto_loss = None
        if bool(cfg["model"].get("use_prototype_loss", True)):
            loss_module = importlib.import_module("Model.modelv5.loss_proto")
            proto_loss = loss_module.PixelPrototypeCELoss()
        return PLHNWrapper(
            model,
            proto_loss=proto_loss,
            output_index=output_index,
        )


BUILDERS = {
    "transunet": _build_transunet,
    "mobile_uvit": _build_mobile_uvit,
    "emcad": _build_emcad,
    "deeplabv3plus": _build_deeplab,
    "pytorch_unet": _build_pytorch_unet,
    "msdahnet": _build_msdahnet,
    "attention_gated": _build_attention_gated,
    "pdpnet": _build_pdpnet,
    "unetplusplus": lambda model_dir, cfg: UNetPlusPlus2D(
        in_channels=cfg["model"]["input_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=int(cfg["model"].get("base_channels", 32)),
    ),
    "hcrt": _build_hcrt,
    "plhn": _build_plhn,
}


def build_model(model_key: str, cfg: Dict[str, Any], model_dir: str | Path) -> nn.Module:
    key = model_key.lower().replace("-", "_")
    if key not in BUILDERS:
        raise KeyError(f"Unsupported model key {model_key!r}. Available: {sorted(BUILDERS)}")
    return BUILDERS[key](Path(model_dir), cfg)
