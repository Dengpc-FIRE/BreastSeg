import importlib.util
from pathlib import Path


OFFICIAL_MODEL_PATH = Path(__file__).resolve().parents[2] / "PDF-UNet.py"


def build_pdfunet(cfg):
    spec = importlib.util.spec_from_file_location("official_pdfunet", OFFICIAL_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PDFUNet(
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        feature_channels=list(cfg["model"]["feature_channels"]),
        bilinear=bool(cfg["model"]["bilinear"]),
        dropout=list(cfg["model"]["dropout"]),
        use_residual_projection=bool(cfg["model"].get("use_residual_projection", True)),
    )

