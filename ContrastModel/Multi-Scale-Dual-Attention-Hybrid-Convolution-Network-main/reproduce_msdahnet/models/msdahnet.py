import sys
from pathlib import Path


OFFICIAL_ROOT = Path(__file__).resolve().parents[2]
if str(OFFICIAL_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_ROOT))

from resunet import DualA_Net  # noqa: E402


def build_msdahnet(in_channels: int = 1, num_classes: int = 1):
    """Build MSDAHNet from the official released implementation.

    The reproduction keeps the official forward path. Only input/output channel
    configurability and nn.Module inheritance are patched in resunet.py.
    """
    return DualA_Net(in_channels=in_channels, num_classes=num_classes)
