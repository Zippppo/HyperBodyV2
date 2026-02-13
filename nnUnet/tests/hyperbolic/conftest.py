import os
import sys
from pathlib import Path


NNUNET_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = NNUNET_ROOT / "nnUNet_data"

if str(NNUNET_ROOT) not in sys.path:
    sys.path.insert(0, str(NNUNET_ROOT))

os.environ.setdefault("nnUNet_raw", str(DATA_ROOT / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(DATA_ROOT / "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", str(DATA_ROOT / "nnUNet_results"))
