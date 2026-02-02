import platform
import os
from pathlib import Path
from typing import Optional, Tuple, List


def get_device_and_compute_type(
    preferred_device: Optional[str] = None, preferred_compute: Optional[str] = None
) -> Tuple[str, List[str]]:
    """Select device and an ordered list of compute_type fallbacks.

    On macOS we prefer running under CPU so that PyTorch can use MPS where
    available. We attempt compute types in order: float16 -> int8_float16 -> int8.
    The caller should try each compute_type when loading the model and fall
    back on failure.
    """
    # Device preference: allow override but default to 'cpu' on macOS so MPS
    # path can be used by PyTorch; otherwise prefer 'auto'.
    if preferred_device:
        device = preferred_device
    else:
        if platform.system() == "Darwin":
            device = "cpu"
        else:
            device = "auto"

    # Candidate compute types to try in order (most performant first)
    # Note: CTranslate2/faster-whisper supports: float16, int8_float16, int8
    if preferred_compute:
        compute_candidates = [preferred_compute]
    else:
        compute_candidates = ["float16", "int8_float16", "int8"]

    return device, compute_candidates


def get_app_models_dir() -> str:
    """Get application-specific models directory"""
    if platform.system() == "Darwin":  # macOS
        models_dir = (
            Path.home() / "Library" / "Application Support" / "SubsGenie" / "models"
        )
    elif platform.system() == "Windows":
        models_dir = Path(os.environ.get("APPDATA", "")) / "SubsGenie" / "models"
    else:  # Linux
        models_dir = Path.home() / ".config" / "SubsGenie" / "models"

    models_dir.mkdir(parents=True, exist_ok=True)
    return str(models_dir)
