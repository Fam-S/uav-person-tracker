

from pathlib import Path
import torch

from models.tracker.siamese_tracker import SiameseTracker


def load_model(checkpoint_path: str, device: str = None):
    """
    Load trained SiameseTracker model exactly as in training config.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ---- device ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    # ---- build model (IMPORTANT: same as training) ----
    model = SiameseTracker(
        backbone_variant="small",
        pretrained_backbone=True,
        feature_channels=96,
        freeze_backbone=True,
        normalize_input=True,
    )

    # ---- load checkpoint ----
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint missing 'model_state_dict'")

    state_dict = checkpoint["model_state_dict"]

    # ---- remove DataParallel prefix if exists ----
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    # ---- load weights ----
    model.load_state_dict(cleaned_state_dict, strict=False)

    # ---- device + eval ----
    model.to(device)
    model.eval()

    print(f"[OK] Model loaded from: {checkpoint_path}")
    print(f"[OK] Device: {device}")

    return model, device