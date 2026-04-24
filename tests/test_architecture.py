from __future__ import annotations

import torch

from models import SiamAPNppMobileOne
from models.backbone import MobileOneS2Backbone


DEVICE = torch.device("cpu")
BATCH = 2


def test_mobileone_backbone_outputs_two_feature_levels():
    backbone = MobileOneS2Backbone(pretrained_path=None, normalize_input=True).to(DEVICE)
    x = torch.rand(BATCH, 3, 287, 287, device=DEVICE)
    low_level, high_level = backbone(x)
    assert low_level.shape == (BATCH, 384, 28, 28)
    assert high_level.shape == (BATCH, 256, 26, 26)


def test_siamapn_forward_shapes():
    model = SiamAPNppMobileOne(feature_channels=96, pretrained_path=None).to(DEVICE)
    template = torch.rand(BATCH, 3, 127, 127, device=DEVICE)
    search = torch.rand(BATCH, 3, 287, 287, device=DEVICE)
    outputs = model(template, search)
    assert outputs["cls1"].shape == (BATCH, 2, 21, 21)
    assert outputs["cls2"].shape == (BATCH, 2, 21, 21)
    assert outputs["cls3"].shape == (BATCH, 1, 21, 21)
    assert outputs["loc"].shape == (BATCH, 4, 21, 21)
    assert outputs["ranchors"].shape == (BATCH, 4, 21, 21)


def test_loss_stays_finite():
    model = SiamAPNppMobileOne(feature_channels=96, pretrained_path=None).to(DEVICE)
    template = torch.rand(BATCH, 3, 127, 127, device=DEVICE)
    search = torch.rand(BATCH, 3, 287, 287, device=DEVICE)
    batch = {
        "template": template,
        "search": search,
        "bbox": torch.tensor([[120.0, 122.0, 166.0, 170.0], [118.0, 119.0, 170.0, 174.0]], device=DEVICE),
        "label_cls2": torch.zeros(BATCH, 1, 21, 21, device=DEVICE),
        "labelxff": torch.zeros(BATCH, 4, 21, 21, device=DEVICE),
        "labelcls3": torch.zeros(BATCH, 1, 21, 21, device=DEVICE),
        "weightxff": torch.ones(BATCH, 1, 21, 21, device=DEVICE),
    }
    outputs = model(batch)
    assert torch.isfinite(outputs["total_loss"])
    assert torch.isfinite(outputs["loc_loss"])
