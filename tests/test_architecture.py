from __future__ import annotations

import torch

from models import SiamAPNppMobileOne
from models.backbone import MobileOneS2Backbone
from models.losses import SiamAPNLoss


DEVICE = torch.device("cpu")
BATCH = 2


def test_mobileone_backbone_outputs_two_feature_levels():
    backbone = MobileOneS2Backbone(pretrained_path=None, normalize_input=True).to(DEVICE)
    x = torch.rand(BATCH, 3, 255, 255, device=DEVICE)
    low_level, high_level = backbone(x)
    assert low_level.shape == (BATCH, 256, 32, 32)
    assert high_level.shape == (BATCH, 640, 16, 16)


def test_siamapn_forward_shapes():
    model = SiamAPNppMobileOne(feature_channels=96, pretrained_path=None).to(DEVICE)
    template = torch.rand(BATCH, 3, 127, 127, device=DEVICE)
    search = torch.rand(BATCH, 3, 255, 255, device=DEVICE)
    outputs = model(template, search)
    assert outputs["bbox_pred"].shape == (BATCH, 4)


def test_loss_stays_finite():
    model = SiamAPNppMobileOne(feature_channels=96, pretrained_path=None).to(DEVICE)
    criterion = SiamAPNLoss(search_size=255).to(DEVICE)
    template = torch.rand(BATCH, 3, 127, 127, device=DEVICE)
    search = torch.rand(BATCH, 3, 255, 255, device=DEVICE)
    search_bbox = torch.tensor(
        [[60.0, 70.0, 50.0, 80.0], [40.0, 30.0, 70.0, 90.0]],
        device=DEVICE,
    )
    outputs = model(template, search)
    loss_out = criterion(bbox_pred=outputs["bbox_pred"], search_bbox=search_bbox)
    assert torch.isfinite(loss_out.total_loss)
    assert torch.isfinite(loss_out.reg_loss)
