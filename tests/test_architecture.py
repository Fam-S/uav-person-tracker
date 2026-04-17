"""Architecture shape tests — run with: uv run pytest tests/test_architecture.py -v"""
from __future__ import annotations

import pytest
import torch

from models.backbone import MobileNetV3Backbone
from models.siamese import SiameseTracker, DepthwiseCrossCorrelation, SiameseHead
from models.losses import SiameseLoss, build_targets


DEVICE = torch.device("cpu")
B = 2  # batch size used in all tests


class TestBackbone:
    def test_output_shape_template(self):
        backbone = MobileNetV3Backbone(variant="small", pretrained=False).to(DEVICE)
        x = torch.rand(B, 3, 127, 127)
        out = backbone(x)
        assert out.shape == (B, 96, 4, 4)

    def test_output_shape_search(self):
        backbone = MobileNetV3Backbone(variant="small", pretrained=False).to(DEVICE)
        x = torch.rand(B, 3, 255, 255)
        out = backbone(x)
        assert out.shape == (B, 96, 8, 8)

    def test_frozen_by_default(self):
        backbone = MobileNetV3Backbone(variant="small", pretrained=False, trainable=False)
        for param in backbone.features.parameters():
            assert not param.requires_grad

    def test_projection_trainable(self):
        backbone = MobileNetV3Backbone(variant="small", pretrained=False)
        for param in backbone.projection.parameters():
            assert param.requires_grad


class TestCorrelation:
    def test_output_shape(self):
        corr = DepthwiseCrossCorrelation()
        z = torch.rand(B, 96, 4, 4)
        x = torch.rand(B, 96, 8, 8)
        out = corr(z, x)
        assert out.shape == (B, 96, 5, 5)

    def test_batch_size_mismatch_raises(self):
        corr = DepthwiseCrossCorrelation()
        z = torch.rand(1, 96, 4, 4)
        x = torch.rand(2, 96, 8, 8)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            corr(z, x)


class TestHead:
    def test_output_shapes(self):
        head = SiameseHead(in_channels=96)
        x = torch.rand(B, 96, 5, 5)
        out = head(x)
        assert out["cls_logits"].shape == (B, 1, 5, 5)
        assert out["bbox_pred"].shape == (B, 4, 5, 5)


class TestSiameseTracker:
    @pytest.fixture
    def model(self):
        return SiameseTracker(
            backbone_variant="small",
            pretrained_backbone=False,
            feature_channels=96,
            freeze_backbone=True,
        ).to(DEVICE)

    def test_forward_output_keys(self, model):
        template = torch.rand(B, 3, 127, 127)
        search = torch.rand(B, 3, 255, 255)
        out = model(template, search)
        assert set(out.keys()) == {"template_features", "search_features", "response_map", "cls_logits", "bbox_pred"}

    def test_cls_logits_shape(self, model):
        template = torch.rand(B, 3, 127, 127)
        search = torch.rand(B, 3, 255, 255)
        out = model(template, search)
        assert out["cls_logits"].shape == (B, 1, 5, 5)

    def test_bbox_pred_shape(self, model):
        template = torch.rand(B, 3, 127, 127)
        search = torch.rand(B, 3, 255, 255)
        out = model(template, search)
        assert out["bbox_pred"].shape == (B, 4, 5, 5)

    def test_batch_mismatch_raises(self, model):
        template = torch.rand(1, 3, 127, 127)
        search = torch.rand(2, 3, 255, 255)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            model(template, search)


class TestLoss:
    @pytest.fixture
    def model_outputs(self):
        model = SiameseTracker(backbone_variant="small", pretrained_backbone=False)
        template = torch.rand(B, 3, 127, 127)
        search = torch.rand(B, 3, 255, 255)
        with torch.no_grad():
            return model(template, search)

    def test_loss_is_scalar(self, model_outputs):
        criterion = SiameseLoss(search_size=255)
        search_bbox = torch.tensor([[50.0, 60.0, 80.0, 100.0], [70.0, 80.0, 60.0, 90.0]])
        loss_out = criterion(
            cls_logits=model_outputs["cls_logits"],
            bbox_pred=model_outputs["bbox_pred"],
            search_bbox=search_bbox,
        )
        assert loss_out.total_loss.shape == ()
        assert loss_out.total_loss.item() > 0

    def test_loss_components_finite(self, model_outputs):
        criterion = SiameseLoss(search_size=255)
        search_bbox = torch.tensor([[50.0, 60.0, 80.0, 100.0], [70.0, 80.0, 60.0, 90.0]])
        loss_out = criterion(
            cls_logits=model_outputs["cls_logits"],
            bbox_pred=model_outputs["bbox_pred"],
            search_bbox=search_bbox,
        )
        assert torch.isfinite(loss_out.total_loss)
        assert torch.isfinite(loss_out.cls_loss)
        assert torch.isfinite(loss_out.reg_loss)
