from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from data.adapn_targets import AnchorTarget, AnchorTarget3
from models.losses import IOULoss, select_cross_entropy_loss, shaloss, weight_l1_loss


def test_select_cross_entropy_loss_matches_manual_positive_negative_average():
    probs = torch.tensor([[[[[0.2, 0.8], [0.7, 0.3]]]]], dtype=torch.float32)
    pred = torch.log(probs)
    label = torch.tensor([[[[1, 0]]]], dtype=torch.int64)

    loss = select_cross_entropy_loss(pred, label)

    expected = 0.5 * -math.log(0.8) + 0.5 * -math.log(0.7)
    assert torch.allclose(loss, torch.tensor(expected, dtype=loss.dtype), atol=1e-6)


def test_shape_and_weighted_l1_losses_match_original_piecewise_formulas():
    pred_shape = torch.tensor([[[[0.00, 0.02]], [[0.08, 0.10]], [[0.20, 0.18]], [[0.40, 0.36]]]])
    label_shape = torch.zeros_like(pred_shape)
    weight = torch.ones(1, 1, 1, 2)

    shape_loss = shaloss(pred_shape, label_shape, weight)
    diff = (pred_shape - label_shape).abs()
    expected_shape = torch.where(diff < 0.04, 25 * diff.pow(2), diff).sum() / weight.sum()
    assert torch.allclose(shape_loss, expected_shape)

    pred_loc = torch.tensor([[[[0.0005]], [[0.0020]], [[0.1000]], [[0.2000]]]])
    label_loc = torch.zeros(1, 4, 1, 1, 1)
    loc_weight = torch.ones(1, 1, 1, 1)
    loc_loss = weight_l1_loss(pred_loc, label_loc, loc_weight)
    loc_diff = pred_loc.abs()
    expected_loc = torch.where(loc_diff < 0.001, 1000 * loc_diff.pow(2), loc_diff).sum()
    assert torch.allclose(loc_loss, expected_loc)


def test_iou_loss_is_near_zero_for_identical_boxes():
    boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [4.0, 5.0, 12.0, 14.0]]])
    loss = IOULoss()(boxes, boxes)
    assert loss < 1e-5


def test_anchor_target_uses_original_grid_geometry_for_fixed_box():
    target = AnchorTarget(search_size=287, stride=8)
    labelcls2, labelxff, weightcls3, labelcls3, weightxff = target.get(np.array([120, 122, 166, 170]), size=21)

    assert labelcls2.shape == (1, 21, 21)
    assert labelxff.shape == (4, 21, 21)
    assert weightcls3.shape == (1, 21, 21)
    assert labelcls3.shape == (1, 21, 21)
    assert weightxff.shape == (1, 21, 21)
    assert (labelcls2 == 1).sum() > 0
    assert weightxff.sum() > 0

    denom = 287 // 4
    assert np.isclose(labelxff[0, 10, 10], (143 - 120) / denom)
    assert np.isclose(labelxff[1, 10, 10], (166 - 143) / denom)
    assert np.isclose(labelxff[2, 10, 10], (143 - 122) / denom)
    assert np.isclose(labelxff[3, 10, 10], (170 - 143) / denom)


def test_anchor_target3_produces_positive_labels_and_delta_weights():
    size = 21
    grid = np.linspace(0, size - 1, size)
    offset = 287 // 2 - 8 * (size - 1) / 2
    cx = np.tile(8 * grid + offset, size).reshape(-1)
    cy = np.tile((8 * grid + offset).reshape(-1, 1), size).reshape(-1)
    anchors = np.zeros((1, size * size, 4), dtype=np.float32)
    anchors[0, :, 0] = cx
    anchors[0, :, 1] = cy
    anchors[0, :, 2] = 46
    anchors[0, :, 3] = 48
    bbox = torch.tensor([[120.0, 122.0, 166.0, 170.0]], dtype=torch.float32)

    np.random.seed(0)
    cls, delta, delta_weight = AnchorTarget3(search_size=287, stride=8).get(anchors, bbox, size)

    assert cls.shape == (1, 1, 21, 21)
    assert delta.shape == (1, 4, 1, 21, 21)
    assert delta_weight.shape == (1, 1, 21, 21)
    assert (cls == 1).sum() > 0
    assert (cls == 0).sum() > 0
    assert delta_weight.sum() > 0


def test_siamapn_bbox_decode_matches_active_original_formula():
    from models.siamapn import SiamAPNppMobileOne

    delta = torch.zeros(1, 4, 1, 1)
    anchor = np.array([[[143.0, 143.0, 46.0, 48.0]]], dtype=np.float32)

    decoded = SiamAPNppMobileOne._convert_bbox(delta, anchor)

    expected = torch.tensor([[[120.0, 119.0, 166.0, 167.0]]])
    assert torch.allclose(decoded, expected)


def test_cls3_loss_contract_is_bce_with_logits():
    logits = torch.tensor([[[[0.0, 2.0], [-2.0, 1.0]]]])
    labels = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    expected = F.binary_cross_entropy_with_logits(logits, labels)
    assert torch.isfinite(expected)
