from __future__ import annotations

import numpy as np

from data.crop_utils import project_box_from_crop, project_box_to_crop


def test_project_box_round_trip_stays_consistent():
    reference_box = np.asarray([40.0, 30.0, 50.0, 80.0], dtype=np.float32)
    search_box = np.asarray([55.0, 45.0, 45.0, 70.0], dtype=np.float32)

    crop_box = project_box_to_crop(
        search_box_xywh=search_box,
        reference_box_xywh=reference_box,
        out_size=255,
        context_amount=0.5,
        center_override=(80.0, 85.0),
        area_scale=2.0,
    )
    restored_box = project_box_from_crop(
        crop_box_xywh=crop_box,
        reference_box_xywh=reference_box,
        out_size=255,
        context_amount=0.5,
        center_override=(80.0, 85.0),
        area_scale=2.0,
    )

    assert np.allclose(np.asarray(restored_box, dtype=np.float32), search_box, atol=1.0)


def test_project_box_to_crop_clips_edges_consistently():
    reference_box = np.asarray([50.0, 50.0, 20.0, 20.0], dtype=np.float32)
    search_box = np.asarray([20.0, 50.0, 40.0, 20.0], dtype=np.float32)

    crop_box = project_box_to_crop(
        search_box_xywh=search_box,
        reference_box_xywh=reference_box,
        out_size=255,
        context_amount=0.5,
        center_override=(60.0, 60.0),
        area_scale=1.0,
    )

    assert np.allclose(crop_box, np.asarray([0.0, 63.75, 127.5, 127.5], dtype=np.float32), atol=1e-4)


def test_crop_and_resize_handles_fractional_left_top_padding():
    from data.crop_utils import crop_and_resize

    frame = np.full((8, 8, 3), fill_value=127, dtype=np.uint8)
    box = np.asarray([0.0, 0.0, 2.0, 2.0], dtype=np.float32)

    crop = crop_and_resize(
        frame,
        box,
        out_size=16,
        context_amount=0.0,
        center_override=(0.4, 0.4),
        area_scale=1.0,
    )

    assert crop.shape == (16, 16, 3)
    assert crop.dtype == np.uint8
