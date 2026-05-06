import numpy as np
import torch

from yolo.utils.dataset_utils import rasterize_masks


def test_rasterize_single_square_polygon():
    # square polygon covering [0.25, 0.5] in normalized coords on a 640x640 image,
    # rasterized at stride 4 → mask shape (160, 160)
    poly = np.array([[0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5]])
    masks = rasterize_masks([poly], image_size=(640, 640), mask_ratio=4)
    assert masks.shape == (1, 160, 160)
    assert masks.dtype == torch.uint8
    # Pixels inside [40, 80] x [40, 80] should be 1
    assert masks[0, 60, 60] == 1
    assert masks[0, 30, 30] == 0
    assert masks[0, 90, 90] == 0


def test_rasterize_empty_returns_zero_tensor():
    masks = rasterize_masks([], image_size=(640, 640), mask_ratio=4)
    assert masks.shape == (0, 160, 160)


def test_rasterize_multiple_polygons():
    polys = [
        np.array([[0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1]]),
        np.array([[0.9, 0.9, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0]]),
    ]
    masks = rasterize_masks(polys, image_size=(640, 640), mask_ratio=4)
    assert masks.shape == (2, 160, 160)
    assert masks[0].sum() > 0   # top-left corner has pixels
    assert masks[1].sum() > 0   # bottom-right corner has pixels
