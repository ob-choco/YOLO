import numpy as np
import torch
from PIL import Image

from yolo.tools.data_augmentation import HorizontalFlip, VerticalFlip


def _img():
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


def test_horizontal_flip_mirrors_polygon_x():
    img = _img()
    boxes = torch.tensor([[0.0, 0.2, 0.3, 0.4, 0.5]])  # cls + xyxy normalized
    polygons = [np.array([[0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.2, 0.5]], dtype=np.float32)]
    flip = HorizontalFlip(prob=1.0)
    out = flip(img, boxes, polygons=polygons)
    assert len(out) == 3, "polygons-aware call should return a 3-tuple"
    _, _, polygons_out = out
    pts = polygons_out[0].reshape(-1, 2)
    np.testing.assert_allclose(pts[:, 0], 1 - np.array([0.2, 0.4, 0.4, 0.2]), atol=1e-6)
    np.testing.assert_allclose(pts[:, 1], [0.3, 0.3, 0.5, 0.5], atol=1e-6)


def test_horizontal_flip_does_not_break_detection_callers():
    """Without polygons, signature stays a 2-tuple (image, boxes)."""
    img = _img()
    boxes = torch.tensor([[0.0, 0.2, 0.3, 0.4, 0.5]])
    flip = HorizontalFlip(prob=1.0)
    out = flip(img, boxes)
    assert len(out) == 2, "detection-only call should remain a 2-tuple"


def test_vertical_flip_mirrors_polygon_y():
    img = _img()
    boxes = torch.tensor([[0.0, 0.2, 0.3, 0.4, 0.5]])
    polygons = [np.array([[0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.2, 0.5]], dtype=np.float32)]
    flip = VerticalFlip(prob=1.0)
    out = flip(img, boxes, polygons=polygons)
    assert len(out) == 3
    _, _, polygons_out = out
    pts = polygons_out[0].reshape(-1, 2)
    np.testing.assert_allclose(pts[:, 1], 1 - np.array([0.3, 0.3, 0.5, 0.5]), atol=1e-6)
    np.testing.assert_allclose(pts[:, 0], [0.2, 0.4, 0.4, 0.2], atol=1e-6)


def test_vertical_flip_no_polygons_returns_2tuple():
    img = _img()
    boxes = torch.tensor([[0.0, 0.2, 0.3, 0.4, 0.5]])
    flip = VerticalFlip(prob=1.0)
    out = flip(img, boxes)
    assert len(out) == 2
