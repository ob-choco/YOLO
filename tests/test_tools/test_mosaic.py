"""Tests for Mosaic augmentation, polygon-aware extension and detection backward-compat."""
import numpy as np
import torch
from PIL import Image

from yolo.tools.data_augmentation import Mosaic


class _FakeDataset:
    """Stand-in for the dataset's parent reference. Provides base_size and
    get_more_data() returning either 2-tuples (detection) or 3-tuples (segmentation).
    """

    def __init__(self, samples, base_size=64):
        self.samples = samples
        self.base_size = base_size

    def get_more_data(self, n):
        return self.samples[:n]


def _img(color):
    return Image.new("RGB", (64, 64), color=color)


def _box():
    return torch.tensor([[0.0, 0.1, 0.1, 0.5, 0.5]], dtype=torch.float32)


def _poly():
    return [np.array([[0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5]], dtype=np.float32)]


def test_mosaic_with_polygons_returns_three_tuple_and_clips_to_unit_square():
    samples = [(_img((20, 0, 0)), _box(), _poly()) for _ in range(3)]
    dataset = _FakeDataset(samples=samples, base_size=64)
    m = Mosaic(prob=1.0)
    m.set_parent(dataset)
    out = m(_img((50, 50, 50)), _box(), polygons=_poly())
    assert len(out) == 3, "polygons-mode call should return a 3-tuple"
    img_out, boxes_out, polys_out = out
    assert isinstance(img_out, Image.Image)
    assert boxes_out.shape[0] == 4   # 4 images × 1 box each
    assert len(polys_out) == 4
    for poly in polys_out:
        pts = poly.reshape(-1, 2)
        assert (pts >= 0.0).all() and (pts <= 1.0).all()


def test_mosaic_polygons_and_boxes_in_same_order():
    """Per-instance correspondence: boxes_out[i] and polys_out[i] originate from the
    same source instance. Verify that each polygon's bbox is consistent with the
    output box."""
    samples = [(_img((20, 0, 0)), _box(), _poly()) for _ in range(3)]
    dataset = _FakeDataset(samples=samples, base_size=64)
    m = Mosaic(prob=1.0)
    m.set_parent(dataset)
    _, boxes_out, polys_out = m(_img((50, 50, 50)), _box(), polygons=_poly())
    for box, poly in zip(boxes_out, polys_out):
        x1, y1, x2, y2 = box[1].item(), box[2].item(), box[3].item(), box[4].item()
        pts = poly.reshape(-1, 2)
        # polygon bounding box should match the output box (within float tolerance)
        np.testing.assert_allclose(pts[:, 0].min(), x1, atol=1e-4)
        np.testing.assert_allclose(pts[:, 0].max(), x2, atol=1e-4)
        np.testing.assert_allclose(pts[:, 1].min(), y1, atol=1e-4)
        np.testing.assert_allclose(pts[:, 1].max(), y2, atol=1e-4)


def test_mosaic_detection_mode_unchanged():
    """Without polygons, the call returns (image, boxes) — backward compat."""
    samples = [(_img((20, 0, 0)), _box()) for _ in range(3)]
    dataset = _FakeDataset(samples=samples, base_size=64)
    m = Mosaic(prob=1.0)
    m.set_parent(dataset)
    out = m(_img((50, 50, 50)), _box())
    assert len(out) == 2
    img_out, boxes_out = out
    assert isinstance(img_out, Image.Image)
    assert boxes_out.shape[0] == 4   # still 4 images × 1 box


def test_mosaic_skipped_when_prob_zero():
    samples = [(_img((20, 0, 0)), _box(), _poly()) for _ in range(3)]
    dataset = _FakeDataset(samples=samples, base_size=64)
    m = Mosaic(prob=0.0)
    m.set_parent(dataset)
    image_in = _img((50, 50, 50))
    boxes_in = _box()
    out = m(image_in, boxes_in, polygons=_poly())
    # When skipped, return signature drops to 2-tuple (matches detection path).
    # This matches existing behavior of the no-op early return.
    assert len(out) == 2
    img_out, boxes_out = out
    assert img_out is image_in
    assert torch.equal(boxes_out, boxes_in)
