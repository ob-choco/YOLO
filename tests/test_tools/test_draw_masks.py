"""Sanity test for `draw_masks` overlay helper."""
import torch
from PIL import Image

from yolo.tools.drawer import draw_masks


def _gray_canvas(size=(64, 64)):
    return Image.new("RGB", size, color=(128, 128, 128))


def test_draw_masks_returns_pil_image_same_size():
    img = _gray_canvas()
    m = torch.zeros((64, 64), dtype=torch.uint8)
    m[10:30, 10:30] = 1
    out = draw_masks(img, [m], [[10, 10, 30, 30, 0]], idx2label=["hold", "volume"])
    assert isinstance(out, Image.Image)
    assert out.size == img.size


def test_draw_masks_blends_pixels_inside_mask():
    """Pixels inside the mask region should change color from gray."""
    img = _gray_canvas()
    m = torch.zeros((64, 64), dtype=torch.uint8)
    m[10:30, 10:30] = 1
    out = draw_masks(img, [m], [[10, 10, 30, 30, 0]], idx2label=["hold", "volume"])
    out_arr = torch.tensor(list(out.getdata())).reshape(64, 64, 3)
    # Pick a pixel well inside the mask but not on the bbox outline (which alters
    # pixels at exactly y1, y2-1, x1, x2-1). (15, 15) is inside both.
    inside = out_arr[15, 15].tolist()
    outside = out_arr[40, 40].tolist()
    assert inside != [128, 128, 128]
    assert outside == [128, 128, 128]


def test_draw_masks_resizes_low_resolution_mask_to_image():
    """Mask at stride-4 (16x16) is resized to image (64x64) before overlay."""
    img = _gray_canvas(size=(64, 64))
    m = torch.zeros((16, 16), dtype=torch.uint8)
    m[2:6, 2:6] = 1   # corresponds to roughly [8..24]x[8..24] on the 64x64 image
    out = draw_masks(img, [m], [[8, 8, 24, 24, 0]])
    assert out.size == (64, 64)


def test_draw_masks_handles_class_label_lookup():
    img = _gray_canvas()
    m = torch.zeros((64, 64), dtype=torch.uint8); m[5:15, 5:15] = 1
    out = draw_masks(img, [m], [[5, 5, 15, 15, 1]], idx2label=["hold", "volume"])
    # Just sanity — no exception raised, output is image-sized.
    assert out.size == (64, 64)
