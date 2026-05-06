import torch

from yolo.model.module import MultiheadSegmentation


def test_forward_returns_dict_with_detect_mask_coefs_and_proto():
    """Forward emits structured outputs: detection per FPN, mask coef per FPN, single proto."""
    torch.manual_seed(0)
    in_channels = [64, 96, 128, 128]   # P3, P4, P5, proto-source channels (matches v9-t-seg)
    head = MultiheadSegmentation(in_channels=in_channels, num_classes=2, num_maskes=32, reg_max=16)
    head.eval()

    # Build dummy feature pyramid + proto source matching the channel widths.
    feats = [
        torch.randn(2, 64, 80, 80),    # P3 stride 8 at 640 input
        torch.randn(2, 96, 40, 40),    # P4 stride 16
        torch.randn(2, 128, 20, 20),   # P5 stride 32
        torch.randn(2, 128, 160, 160), # prototype source — already upsampled to stride 4
    ]
    out = head(feats)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"detect", "mask_coefs", "proto"}

    # detect: list of 3 (class_x, anchor_x, vector_x) tuples per FPN level
    # — verifying tuple shape (not just count) catches regressions where
    # MultiheadDetection silently returns Nones or wrong-shape tensors.
    assert len(out["detect"]) == 3
    for level_out in out["detect"]:
        assert len(level_out) == 3, "each detect level should be a (cls, anc, vec) tuple"
        cls_x, anc_x, vec_x = level_out
        assert cls_x.ndim == 4 and cls_x.shape[0] == 2 and cls_x.shape[1] == 2  # (B, num_classes, h, w)
        assert anc_x.ndim == 5 and anc_x.shape[0] == 2                          # (B, R, P, h, w)
        assert vec_x.ndim == 4 and vec_x.shape[0] == 2                          # (B, P, h, w)

    # mask_coefs: list of 3 tensors, (B, 32, h, w) per FPN level
    assert len(out["mask_coefs"]) == 3
    assert out["mask_coefs"][0].shape == (2, 32, 80, 80)
    assert out["mask_coefs"][1].shape == (2, 32, 40, 40)
    assert out["mask_coefs"][2].shape == (2, 32, 20, 20)

    # proto: single tensor (B, 32, H/4, W/4)
    assert out["proto"].shape == (2, 32, 160, 160)
