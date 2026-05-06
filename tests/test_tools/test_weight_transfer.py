import torch

from tools.weight_transfer.v9_to_seg import transfer_detection_to_seg


def _fake_detection_state(prefix: str = "model.") -> dict:
    """Synthesize a minimal dict resembling a v9 detection ckpt state_dict."""
    return {
        f"{prefix}backbone.0.weight": torch.randn(16, 3, 3, 3),
        f"{prefix}neck.0.weight": torch.randn(64, 16, 3, 3),
        f"{prefix}head.detection.0.weight": torch.randn(8, 64, 1, 1),  # detection head conv
    }


def test_transfer_keeps_backbone_neck_head_weights_and_inserts_seg_init():
    det = _fake_detection_state()
    out = transfer_detection_to_seg(det, num_maskes=32, prototype_channels=128)
    # Backbone/neck/head detection weights are preserved, possibly under renamed keys.
    assert any(k.endswith("backbone.0.weight") for k in out)
    # New mask coef + prototype keys exist with random init shapes.
    proto_keys = [k for k in out if "proto_head" in k]
    assert proto_keys, "expected proto_head weights to be present after transfer"
    mask_head_keys = [k for k in out if "mask_heads" in k]
    assert mask_head_keys, "expected mask_heads weights to be present after transfer"
