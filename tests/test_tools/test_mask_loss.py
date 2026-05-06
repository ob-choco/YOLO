import torch

from yolo.tools.loss_functions import MaskLoss


def test_mask_loss_returns_finite_scalar_with_grad():
    nm = 32
    B, A, H, W = 2, 16, 64, 64           # 16 anchors total, 64x64 mask res
    coefs = torch.randn(B, A, nm, requires_grad=True)
    proto = torch.randn(B, nm, H, W, requires_grad=True)
    gt_masks = torch.randint(0, 2, (B, 4, H, W), dtype=torch.uint8)   # 4 GT per image
    target_bbox = torch.tensor([
        [[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6], [0.3, 0.3, 0.7, 0.7], [0.4, 0.4, 0.8, 0.8]],
        [[0.0, 0.0, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6], [0.3, 0.3, 0.7, 0.7]],
    ])
    # match table: for each anchor, which GT (or -1 for negative)
    match_idx = torch.full((B, A), -1, dtype=torch.long)
    match_idx[:, :4] = torch.arange(4)   # first 4 anchors in each batch matched to GT 0..3

    loss_fn = MaskLoss(bce_weight=0.5, dice_weight=0.5)
    loss = loss_fn(coefs, proto, gt_masks, target_bbox, match_idx)
    assert torch.isfinite(loss)
    loss.backward()
    assert coefs.grad is not None and torch.isfinite(coefs.grad).all()
    assert proto.grad is not None and torch.isfinite(proto.grad).all()


def test_mask_loss_zero_when_no_positives():
    nm = 32
    B, A, H, W = 1, 8, 32, 32
    coefs = torch.randn(B, A, nm)
    proto = torch.randn(B, nm, H, W)
    gt_masks = torch.zeros(B, 0, H, W, dtype=torch.uint8)
    target_bbox = torch.zeros(B, 0, 4)
    match_idx = torch.full((B, A), -1, dtype=torch.long)
    loss = MaskLoss()(coefs, proto, gt_masks, target_bbox, match_idx)
    assert loss.item() == 0.0
