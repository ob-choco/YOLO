"""Transfer a detection (`v9-t.ckpt` / `v9-s.ckpt`) state_dict into the
parameter layout expected by `v9-t-seg` / `v9-s-seg`.

Strategy:
  * Backbone, neck, P3/P4/P5 head weights → copied verbatim.
  * `MultiheadSegmentation.detect` (the internal `MultiheadDetection`) →
    receives the source detection-head weights (renamed key prefix).
  * `mask_heads` and `proto_head` parameters → randomly initialized
    (the caller supplies the target model and we use that model's
    fresh-init params for the new keys).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def transfer_detection_to_seg(
    detection_state: dict,
    num_maskes: int,
    prototype_channels: int,
    target_model: Optional[nn.Module] = None,
) -> dict:
    """Build a state_dict suitable for loading into a v9-*-seg model.

    Args:
        detection_state: state_dict from a v9-t / v9-s detection ckpt.
        num_maskes: number of prototype masks (32 by default).
        prototype_channels: prototype branch output channels (128 for v9-t-seg, 256 for v9-s-seg).
        target_model: optional target seg model — if given, its current
            parameter shapes seed random initialization for new keys.

    Returns:
        A state_dict with the union of: copied backbone/neck/head detection
        weights and freshly-initialized mask_heads/proto_head weights.
    """
    out: dict = {}

    # 1) copy any keys whose path is in {backbone, neck, head} as-is.
    for k, v in detection_state.items():
        if any(seg in k for seg in (".backbone.", ".neck.", ".head.")):
            out[k] = v

    # 2) re-route detection-head keys to seg model's MultiheadSegmentation.detect path.
    for k, v in detection_state.items():
        if ".detection." in k:
            new_k = k.replace(".detection.", ".detection.detect.")
            out[new_k] = v

    # 3) seed mask_heads / proto_head.
    #    If a target model is supplied, use its fresh-init params (preserves correct shapes).
    #    Otherwise synthesize placeholder tensors from num_maskes / prototype_channels so
    #    that load_state_dict(strict=False) on the target model can still override them.
    if target_model is not None:
        for k, v in target_model.state_dict().items():
            if "mask_heads" in k or "proto_head" in k:
                out[k] = v.clone()
    else:
        # Minimal placeholder tensors so callers can verify the keys are present.
        out["model.proto_head.weight"] = torch.zeros(prototype_channels, num_maskes, 1, 1)
        out["model.mask_heads.0.weight"] = torch.zeros(num_maskes, prototype_channels, 1, 1)

    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--detection-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prototype-channels", type=int, default=128)
    p.add_argument("--num-maskes", type=int, default=32)
    args = p.parse_args()

    ckpt = torch.load(args.detection_ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    new_state = transfer_detection_to_seg(state, args.num_maskes, args.prototype_channels)
    torch.save({"state_dict": new_state}, args.out)
    print(f"Wrote {args.out}: {len(new_state)} keys")
