from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss

from yolo.config.config import Config, LossConfig
from yolo.utils.bounding_box_utils import BoxMatcher, Vec2Box, calculate_iou
from yolo.utils.logger import logger


class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Refactor the device, should be assign by config
        # TODO: origin v9 assing pos_weight == 1?
        self.bce = BCEWithLogitsLoss(reduction="none")

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Any:
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm


class BoxLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, predicts_bbox: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
        loss_iou = 1.0 - iou
        loss_iou = (loss_iou * box_norm).sum() / cls_norm
        return loss_iou


class DFLoss(nn.Module):
    def __init__(self, vec2box: Vec2Box, reg_max: int) -> None:
        super().__init__()
        self.anchors_norm = (vec2box.anchor_grid / vec2box.scaler[:, None])[None]
        self.reg_max = reg_max

    def forward(
        self, predicts_anc: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)
        targets_dist = torch.cat(((self.anchors_norm - bbox_lt), (bbox_rb - self.anchors_norm)), -1).clamp(
            0, self.reg_max - 1.01
        )
        picked_targets = targets_dist[valid_bbox].view(-1)
        picked_predict = predicts_anc[valid_bbox].view(-1, self.reg_max)

        label_left, label_right = picked_targets.floor(), picked_targets.floor() + 1
        weight_left, weight_right = label_right - picked_targets, picked_targets - label_left

        loss_left = F.cross_entropy(picked_predict, label_left.to(torch.long), reduction="none")
        loss_right = F.cross_entropy(picked_predict, label_right.to(torch.long), reduction="none")
        loss_dfl = loss_left * weight_left + loss_right * weight_right
        loss_dfl = loss_dfl.view(-1, 4).mean(-1)
        loss_dfl = (loss_dfl * box_norm).sum() / cls_norm
        return loss_dfl


class YOLOLoss:
    def __init__(self, loss_cfg: LossConfig, vec2box: Vec2Box, class_num: int = 80, reg_max: int = 16) -> None:
        self.class_num = class_num
        self.vec2box = vec2box

        self.cls = BCELoss()
        self.dfl = DFLoss(vec2box, reg_max)
        self.iou = BoxLoss()

        self.matcher = BoxMatcher(loss_cfg.matcher, self.class_num, vec2box, reg_max)

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.vec2box.scaler[None, :, None]
        return anchors_cls, anchors_box

    def __call__(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        predicts_cls, predicts_anc, predicts_box = predicts
        # For each predicted targets, assign a best suitable ground truth box.
        align_targets, valid_masks = self.matcher(targets, (predicts_cls.detach(), predicts_box.detach()))

        targets_cls, targets_bbox = self.separate_anchor(align_targets)
        predicts_box = predicts_box / self.vec2box.scaler[None, :, None]

        cls_norm = max(targets_cls.sum(), 1)
        box_norm = targets_cls.sum(-1)[valid_masks]

        ## -- CLS -- ##
        loss_cls = self.cls(predicts_cls, targets_cls, cls_norm)
        ## -- IOU -- ##
        loss_iou = self.iou(predicts_box, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- DFL -- ##
        loss_dfl = self.dfl(predicts_anc, targets_bbox, valid_masks, box_norm, cls_norm)

        return loss_iou, loss_dfl, loss_cls


class DualLoss:
    def __init__(self, cfg: Config, vec2box) -> None:
        loss_cfg = cfg.task.loss
        self.loss = YOLOLoss(loss_cfg, vec2box, class_num=cfg.dataset.class_num, reg_max=cfg.model.anchor.reg_max)

        self.aux_rate = loss_cfg.aux

        self.iou_rate = loss_cfg.objective["BoxLoss"]
        self.dfl_rate = loss_cfg.objective["DFLoss"]
        self.cls_rate = loss_cfg.objective["BCELoss"]

        self.task_type = getattr(cfg.model, "task_type", "detection")
        if self.task_type == "segmentation":
            mask_cfg = getattr(loss_cfg, "mask", None)
            bce_w = getattr(mask_cfg, "bce_weight", 0.5) if mask_cfg is not None else 0.5
            dice_w = getattr(mask_cfg, "dice_weight", 0.5) if mask_cfg is not None else 0.5
            self.mask_loss = MaskLoss(bce_weight=bce_w, dice_weight=dice_w)
            self.mask_rate = loss_cfg.objective.get("MaskLoss", 7.5)

    def __call__(
        self, aux_predicts: List[Tensor], main_predicts: List[Tensor], targets: Tensor, mask_inputs=None
    ) -> Tuple[Tensor, Dict[str, float]]:
        # TODO: Need Refactor this region, make it flexible!
        aux_iou, aux_dfl, aux_cls = self.loss(aux_predicts, targets)
        main_iou, main_dfl, main_cls = self.loss(main_predicts, targets)

        total_loss = [
            self.iou_rate * (aux_iou * self.aux_rate + main_iou),
            self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
            self.cls_rate * (aux_cls * self.aux_rate + main_cls),
        ]
        loss_dict = {
            f"Loss/{name}Loss": value.detach().item() for name, value in zip(["Box", "DFL", "BCE"], total_loss)
        }

        # segmentation (main only)
        if self.task_type == "segmentation" and mask_inputs is not None:
            coefs, proto, gt_masks, target_bbox, match_idx = mask_inputs
            ml = self.mask_loss(coefs, proto, gt_masks, target_bbox, match_idx)
            total_loss.append(self.mask_rate * ml)
            loss_dict["Loss/MaskLoss"] = ml.detach().item()

        return sum(total_loss), loss_dict


class MaskLoss(nn.Module):
    """YOLACT-style mask loss: BCE + Dice with bbox crop, averaged over positives.

    Args (forward):
        coefs:        (B, A, nm) per-anchor mask coefficient predictions
        proto:        (B, nm, H, W) shared prototype masks per image
        gt_masks:     (B, N_max, H, W) GT masks (uint8 or float)
        target_bbox:  (B, N_max, 4) GT bbox xyxy in normalized [0,1] coords
        match_idx:    (B, A) GT index each anchor is matched to (-1 = negative)

    Returns:
        scalar loss tensor.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    @staticmethod
    def _crop(mask: Tensor, bbox: Tensor) -> Tensor:
        """Zero out everything outside the bbox region. mask (H, W); bbox xyxy in [0,1]."""
        H, W = mask.shape[-2:]
        x1 = (bbox[0] * W).clamp(0, W).long()
        y1 = (bbox[1] * H).clamp(0, H).long()
        x2 = (bbox[2] * W).clamp(0, W).long()
        y2 = (bbox[3] * H).clamp(0, H).long()
        cropped = torch.zeros_like(mask)
        cropped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
        return cropped

    def forward(self, coefs, proto, gt_masks, target_bbox, match_idx):
        device = coefs.device
        positives = match_idx >= 0   # (B, A)
        if not positives.any():
            return torch.zeros((), device=device)

        # gather GT-side data per positive
        b_idx, a_idx = positives.nonzero(as_tuple=True)
        gt_idx = match_idx[b_idx, a_idx]   # (P,)
        coef = coefs[b_idx, a_idx]          # (P, nm)
        proto_b = proto[b_idx]              # (P, nm, H, W)
        gt = gt_masks[b_idx, gt_idx].float()       # (P, H, W)
        bbox = target_bbox[b_idx, gt_idx]          # (P, 4)

        # synthesize predicted masks
        # einsum("pn,pnhw->phw")
        pred_logit = torch.einsum("pn,pnhw->phw", coef, proto_b)   # (P, H, W)

        # crop both pred and gt to bbox region
        cropped_pred, cropped_gt = [], []
        for p in range(pred_logit.shape[0]):
            cp = self._crop(pred_logit[p], bbox[p])
            cg = self._crop(gt[p], bbox[p])
            cropped_pred.append(cp)
            cropped_gt.append(cg)
        cropped_pred = torch.stack(cropped_pred)
        cropped_gt = torch.stack(cropped_gt)

        # BCE on logits, mean over (H, W)
        bce_per = self.bce(cropped_pred, cropped_gt).mean(dim=(-1, -2))

        # Dice on sigmoid
        prob = cropped_pred.sigmoid()
        intersection = (prob * cropped_gt).sum(dim=(-1, -2))
        union = prob.sum(dim=(-1, -2)) + cropped_gt.sum(dim=(-1, -2))
        dice = 1 - (2 * intersection + 1) / (union + 1)

        return (self.bce_w * bce_per + self.dice_w * dice).mean()


def create_loss_function(cfg: Config, vec2box) -> DualLoss:
    # TODO: make it flexible, if cfg doesn't contain aux, only use SingleLoss
    loss_function = DualLoss(cfg, vec2box)
    logger.info(":white_check_mark: Success load loss function")
    return loss_function
