from math import ceil
from pathlib import Path

import torch
from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler


class BaseModel(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)

    def forward(self, x):
        return self.model(x)


class ValidateModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.task_type = getattr(cfg.model, "task_type", "detection")
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        self.ema = self.model

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        # Tail-flexible unpack: detection collate emits 5-tuple
        # (batch_size, images, targets, rev, paths); seg collate emits 6-tuple
        # with an extra gt_masks slot. Bbox metric path doesn't consume masks
        # so we just pull batch_size/images/targets from the head and rev/paths
        # from the tail. Works for both phases independent of model.task_type vs
        # dataset.task_type alignment.
        batch_size, images, targets = batch[0], batch[1], batch[2]
        rev_tensor, img_paths = batch[-2], batch[-1]
        H, W = images.shape[2:]
        predicts = self.post_process(self.ema(images), image_size=[W, H])
        mAP = self.metric(
            [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
        )
        return predicts, mAP

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
        del epoch_metrics["classes"]
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(
            {"PyCOCO/AP @ .5:.95": epoch_metrics["map"], "PyCOCO/AP @ .5": epoch_metrics["map_50"]},
            sync_dist=True,
            rank_zero_only=True,
        )
        self.metric.reset()


class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.task_type = getattr(cfg.model, "task_type", "detection")
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)

    def setup(self, stage):
        super().setup(stage)
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)

    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        self.trainer.optimizers[0].next_epoch(
            ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
        )
        self.vec2box.update(self.cfg.image_size)

    def _compute_match_idx(self, main_predicts, targets):
        """Returns (B, A) long tensor with GT index per anchor (-1 = negative).

        Reuses BoxMatcher (already instantiated inside YOLOLoss via self.loss_fn.loss.matcher)
        to produce the GT-index assignment for each anchor.  The optional
        return_match_idx=True flag was added to BoxMatcher.__call__ specifically to
        surface the unique_indices tensor that is already computed internally but not
        normally returned.
        """
        preds_cls, _, preds_box = main_predicts
        matcher = self.loss_fn.loss.matcher
        _, _, match_idx = matcher(
            targets, (preds_cls.detach(), preds_box.detach()), return_match_idx=True
        )
        return match_idx

    def training_step(self, batch, batch_idx):
        lr_dict = self.trainer.optimizers[0].next_batch()
        if self.task_type == "segmentation":
            batch_size, images, targets, gt_masks, *_ = batch
            raw = self(images)
            # AUX is unchanged — list of (cls, anc, box) tuples
            aux_predicts = self.vec2box(raw["AUX"])
            # Main is a dict; route detect through Vec2Box, keep mask outputs
            main_dict = raw["Main"]
            main_detect = self.vec2box(main_dict["detect"])

            # Gather per-anchor coefficients across the 3 FPN levels into (B, A, nm).
            # mask_coefs is a list of (B, nm, h, w) — flatten + concat across spatial dim.
            coefs = torch.cat(
                [c.flatten(2).transpose(1, 2) for c in main_dict["mask_coefs"]],
                dim=1,
            )
            proto = main_dict["proto"]

            # Compute per-anchor GT-index assignment using the matcher already inside loss_fn.
            match_idx = self._compute_match_idx(main_detect, targets)
            target_bbox = targets[..., 1:5]

            loss, loss_item = self.loss_fn(
                aux_predicts, main_detect, targets,
                mask_inputs=(coefs, proto, gt_masks, target_bbox, match_idx),
            )
        else:
            batch_size, images, targets, *_ = batch
            predicts = self(images)
            aux_predicts = self.vec2box(predicts["AUX"])
            main_predicts = self.vec2box(predicts["Main"])
            loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
        self.log_dict(
            loss_item,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
        return loss * batch_size

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
        return [optimizer], [scheduler]


class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, rev_tensor, origin_frame = batch
        predicts = self.post_process(self(images), rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx)
        return img, fps

    def _save_image(self, img, batch_idx):
        save_image_path = Path(self.trainer.default_root_dir) / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")
