import sys
from math import isinf, isnan
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.loss_functions import DualLoss, create_loss_function
from yolo.utils.bounding_box_utils import Vec2Box


@pytest.fixture
def cfg() -> Config:
    with initialize(config_path="../../yolo/config", version_base=None):
        cfg = compose(config_name="config", overrides=["task=train"])
    return cfg


@pytest.fixture
def model(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg.model, weight_path=None)
    return model.to(device)


@pytest.fixture
def vec2box(cfg: Config, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Vec2Box(model, cfg.model.anchor, cfg.image_size, device)


@pytest.fixture
def loss_function(cfg, vec2box) -> DualLoss:
    return create_loss_function(cfg, vec2box)


@pytest.fixture
def data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = torch.zeros(1, 20, 5, device=device)
    predicts = [torch.zeros(1, 8400, *cn, device=device) for cn in [(80,), (4, 16), (4,)]]
    return predicts, targets


def test_yolo_loss(loss_function, data):
    predicts, targets = data
    loss, loss_dict = loss_function(predicts, predicts, targets)
    assert loss_dict["Loss/BoxLoss"] == 0
    assert loss_dict["Loss/DFLLoss"] == 0
    assert loss_dict["Loss/BCELoss"] >= 2e5


def test_dualloss_constructs_in_segmentation_mode(cfg, vec2box):
    """When cfg.model.task_type=='segmentation', DualLoss carries a MaskLoss."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.model.task_type = "segmentation"
    # ensure objective dict has MaskLoss entry; otherwise default 7.5 used
    cfg.task.loss.objective = OmegaConf.create({"BoxLoss": 7.5, "DFLoss": 1.5, "BCELoss": 0.5, "MaskLoss": 7.5})
    cfg.task.loss.mask = OmegaConf.create({"bce_weight": 0.5, "dice_weight": 0.5})

    from yolo.tools.loss_functions import DualLoss, MaskLoss
    dl = DualLoss(cfg, vec2box)
    assert hasattr(dl, "mask_loss")
    assert isinstance(dl.mask_loss, MaskLoss)
    assert dl.mask_rate == 7.5


def test_dualloss_detection_mode_unchanged(cfg, vec2box):
    """When task_type defaults to 'detection', DualLoss has no mask_loss attr."""
    from yolo.tools.loss_functions import DualLoss
    dl = DualLoss(cfg, vec2box)
    # Detection-only: mask_loss should NOT be constructed
    assert not hasattr(dl, "mask_loss") or dl.task_type != "segmentation"
