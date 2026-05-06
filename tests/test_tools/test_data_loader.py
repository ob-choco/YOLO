import sys
from pathlib import Path

from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.data_loader import StreamDataLoader, create_dataloader


def test_create_dataloader_cache(train_cfg: Config):
    train_cfg.task.data.shuffle = False
    train_cfg.task.data.batch_size = 2

    cache_file = Path("tests/data/train.cache")
    cache_file.unlink(missing_ok=True)

    make_cache_loader = create_dataloader(train_cfg.task.data, train_cfg.dataset)
    load_cache_loader = create_dataloader(train_cfg.task.data, train_cfg.dataset)
    m_batch_size, m_images, _, m_reverse_tensors, m_image_paths = next(iter(make_cache_loader))
    l_batch_size, l_images, _, l_reverse_tensors, l_image_paths = next(iter(load_cache_loader))
    assert m_batch_size == l_batch_size
    assert m_images.shape == l_images.shape
    assert m_reverse_tensors.shape == l_reverse_tensors.shape
    assert m_image_paths == l_image_paths


def test_training_data_loader_correctness(train_dataloader: DataLoader):
    """Test that the training data loader produces correctly shaped data and metadata."""
    batch_size, images, _, reverse_tensors, image_paths = next(iter(train_dataloader))
    assert batch_size == 2
    assert images.shape == (2, 3, 640, 640)
    assert reverse_tensors.shape == (2, 5)
    expected_paths = [
        Path("tests/data/images/train/000000050725.jpg"),
        Path("tests/data/images/train/000000167848.jpg"),
    ]
    assert list(image_paths) == list(expected_paths)


def test_validation_data_loader_correctness(validation_dataloader: DataLoader):
    batch_size, images, targets, reverse_tensors, image_paths = next(iter(validation_dataloader))
    assert batch_size == 5
    assert images.shape == (5, 3, 640, 640)
    assert targets.shape == (5, 18, 5)
    assert reverse_tensors.shape == (5, 5)
    expected_paths = [
        Path("tests/data/images/val/000000151480.jpg"),
        Path("tests/data/images/val/000000284106.jpg"),
        Path("tests/data/images/val/000000323571.jpg"),
        Path("tests/data/images/val/000000556498.jpg"),
        Path("tests/data/images/val/000000570456.jpg"),
    ]
    assert list(image_paths) == list(expected_paths)


def test_file_stream_data_loader_frame(file_stream_data_loader: StreamDataLoader):
    """Test the frame output from the file stream data loader."""
    frame, rev_tensor, origin_frame = next(iter(file_stream_data_loader))
    assert frame.shape == (1, 3, 640, 640)
    assert rev_tensor.shape == (1, 5)
    assert origin_frame.size == (1024, 768)


def test_directory_stream_data_loader_frame(directory_stream_data_loader: StreamDataLoader):
    """Test the frame output from the directory stream data loader."""
    frame, rev_tensor, origin_frame = next(iter(directory_stream_data_loader))
    assert frame.shape == (1, 3, 640, 640)
    assert rev_tensor.shape == (1, 5)
    assert origin_frame.size != (640, 640)


import numpy as np
from yolo.tools.data_loader import YoloDataset


def test_load_valid_labels_returns_polygons_when_segmentation_task():
    dataset = YoloDataset.__new__(YoloDataset)
    dataset.task_type = "segmentation"
    seg = [[0, 0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5]]
    bboxes, polygons = dataset.load_valid_labels("img1", seg)
    assert bboxes.shape == (1, 5)
    assert isinstance(polygons, list)
    assert polygons[0].shape == (1, 8)


def test_load_valid_labels_returns_none_polygons_when_detection_task():
    dataset = YoloDataset.__new__(YoloDataset)
    dataset.task_type = "detection"
    seg = [[0, 0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5]]
    bboxes, polygons = dataset.load_valid_labels("img1", seg)
    assert bboxes.shape == (1, 5)
    assert polygons is None


def test_load_valid_labels_empty_input_segmentation_returns_empty_polygons():
    dataset = YoloDataset.__new__(YoloDataset)
    dataset.task_type = "segmentation"
    bboxes, polygons = dataset.load_valid_labels("img1", [])
    assert bboxes.shape == (0, 5)
    assert polygons == []


def test_collate_fn_seg_pads_targets_and_rasterizes_masks():
    """Verify batch padding for bboxes + on-the-fly polygon→mask rasterization."""
    import numpy as np
    import torch
    from yolo.tools.data_loader import collate_fn_seg

    # Two samples with different number of instances (1 and 2). image_size 64x64.
    img = torch.zeros(3, 64, 64)
    rev = torch.zeros(4)
    box1 = torch.tensor([[0.0, 0.1, 0.1, 0.5, 0.5]])
    poly1 = [np.array([[0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5]], dtype=np.float32)]
    box2 = torch.tensor([
        [1.0, 0.2, 0.2, 0.4, 0.4],
        [0.0, 0.3, 0.3, 0.6, 0.6],
    ])
    poly2 = [
        np.array([[0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.2, 0.4]], dtype=np.float32),
        np.array([[0.3, 0.3, 0.6, 0.3, 0.6, 0.6, 0.3, 0.6]], dtype=np.float32),
    ]
    s1 = (img, box1, poly1, rev, "a")
    s2 = (img, box2, poly2, rev, "b")

    bs, images, targets, masks, revs, paths = collate_fn_seg([s1, s2])
    assert bs == 2
    assert images.shape == (2, 3, 64, 64)
    # N_max = max(1, 2) = 2
    assert targets.shape == (2, 2, 5)
    # mask resolution 64/4 = 16
    assert masks.shape == (2, 2, 16, 16)
    assert masks.dtype == torch.uint8
    # padding row in sample 0 (index 1) is all zero — mask AND target cls = -1
    assert (masks[0, 1] == 0).all()
    assert targets[0, 1, 0] == -1
    # populated entries are non-zero
    assert masks[0, 0].sum() > 0
    assert masks[1, 0].sum() > 0
    assert masks[1, 1].sum() > 0
    assert paths == ("a", "b")


def test_collate_fn_seg_handles_zero_instance_sample():
    """A sample with no annotations must pad to N_max from the other sample."""
    import numpy as np
    import torch
    from yolo.tools.data_loader import collate_fn_seg

    img = torch.zeros(3, 64, 64)
    rev = torch.zeros(4)
    s_empty = (img, torch.zeros((0, 5)), [], rev, "empty")
    s_one = (img, torch.tensor([[0.0, 0.1, 0.1, 0.5, 0.5]]),
             [np.array([[0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1, 0.5]], dtype=np.float32)],
             rev, "one")

    bs, images, targets, masks, revs, paths = collate_fn_seg([s_empty, s_one])
    # N_max = max(0, 1) = 1
    assert targets.shape == (2, 1, 5)
    assert masks.shape == (2, 1, 16, 16)
    # empty sample row 0 is all-zero mask + cls=-1 target
    assert (masks[0, 0] == 0).all()
    assert targets[0, 0, 0] == -1


def test_create_dataloader_routes_to_seg_collate_when_task_type_segmentation():
    """create_dataloader picks collate_fn_seg when dataset_cfg.task_type == 'segmentation'."""
    from types import SimpleNamespace
    from yolo.tools.data_loader import collate_fn_seg

    # Minimal dataset_cfg with task_type — we don't actually call create_dataloader on
    # a real dataset; we verify the helper used to pick the collate is exposed and
    # branches correctly.
    dataset_cfg_seg = SimpleNamespace(task_type="segmentation")
    dataset_cfg_det = SimpleNamespace(task_type="detection")

    # Minimal expectation: the chosen collate is seg-specific in seg mode.
    # We expose pick_collate_fn(dataset_cfg) for testability — implementation can
    # also inline the choice in create_dataloader; if so adapt the test to read
    # the .collate_fn attribute of a constructed DataLoader.
    from yolo.tools.data_loader import pick_collate_fn
    assert pick_collate_fn(dataset_cfg_seg) is collate_fn_seg
    from yolo.tools.data_loader import collate_fn as collate_fn_det
    assert pick_collate_fn(dataset_cfg_det) is collate_fn_det
