import json
from pathlib import Path

import pytest

from tools.data_prep.via_to_coco import convert_via_to_coco


@pytest.fixture
def via_json(tmp_path: Path) -> Path:
    """Minimal VIA JSON with two regions: one hold polygon, one volume polygon."""
    via = {
        "_via_img_metadata": {
            "img1.jpg100": {
                "filename": "img1.jpg",
                "size": 100,
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [10, 20, 20, 10],
                            "all_points_y": [10, 10, 20, 20],
                        },
                        "region_attributes": {"hold_type": "hold"},
                    },
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [30, 40, 40, 30],
                            "all_points_y": [30, 30, 40, 40],
                        },
                        "region_attributes": {"hold_type": "volume"},
                    },
                ],
            },
            "img2.jpg200": {
                "filename": "img2.jpg",
                "size": 200,
                "regions": [],  # no labels — must be skipped
            },
        }
    }
    path = tmp_path / "via.json"
    path.write_text(json.dumps(via))
    return path


def test_convert_via_to_coco_writes_coco_with_two_categories(via_json: Path, tmp_path: Path):
    """Converter emits standard COCO instance-seg JSON with hold(0)/volume(1)."""
    out_path = tmp_path / "coco.json"
    # image_size_lookup maps filename -> (width, height) so the test does not need real images.
    # Image is 50×50 here so polygon max coord (40) is ≥ 60% of image dim → no auto-scale.
    convert_via_to_coco(
        via_json=via_json,
        output_json=out_path,
        image_size_lookup={"img1.jpg": (50, 50)},
    )
    coco = json.loads(out_path.read_text())
    assert {c["name"] for c in coco["categories"]} == {"hold", "volume"}
    assert {c["id"] for c in coco["categories"]} == {0, 1}
    assert len(coco["images"]) == 1   # img2 dropped (zero regions)
    assert len(coco["annotations"]) == 2
    # check polygon flattened correctly
    ann = coco["annotations"][0]
    assert ann["segmentation"] == [[10, 10, 20, 10, 20, 20, 10, 20]]
    assert ann["bbox"] == [10, 10, 10, 10]   # x, y, w, h
    assert ann["iscrowd"] == 0


def test_convert_via_to_coco_scales_polygons_when_labeling_resolution_is_downsampled(tmp_path: Path):
    """If polygon coords reach only a fraction of image size, infer integer ×N scale.

    Emulates the bh-phone case where VIA polygons were drawn on a 1/2-resolution
    EXIF-rotated image but the on-disk file is at full resolution.
    """
    via = {
        "_via_img_metadata": {
            "img.jpg100": {
                "filename": "img.jpg",
                "size": 100,
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [10, 90, 90, 10],   # max coord 90
                            "all_points_y": [10, 10, 90, 90],
                        },
                        "region_attributes": {"hold_type": "hold"},
                    }
                ],
            }
        }
    }
    via_path = tmp_path / "via.json"
    via_path.write_text(json.dumps(via))
    out_path = tmp_path / "coco.json"
    # Image (200, 200) — ratio = 200/90 ≈ 2.22 → scale = 2 → coords doubled
    convert_via_to_coco(
        via_json=via_path,
        output_json=out_path,
        image_size_lookup={"img.jpg": (200, 200)},
    )
    coco = json.loads(out_path.read_text())
    ann = coco["annotations"][0]
    # Polygons multiplied by 2; flattened
    assert ann["segmentation"] == [[20, 20, 180, 20, 180, 180, 20, 180]]
    assert ann["bbox"] == [20, 20, 160, 160]


def test_convert_via_to_coco_skips_polygons_with_fewer_than_three_points(tmp_path: Path):
    via = {
        "_via_img_metadata": {
            "a.jpg10": {
                "filename": "a.jpg",
                "size": 10,
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [1, 2],   # only 2 points — invalid
                            "all_points_y": [1, 2],
                        },
                        "region_attributes": {"hold_type": "hold"},
                    }
                ],
            }
        }
    }
    via_path = tmp_path / "via.json"
    via_path.write_text(json.dumps(via))
    out_path = tmp_path / "coco.json"
    convert_via_to_coco(
        via_json=via_path,
        output_json=out_path,
        image_size_lookup={"a.jpg": (50, 50)},
    )
    coco = json.loads(out_path.read_text())
    assert coco["annotations"] == []
    assert coco["images"] == []   # no valid annotations means image is dropped
