"""Convert VIA (VGG Image Annotator) JSON to COCO instance-segmentation JSON.

Used to convert hand-labelled climbing wall photos into the format consumed
by yolo/tools/data_loader.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

CATEGORY_MAP = {"hold": 0, "volume": 1}


def convert_via_to_coco(
    via_json: Path,
    output_json: Path,
    image_size_lookup: Optional[Dict[str, Tuple[int, int]]] = None,
    images_dir: Optional[Path] = None,
) -> None:
    """Read a VIA JSON file and write a COCO instance-seg JSON.

    Args:
        via_json: Path to VIA JSON file.
        output_json: Where to write the COCO JSON.
        image_size_lookup: Optional pre-computed {filename: (width, height)}.
            If absent and `images_dir` is given, sizes are read with PIL.
        images_dir: Directory containing the image files; used only if
            `image_size_lookup` is None.
    """
    via = json.loads(Path(via_json).read_text())
    metadata = via["_via_img_metadata"]

    images, annotations = [], []
    image_id = 0
    ann_id = 0

    for entry in metadata.values():
        filename = entry["filename"]
        regions = entry.get("regions", [])
        if not regions:
            continue

        valid_anns = []
        for r in regions:
            shape = r.get("shape_attributes", {})
            if shape.get("name") != "polygon":
                continue
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if len(xs) < 3 or len(ys) < 3:
                continue
            hold_type = r.get("region_attributes", {}).get("hold_type")
            if hold_type not in CATEGORY_MAP:
                continue
            valid_anns.append((xs, ys, hold_type))

        if not valid_anns:
            continue

        # resolve image size
        if image_size_lookup and filename in image_size_lookup:
            width, height = image_size_lookup[filename]
        elif images_dir is not None:
            from PIL import Image  # local import: only needed if no lookup
            with Image.open(Path(images_dir) / filename) as im:
                width, height = im.size
        else:
            raise ValueError(
                f"image_size for {filename} unresolved (provide image_size_lookup or images_dir)"
            )

        images.append(
            {"id": image_id, "file_name": filename, "width": width, "height": height}
        )

        for xs, ys, hold_type in valid_anns:
            seg_flat = [v for pair in zip(xs, ys) for v in pair]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w = x_max - x_min
            h = y_max - y_min
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_MAP[hold_type],
                    "segmentation": [seg_flat],
                    "bbox": [x_min, y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cid, "name": name, "supercategory": "climbing"}
            for name, cid in CATEGORY_MAP.items()
        ],
    }
    Path(output_json).write_text(json.dumps(coco))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--via", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    args = parser.parse_args()
    convert_via_to_coco(args.via, args.out, images_dir=args.images_dir)
    print(f"Wrote {args.out}")
