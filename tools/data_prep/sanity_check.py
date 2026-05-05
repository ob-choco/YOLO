"""Render polygons from a COCO instance-seg JSON onto images for visual QA."""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np


def render_sanity(
    coco_json: Path, images_dir: Path, output_dir: Path, sample: int = 4
) -> None:
    coco = json.loads(Path(coco_json).read_text())
    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    color = {0: (0, 255, 0), 1: (0, 0, 255)}  # hold green, volume red
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled = random.sample(list(images_by_id.values()), min(sample, len(images_by_id)))
    for img in sampled:
        path = images_dir / img["file_name"]
        canvas = cv2.imread(str(path))
        if canvas is None:
            continue
        for a in anns_by_image.get(img["id"], []):
            poly = np.array(a["segmentation"][0], dtype=np.int32).reshape(-1, 2)
            cv2.polylines(canvas, [poly], isClosed=True, color=color[a["category_id"]], thickness=2)
        out = output_dir / img["file_name"]
        cv2.imwrite(str(out), canvas)
        print(f"  {out.name}  ({len(anns_by_image.get(img['id'], []))} polys, {cats})")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--coco", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample", type=int, default=4)
    args = p.parse_args()
    render_sanity(args.coco, args.images, args.out, args.sample)
