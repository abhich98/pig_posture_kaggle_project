from __future__ import annotations

from dataclasses import dataclass


REQUIRED_TRAIN_COLUMNS = ["row_id", "image_id", "width", "height", "bbox", "class_id"]
REQUIRED_TEST_COLUMNS = ["row_id", "image_id", "width", "height", "bbox"]


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float


def parse_bbox_string(bbox_str: str) -> BBox:
    s = str(bbox_str).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    s = s.replace(",", " ")
    parts = [p for p in s.split() if p]
    if len(parts) != 4:
        raise ValueError(f"Bad bbox: {bbox_str}")
    x, y, w, h = map(float, parts)
    return BBox(x=x, y=y, w=w, h=h)
