from pathlib import Path

import numpy as np


CLASS_NAMES = {
    0: "Fishing",
    1: "Tanker",
    2: "Cargo",
    3: "Other Type",
}

CLASS_BBOX_PAD = {
    0: 5.0,  # Fishing is sparse, so use a slightly more forgiving box.
    1: 2.0,
    2: 2.0,
    3: 2.0,
}


def map_ship_class(row):
    ship_type = str(row.get("category", ""))
    if ship_type == "Fishing":
        return 0
    if "Tanker" in ship_type:
        return 1
    if "Cargo" in ship_type:
        return 2
    return 3


def patch_scene_bounds(row):
    if all(k in row for k in ("UpperLeft_x", "UpperLeft_y", "LowerRight_x", "LowerRight_y")):
        min_x = float(min(row["UpperLeft_x"], row["LowerRight_x"]))
        max_x = float(max(row["UpperLeft_x"], row["LowerRight_x"]))
        min_y = float(min(row["UpperLeft_y"], row["LowerRight_y"]))
        max_y = float(max(row["UpperLeft_y"], row["LowerRight_y"]))
        return min_x, min_y, max_x, max_y

    center_x = float(row["Center_x"])
    center_y = float(row["Center_y"])
    half = 42.0
    return center_x - half, center_y - half, center_x + half, center_y + half


def patch_scene_size(row):
    min_x, min_y, max_x, max_y = patch_scene_bounds(row)
    return (max_x - min_x + 1.0), (max_y - min_y + 1.0)


def head_tail_local(row):
    min_x, min_y, _, _ = patch_scene_bounds(row)
    head_x = float(row.get("Head_x", row["Center_x"])) - min_x
    head_y = float(row.get("Head_y", row["Center_y"])) - min_y
    tail_x = float(row.get("Tail_x", row["Center_x"])) - min_x
    tail_y = float(row.get("Tail_y", row["Center_y"])) - min_y
    return np.array([head_x, head_y], dtype=np.float32), np.array([tail_x, tail_y], dtype=np.float32)


def _safe_ratio(width_value, length_value):
    try:
        width_value = float(width_value)
        length_value = float(length_value)
    except (TypeError, ValueError):
        return None
    if width_value <= 0 or length_value <= 0:
        return None
    return width_value / length_value


def estimate_ship_width_pixels(row, head_xy, tail_xy, default_ratio=0.18, min_width=3.0):
    pixel_length = float(np.hypot(*(tail_xy - head_xy)))
    ratios = [
        _safe_ratio(row.get("AIS_Width"), row.get("AIS_Length")),
        _safe_ratio(row.get("Breadth_extreme"), row.get("Length_overall")),
    ]
    ratio = next((r for r in ratios if r is not None), default_ratio)
    class_id = map_ship_class(row)
    if class_id == 2:
        min_width = max(min_width, 5.0)
    return max(min_width, pixel_length * ratio)


def oriented_box_corners_local(row):
    head_xy, tail_xy = head_tail_local(row)
    vector = tail_xy - head_xy
    norm = float(np.hypot(*vector))
    if norm < 1.0:
        center_x = float(row["Center_x"])
        center_y = float(row["Center_y"])
        min_x, min_y, _, _ = patch_scene_bounds(row)
        cx = center_x - min_x
        cy = center_y - min_y
        half = 6.0
        return np.array(
            [
                [cx - half, cy - half],
                [cx + half, cy - half],
                [cx + half, cy + half],
                [cx - half, cy + half],
            ],
            dtype=np.float32,
        )

    unit = vector / norm
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)
    half_width = estimate_ship_width_pixels(row, head_xy, tail_xy) / 2.0
    corners = np.array(
        [
            head_xy + perp * half_width,
            head_xy - perp * half_width,
            tail_xy - perp * half_width,
            tail_xy + perp * half_width,
        ],
        dtype=np.float32,
    )
    return corners


def axis_aligned_bbox_local(row, image_shape=None, pad=None):
    if pad is None:
        class_id = map_ship_class(row)
        pad = CLASS_BBOX_PAD.get(class_id, 2.0)
    corners = oriented_box_corners_local(row)
    x1 = float(corners[:, 0].min() - pad)
    y1 = float(corners[:, 1].min() - pad)
    x2 = float(corners[:, 0].max() + pad)
    y2 = float(corners[:, 1].max() + pad)

    scene_w, scene_h = patch_scene_size(row)
    if image_shape is not None:
        img_h, img_w = image_shape[:2]
        sx = float(img_w) / float(scene_w)
        sy = float(img_h) / float(scene_h)
        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy
        x1 = np.clip(x1, 0.0, img_w - 1.0)
        x2 = np.clip(x2, 0.0, img_w - 1.0)
        y1 = np.clip(y1, 0.0, img_h - 1.0)
        y2 = np.clip(y2, 0.0, img_h - 1.0)

    return x1, y1, x2, y2


def bbox_to_yolo(bbox_xyxy, image_shape):
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return (
        center_x / img_w,
        center_y / img_h,
        width / img_w,
        height / img_h,
    )


def sar_to_rgb_uint8(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 2, 3, 4):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 2:
        img = np.concatenate([img, np.zeros_like(img[..., :1])], axis=-1)
    elif img.shape[-1] > 3:
        img = img[..., :3]

    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def patch_name_from_any(value):
    return Path(str(value)).name
