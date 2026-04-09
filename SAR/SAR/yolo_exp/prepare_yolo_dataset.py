import argparse
import math
from pathlib import Path
import shutil

from matplotlib import image as mpimg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tifffile as tiff

from bbox_utils import CLASS_NAMES, axis_aligned_bbox_local, bbox_to_yolo, map_ship_class, patch_name_from_any, sar_to_rgb_uint8


def filter_metadata(df, image_dir):
    rows = []
    for _, row in df.iterrows():
        class_id = map_ship_class(row)
        if class_id is None:
            continue
        patch_name = patch_name_from_any(row["patch_cal"])
        image_path = image_dir / patch_name
        if not image_path.exists():
            continue
        row = row.copy()
        row["class_id"] = class_id
        row["patch_name"] = patch_name
        row["image_path"] = str(image_path)
        rows.append(row)
    return pd.DataFrame(rows)


def assign_splits(df, random_state):
    labels = df["class_id"].astype(int)
    train_val_idx, val_idx = train_test_split(
        df.index.to_numpy(),
        train_size=0.8,
        stratify=labels.to_numpy(),
        random_state=random_state,
    )
    train_idx, test_idx = train_test_split(
        train_val_idx,
        train_size=0.8,
        stratify=df.loc[train_val_idx, "class_id"].astype(int).to_numpy(),
        random_state=random_state,
    )
    df = df.copy()
    df["split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def cleanup_output_dir(out_dir):
    for name in ("images", "labels"):
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)
    for name in ("manifest.csv", "data.yaml", "split_summary.csv"):
        path = out_dir / name
        if path.exists():
            path.unlink()


def transform_points(points, image_shape, transform_name):
    img_h, img_w = image_shape[:2]
    points = np.asarray(points, dtype=np.float32)

    if transform_name == "orig":
        return points
    if transform_name == "hflip":
        out = points.copy()
        out[:, 0] = (img_w - 1) - out[:, 0]
        return out
    if transform_name == "vflip":
        out = points.copy()
        out[:, 1] = (img_h - 1) - out[:, 1]
        return out
    if transform_name == "hvflip":
        out = points.copy()
        out[:, 0] = (img_w - 1) - out[:, 0]
        out[:, 1] = (img_h - 1) - out[:, 1]
        return out
    if transform_name == "rot90":
        out = points.copy()
        x = out[:, 0].copy()
        y = out[:, 1].copy()
        out[:, 0] = y
        out[:, 1] = (img_w - 1) - x
        return out
    if transform_name == "rot270":
        out = points.copy()
        x = out[:, 0].copy()
        y = out[:, 1].copy()
        out[:, 0] = (img_h - 1) - y
        out[:, 1] = x
        return out
    raise ValueError(f"Unsupported transform: {transform_name}")


def transform_bbox_xyxy(bbox_xyxy, image_shape, transform_name):
    """
    Apply geometric transformations to bounding box coordinates.
    Pixel-level transforms (noise/bright) do not affect coordinates.
    """
    x1, y1, x2, y2 = bbox_xyxy
    current_bbox = [x1, y1, x2, y2]
    
    # Handle geometric components in order
    if "hflip" in transform_name:
        corners = np.array([[current_bbox[0], current_bbox[1]], [current_bbox[2], current_bbox[1]], 
                           [current_bbox[2], current_bbox[3]], [current_bbox[0], current_bbox[3]]], dtype=np.float32)
        transformed = transform_points(corners, image_shape, "hflip")
        current_bbox = [float(transformed[:, 0].min()), float(transformed[:, 1].min()), 
                        float(transformed[:, 0].max()), float(transformed[:, 1].max())]
        
    if "vflip" in transform_name:
        corners = np.array([[current_bbox[0], current_bbox[1]], [current_bbox[2], current_bbox[1]], 
                           [current_bbox[2], current_bbox[3]], [current_bbox[0], current_bbox[3]]], dtype=np.float32)
        transformed = transform_points(corners, image_shape, "vflip")
        current_bbox = [float(transformed[:, 0].min()), float(transformed[:, 1].min()), 
                        float(transformed[:, 0].max()), float(transformed[:, 1].max())]

    if "rot90" in transform_name:
        corners = np.array([[current_bbox[0], current_bbox[1]], [current_bbox[2], current_bbox[1]], 
                           [current_bbox[2], current_bbox[3]], [current_bbox[0], current_bbox[3]]], dtype=np.float32)
        transformed = transform_points(corners, image_shape, "rot90")
        current_bbox = [float(transformed[:, 0].min()), float(transformed[:, 1].min()), 
                        float(transformed[:, 0].max()), float(transformed[:, 1].max())]
        # After 90 deg rotation, image_shape effectively swaps W and H for the NEXT transform
        image_shape = (image_shape[1], image_shape[0], image_shape[2])

    if "rot270" in transform_name:
        corners = np.array([[current_bbox[0], current_bbox[1]], [current_bbox[2], current_bbox[1]], 
                           [current_bbox[2], current_bbox[3]], [current_bbox[0], current_bbox[3]]], dtype=np.float32)
        transformed = transform_points(corners, image_shape, "rot270")
        current_bbox = [float(transformed[:, 0].min()), float(transformed[:, 1].min()), 
                        float(transformed[:, 0].max()), float(transformed[:, 1].max())]
        image_shape = (image_shape[1], image_shape[0], image_shape[2])

    return tuple(current_bbox)


def transform_image(image, transform_name):
    """
    Apply image transformations including geometric and pixel-level changes.
    """
    img = image.copy()
    
    # 1. Geometric Transforms
    if "hflip" in transform_name:
        img = np.fliplr(img).copy()
    if "vflip" in transform_name:
        img = np.flipud(img).copy()
    if "rot90" in transform_name:
        img = np.rot90(img, 1).copy()
    if "rot270" in transform_name:
        img = np.rot90(img, 3).copy()
        
    # 2. Pixel-level Transforms (Intensity/Noise)
    if "noise" in transform_name:
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if "bright" in transform_name:
        factor = np.random.uniform(0.7, 1.3)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if "blur" in transform_name:
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=(0.5, 0.5, 0))
        
    return img


def build_train_sampling_plan(train_df, focus_class_id=0, focus_multiplier=1.5):
    train_df = train_df.copy()
    counts = train_df["class_id"].value_counts().to_dict()
    majority = max(counts.values())
    targets = {class_id: majority for class_id in counts}
    if focus_class_id in targets:
        # Boost Fishing heavily
        targets[focus_class_id] = int(math.ceil(majority * focus_multiplier))

    augmented_rows = []
    # Richer variety for Fishing
    base_geos = ["orig", "hflip", "vflip", "rot90", "rot270"]
    effects = ["", "_noise", "_bright", "_blur", "_noise_bright"]
    
    heavy_cycle = []
    for g in base_geos:
        for e in effects:
            heavy_cycle.append(g + e)

    for class_id, class_df in train_df.groupby("class_id", sort=True):
        rows = class_df.to_dict(orient="records")
        target_count = targets[class_id]
        
        # Use heavy_cycle for Fishing, base_geos for others
        current_cycle = heavy_cycle if class_id == focus_class_id else base_geos
        
        for i in range(target_count):
            row = dict(rows[i % len(rows)])
            row["transform_name"] = current_cycle[i % len(current_cycle)]
            row["copy_index"] = i
            augmented_rows.append(row)
    return pd.DataFrame(augmented_rows), targets


def write_data_yaml(out_dir):
    yaml_text = "\n".join(
        [
            f"path: {out_dir.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: Fishing",
            "  1: Tanker",
            "  2: Cargo",
            "  3: Other Type",
            "",
        ]
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    base_dir = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_csv", default=str(base_dir / "metadata.csv"))
    ap.add_argument("--image_dir", default=str(base_dir / "dataset" / "PATCH_CAL"))
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "dataset"))
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="0 = use all samples")
    ap.add_argument("--focus_class_id", type=int, default=0, help="Default: Fishing class")
    ap.add_argument("--focus_multiplier", type=float, default=1.5, help="Oversample factor for focus class in train split")
    ap.add_argument("--balance_train", action="store_true", default=True)
    ap.add_argument("--no-balance_train", dest="balance_train", action="store_false")
    args = ap.parse_args()

    metadata_csv = Path(args.metadata_csv)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)

    df = pd.read_csv(metadata_csv)
    df = filter_metadata(df, image_dir)
    if df.empty:
        raise RuntimeError("No usable rows found for YOLO dataset generation.")

    if args.limit and len(df) > args.limit:
        df = df.groupby("class_id", group_keys=False).head(max(1, args.limit // max(len(CLASS_NAMES), 1)))
        df = df.reset_index(drop=True)

    if len(df["class_id"].value_counts()) == len(CLASS_NAMES) and df["class_id"].value_counts().min() >= 2:
        df = assign_splits(df, args.random_state)
    else:
        df = df.copy()
        df["split"] = "train"

    cleanup_output_dir(out_dir)

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_df = df[df["split"] == "train"].copy()
    other_df = df[df["split"] != "train"].copy()
    if args.balance_train and not train_df.empty:
        train_df, train_targets = build_train_sampling_plan(
            train_df,
            focus_class_id=args.focus_class_id,
            focus_multiplier=args.focus_multiplier,
        )
    else:
        train_df = train_df.copy()
        train_df["transform_name"] = "orig"
        train_df["copy_index"] = 0
        train_targets = train_df["class_id"].value_counts().to_dict()

    other_df["transform_name"] = "orig"
    other_df["copy_index"] = 0
    df_to_write = pd.concat([train_df, other_df], ignore_index=True)

    manifest_rows = []
    for _, row in df_to_write.iterrows():
        image_path = Path(row["image_path"])
        image = tiff.imread(str(image_path))
        rgb = sar_to_rgb_uint8(image)
        bbox = axis_aligned_bbox_local(row, image_shape=rgb.shape)
        transform_name = str(row.get("transform_name", "orig"))
        rgb = transform_image(rgb, transform_name)
        bbox = transform_bbox_xyxy(bbox, image_shape=image.shape, transform_name=transform_name)
        yolo_box = bbox_to_yolo(bbox, rgb.shape)

        split = row["split"]
        stem = image_path.stem
        suffix = ""
        if transform_name != "orig" or int(row.get("copy_index", 0)) > 0:
            suffix = f"__{transform_name}_{int(row.get('copy_index', 0)):04d}"
        image_out = out_dir / "images" / split / f"{stem}{suffix}.png"
        label_out = out_dir / "labels" / split / f"{stem}{suffix}.txt"

        mpimg.imsave(image_out, rgb)
        label_out.write_text(
            f"{int(row['class_id'])} "
            f"{yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n",
            encoding="utf-8",
        )

        manifest_rows.append(
            {
                "split": split,
                "image_png": str(image_out),
                "label_txt": str(label_out),
                "class_id": int(row["class_id"]),
                "class_name": CLASS_NAMES[int(row["class_id"])],
                "source_tiff": str(image_path),
                "transform_name": transform_name,
            }
        )

    write_data_yaml(out_dir)
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(out_dir / "manifest.csv", index=False)
    split_summary = (
        manifest_df.groupby(["split", "class_name"]).size().rename("count").reset_index()
    )
    split_summary.to_csv(out_dir / "split_summary.csv", index=False)

    print(f"Generated YOLO dataset in: {out_dir}")
    print(manifest_df["split"].value_counts().to_dict())
    print(split_summary)


if __name__ == "__main__":
    main()
