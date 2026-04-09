import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import tifffile as tiff

from bbox_utils import axis_aligned_bbox_local, head_tail_local, sar_to_rgb_uint8


def main():
    base_dir = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_csv", default=str(base_dir / "metadata.csv"))
    ap.add_argument("--image_dir", default=str(base_dir / "dataset" / "PATCH_CAL"))
    ap.add_argument("--image_name", required=True, help="Example: Cargo_x1673_y6973.tif")
    ap.add_argument("--out_path", default="")
    ap.add_argument("--show_head_tail", action="store_true", default=True)
    ap.add_argument("--no-show_head_tail", dest="show_head_tail", action="store_false")
    args = ap.parse_args()

    metadata_csv = Path(args.metadata_csv)
    image_dir = Path(args.image_dir)
    image_name = Path(args.image_name).name
    out_path = Path(args.out_path) if args.out_path else Path(__file__).resolve().parent / "outputs" / f"{Path(image_name).stem}_bbox.png"

    df = pd.read_csv(metadata_csv)
    matched = df[df["patch_cal"].astype(str) == image_name]
    if matched.empty:
        raise FileNotFoundError(f"{image_name} not found in {metadata_csv}")
    row = matched.iloc[0]

    image_path = image_dir / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = tiff.imread(str(image_path))
    rgb = sar_to_rgb_uint8(image)
    x1, y1, x2, y2 = axis_aligned_bbox_local(row, image_shape=rgb.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="#ff6b00",
            linewidth=2.0,
        )
    )

    if args.show_head_tail:
        head_xy, tail_xy = head_tail_local(row)
        scene_h, scene_w = image.shape[:2]
        src_w = max(float(scene_w), 1.0)
        src_h = max(float(scene_h), 1.0)
        patch_w = max(float(row["UpperLeft_x"] - row["LowerRight_x"] + 1), 1.0) if "UpperLeft_x" in row and "LowerRight_x" in row else src_w
        patch_h = max(float(row["UpperLeft_y"] - row["LowerRight_y"] + 1), 1.0) if "UpperLeft_y" in row and "LowerRight_y" in row else src_h
        ax.plot(
            [head_xy[0] * src_w / patch_w, tail_xy[0] * src_w / patch_w],
            [head_xy[1] * src_h / patch_h, tail_xy[1] * src_h / patch_h],
            color="#00f5d4",
            linewidth=1.5,
        )

    ax.set_title(f"{image_name}\nBBox from metadata")
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to: {out_path}")


if __name__ == "__main__":
    main()
