import pandas as pd
import os
import shutil
import numpy as np
import tifffile
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
import sys

# Import logika PRO
sys.path.append(os.path.abspath("../yolo_exp"))
from bbox_utils import axis_aligned_bbox_local, bbox_to_yolo, sar_to_rgb_uint8

def prepare_data_clean():
    root_dir = Path("..").resolve()
    metadata_path = root_dir / "metadata.csv"
    image_source_dir = root_dir / "resized_new"
    det_dir = Path("dataset_yolo_det").resolve()
    
    df = pd.read_csv(metadata_path)
    class_names = ['Bulk Carrier', 'Container Ship', 'Fishing', 'Tanker']
    prepared_data = []

    print("Mengekstrak data dengan BBox PRO (No Oversampling)...")

    for _, row in df.iterrows():
        ship_type = str(row.get("category", ""))
        el_type = str(row.get("Elaborated_type", ""))
        patch_name = str(row.get("patch_cal", ""))
        
        label = None
        if "Cargo" in ship_type:
            if el_type == "Bulk Carrier": label = 0
            elif el_type == "Container Ship": label = 1
        elif ship_type == "Fishing": label = 2
        elif "Tanker" in ship_type: label = 3
            
        if label is not None and patch_name != "":
            img_path = image_source_dir / patch_name
            if img_path.exists():
                img_tiff = tifffile.imread(img_path)
                # Gunakan padding standar PRO (pad=2.0)
                bbox_xyxy = axis_aligned_bbox_local(row, image_shape=img_tiff.shape, pad=2.0)
                yolo_coords = bbox_to_yolo(bbox_xyxy, image_shape=img_tiff.shape)
                
                prepared_data.append({
                    "img_path": img_path, "patch_name": patch_name, "label": label,
                    "yolo_bbox": f"{label} {' '.join([f'{c:.6f}' for c in yolo_coords])}"
                })

    df_filtered = pd.DataFrame(prepared_data)
    train_df, val_df = train_test_split(df_filtered, test_size=0.2, stratify=df_filtered['label'], random_state=42)

    for split_name, split_df in [('train', train_df), ('val', val_df)]:
        shutil.rmtree(det_dir / split_name, ignore_errors=True)
        (det_dir / split_name / 'images').mkdir(parents=True, exist_ok=True)
        (det_dir / split_name / 'labels').mkdir(parents=True, exist_ok=True)
        
        print(f"Memproses {split_name} ({len(split_df)} sampel)...")
        for _, row in split_df.iterrows():
            img_rgb = sar_to_rgb_uint8(tifffile.imread(row['img_path']))
            new_name = Path(row['patch_name']).with_suffix('.png').name
            Image.fromarray(img_rgb).save(det_dir / split_name / 'images' / new_name)
            with open(det_dir / split_name / 'labels' / (Path(new_name).stem + ".txt"), "w") as f:
                f.write(row['yolo_bbox'] + "\n")

    with open(det_dir / "data.yaml", "w") as f:
        f.write(f"path: {det_dir}\ntrain: train/images\nval: val/images\nnc: 4\nnames: {class_names}")
    print("\nSelesai! Dataset kembali ke kondisi bersih (Original Split).")

if __name__ == "__main__":
    prepare_data_clean()
