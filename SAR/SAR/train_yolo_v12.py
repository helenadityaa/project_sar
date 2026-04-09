import argparse
from ultralytics import YOLO
import os

def train_yolo(variant, task, epochs, imgsz, batch):
    # Tentukan nama model berdasarkan varian dan tugas
    # Contoh: yolov12n.pt (detect) atau yolov12n-cls.pt (classify)
    suffix = "-cls" if task == "classify" else ""
    model_name = f"yolov12{variant}{suffix}.pt"
    
    print(f"\n{'='*50}")
    print(f"MENJALANKAN TRAINING YOLOv12")
    print(f"Varian: {variant.upper()}")
    print(f"Tugas : {task.upper()}")
    print(f"Model : {model_name}")
    print(f"{'='*50}\n")

    # Load Model
    model = YOLO(model_name)

    # Tentukan file data (YAML untuk deteksi, folder untuk klasifikasi)
    # Catatan: Untuk klasifikasi, YOLO butuh dataset dalam format folder per kelas
    data_path = 'yolo_exp/dataset/data.yaml' if task == "detect" else 'yolo_exp/dataset_crop'

    # Jalankan Training
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,
        device=0, # GPU H100
        project='yolo_exp/runs',
        name=f"v12_{variant}_{task}_4class",
        save_json=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training YOLOv12 Flexible")
    parser.add_argument("--variant", type=str, default="n", choices=["n", "s", "m", "l", "x"], 
                        help="Varian model: n, s, m, l, x")
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "classify"], 
                        help="Tugas: detect atau classify")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)

    args = parser.parse_args()
    train_yolo(args.variant, args.task, args.epochs, args.imgsz, args.batch)
