import argparse
import os
from ultralytics import YOLO
from pathlib import Path

def train_yolo():
    parser = argparse.ArgumentParser(description="Universal YOLO Training Script")
    parser.add_argument("--version", type=str, default="8")
    parser.add_argument("--variant", type=str, default="s")
    parser.add_argument("--task", type=str, default="detect")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=64) # Gunakan 64 sesuai ukuran patch
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--output", type=str, default="runs/detect/yolo_result")

    args = parser.parse_args()

    model_map = {"8": "yolov8", "11": "yolo11", "12": "yolov12"}
    base_name = f"{model_map[args.version]}{args.variant}"
    model_pt = f"{base_name}{'-cls' if args.task == 'classify' else ''}.pt"

    print(f"\n--- Memulai Training YOLOv{args.version}{args.variant} ---")
    print(f"Epochs  : {args.epochs}")
    print(f"Output  : {args.output}")
    print(f"------------------------\n")

    model = YOLO(model_pt)
    
    # Path output mutlak agar tidak ada nesting
    project_path = os.path.dirname(os.path.abspath(args.output))
    folder_name = os.path.basename(args.output)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=project_path,
        name=folder_name,
        exist_ok=True,
        plots=True
    )

if __name__ == "__main__":
    train_yolo()
