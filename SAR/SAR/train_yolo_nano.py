from ultralytics import YOLO
import os

# 1. Load model YOLO11n (otomatis download jika belum ada)
model = YOLO('yolo11n.pt')

# 2. Jalankan training 50 epoch
# Menggunakan batch=4 agar muat di sisa memori GPU 1.4GB
model.train(
    data='yolo_exp/dataset/data.yaml',
    epochs=50,
    imgsz=320,
    batch=4,
    workers=4,
    device=0,
    project='yolo_exp/runs',
    name='sar_fishing_v11n_fast'
)
