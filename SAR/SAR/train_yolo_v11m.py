from ultralytics import YOLO
import os

# Memastikan model ter-load
model = YOLO('yolo11m.pt')

# Jalankan training dengan setting 50 epoch
# Menaikkan batch size ke 32 agar optimal di H100 (cepat & tidak berat)
model.train(
    data='yolo_exp/dataset/data.yaml',
    epochs=50,
    imgsz=320,
    batch=32,
    workers=8,
    device=0,
    project='yolo_exp/runs',
    name='sar_fishing_v11m_50epochs'
)
