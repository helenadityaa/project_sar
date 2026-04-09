from ultralytics import YOLO
import os

# 1. Load Model YOLOv11m (Medium)
# Model ini punya performa bagus, cocok buat deteksi kapal di citra satelit
model = YOLO('yolo11m.pt')

# 2. Jalankan Training
# Karena kamu punya H100, kita set batch size ke 32-64 biar ngebut!
model.train(
    data='yolo_exp/dataset/data.yaml',
    epochs=100, # Tambah epoch untuk akurasi lebih baik
    imgsz=640,  # Resolusi 640 jauh lebih bagus untuk deteksi detail kapal (Bulk vs Container)
    batch=64,   # Batch size tinggi untuk H100
    patience=15, # Stop kalau 15 epoch ga ada progres biar ga overfitting
    workers=8,
    device=0,   # GPU H100
    project='yolo_exp/runs',
    name='sar_fishing_4class_v11m_100e_640', # Nama folder hasil training lebih deskriptif
    save_json=True  # Otomatis buat summary JSON di folder output nanti
)
