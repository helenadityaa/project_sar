import os
from ultralytics import YOLO
from pathlib import Path

def run_predict():
    # Path model best.pt dari training tadi
    model_path = "runs/detect/YOLOV8M_64_E50_B8/weights/best.pt"
    # Path folder dataset validasi
    source_path = "dataset_yolo_det/val/images"
    # Output folder untuk hasil prediksi
    output_project = "runs/detect"
    output_name = "PREDICT_V8M_64_FINAL"

    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} tidak ditemukan!")
        return

    print(f"\n--- Menjalankan Prediksi YOLOv8m (imgsz=64) ---")
    model = YOLO(model_path)
    
    # Jalankan prediksi
    results = model.predict(
        source=source_path,
        conf=0.25,        # Threshold confidence standar
        imgsz=64,         # WAJIB sama dengan ukuran training
        save=True,        # Simpan hasil gambar (.jpg)
        save_txt=True,    # Simpan koordinat bbox (.txt)
        project=output_project,
        name=output_name,
        exist_ok=True
    )
    
    print(f"\nPrediksi selesai! Hasil disimpan di: {output_project}/{output_name}")

if __name__ == "__main__":
    run_predict()
