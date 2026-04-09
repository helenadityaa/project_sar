from ultralytics import YOLO
import json
import os

# 1. Load Model (Gunakan YOLO11n agar cepat & muat di memori)
model = YOLO('yolo11n.pt')

# 2. Training dengan menonjolkan Fishing (cls_weight lebih besar)
# Serta 50 Epoch sesuai permintaan
results = model.train(
    data='yolo_exp/dataset_2class/data_2class.yaml',
    epochs=50,
    imgsz=320,
    batch=4,      # Tetap kecil biar muat di sisa memori GPU
    workers=4,
    device=0,
    project='yolo_exp/runs',
    name='sar_fishing_2class_v11n',
    cls=2.0       # Menaikkan bobot loss klasifikasi agar lebih akurat
)

# 3. Jalankan Validasi Akhir untuk dapat detail per kelas
print("\n" + "="*50)
print("HASIL EVALUASI DETAIL PER KELAS")
print("="*50)
val_results = model.val()

# Menyusun summary seperti permintaan Anda
summary = {
    "Val mAP50": f"{val_results.results_dict['metrics/mAP50(B)']*100:.2f}%",
    "Val mAP50-95": f"{val_results.results_dict['metrics/mAP50-95(B)']*100:.2f}%",
    "Best weight": os.path.join(results.save_dir, 'weights/best.pt')
}

# Print ringkasan akhir
print("\nSummary disimpan:")
print(json.dumps(summary, indent=4))

with open(os.path.join(results.save_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=4)
