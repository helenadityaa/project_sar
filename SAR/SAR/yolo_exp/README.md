# YOLO Experiment

Folder ini sengaja dipisah dari pipeline classifier `main.py` supaya kamu bisa coba bounding box dan YOLO tanpa mengganggu training ResNet yang sekarang.

## Isi

- `visualize_sar_bbox.py`
  - Menampilkan bounding box pada patch SAR dari metadata yang sudah ada.
- `prepare_yolo_dataset.py`
  - Mengubah patch TIFF + metadata menjadi dataset YOLO terpisah.
- `bbox_utils.py`
  - Fungsi shared untuk konversi metadata menjadi bounding box.

## Penting

Bounding box di sini bersifat `metadata-derived`, bukan hasil anotasi manual. Box dihitung dari `Head/Tail` kapal dan rasio lebar kapal dari metadata. Jadi ini cocok untuk eksperimen awal YOLO, tapi belum setara ground-truth box manual.

## 1. Visualisasi Bounding Box

Jalankan dari folder project:

```bash
cd /home/helena/project_sar/SAR/SAR
PYTHONPATH=/home/helena/project_sar/.deps python3 yolo_exp/visualize_sar_bbox.py \
  --image_name Cargo_x1673_y6973.tif
```

Output default:

`yolo_exp/outputs/Cargo_x1673_y6973_bbox.png`

## 2. Generate Dataset YOLO

Command:

```bash
cd /home/helena/project_sar/SAR/SAR
PYTHONPATH=/home/helena/project_sar/.deps python3 yolo_exp/prepare_yolo_dataset.py
```

Output default:

- `yolo_exp/dataset/images/train`
- `yolo_exp/dataset/images/val`
- `yolo_exp/dataset/images/test`
- `yolo_exp/dataset/labels/train`
- `yolo_exp/dataset/labels/val`
- `yolo_exp/dataset/labels/test`
- `yolo_exp/dataset/data.yaml`
- `yolo_exp/dataset/manifest.csv`

## 3. Train YOLO

Kalau environment kamu sudah punya CLI YOLO, contoh command:

```bash
yolo detect train data=/home/helena/project_sar/SAR/SAR/yolo_exp/dataset/data.yaml model=yolo12s.pt imgsz=640 epochs=100
```

Kalau model `yolo12s.pt` belum ada di install kamu, pakai model kecil YOLO yang tersedia di environment tersebut.

## Catatan Praktis

- Dataset ini memakai patch SAR yang sudah terpotong, jadi satu gambar umumnya hanya punya satu kapal.
- YOLO tetap bisa dicoba, tapi ini bukan setup deteksi paling ideal.
- Kalau nanti kamu punya citra SAR yang lebih besar dan anotasi box manual, hasil YOLO biasanya akan jauh lebih masuk akal.
