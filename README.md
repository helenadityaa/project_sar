# SAR Ship Detection and Classification Project

This repository contains a comprehensive pipeline for ship detection and classification in Synthetic Aperture Radar (SAR) imagery, specifically using the **OpenSARShip** dataset. The project implements two major approaches: **ResNet50 with TensorRT optimization** for classification and **YOLOv8m** for detection, optimized for small-scale SAR patches.

## 🚀 Key Features

*   **ResNet50RT**: High-performance classification using ResNet50, converted to ONNX and optimized with NVIDIA TensorRT for low-latency inference.
*   **YOLO New Gen**: Specialized YOLOv8m implementation optimized for 64x64 SAR patches, achieving significant mAP improvements (especially for rare classes like 'Fishing').
*   **Preprocessing Pipeline**: custom BBox extraction, SAR-to-RGB conversion, and automated dataset balancing (Oversampling).
*   **Optimization**: Inference speedup using TensorRT and FP16 precision.

---

## 📁 Repository Structure

```text
.
├── SAR/SAR/
│   ├── resnet_full_experiment/  # ResNet50 training & evaluation scripts
│   ├── yolo_new_gen/            # Optimized YOLOv8 detection pipeline
│   ├── yolo_exp/                # Shared utilities & BBox logic
│   ├── build_trt_engine.py      # Script to convert ONNX to TensorRT (.trt)
│   ├── export_onnx.py           # Script to export PyTorch models to ONNX
│   ├── predict_trt.py           # Inference script for TensorRT engines
│   ├── metadata.csv             # Dataset metadata & labels
│   └── requirements.txt         # Project dependencies
└── .gitignore                   # Optimized for large SAR data management
```

---

## 📊 Experimental Results (YOLOv8m)

By adjusting the `imgsz` to **64** (matching the native SAR patch resolution) and implementing targeted data preparation, we achieved high detection accuracy across all classes:

| Class           | mAP50  | Precision | Recall |
|-----------------|--------|-----------|--------|
| **Overall**     | **0.694** | 0.606     | 0.639  |
| Bulk Carrier    | 0.767  | 0.674     | 0.747  |
| Container Ship  | 0.601  | 0.656     | 0.336  |
| **Fishing**     | **0.664** | 0.487     | 0.658  |
| Tanker          | 0.743  | 0.610     | 0.814  |

---

## 🛠️ Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/helenadityaa/project_sar.git
    cd project_sar
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r SAR/SAR/requirements.txt
    ```

3.  **Prepare the dataset**:
    Navigate to the YOLO New Gen folder and run the data preparation script:
    ```bash
    cd SAR/SAR/yolo_new_gen
    python3 prepare_data.py
    ```

---

## 🏋️ Training & Inference

### YOLOv8m (Detection)
To train the model with optimized settings for 64x64 patches:
```bash
python3 train_yolo.py --version 8 --variant m --epochs 50 --batch 8 --imgsz 64
```

### ResNet50RT (Classification & TensorRT)
1. **Train ResNet**: Run scripts in `resnet_full_experiment/`.
2. **Export to ONNX**: `python3 export_onnx.py`
3. **Build TRT Engine**:
   ```bash
   python3 build_trt_engine.py --onnx_path model.onnx --engine_path model.trt --fp16
   ```

---

## 📝 Authors
*   **Helena** - *Main Developer*

---
*Generated for academic and research purposes in the field of Remote Sensing and Deep Learning.*
