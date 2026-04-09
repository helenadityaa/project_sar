#!/bin/bash
# Pastikan library di folder .deps dan .deps_yolo terdeteksi
export PYTHONPATH=../../.deps_yolo:../../.deps

# Jalankan training YOLOv11m 4 Class
python3 train_yolo_v11m_4class.py
