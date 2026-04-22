import argparse
import os
import cv2
from ultralytics import YOLO
from pathlib import Path

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = YOLO(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Predicting with Upscale (untuk label jelas) on {args.device}...")
    
    # Ambil daftar gambar
    image_files = list(Path(args.source).glob("*.png")) + list(Path(args.source).glob("*.jpg"))
    
    for img_path in image_files[:50]: # Batasi 50 gambar saja agar cepat
        results = model.predict(source=str(img_path), conf=args.conf, device=args.device, verbose=False)
        
        # Baca gambar asli
        img = cv2.imread(str(img_path))
        
        # PERBESAR GAMBAR (dari 64x64 ke 512x512) agar label kelihatan
        img_large = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        scale = 512 / 64
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Ambil koordinat dan skala ke 512
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
                
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                
                # Gambar Kotak
                cv2.rectangle(img_large, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Gambar Label (Teks Putih dengan Background Hitam agar kontras)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(img_large, (x1, y1 - 20), (x1 + w, y1), (0, 0, 0), -1)
                cv2.putText(img_large, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)

        # Simpan hasil
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), img_large)

    print(f"\nSelesai! 50 gambar dengan label besar disimpan di: {args.output}")

if __name__ == "__main__":
    predict()
