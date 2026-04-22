import subprocess
import pandas as pd
import os
from pathlib import Path

def run_compare():
    # Daftar eksperimen yang ingin dicoba
    # (Version, Variant)
    experiments = [
        ("8", "n"),
        ("11", "n"),
        ("12", "n"),
        ("11", "s"),
    ]
    
    results = []
    dataset_yaml = "dataset_yolo/data.yaml"
    
    if not os.path.exists(dataset_yaml):
        print(f"Error: {dataset_yaml} tidak ditemukan! Jalankan prepare_data.py dulu.")
        return

    for version, variant in experiments:
        exp_name = f"compare_v{version}{variant}"
        print(f"\n>>> Menjalankan Eksperimen: YOLOv{version} {variant}...")
        
        cmd = [
            "python3", "train_yolo.py",
            "--version", version,
            "--variant", variant,
            "--task", "detect",
            "--data", dataset_yaml,
            "--epochs", "30",  # Epoch sedikit saja untuk komparasi cepat
            "--batch", "16",
            "--name", exp_name
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Cari file results.csv hasil training ultralytics
            # Biasanya di yolo_new_gen/runs/compare_v.../results.csv
            run_dir = Path(f"runs/{exp_name}_yolo{version if version != '8' else 'v8'}{variant}")
            csv_path = run_dir / "results.csv"
            
            if csv_path.exists():
                df_results = pd.read_csv(csv_path)
                # Ambil baris terakhir (epoch terakhir)
                last_metrics = df_results.iloc[-1]
                
                results.append({
                    "Model": f"YOLOv{version}{variant}",
                    "mAP50": last_metrics.get("metrics/mAP50(B)", 0),
                    "Precision": last_metrics.get("metrics/precision(B)", 0),
                    "Recall": last_metrics.get("metrics/recall(B)", 0),
                    "Fitness": last_metrics.get("fitness", 0)
                })
                print(f"Selesai: {exp_name} tercatat.")
            else:
                print(f"Warning: File hasil {csv_path} tidak ditemukan.")
                
        except Exception as e:
            print(f"Gagal menjalankan {exp_name}: {e}")

    # Simpan ringkasan
    if results:
        summary_df = pd.DataFrame(results)
        summary_df.to_csv("summary_comparison.csv", index=False)
        print("\n--- RINGKASAN HASIL ---")
        print(summary_df)
        print("\nHasil disimpan di: summary_comparison.csv")

if __name__ == "__main__":
    run_compare()
