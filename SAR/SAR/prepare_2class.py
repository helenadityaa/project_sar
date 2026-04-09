import os

def convert_to_2class(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"Folder tidak ditemukan: {input_dir}, skip.")
        return
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    count = 0
    total = len(files)
    
    for filename in files:
        with open(os.path.join(input_dir, filename), 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) > 0:
                cls = int(parts[0])
                # 0 tetap 0 (Fishing), selain itu jadi 1 (Non-Fishing)
                new_cls = 0 if cls == 0 else 1
                parts[0] = str(new_cls)
                new_lines.append(" ".join(parts))
        
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write("\n".join(new_lines))
        
        count += 1
        if count % 500 == 0:
            print(f"  Proses: {count}/{total} file...")

# Jalankan konversi
base_path = "yolo_exp/dataset/labels"
out_base = "yolo_exp/dataset_2class/labels"

print("Mengonversi labels ke 2-class (Fishing vs Non-Fishing)...")
convert_to_2class(os.path.join(base_path, "train"), os.path.join(out_base, "train"))
convert_to_2class(os.path.join(base_path, "val"), os.path.join(out_base, "val"))
convert_to_2class(os.path.join(base_path, "test"), os.path.join(out_base, "test"))

# Buat data_2class.yaml
# Memastikan path menggunakan path absolut yang benar
abs_path = os.path.abspath("yolo_exp/dataset")
yaml_content = f"""
path: {abs_path}
train: images/train
val: images/val
test: images/test

# Mengarahkan label ke folder baru (Path Absolut)
labels: 
  train: {os.path.abspath(out_base + '/train')}
  val: {os.path.abspath(out_base + '/val')}
  test: {os.path.abspath(out_base + '/test')}

names:
  0: Fishing
  1: Non-Fishing
"""
os.makedirs("yolo_exp/dataset_2class", exist_ok=True)
with open("yolo_exp/dataset_2class/data_2class.yaml", "w") as f:
    f.write(yaml_content)
print("Selesai! File yaml: yolo_exp/dataset_2class/data_2class.yaml")
