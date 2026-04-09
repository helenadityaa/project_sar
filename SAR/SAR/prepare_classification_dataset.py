import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# 1. Konfigurasi Path
metadata_path = "SAR/SAR/metadata.csv"
patch_dir = "SAR/SAR/dataset/PATCH"
output_dir = "SAR/SAR/yolo_exp/dataset_classification"

# 2. Pemetaan Elaborated_type ke 4 Kelas Utama
# Berdasarkan data.yaml sebelumnya: 0:Fishing, 1:Tanker, 2:Bulk carrier, 3:Container
def map_category(elaborated_type, category):
    et = str(elaborated_type).lower()
    cat = str(category).lower()
    
    if 'fishing' in cat or 'fishing' in et:
        return 'Fishing'
    elif 'tanker' in et or 'tanker' in cat:
        return 'Tanker'
    elif 'bulk carrier' in et:
        return 'Bulk_carrier'
    elif 'container' in et:
        return 'Container'
    else:
        # Jika tidak masuk 4 besar, kita abaikan atau masukkan ke kategori terdekat
        return None

# 3. Baca Metadata
print("Membaca metadata...")
df = pd.read_csv(metadata_path)

# 4. Filter dan Petakan Kelas
df['target_class'] = df.apply(lambda row: map_category(row['Elaborated_type'], row['category']), axis=1)
df = df.dropna(subset=['target_class'])

print(f"Total data setelah difilter: {len(df)}")
print(df['target_class'].value_counts())

# 5. Split Dataset (Train 80%, Val 10%, Test 10%)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['target_class'], random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df['target_class'], random_state=42)

def setup_folders(df_split, split_name):
    for _, row in df_split.iterrows():
        class_name = row['target_class']
        file_name = row['patch'] # Nama file .tif
        
        src_path = os.path.join(patch_dir, file_name)
        dest_dir = os.path.join(output_dir, split_name, class_name)
        
        if os.path.exists(src_path):
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(src_path, os.path.join(dest_dir, file_name))

# 6. Jalankan Penyalinan File
print("Menyusun folder dataset klasifikasi...")
setup_folders(train_df, 'train')
setup_folders(val_df, 'val')
setup_folders(test_df, 'test')

print(f"Selesai! Dataset klasifikasi siap di: {output_dir}")
