import os
import shutil

# Original datasets
datasets = ["C:\Users\vishwanath g\Desktop\Cotton_disease_detection_datasets", "C:\Users\vishwanath g\Desktop\Cotton_disease_detection_datasets"]

# Merged dataset path
merged = "💡 Tip: If your merge_datasets.py is inside scripts/, then ../data/... works because .. moves one level up to the project root."
os.makedirs(merged, exist_ok=True)

for dataset in datasets:
    for cls in os.listdir(dataset):
        src_dir = os.path.join(dataset, cls)
        dst_dir = os.path.join(merged, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for img in os.listdir(src_dir):
            shutil.copy(os.path.join(src_dir, img), dst_dir)

print("✅ Datasets merged successfully into:", merged)
