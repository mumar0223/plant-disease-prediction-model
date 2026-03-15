#  Input  Structure : Dataset/<class_name>/<image_files>
#  Output Structure :
#       Dataset/
#       ├── train/   (70%)
#       ├── val/     (15%)
#       └── test/    (15%)

import os
import shutil
import random
import json
from pathlib import Path

SOURCE_DIR = r'\Dataset\Dataset'      # Your original flat dataset folder
OUTPUT_DIR  = 'Dataset_Split'    # Where train/val/test will be created

TRAIN_RATIO = 0.70               # 70% → train
VAL_RATIO   = 0.15               # 15% → val
TEST_RATIO  = 0.15               # 15% → test

RANDOM_SEED = 42                 # For reproducibility

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "Ratios must sum to 1.0"

# Only pick directories that contain actual image files
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

class_dirs = sorted([
    d for d in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, d))
    and not d in ('train', 'val', 'test')   # Skip if already split
])

if len(class_dirs) == 0:
    raise ValueError(f" No class subdirectories found in '{SOURCE_DIR}'")

print(f"✅ Found {len(class_dirs)} classes:")
for cls in class_dirs:
    print(f"   → {cls}")

splits = ['train', 'val', 'test']

for split in splits:
    for cls in class_dirs:
        split_cls_dir = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(split_cls_dir, exist_ok=True)

print(f"\n Output directory structure created under '{OUTPUT_DIR}/'")

split_summary = {}
total_train = total_val = total_test = 0

random.seed(RANDOM_SEED)

print("\n" + "="*60)
print("  Splitting Dataset...")
print("="*60)

for cls in class_dirs:
    cls_src = os.path.join(SOURCE_DIR, cls)

    # Collect all valid image files
    all_images = sorted([
        f for f in os.listdir(cls_src)
        if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if len(all_images) == 0:
        print(f"  Skipping '{cls}' — no images found.")
        continue

    random.shuffle(all_images)

    # Calculate split indices
    n_total = len(all_images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)
    n_test  = n_total - n_train - n_val     # Remaining → test

    train_files = all_images[:n_train]
    val_files   = all_images[n_train : n_train + n_val]
    test_files  = all_images[n_train + n_val:]

    # Copy files to respective folders
    for fname in train_files:
        shutil.copy2(
            os.path.join(cls_src, fname),
            os.path.join(OUTPUT_DIR, 'train', cls, fname)
        )
    for fname in val_files:
        shutil.copy2(
            os.path.join(cls_src, fname),
            os.path.join(OUTPUT_DIR, 'val', cls, fname)
        )
    for fname in test_files:
        shutil.copy2(
            os.path.join(cls_src, fname),
            os.path.join(OUTPUT_DIR, 'test', cls, fname)
        )

    split_summary[cls] = {
        'total' : n_total,
        'train' : len(train_files),
        'val'   : len(val_files),
        'test'  : len(test_files)
    }

    total_train += len(train_files)
    total_val   += len(val_files)
    total_test  += len(test_files)

    print(f"  {cls:<55} total={n_total:>5} | "
          f"train={len(train_files):>4} | "
          f"val={len(val_files):>4} | "
          f"test={len(test_files):>4}")

grand_total = total_train + total_val + total_test

print("\n" + "="*60)
print("  SPLIT SUMMARY")
print("="*60)
print(f"  Total Images : {grand_total}")
print(f"  Train        : {total_train}  ({total_train/grand_total*100:.1f}%)")
print(f"  Val          : {total_val}  ({total_val/grand_total*100:.1f}%)")
print(f"  Test         : {total_test}  ({total_test/grand_total*100:.1f}%)")
print("="*60)

print(f"\n Dataset split complete!")
print(f"   Train : {OUTPUT_DIR}/train/")
print(f"   Val   : {OUTPUT_DIR}/val/")
print(f"   Test  : {OUTPUT_DIR}/test/")

split_info = {
    'source_dir'  : SOURCE_DIR,
    'output_dir'  : OUTPUT_DIR,
    'train_ratio' : TRAIN_RATIO,
    'val_ratio'   : VAL_RATIO,
    'test_ratio'  : TEST_RATIO,
    'random_seed' : RANDOM_SEED,
    'total_images': grand_total,
    'total_train' : total_train,
    'total_val'   : total_val,
    'total_test'  : total_test,
    'classes'     : class_dirs,
    'per_class'   : split_summary
}

with open('split_info.json', 'w') as f:
    json.dump(split_info, f, indent=4)

print("\n Split info saved: split_info.json")
print("\n Now run: python train.py")