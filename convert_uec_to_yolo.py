import os
import shutil
import random
from PIL import Image
import argparse
import sys
from tqdm import tqdm

def convert_uec_to_yolo(dataset="UEC_Food_256"):
    # Use absolute paths and forward slashes, fix folder name
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
    # Correct subfolder name for dataset
    if dataset == 'UEC_Food_100':
        SRC_ROOT = os.path.join(DATA_DIR, dataset, 'UECFOOD100')
    elif dataset == 'UEC_Food_256':
        SRC_ROOT = os.path.join(DATA_DIR, dataset, 'UECFOOD256')
    else:
        raise ValueError('Unknown dataset')
    DST_IMAGES = os.path.join(DATA_DIR, dataset, 'yolo', 'images')
    DST_LABELS = os.path.join(DATA_DIR, dataset, 'yolo', 'labels')
    CATEGORY_FILE = os.path.join(SRC_ROOT, 'category.txt')

    # Convert all paths to use forward slashes
    DST_IMAGES = DST_IMAGES.replace('\\', '/')
    DST_LABELS = DST_LABELS.replace('\\', '/')
    CATEGORY_FILE = CATEGORY_FILE.replace('\\', '/')
    SRC_ROOT = SRC_ROOT.replace('\\', '/')
    DATA_DIR = DATA_DIR.replace('\\', '/')

    os.makedirs(DST_IMAGES, exist_ok=True)
    os.makedirs(DST_LABELS, exist_ok=True)

    # 1. Read class names
    class_names = []
    with open(CATEGORY_FILE, encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('id'):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    class_names.append(parts[1])

    # 2. Process each class folder and collect all image-label pairs
    img_label_pairs = []
    class_count = len(class_names) + 1
    for class_idx in tqdm(range(1, class_count), desc="Processing classes", unit="class", 
                          position=0, leave=True, dynamic_ncols=True):
        class_folder = os.path.join(SRC_ROOT, str(class_idx)).replace('\\', '/')
        bb_file = os.path.join(class_folder, 'bb_info.txt').replace('\\', '/')
        if not os.path.exists(bb_file):
            print(f"Warning: No bb_info.txt found in class folder {class_idx}")
            continue
        with open(bb_file, encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('img') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            img_id, x1, y1, x2, y2 = parts
            img_file = os.path.join(class_folder, f'{img_id}.jpg').replace('\\', '/')
            if not os.path.exists(img_file):
                continue
            new_img_name = f'{class_idx}_{img_id}.jpg'
            # Get image size
            with Image.open(img_file) as im:
                w, h = im.size
            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            yolo_class_id = class_idx - 1
            label_content = f'{yolo_class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n'
            img_label_pairs.append((img_file, new_img_name, label_content))

    # Shuffle and split
    random.shuffle(img_label_pairs)
    split_idx = int(0.8 * len(img_label_pairs))
    train_pairs = img_label_pairs[:split_idx]
    val_pairs = img_label_pairs[split_idx:]

    # Prepare folders
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DST_IMAGES, split), exist_ok=True)
        os.makedirs(os.path.join(DST_LABELS, split), exist_ok=True)

    # Copy and write files
    print(f"\nSplitting dataset: {len(train_pairs)} training images, {len(val_pairs)} validation images")
    for position, (split, pairs) in enumerate([('train', train_pairs), ('val', val_pairs)]):
        with tqdm(total=len(pairs), desc=f"Processing {split}", unit="img", 
                 position=position, leave=True, dynamic_ncols=True, 
                 file=sys.stdout) as pbar:
            for img_file, new_img_name, label_content in pairs:
                dst_img_path = os.path.join(DST_IMAGES, split, new_img_name).replace('\\', '/')
                dst_label_path = os.path.join(DST_LABELS, split, new_img_name.replace('.jpg', '.txt')).replace('\\', '/')
                shutil.copyfile(img_file, dst_img_path)
                with open(dst_label_path, 'w') as lf:
                    lf.write(label_content)
                pbar.update(1)

    # 3. Write data.yaml
    yaml_path = os.path.join(DATA_DIR, dataset, 'yolo', 'data.yaml').replace('\\', '/')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write('path: .\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write(f'nc: {len(class_names)}\n')
        f.write('names: [')
        f.write(', '.join([f'"{name}"' for name in class_names]))
        f.write(']\n')

    print(f'Conversion complete! YOLOv8 dataset is in data/{dataset}/yolo/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert UEC Food dataset to YOLO format')
    parser.add_argument('--dataset', type=str, default='UEC_Food_256', 
                        choices=['UEC_Food_100', 'UEC_Food_256'],
                        help='Dataset to convert: UEC_Food_100 or UEC_Food_256')
    args = parser.parse_args()
    
    convert_uec_to_yolo(args.dataset)