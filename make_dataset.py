import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os

# 支持的图片扩展名
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)

def read_and_resize_gray(img_path, size):
    """读取并转为单通道灰度图 (用于输入数据堆叠)"""
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_CUBIC)
    return im

def read_label_and_resize(img_path, size):
    """读取标签并二值化"""
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if im is None: return None
    # 使用最近邻插值，保证标签只有0和255，不产生中间灰度
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_NEAREST)
    _, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
    return im

def process_patient(img_patient_dir, lbl_patient_dir, out_data_dir, out_gt_dir, size):
    patient_id = img_patient_dir.name
    
    # 1. 定位 Input 关键文件夹 (在 images 下)
    dir_vibrant = img_patient_dir / "VIBRANT"
    
    # 2. 定位 Label 关键文件夹 (在 labels 下)
    # 【核心修改】：将标签来源锁定为 VIBRANT 文件夹
    dir_label_target = lbl_patient_dir / "VIBRANT"

    if not dir_vibrant.exists():
        # print(f"Skip {patient_id}: No VIBRANT folder")
        return
    
    if not dir_label_target.exists():
        # 如果这个病人没有 VIBRANT 的标签文件夹，说明无法训练，跳过
        # print(f"Skip {patient_id}: No VIBRANT label folder")
        return

    # 3. 预先定位 SUB1 到 SUB8 文件夹 (用于构建9通道输入)
    sub_dirs = []
    for i in range(1, 9):
        d = img_patient_dir / f"SUB{i}"
        sub_dirs.append(d) # 即使不存在也先存个路径对象，后面判断

    # 4. 遍历切片 (以 VIBRANT 为基准)
    slices = sorted([p for p in dir_vibrant.iterdir() if is_image_file(p.name)])
    
    for slice_path in slices:
        slice_name = slice_path.name       # e.g. "p-032.jpg"
        slice_stem = slice_path.stem       # e.g. "p-032"
        
        # --- A. 构建 9 通道输入数据 ---
        channels = []
        
        # Channel 0: VIBRANT
        img_pre = read_and_resize_gray(slice_path, size)
        if img_pre is None: continue
        channels.append(img_pre)
        
        # Channel 1-8: SUB1-SUB8
        for s_dir in sub_dirs:
            s_path = s_dir / slice_name
            if s_dir.exists() and s_path.exists():
                img_sub = read_and_resize_gray(s_path, size)
                if img_sub is not None:
                    channels.append(img_sub)
                else:
                    channels.append(np.zeros((size, size), dtype=np.uint8))
            else:
                # 缺失序列补全黑
                channels.append(np.zeros((size, size), dtype=np.uint8))
        
        if len(channels) != 9: continue

        # 堆叠通道: (H, W, 9)
        stacked_data = np.stack(channels, axis=-1) 
        
        # --- B. 获取标签 (从 labels/../VIBRANT 中找) ---
        
        label_file = None
        
        # 尝试1: 直接找同名文件 (p-032.jpg / p-032.png)
        potential_files = [
            dir_label_target / f"{slice_stem}.png",
            dir_label_target / f"{slice_stem}.jpg",
            dir_label_target / f"{slice_stem}.bmp"
        ]
        
        for p in potential_files:
            if p.exists():
                label_file = p
                break
        
        # 尝试2: 如果找不到，可能是 p-032_mask.png 这种格式，再扫一遍文件夹
        if label_file is None:
            for f in dir_label_target.iterdir():
                if f.stem == slice_stem or f.stem == f"{slice_stem}_mask":
                    label_file = f
                    break
        
        # 只有找到对应 VIBRANT 标签的切片才保存
        if label_file and label_file.exists():
            gt_img = read_label_and_resize(label_file, size)
            if gt_img is not None:
                # --- 保存 ---
                # 保存 .npy
                out_name_npy = f"{patient_id}_{slice_stem}.npy"
                np.save(str(out_data_dir / out_name_npy), stacked_data)
                
                # 保存 .png 标签
                out_name_gt = f"{patient_id}_{slice_stem}.png"
                cv2.imwrite(str(out_gt_dir / out_name_gt), gt_img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='数据集根目录 (包含 train/val/test)')
    parser.add_argument('--output_root', type=str, default='./processed_9ch_vibrant_label', help='输出目录')
    parser.add_argument('--size', type=int, default=256, help='Resize大小')
    args = parser.parse_args()

    root_path = Path(args.dataset_root)
    output_root = Path(args.output_root)

    for split in ['train', 'val', 'test']:
        # 关键路径：images 和 labels 分开
        split_img_dir = root_path / split / 'images'
        split_lbl_dir = root_path / split / 'labels'
        
        if not split_img_dir.exists():
            print(f"[Info] {split}/images 不存在，跳过")
            continue
        
        if not split_lbl_dir.exists():
            print(f"[Warning] {split}/labels 不存在，无法提取标签！跳过")
            continue
            
        print(f"正在处理 {split} ...")
        
        # 输出目录结构
        split_out_data = output_root / split / 'data' # 存放 .npy
        split_out_gt = output_root / split / 'GT'     # 存放 .png
        split_out_data.mkdir(parents=True, exist_ok=True)
        split_out_gt.mkdir(parents=True, exist_ok=True)
        
        # 获取病人列表 (基于 images 目录)
        patients = [p for p in split_img_dir.iterdir() if p.is_dir()]
        
        for img_patient_dir in tqdm(patients):
            # 对应的标签文件夹路径
            lbl_patient_dir = split_lbl_dir / img_patient_dir.name
            
            process_patient(img_patient_dir, lbl_patient_dir, split_out_data, split_out_gt, args.size)

    print(f"\n完成！")
    print(f"输入数据 (9通道 .npy) 保存在: {output_root}/<split>/data")
    print(f"标签数据 (VIBRANT .png) 保存在: {output_root}/<split>/GT")

if __name__ == '__main__':
    main()