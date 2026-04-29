import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================= 配置区域 =================
# 你的原始数据根目录 (包含 images 和 labels 文件夹的上一级目录)
# 比如你的目录结构是 /home/data/BreastDM/images 和 /home/data/BreastDM/labels
# 这里的 SRC_ROOT 就填 /home/data/BreastDM/
SRC_ROOT = './test'  

# 输出的 Swin-Unet 格式数据路径
DST_ROOT = './BreastDM_SwinUnet_Ready'

# Swin-Unet 默认输入尺寸
TARGET_SIZE = (224, 224) 

# 想要处理的序列类型？
# 如果只想跑特定序列（如减影序列 SUB），可以在这里过滤。
# 如果想跑所有文件夹，保持为空列表 []
TARGET_SEQUENCES = [] # 例如: ['SUB1', 'VIBRANT+C1']，留空则处理所有发现的文件夹

# ===========================================

def make_dirs():
    os.makedirs(os.path.join(DST_ROOT, 'train_npz'), exist_ok=True)
    os.makedirs(os.path.join(DST_ROOT, 'test_npz'), exist_ok=True)
    os.makedirs(os.path.join(DST_ROOT, 'lists'), exist_ok=True)

def process_case(case_path, is_train):
    """
    处理单个病例文件夹
    case_path: .../images/BreaDM-Be-1801
    """
    case_id = os.path.basename(case_path)
    
    # 对应的 label 文件夹路径
    # 假设 images 和 labels 平级
    label_case_path = case_path.replace('/images/', '/labels/').replace('\\images\\', '\\labels\\')
    
    if not os.path.exists(label_case_path):
        print(f"Skipping {case_id}: No corresponding label folder found.")
        return []

    processed_files = []
    
    # 遍历该病例下的所有序列文件夹 (SUB1, VIBRANT...)
    for seq_name in os.listdir(case_path):
        seq_img_dir = os.path.join(case_path, seq_name)
        seq_label_dir = os.path.join(label_case_path, seq_name)
        
        # 过滤序列
        if TARGET_SEQUENCES and seq_name not in TARGET_SEQUENCES:
            continue
            
        if not os.path.isdir(seq_img_dir): continue
        if not os.path.exists(seq_label_dir): continue # 如果这个序列没有标注，跳过

        # 遍历序列下的所有 jpg 图片
        img_files = glob.glob(os.path.join(seq_img_dir, '*.jpg'))
        
        for img_file in img_files:
            file_name = os.path.basename(img_file) # e.g., 001.jpg
            file_base = os.path.splitext(file_name)[0]
            
            # 构造对应的 mask 文件路径 (注意后缀是 png)
            mask_file = os.path.join(seq_label_dir, file_base + '.png')
            
            if not os.path.exists(mask_file):
                # 尝试找同名 jpg (防止后缀不一致)
                mask_file = os.path.join(seq_label_dir, file_name)
                if not os.path.exists(mask_file):
                    continue # 没找到对应的 Mask，跳过
            
            # --- 读取与处理 ---
            try:
                # 1. 读取 (转为灰度 'L', 如果需要 RGB 改为 'RGB')
                img = Image.open(img_file).convert('RGB') 
                mask = Image.open(mask_file).convert('L')
                
                # 2. Resize
                img = img.resize(TARGET_SIZE, resample=Image.BICUBIC)
                mask = mask.resize(TARGET_SIZE, resample=Image.NEAREST) # Mask 必须最近邻
                
                # 3. 转 Numpy & 归一化
                img_arr = np.array(img).astype(np.float32) / 255.0
                mask_arr = np.array(mask).astype(np.float32)
                
                # 4. 二值化 Mask (确保只有 0 和 1)
                mask_arr[mask_arr > 127] = 1
                mask_arr[mask_arr <= 127] = 0
                
                # 5. 生成唯一文件名: Case_Seq_File
                # 例如: BreaDM-Be-1801_SUB1_005
                save_filename = f"{case_id}_{seq_name}_{file_base}"
                
                save_dir = 'train_npz' if is_train else 'test_npz'
                save_path = os.path.join(DST_ROOT, save_dir, save_filename + '.npz')
                
                # 6. 保存 (label 必须是 key 之一)
                np.savez(save_path, image=img_arr, label=mask_arr)
                
                processed_files.append(save_filename)
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

    return processed_files

def main():
    make_dirs()
    
    # 获取 images 下的所有病例文件夹
    images_root = os.path.join(SRC_ROOT, 'images')
    all_cases = [os.path.join(images_root, d) for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))]
    
    # 划分训练集和测试集 (按病例划分，防止同一病人的不同切片泄露到测试集)
    train_cases, test_cases = train_test_split(all_cases, test_size=0.2, random_state=42)
    
    train_list = []
    test_list = []
    
    print(f"Total Cases: {len(all_cases)}. Train: {len(train_cases)}, Test: {len(test_cases)}")
    
    print("Processing Train Set...")
    for case in tqdm(train_cases):
        files = process_case(case, is_train=True)
        train_list.extend(files)
        
    print("Processing Test Set...")
    for case in tqdm(test_cases):
        files = process_case(case, is_train=False)
        test_list.extend(files)
        
    # 保存列表文件
    with open(os.path.join(DST_ROOT, 'lists', 'train.txt'), 'w') as f:
        f.write('\n'.join(train_list))
        
    with open(os.path.join(DST_ROOT, 'lists', 'test_vol.txt'), 'w') as f:
        f.write('\n'.join(test_list)) # Swin-Unet 常用 test_vol.txt 命名
        
    print("Done! 数据准备完成。")
    print(f"Train samples: {len(train_list)}")
    print(f"Test samples: {len(test_list)}")

if __name__ == '__main__':
    main()