import os
import shutil

# 原始文件夹路径
source_folder = "/media/dell/SATA1/ECL/checkpoints/BASE/Predyn100"
# 目标文件夹路径
destination_folder = "/media/dell/D/breastTumorSeg/YNpre/mask"

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 遍历每个子文件夹
for subdir in os.listdir(source_folder):
    subdir_path = os.path.join(source_folder, subdir)
    
    # 检查子文件夹是否存在并包含pred.nii.gz
    if os.path.isdir(subdir_path):
        pred_file_path = os.path.join(subdir_path, "pred.nii.gz")
        
        # 如果pred.nii.gz文件存在，进行复制
        if os.path.exists(pred_file_path):
            # 设置目标文件路径，并命名为子文件夹名
            dest_file_path = os.path.join(destination_folder, f"{subdir}.nii.gz")
            shutil.copy(pred_file_path, dest_file_path)
            print(f"Copied {pred_file_path} to {dest_file_path}")
        else:
            print(f"No pred.nii.gz found in {subdir_path}")
