import os
import shutil

# 设置源文件夹和目标文件夹路径
source_dir = '/media/dell/D/breastTumorSeg/TMI/pes'  # data文件夹的路径
target_dir = '/media/dell/D/breastTumorSeg/TMI/pre_mask'  # 目标文件夹的路径

# 遍历data文件夹下的每个子文件夹
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    
    # 确保它是一个文件夹
    if os.path.isdir(folder_path):
        # 查找该文件夹中的nii文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.nii.gz'):
                nii_file_path = os.path.join(folder_path, file_name)
                
                # 生成新的文件名（以文件夹名命名）
                new_file_name = folder_name + '.nii.gz'
                new_file_path = os.path.join(target_dir, new_file_name)
                
                # 移动文件
                shutil.move(nii_file_path, new_file_path)
                print(f'文件 {file_name} 已移动到 {new_file_path}')
