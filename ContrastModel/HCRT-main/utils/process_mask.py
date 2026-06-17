import nibabel as nib
import numpy as np
import os
from scipy.ndimage import binary_dilation


def mask_nii_data_with_margin(image_path, mask_path, output_path, margin=100):
    """
    使用粗糙mask遮罩nii图像数据, 并将mask外围一定区域置为0。

    Args:
      image_path (str): 原始nii图像路径。
      mask_path (str): 粗糙mask路径。
      output_path (str): 遮罩后图像保存路径。
      margin (int): mask外围需要置0的区域大小 (单位: 像素)。
    """

    # 加载图像和mask数据
    image_data = nib.load(image_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()

    # 确保图像和mask尺寸一致
    if image_data.shape != mask_data.shape:
        raise ValueError("图像和mask尺寸不一致！")

    # 对mask进行膨胀操作
    dilated_mask = binary_dilation(mask_data, iterations=margin)

    # 使用膨胀后的mask遮罩图像
    masked_data = image_data * dilated_mask

    # 保存遮罩后的图像
    masked_img = nib.Nifti1Image(masked_data, nib.load(image_path).affine)
    nib.save(masked_img, output_path)
    print(f"保存成功：{output_path.split('/')[-1]}")


if __name__ == '__main__':
    # Specify the folder path
    image_path = "/data/shaofengzou/PanSTEEL/breastTumorSeg/YNpre/image_p0"
    mask_path = "/data/shaofengzou/PanSTEEL/breastTumorSeg/YNpre/Finetuning"
    output_path = "/data/shaofengzou/PanSTEEL/breastTumorSeg/YNpre/mask_p0"
    file_list = os.listdir(image_path)
    file_list = sorted(file_list)
    for file_name in file_list:
        if file_name.endswith('.nii.gz'):
            image = os.path.join(image_path, file_name)
            name = file_name.replace('P0', 'SAM')
            mask = os.path.join(mask_path, name)
            output = os.path.join(output_path, file_name)
            mask_nii_data_with_margin(image, mask, output, margin=75)
