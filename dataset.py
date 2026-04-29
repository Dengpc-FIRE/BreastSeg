import os
from typing import Optional, List, Callable, Dict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class BreastDMDataset(Dataset):
    """
    Dataset for Breast DCE-MRI preprocessed .npy files.

    Each .npy file: [H, W, 9] where channel 0 is anatomy and channels 1-8 are subtraction (time) maps.

    Returns dict with keys: 'image' (Tensor [9,H,W]), 'mask' (Tensor [1,H,W]), 'filename' (str)
    """

    def __init__(
        self,
        data_dir: str,
        gt_dir: str,
        transform: Optional[Callable] = None,
        file_list: Optional[List[str]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.transform = transform

        if file_list is None:
            # collect .npy files
            files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.npy')])
        else:
            files = file_list

        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        fname = self.files[idx]
        fpath = os.path.join(self.data_dir, fname)

        # Load array: shape [H, W, 9]
        arr = np.load(fpath)
        if arr.ndim != 3 or arr.shape[2] != 9:
            raise ValueError(f"Expect .npy with shape [H,W,9], got {arr.shape} for {fname}")

        H, W, C = arr.shape

        # Transpose to [C, H, W]
        img = np.transpose(arr, (2, 0, 1)).astype(np.float32)

        # Normalize to [0,1] per-sample (robust): shift min to 0 then scale
        img = img - img.min()
        maxv = img.max()
        if maxv > 0:
            img = img / maxv

        # Load mask: try png/jpg with same base name but typical extensions
        base = os.path.splitext(fname)[0]
        # prefer exact basename + common ext
        mask_path = None
        for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif'):
            p = os.path.join(self.gt_dir, base + ext)
            if os.path.exists(p):
                mask_path = p
                break

        if mask_path is None:
            # fallback: try same filename but .png
            candidate = os.path.join(self.gt_dir, base + '.png')
            if os.path.exists(candidate):
                mask_path = candidate

        if mask_path is None:
            raise FileNotFoundError(f"Mask for {fname} not found in {self.gt_dir}")

        mask = Image.open(mask_path).convert('L')
        # resize to match H,W using nearest neighbor
        mask = mask.resize((W, H), resample=Image.NEAREST)
        mask_np = np.array(mask, dtype=np.uint8)

        # Convert mask values: 255 -> 1, else 0
        mask_bin = (mask_np == 255).astype(np.float32)

        # Apply albumentations-style transform if provided
        if self.transform is not None:
            # albumentations expects HWC image
            image_hwc = np.transpose(img, (1, 2, 0))  # H,W,C
            augmented = self.transform(image=image_hwc, mask=mask_bin)
            image_hwc = augmented['image']
            mask_bin = augmented['mask']
            # back to CHW
            img = np.transpose(image_hwc, (2, 0, 1)).astype(np.float32)

        # To tensor
        img_t = torch.from_numpy(img).float()
        mask_t = torch.from_numpy(mask_bin).unsqueeze(0).float()  # [1,H,W]

        return {'image': img_t, 'mask': mask_t, 'filename': fname}


if __name__ == '__main__':
    # quick local test snippet
    ds = BreastDMDataset(data_dir='processed_9ch_sub2label/test/data', gt_dir='processed_9ch_sub2label/test/GT')
    if len(ds) > 0:
        sample = ds[0]
        print('image.shape =', sample['image'].shape)
        print('mask.shape  =', sample['mask'].shape)
        print('filename    =', sample['filename'])
