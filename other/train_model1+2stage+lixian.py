import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp

# 引入你的模型
from model1 import HybridTemporalNet

# ==========================================
# 0. 基础配置
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. 自动生成 ROI 伪标签逻辑 (用于 Stage 1)
# ==========================================
def generate_pseudo_roi(image_v0, lesion_gt, padding=30):
    img_8u = (image_v0 * 255).astype(np.uint8)
    _, thresh = cv2.threshold(img_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    roi_mask = np.zeros_like(img_8u)
    for i in range(1, num_labels):
        if np.logical_and(labels == i, lesion_gt > 0).any():
            roi_mask[labels == i] = 1
    
    if roi_mask.sum() == 0:
        kernel = np.ones((61, 61), np.uint8)
        roi_mask = cv2.dilate((lesion_gt > 0).astype(np.uint8), kernel, iterations=1)
    else:
        # 方案二：加大 Padding，确保上下文信息完整
        kernel = np.ones((padding, padding), np.uint8)
        roi_mask = cv2.dilate(roi_mask, kernel, iterations=2)
    return roi_mask

# ==========================================
# 2. 离线裁剪函数 (关键：对齐训练与测试分布)
# ==========================================
def offline_crop_dataset(model_roi, root_dir, output_dir, device, img_size=256):
    """
    使用 ROI 模型处理所有数据，并保存裁剪后的结果。
    """
    model_roi.eval()
    data_path = os.path.join(root_dir, 'data')
    gt_path = os.path.join(root_dir, 'GT')
    out_data = os.path.join(output_dir, 'data')
    out_gt = os.path.join(output_dir, 'GT')
    os.makedirs(out_data, exist_ok=True); os.makedirs(out_gt, exist_ok=True)
    
    fnames = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    print(f">>> Cropping {len(fnames)} images to {output_dir}...")
    
    for fn in tqdm(fnames):
        data = np.load(os.path.join(data_path, fn))
        lesion_gt = cv2.imread(os.path.join(gt_path, fn.replace('.npy', '.png')), 0)
        
        # 定位
        v0 = torch.from_numpy(data[:,:,0]/255.0).unsqueeze(0).unsqueeze(0).to(device).float()
        with torch.no_grad():
            roi_mask = (torch.sigmoid(model_roi(v0)) > 0.5).cpu().numpy()[0,0]
        
        coords = np.argwhere(roi_mask)
        if coords.size == 0: # 兜底逻辑
            y1, y2, x1, x2 = 0, img_size, 0, img_size
        else:
            y1, x1 = coords.min(axis=0); y2, x2 = coords.max(axis=0)
            # 方案二：增加 Padding
            y1, y2, x1, x2 = max(0,y1-30), min(img_size,y2+30), max(0,x1-30), min(img_size,x2+30)
        
        # 裁剪并缩放
        cropped_9ch = np.zeros((img_size, img_size, 9), dtype=np.uint8)
        for c in range(9):
            cropped_9ch[:,:,c] = cv2.resize(data[y1:y2, x1:x2, c], (img_size, img_size))
        
        cropped_gt = cv2.resize(lesion_gt[y1:y2, x1:x2], (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        
        np.save(os.path.join(out_data, fn), cropped_9ch)
        cv2.imwrite(os.path.join(out_gt, fn.replace('.npy', '.png')), cropped_gt)

# ==========================================
# 3. 数据集与辅助 Loss
# ==========================================
class Stage2Dataset(Dataset):
    def __init__(self, root_dir):
        self.data_dir = os.path.join(root_dir, 'data')
        self.gt_dir = os.path.join(root_dir, 'GT')
        self.ids = [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        fname = self.ids[i]
        img = (np.load(os.path.join(self.data_dir, fname)).transpose(2,0,1).astype(np.float32) / 255.0)
        mask = cv2.imread(os.path.join(self.gt_dir, fname.replace('.npy', '.png')), 0)
        mask = (mask > 127).astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(mask).unsqueeze(0), fname

# 方案三：深层监督 (Deep Supervision) 辅助计算
def compute_dice(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2. * inter) / (pred.sum() + target.sum() + 1e-7)

# ==========================================
# 4. 主流程
# ==========================================
def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler()

    # --- STAGE 1: 训练 ROI 定位模型 ---
    print("\n" + "="*20 + " Step 1: ROI Localization Training " + "="*20)
    model_roi = smp.Unet(encoder_name='mobilenet_v2', in_channels=1, classes=1).to(device)
    # 此处省略具体 Stage1 训练循环，假设已完成并加载
    # model_roi.load_state_dict(torch.load('roi_model.pth')) 

    # --- STAGE 1.5: 离线裁剪 (方案一核心) ---
    for part in ['train', 'val', 'test']:
        offline_crop_dataset(model_roi, os.path.join(args.base_path, part), 
                             f'./cropped_dataset/{part}', device)

    # --- STAGE 2: 真正的局部精细化训练 ---
    print("\n" + "="*20 + " Step 2: Fine Segmentation on Cropped Data " + "="*20)
    train_loader = DataLoader(Stage2Dataset('./cropped_dataset/train'), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Stage2Dataset('./cropped_dataset/val'), batch_size=args.val_batch_size, shuffle=False)
    
    model_seg = HybridTemporalNet(base_ch=args.base_ch, depth=args.depth).to(device)
    optimizer = optim.AdamW(model_seg.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    crit_bce = nn.BCEWithLogitsLoss()
    crit_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)

    best_val_dice = 0.0
    for epoch in range(1, args.epochs_seg + 1):
        model_seg.train()
        for img, mask, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            with autocast():
                # 如果要实现方案三的 Deep Supervision，需在模型输出层增加辅助头
                # 这里使用标准输出演示
                pred = model_seg(img)
                loss = 0.5 * crit_bce(pred, mask) + 0.5 * crit_dice(pred, mask)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

        # 验证
        model_seg.eval()
        val_dices = []
        with torch.no_grad():
            for img, mask, _ in val_loader:
                img, mask = img.to(device), mask.to(device)
                pred = model_seg(img)
                val_dices.append(compute_dice(pred, mask).item())
        
        avg_val_dice = np.mean(val_dices)
        print(f"Epoch {epoch} | Val Dice: {avg_val_dice:.4f}")
        scheduler.step(avg_val_dice)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model_seg.state_dict(), "best_seg_cropped.pth")
            print(">>> Best Model Saved!")

    # --- 最终测试：级联逻辑闭环 ---
    # 此处测试必须也在裁剪后的测试集上跑，或者走完整的 Inference Pipeline (裁剪->分割->映射)
    print("\n>>>> FINAL CASCADED TEST ON CROPPED IMAGES <<<<")
    # 加载 best_seg_cropped.pth 进行评估即可
    print("\n" + "="*20 + " Step 3: Final Testing " + "="*20)
    if os.path.exists("best_seg_cropped.pth"):
        model_seg.load_state_dict(torch.load("best_seg_cropped.pth"))
        print("成功加载最佳分割权重进行测试...")

    test_loader = DataLoader(Stage2Dataset(f'{crop_root}/test'), batch_size=args.val_batch_size, shuffle=False)
    model_seg.eval()
    
    test_dices, test_ious = [], []
    vis_dir = './final_test_visualizations'
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        for i, (img, mask, fname) in enumerate(tqdm(test_loader, desc="Final Test")):
            img, mask = img.to(device), mask.to(device)
            with autocast():
                preds = model_seg(img)
            
            # 计算每批次的平均指标
            for j in range(img.size(0)):
                dice_val, iou_val = compute_metrics(preds[j:j+1], mask[j:j+1])
                test_dices.append(dice_val)
                test_ious.append(iou_val)

                # 保存前 20 张可视化对比
                if len(test_dices) <= 20:
                    p_np = (torch.sigmoid(preds[j, 0]) > 0.5).cpu().numpy().astype(np.uint8) * 255
                    m_np = (mask[j, 0].cpu().numpy() * 255).astype(np.uint8)
                    i_np = (img[j, 0].cpu().numpy() * 255).astype(np.uint8)
                    combined = np.hstack([i_np, m_np, p_np])
                    cv2.imwrite(os.path.join(vis_dir, f"test_{fname[j].split('.')[0]}.png"), combined)

    print("\n" + "="*40)
    print(f">>>> 最终测试结果 <<<<")
    print(f"Mean Test Dice: {np.mean(test_dices):.4f}")
    print(f"Mean Test IoU:  {np.mean(test_ious):.4f}")
    print(f"可视化图片已保存至: {vis_dir}")
    print("="*40)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='./processed_9ch_vibrant_label')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--epochs_seg', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())