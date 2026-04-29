import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import argparse
import segmentation_models_pytorch as smp # 仍用于计算 Loss 和 Metrics

# --- 【必须】安装并导入 MONAI ---
# pip install monai
from monai.networks.nets import SwinUNETR

# ==========================================
# 1. Dataset (与上面完全相同)
# ==========================================
class BreastDM9ChDataset(Dataset):
    def __init__(self, data_dir, gt_dir, img_size=256):
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        if os.path.exists(data_dir):
            self.ids = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        else:
            self.ids = []
            print(f"[Warning] Directory not found: {data_dir}")

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        file_name = self.ids[i]
        data = np.load(os.path.join(self.data_dir, file_name)) 
        image = torch.from_numpy((data.astype(np.float32) / 255.0).transpose(2, 0, 1))
        mask_name = file_name.replace('.npy', '.png')
        gt_path = os.path.join(self.gt_dir, mask_name)
        if os.path.exists(gt_path):
            mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape[0] != self.img_size:
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            elif mask is None: mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else: mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        mask = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        return image, torch.from_numpy(np.expand_dims(mask, axis=0)), file_name

# ==========================================
# 2. 模型：Swin UNETR (MONAI)
# ==========================================
def get_model(img_size=256):
    print("Initializing 9-Channel Swin UNETR (2D Mode)...")
    model = SwinUNETR(
        # img_size=(img_size, img_size), # <--- 【删除】新版 MONAI 已弃用该参数
        in_channels=9,       # <--- 【修正】注意这里改成了 in_channels
        out_channels=1,      # <--- 【修正】注意这里改成了 out_channels
        feature_size=24,     
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        norm_name="instance",
        spatial_dims=2       # 依然必须保留 2D 模式
    )
    return model

# ==========================================
# 3. 评估/测试通用函数
# ==========================================
def evaluate(model, loader, device, desc="Validation"):
    model.eval()
    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for img, mask, _ in tqdm(loader, desc=desc, leave=False):
            img, mask = img.to(device), mask.to(device)
            with autocast('cuda'):
                preds = model(img)
            
            # SwinUNETR 输出可能没有经过 Sigmoid
            preds = (torch.sigmoid(preds) > 0.5).float()
            
            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), mask.long(), mode='binary')
            dice_scores.append(smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise").item())
            iou_scores.append(smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item())
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    mean_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)
    
    return mean_dice, std_dice, mean_iou, std_iou

# ==========================================
# 4. 训练主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./processed_9ch_vibrant_label/train')
    parser.add_argument('--val_path', type=str, default='./processed_9ch_vibrant_label/val')
    parser.add_argument('--test_path', type=str, default='./processed_9ch_vibrant_label/test')
    
    # 修改了默认输出路径
    parser.add_argument('--output_path', type=str, default='./results_swinunetr_9ch')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16) # Transformer 显存占用大，可能需要调小 batch_size
    parser.add_argument('--lr', type=float, default=1e-4)     # Transformer 通常需要更小的初始学习率
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Datasets...")
    train_ds = BreastDM9ChDataset(os.path.join(args.train_path, 'data'), os.path.join(args.train_path, 'GT'))
    val_ds = BreastDM9ChDataset(os.path.join(args.val_path, 'data'), os.path.join(args.val_path, 'GT'))
    test_ds = BreastDM9ChDataset(os.path.join(args.test_path, 'data'), os.path.join(args.test_path, 'GT'))
    
    if len(train_ds) == 0: return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = get_model(img_size=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
    scaler = GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_dice = 0.0

    print("Start Training Swin UNETR...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img, mask, _ in pbar:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                loss = loss_fn(model(img), mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        scheduler.step()
        
        val_dice,val_dice_std,val_iou,val_iou_std = evaluate(model, val_loader, device, desc="Validating")
        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.output_path, 'best_model.pth'))
            print(f">>> 💾 Best Model Saved! (Dice: {best_dice:.4f})")

    # 测试
    print("\n=======================================")
    best_model_path = os.path.join(args.output_path, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        test_dice_mean, test_dice_std, test_iou_mean, test_iou_std = evaluate(model, test_loader, device, desc="Testing")

        print(f"Test Dice Score: {test_dice_mean:.4f} ± {test_dice_std:.4f}")
        print(f"Test IoU Score:  {test_iou_mean:.4f} ± {test_iou_std:.4f}")
        
        print(f"\n>>>> FINAL TEST RESULTS (Swin UNETR) <<<<\nTest Dice: {test_dice:.4f} | Test IoU: {test_iou:.4f}\n=======================================")

if __name__ == '__main__':
    main()