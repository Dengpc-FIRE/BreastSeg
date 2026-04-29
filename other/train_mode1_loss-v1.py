import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import cv2

import segmentation_models_pytorch as smp

# 导入你的模型和数据集
from dataset import BreastDMDataset 
from model1 import HybridTemporalNet

# ==========================================
# 0. 环境配置
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
# 1. 核心替换：定义更强大的联合损失函数
# ==========================================
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        # Focal Loss: 专注于难分类样本
        self.focal = smp.losses.FocalLoss(mode='binary')
        # Tversky Loss: 针对小目标分割优化
        # alpha=0.3, beta=0.7 表示更关注 False Negatives (漏诊)
        self.tversky = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7, from_logits=True)

    def forward(self, y_pred, y_true):
        return 0.5 * self.focal(y_pred, y_true) + 0.5 * self.tversky(y_pred, y_true)

# ==========================================
# 2. 评估与可视化函数
# ==========================================
def evaluate(model, loader, device):
    model.eval()
    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for batch in loader:
            imgs, masks = batch['image'].to(device), batch['mask'].to(device)
            with autocast():
                preds = model(imgs)
            preds = (torch.sigmoid(preds) > 0.5).float()
            
            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), masks.long(), mode='binary')
            dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
    return np.mean(dice_scores), np.mean(iou_scores)

def save_test_visuals(model, loader, device, save_dir, n=20):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for batch in loader:
            imgs, masks, fnames = batch['image'].to(device), batch['mask'].to(device), batch['filename']
            preds = torch.sigmoid(model(imgs))
            
            for i in range(imgs.shape[0]):
                if count >= n: return
                # 可视化：原图(第0通道) | 真值 | 预测
                img_bg = (imgs[i, 0].cpu().numpy() * 255).astype(np.uint8)
                gt = (masks[i, 0].cpu().numpy() * 255).astype(np.uint8)
                pr = ((preds[i, 0].cpu().numpy() > 0.5) * 255).astype(np.uint8)
                
                res = np.hstack([cv2.cvtColor(img_bg, cv2.COLOR_GRAY2BGR), 
                                 cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR), 
                                 cv2.cvtColor(pr, cv2.COLOR_GRAY2BGR)])
                cv2.imwrite(os.path.join(save_dir, f"test_{fnames[i].split('.')[0]}.png"), res)
                count += 1

# ==========================================
# 3. 训练主程序
# ==========================================
def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_path, exist_ok=True)

    # 数据准备
    train_ds = BreastDMDataset(os.path.join(args.train_path, 'data'), os.path.join(args.train_path, 'GT'))
    val_ds = BreastDMDataset(os.path.join(args.val_path, 'data'), os.path.join(args.val_path, 'GT'))
    test_ds = BreastDMDataset(os.path.join(args.test_path, 'data'), os.path.join(args.test_path, 'GT'))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size, shuffle=False)

    # 模型与新损失函数
    model = HybridTemporalNet(base_ch=args.base_ch, depth=args.depth).to(device)
    criterion = CombinedLoss() # 使用新定义的 Focal + Tversky
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = GradScaler()

    best_dice = 0.0
    early_stop_count = 0

    print(f"Starting Training: {args.model_type} on {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            imgs, masks = batch['image'].to(device), batch['mask'].to(device)
            optimizer.zero_grad()
            with autocast():
                preds = model(imgs)
                loss = criterion(preds, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 验证
        val_dice, val_iou = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        scheduler.step(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            early_stop_count = 0
            torch.save(model.state_dict(), os.path.join(args.output_path, 'best_model.pth'))
            print(">>> Best Model Saved!")
        else:
            early_stop_count += 1

        if early_stop_count >= args.early_stop:
            print("Early Stopping triggered.")
            break

    # 最终测试
    print("\n" + "="*20 + " Final Test " + "="*20)
    model.load_state_dict(torch.load(os.path.join(args.output_path, 'best_model.pth')))
    tdice, tiou = evaluate(model, test_loader, device)
    print(f"TEST RESULTS: Dice={tdice:.4f}, IoU={tiou:.4f}")
    
    # 保存可视化
    save_test_visuals(model, test_loader, device, os.path.join(args.output_path, 'vis_results'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./processed_9ch_vibrant_label/train')
    parser.add_argument('--val_path', type=str, default='./processed_9ch_vibrant_label/val')
    parser.add_argument('--test_path', type=str, default='./processed_9ch_vibrant_label/test')
    parser.add_argument('--output_path', type=str, default='./results_hybrid_v2')
    parser.add_argument('--model_type', type=str, default='hybrid')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early_stop', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--depth', type=int, default=4)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())