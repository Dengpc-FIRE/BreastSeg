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

# 导入你自己的数据集和模型类
from dataset import BreastDMDataset 
from model1 import HybridTemporalNet

# ==========================================
# 0. 固定随机种子
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
# 1. 统一评估函数
# ==========================================
def evaluate(model, loader, device, desc="Evaluating"):
    model.eval()
    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            with autocast():
                preds = model(imgs)
            preds = (torch.sigmoid(preds) > 0.5).float()
            
            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), masks.long(), mode='binary')
            dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
    return np.mean(dice_scores), np.mean(iou_scores)

# ==========================================
# 2. 统一可视化函数 (原图|真值|预测)
# ==========================================
def save_visuals(model, loader, device, save_dir, n_images=10):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            fnames = batch['filename']
            
            with autocast():
                preds = torch.sigmoid(model(imgs))
            
            imgs_np = imgs.cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_np = (preds > 0.5).float().cpu().numpy()
            
            for i in range(imgs_np.shape[0]):
                if count >= n_images: return
                # 取第0通道作为原图展示
                img_show = (imgs_np[i, 0, :, :] * 255).clip(0, 255).astype(np.uint8)
                img_show = cv2.cvtColor(img_show, cv2.COLOR_GRAY2BGR)
                
                gt_m = (masks_np[i, 0, :, :] * 255).astype(np.uint8)
                pd_m = (preds_np[i, 0, :, :] * 255).astype(np.uint8)
                
                # 拼接
                res = np.hstack([img_show, 
                                 cv2.cvtColor(gt_m, cv2.COLOR_GRAY2BGR), 
                                 cv2.cvtColor(pd_m, cv2.COLOR_GRAY2BGR)])
                cv2.imwrite(os.path.join(save_dir, f"vis_{fnames[i].split('.')[0]}.png"), res)
                count += 1

# ==========================================
# 3. 训练主流程
# ==========================================
def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device: {device} | Model: {args.model_type}')

    # 数据集
    train_ds = BreastDMDataset(os.path.join(args.train_path, 'data'), os.path.join(args.train_path, 'GT'))
    val_ds = BreastDMDataset(os.path.join(args.val_path, 'data'), os.path.join(args.val_path, 'GT'))
    test_ds = BreastDMDataset(os.path.join(args.test_path, 'data'), os.path.join(args.test_path, 'GT'))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size, shuffle=False)

    # 模型选择
    if args.model_type == 'hybrid':
        model = HybridTemporalNet(base_ch=args.base_ch, depth=args.depth).to(device)
    else:
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=9, classes=1).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
    scaler = GradScaler()

    best_dice = 0.0
    epochs_no_improve = 0
    os.makedirs(args.output_path, exist_ok=True)

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch in pbar:
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            with autocast():
                preds = model(imgs)
                loss = 0.5 * criterion_bce(preds, masks) + 0.5 * criterion_dice(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 验证
        val_dice, val_iou = evaluate(model, val_loader, device, desc="Validating")
        print(f"Epoch {epoch} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        scheduler.step(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.output_path, 'best_model.pth'))
            print('>>> 💾 Best Model Saved')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.early_stop:
            print(f"Early Stopping at epoch {epoch}")
            break

    # ==========================================
    # 4. 最终测试与可视化对比
    # ==========================================
    print("\n" + "="*30 + "\nStarting Final Test\n" + "="*30)
    model.load_state_dict(torch.load(os.path.join(args.output_path, 'best_model.pth')))
    
    test_dice, test_iou = evaluate(model, test_loader, device, desc="Testing")
    print(f"FINAL TEST RESULTS | Dice: {test_dice:.4f} | IoU: {test_iou:.4f}")

    vis_dir = os.path.join(args.output_path, 'visualizations_test')
    save_visuals(model, test_loader, device, vis_dir, n_images=20)
    print(f"Test visualizations saved to {vis_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument('--train_path', type=str, default='./processed_9ch_vibrant_label/train')
    parser.add_argument('--val_path', type=str, default='./processed_9ch_vibrant_label/val')
    parser.add_argument('--test_path', type=str, default='./processed_9ch_vibrant_label/test')
    parser.add_argument('--output_path', type=str, default='./results_hybrid')
    
    # 对比实验配置
    parser.add_argument('--model_type', type=str, default='hybrid', choices=['hybrid', 'unet'])
    parser.add_argument('--seed', type=int, default=42)
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early_stop', type=int, default=15)
    
    # HybridTemporalNet 专用参数
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--depth', type=int, default=4)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)