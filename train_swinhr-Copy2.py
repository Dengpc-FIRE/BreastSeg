import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler # 使用最新的 AMP 接口
import argparse
from swinhr_v7 import SwinHR # 确保你的模型文件名正确

# ==========================================
# 1. Dataset (针对 9 通道数据的加载逻辑)
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
            print(f"[Warning] 路径未找到: {data_dir}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        file_name = self.ids[i]
        data_path = os.path.join(self.data_dir, file_name)
        data = np.load(data_path) 
        
        # 归一化并转置为 [C, H, W]
        image = data.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))

        mask_name = file_name.replace('.npy', '.png')
        gt_path = os.path.join(self.gt_dir, mask_name)
        
        if os.path.exists(gt_path):
            mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[0] != self.img_size:
                    mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
        mask = mask.astype(np.float32) / 255.0
        mask[mask > 0.5] = 1.0
        mask_tensor = torch.from_numpy(np.expand_dims(mask, axis=0))

        return image, mask_tensor, file_name

# ==========================================
# 2. 模型初始化
# ==========================================
def get_model():
    model = SwinHR(
        in_channels=1,        # VIBRANT 解剖流
        attn_channels=8,      # SUB 1-8 血流流
        out_channels=1,       
        spatial_dims=2        
    )
    return model

# ==========================================
# 3. 评估/测试函数 (关键：需处理双输出)
# ==========================================
def evaluate(model, loader, device, desc="Validation"):
    model.eval()
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for img, mask, _ in tqdm(loader, desc=desc, leave=False):
            img, mask = img.to(device), mask.to(device)
            with autocast('cuda'):
                # 评估时只取第一个主输出，忽略血流分支预测
                preds_main, _ = model(img)
            
            preds = (torch.sigmoid(preds_main) > 0.5).float()
            
            # 计算指标
            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), mask.long(), mode='binary')
            dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
    
    return np.mean(dice_scores), np.mean(iou_scores)

# ==========================================
# 4. 训练主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./processed_9ch_vibrant_label/train')
    parser.add_argument('--val_path', type=str, default='./processed_9ch_vibrant_label/val')
    parser.add_argument('--test_path', type=str, default='./processed_9ch_vibrant_label/test')
    parser.add_argument('--output_path', type=str, default='./results_swinhr_aux')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据加载
    train_ds = BreastDM9ChDataset(os.path.join(args.train_path, 'data'), os.path.join(args.train_path, 'GT'))
    val_ds = BreastDM9ChDataset(os.path.join(args.val_path, 'data'), os.path.join(args.val_path, 'GT'))
    test_ds = BreastDM9ChDataset(os.path.join(args.test_path, 'data'), os.path.join(args.test_path, 'GT'))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 初始化
    model = get_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
    scaler = GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_dice = 0.0

    print(f"训练开始: {len(train_ds)} 样本, 目标 SOTA: 83.2%")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        aux_weight = 0.4 # 辅助损失权重，建议在 0.3-0.5 之间
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img, mask, _ in pbar:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # 1. 接收主输出和辅助输出
                preds_main, preds_aux = model(img)
                
                # 2. 计算双重损失
                loss_main = loss_fn(preds_main, mask)
                loss_aux = loss_fn(preds_aux, mask)
                
                # 3. 联合优化：引导血流分支学会有物理意义的特征
                loss = loss_main + aux_weight * loss_aux
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Main': f"{loss_main.item():.4f}", 'Aux': f"{loss_aux.item():.4f}"})
        
        scheduler.step()
        
        # 验证
        val_dice, val_iou = evaluate(model, val_loader, device, desc="Validating")
        print(f"Epoch {epoch} | Loss: {epoch_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.output_path, 'best_model.pth'))
            print(f">>> 💾 发现更优模型! (Dice: {best_dice:.4f})")

    # 最终测试
    print("\n训练结束，加载最佳权重进行测试...")
    best_model_path = os.path.join(args.output_path, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        test_dice, test_iou = evaluate(model, test_loader, device, desc="Final Testing")
        print(f"\n>>>> 最终测试结果 <<<<")
        print(f"Test Dice: {test_dice:.4f}")
        print(f"Test IoU:  {test_iou:.4f}")
        print("==========================")

if __name__ == '__main__':
    main()