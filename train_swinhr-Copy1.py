import segmentation_models_pytorch as smp
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
from swinhr_v6 import SwinHR
import torch.nn.functional as F

# ==========================================
# 1. Dataset (逻辑保持不变)
# ==========================================
class BreastDM9ChDataset(Dataset):
    def __init__(self, data_dir, gt_dir, img_size=256):
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        # 只有当目录存在时才读取文件列表，防止报错
        if os.path.exists(data_dir):
            self.ids = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        else:
            self.ids = []
            print(f"[Warning] Directory not found: {data_dir}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        file_name = self.ids[i]
        data_path = os.path.join(self.data_dir, file_name)
        data = np.load(data_path) 
        
        # 你的 NPY 是 (256, 256, 9) -> 转置为 (9, 256, 256) 给 PyTorch
        image = data.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1)) # [9, H, W]

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
# 2. 模型：标准 U-Net 但改为 9 通道输入
# ==========================================

def get_model():
    model = SwinHR(
        # img_size=(256, 256), 
        in_channels=1,        
        attn_channels=8,      
        out_channels=1,       
        spatial_dims=2        
    )
    return model

# ==========================================
# 3. 评估/测试通用函数
# ==========================================
def evaluate(model, loader, device, desc="Validation"):
    model.eval()
    dice_scores = []
    # 也可以加 IoU
    iou_scores = []
    
    with torch.no_grad():
        for img, mask, _ in tqdm(loader, desc=desc, leave=False):
            img, mask = img.to(device), mask.to(device)
            with autocast('cuda'):
                preds, _, _ = model(img)
            
            # 预测二值化
            preds = (torch.sigmoid(preds) > 0.5).float()
            
            # 计算指标
            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), mask.long(), mode='binary')
            
            # Dice (F1)
            dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
            # IoU
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
    
    return np.mean(dice_scores), np.mean(iou_scores)

# ==========================================
# 4. 训练主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    # 【修改点 1】默认路径改为新的 'processed_9ch_vibrant_label'
    parser.add_argument('--train_path', type=str, default='./processed_9ch_vibrant_label/train')
    parser.add_argument('--val_path', type=str, default='./processed_9ch_vibrant_label/val')
    # 【修改点 2】新增测试集路径参数
    parser.add_argument('--test_path', type=str, default='./processed_9ch_vibrant_label/test')
    
    parser.add_argument('--output_path', type=str, default='./results_unet_9ch')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4) # Windows下如果不稳定可以改为0
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 准备数据 ---
    print("Loading Datasets...")
    train_ds = BreastDM9ChDataset(os.path.join(args.train_path, 'data'), os.path.join(args.train_path, 'GT'))
    val_ds = BreastDM9ChDataset(os.path.join(args.val_path, 'data'), os.path.join(args.val_path, 'GT'))
    test_ds = BreastDM9ChDataset(os.path.join(args.test_path, 'data'), os.path.join(args.test_path, 'GT'))
    
    if len(train_ds) == 0:
        print("Error: No training data found. Check your paths.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # 【修改点 3】定义测试 Loader
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Train Size: {len(train_ds)}, Val Size: {len(val_ds)}, Test Size: {len(test_ds)}")

    # --- 初始化模型 ---
    print("Initializing 9-Channel U-Net (ResNet34 Backbone)...")
    model = get_model().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
    scaler = GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_dice = 0.0

    # --- 训练循环 ---
    print("Start Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        alpha = 0.1 
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img, mask, _ in pbar:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # 1. 接收三个返回值
                preds, feat_anatomy, feat_hemo = model(img)
                
                # 2. 计算标准的分割 Dice Loss
                seg_loss = loss_fn(preds, mask)
                
                # ==========================================
                # 【新增创新点】：跨模态正交约束损失
                # 强迫 feat_anatomy 和 feat_hemo 学到互不重叠的信息
                # ==========================================
                # 展平空间维度 (B, C, H, W) -> (B, C, H*W)
                b, c, h, w = feat_anatomy.shape
                f_a = feat_anatomy.view(b, c, -1)
                f_h = feat_hemo.view(b, c, -1)
                
                # 对通道进行 L2 归一化 (非常重要，防止尺度不同)
                f_a = F.normalize(f_a, p=2, dim=1)
                f_h = F.normalize(f_h, p=2, dim=1)
                
                # 计算余弦相似度矩阵 (内积)，越接近 0 说明越正交
                # f_a: (B, C, N), f_h.transpose: (B, N, C) -> sim: (B, C, C)
                sim_matrix = torch.bmm(f_a, f_h.transpose(1, 2))
                
                # 正交损失 = 相似度矩阵的 F-范数 (越小越好)
                ortho_loss = torch.norm(sim_matrix, p='fro') / (c * c)
                
                # 3. 组合最终的总 Loss
                loss = seg_loss + alpha * ortho_loss
                # ==========================================
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({'Seg': f"{seg_loss.item():.4f}", 'Ortho': f"{ortho_loss.item():.4f}"})
            
        scheduler.step()
        
        # --- 验证 ---
        val_dice, val_iou = evaluate(model, val_loader, device, desc="Validating")
        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.output_path, 'best_model.pth'))
            print(f">>> 💾 Best Model Saved! (Dice: {best_dice:.4f})")

    # ==========================================
    # 【修改点 4】训练结束后，加载最佳模型进行测试
    # ==========================================
    print("\n=======================================")
    print("Training Finished. Starting Testing...")
    print("=======================================")
    
    best_model_path = os.path.join(args.output_path, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded Best Model from {best_model_path}")
        
        test_dice, test_iou = evaluate(model, test_loader, device, desc="Testing")
        
        print(f"\n>>>> FINAL TEST RESULTS <<<<")
        print(f"Test Set Size: {len(test_ds)}")
        print(f"Test Dice Score: {test_dice:.4f}")
        print(f"Test IoU Score:  {test_iou:.4f}")
        print("=======================================")
    else:
        print("Error: Best model file not found!")

if __name__ == '__main__':
    main()

