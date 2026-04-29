import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import argparse

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
    # 使用 ResNet34 作为主干
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights=None,           # 9通道无法使用ImageNet权重
        in_channels=9,                  # <--- 接收 9 通道输入
        classes=1,                      
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
            with autocast():
                preds = model(img)
            
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
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_dice = 0.0

    # --- 训练循环 ---
    print("Start Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img, mask, _ in pbar:
            img, mask = img.to(device), mask.to(device)
            
            optimizer.zero_grad()
            with autocast():
                preds = model(img)
                loss = loss_fn(preds, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
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





# import segmentation_models_pytorch as smp
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# import numpy as np
# import cv2
# import random
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from torch.cuda.amp import autocast, GradScaler
# import argparse
# import albumentations as A

# # ==========================================
# # 0. 环境配置与随机种子固定
# # ==========================================
# def seed_everything(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     os.environ['OMP_NUM_THREADS'] = '1' # 解决 libgomp 报错
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # ==========================================
# # 1. Dataset 类 (支持 Albumentations 增强)
# # ==========================================
# class BreastDM9ChDataset(Dataset):
#     def __init__(self, data_dir, gt_dir, img_size=256, transform=None):
#         self.data_dir = data_dir
#         self.gt_dir = gt_dir
#         self.img_size = img_size
#         self.transform = transform
        
#         if os.path.exists(data_dir):
#             self.ids = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
#         else:
#             self.ids = []
#             print(f"[Warning] Directory not found: {data_dir}")

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, i):
#         file_name = self.ids[i]
#         # 加载 9 通道 NPY 数据 [H, W, 9]
#         data = np.load(os.path.join(self.data_dir, file_name)) 
        
#         # 加载 Mask
#         mask_name = file_name.replace('.npy', '.png')
#         gt_path = os.path.join(self.gt_dir, mask_name)
#         if os.path.exists(gt_path):
#             mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
#             mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
#         else:
#             mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

#         # 数据增强
#         if self.transform:
#             augmented = self.transform(image=data, mask=mask)
#             data, mask = augmented['image'], augmented['mask']

#         # 归一化与 Tensor 转换
#         image = data.astype(np.float32) / 255.0
#         image = torch.from_numpy(image.transpose(2, 0, 1)) # [9, H, W]

#         mask = mask.astype(np.float32) / 255.0
#         mask[mask > 0.5] = 1.0
#         mask_tensor = torch.from_numpy(mask).unsqueeze(0) # [1, H, W]

#         return image, mask_tensor, file_name

# # ==========================================
# # 2. 增强策略与模型获取
# # ==========================================
# def get_transforms(mode='train'):
#     if mode == 'train':
#         return A.Compose([
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomRotate90(p=0.5),
#             # 使用 Affine 代替被弃用的 ShiftScaleRotate 警告
#             A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, 
#                      scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
#         ])
#     return None

# def get_model(in_channels=9):
#     return smp.Unet(
#         encoder_name="resnet34",
#         encoder_weights=None, 
#         in_channels=in_channels,
#         classes=1
#     )

# # ==========================================
# # 3. 评估与可视化函数
# # ==========================================
# def evaluate(model, loader, device, desc="Validation"):
#     model.eval()
#     dice_scores, iou_scores = [], []
#     with torch.no_grad():
#         for img, mask, _ in tqdm(loader, desc=desc, leave=False):
#             img, mask = img.to(device), mask.to(device)
#             with autocast():
#                 preds = model(img)
#             preds = (torch.sigmoid(preds) > 0.5).float()
            
#             tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), mask.long(), mode='binary')
#             dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
#             iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            
#             dice_scores.append(dice.item())
#             iou_scores.append(iou.item())
#     return np.mean(dice_scores), np.mean(iou_scores)

# def visualize_results(model, loader, device, output_dir, n_images=20):
#     model.eval()
#     os.makedirs(output_dir, exist_ok=True)
#     count = 0
#     with torch.no_grad():
#         for imgs, masks, file_names in loader:
#             imgs, masks = imgs.to(device), masks.to(device)
#             with autocast():
#                 preds = torch.sigmoid(model(imgs))
            
#             imgs_np = imgs.cpu().numpy()
#             masks_np = masks.cpu().numpy()
#             preds_np = (preds > 0.5).float().cpu().numpy()
            
#             for i in range(imgs_np.shape[0]):
#                 if count >= n_images: return
#                 # 取前3通道可视化，如果没有意义请改为取第0通道
#                 img_show = imgs_np[i, :3, :, :].transpose(1, 2, 0)
#                 img_show = (img_show * 255).clip(0, 255).astype(np.uint8)
                
#                 gt_m = (masks_np[i, 0, :, :] * 255).astype(np.uint8)
#                 pd_m = (preds_np[i, 0, :, :] * 255).astype(np.uint8)
                
#                 # 拼接：原图 | 真值 | 预测
#                 res = np.hstack([img_show, 
#                                  cv2.cvtColor(gt_m, cv2.COLOR_GRAY2BGR), 
#                                  cv2.cvtColor(pd_m, cv2.COLOR_GRAY2BGR)])
#                 cv2.imwrite(os.path.join(output_dir, f"test_{file_names[i].split('.')[0]}.png"), res)
#                 count += 1

# # ==========================================
# # 4. 主训练流程
# # ==========================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_path', type=str, default='./processed_9ch_vibrant_label/train')
#     parser.add_argument('--val_path', type=str, default='./processed_9ch_vibrant_label/val')
#     parser.add_argument('--test_path', type=str, default='./processed_9ch_vibrant_label/test')
#     parser.add_argument('--output_path', type=str, default='./results_unet_9ch_optimized')
#     parser.add_argument('--epochs', type=int, default=100)
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--early_stop', type=int, default=15)
#     args = parser.parse_args()

#     seed_everything(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output_path, exist_ok=True)

#     # 数据加载
#     train_ds = BreastDM9ChDataset(os.path.join(args.train_path, 'data'), os.path.join(args.train_path, 'GT'), transform=get_transforms('train'))
#     val_ds = BreastDM9ChDataset(os.path.join(args.val_path, 'data'), os.path.join(args.val_path, 'GT'))
#     test_ds = BreastDM9ChDataset(os.path.join(args.test_path, 'data'), os.path.join(args.test_path, 'GT'))

#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
#     test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

#     # 模型、损失、优化器
#     model = get_model().to(device)
#     dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
#     bce_loss = nn.BCEWithLogitsLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
#     # 移除 verbose=True 避免新版本 PyTorch 报错
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
#     scaler = GradScaler()

#     best_dice = 0.0
#     epochs_no_improve = 0

#     print(f"Training on {device}... Data Size: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         epoch_loss = 0
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
#         for img, mask, _ in pbar:
#             img, mask = img.to(device), mask.to(device)
#             optimizer.zero_grad()
#             with autocast():
#                 output = model(img)
#                 loss = 0.5 * dice_loss(output, mask) + 0.5 * bce_loss(output, mask)
            
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             epoch_loss += loss.item()
#             pbar.set_postfix({'loss': f"{loss.item():.4f}"})

#         val_dice, val_iou = evaluate(model, val_loader, device, desc="Validating")
#         print(f"Epoch {epoch} | Loss: {epoch_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
#         scheduler.step(val_dice)

#         if val_dice > best_dice:
#             best_dice = val_dice
#             epochs_no_improve = 0
#             torch.save(model.state_dict(), os.path.join(args.output_path, 'best_model.pth'))
#             print(">>> 💾 Best Model Saved!")
#         else:
#             epochs_no_improve += 1

#         if epochs_no_improve >= args.early_stop:
#             print(f"Early Stopping at epoch {epoch}")
#             break

#     # --- 最终测试与可视化 ---
#     print("\n" + "="*30 + "\nFinal Testing\n" + "="*30)
#     model.load_state_dict(torch.load(os.path.join(args.output_path, 'best_model.pth')))
#     test_dice, test_iou = evaluate(model, test_loader, device, desc="Testing")
#     print(f"FINAL TEST RESULTS | Dice: {test_dice:.4f} | IoU: {test_iou:.4f}")

#     vis_dir = os.path.join(args.output_path, 'visualizations')
#     visualize_results(model, test_loader, device, vis_dir)
#     print(f"Visualizations saved to {vis_dir}")

# if __name__ == '__main__':
#     main()