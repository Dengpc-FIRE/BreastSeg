import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import segmentation_models_pytorch as smp

# 引入你的模型 (请确保 model1.py 在当前目录)
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
# 1. 核心提分算法：Tversky Loss
# ==========================================
class TverskyLoss(nn.Module):
    """
    Tversky Loss 对假阴性(漏诊)给予更高的惩罚权重 (beta=0.7)
    有助于在小病灶分割中获得更高的 Dice 分数
    """
    def __init__(self, alpha=0.3, beta=0.7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        tversky = (tp + 1e-7) / (tp + self.alpha * fp + self.beta * fn + 1e-7)
        return 1 - tversky

# ==========================================
# 2. 数据增强定义 (解剖学仿生形变)
# ==========================================
def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # 弹性形变：模拟 MRI 拍摄时乳腺受挤压产生的非线性变形 (提分关键)
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
        A.GridDistortion(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])

# ==========================================
# 3. 增强版数据集类
# ==========================================
class Stage2Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data_dir = os.path.join(root_dir, 'data')
        self.gt_dir = os.path.join(root_dir, 'GT')
        self.ids = [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        fname = self.ids[i]
        # 加载 9 通道数据 [256, 256, 9]
        img = np.load(os.path.join(self.data_dir, fname))
        mask = cv2.imread(os.path.join(self.gt_dir, fname.replace('.npy', '.png')), 0)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            # Albumentations 同步增强图像和掩码
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.from_numpy(img), torch.from_numpy(mask).unsqueeze(0), fname

# ==========================================
# 4. 指标计算与评估函数
# ==========================================
def compute_metrics(pred_logits, target):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    tp, fp, fn, tn = smp.metrics.get_stats(pred.long(), target.long(), mode='binary')
    dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
    iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
    return dice.item(), iou.item()

def evaluate(model, loader, device):
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for img, mask, _ in loader:
            img, mask = img.to(device), mask.to(device)
            with autocast('cuda'):
                pred = model(img)
            d, i = compute_metrics(pred, mask)
            dices.append(d); ious.append(i)
    return np.mean(dices), np.mean(ious)

# ==========================================
# 5. 主程序
# ==========================================
def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_path, exist_ok=True)
    scaler = GradScaler()

    # 加载裁剪后的数据 (假定离线裁剪已完成)
    train_dir = './cropped_dataset/train'
    val_dir = './cropped_dataset/val'
    test_dir = './cropped_dataset/test'

    train_loader = DataLoader(Stage2Dataset(train_dir, transform=get_train_transforms()), 
                              batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Stage2Dataset(val_dir), batch_size=args.val_batch_size, shuffle=False)
    test_loader = DataLoader(Stage2Dataset(test_dir), batch_size=args.val_batch_size, shuffle=False)

    # 模型初始化
    model = HybridTemporalNet(base_ch=args.base_ch, depth=args.depth).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)

    # 复合损失函数：BCE + Tversky
    crit_bce = nn.BCEWithLogitsLoss()
    crit_tversky = TverskyLoss(alpha=0.3, beta=0.7)

    best_val_dice = 0.0
    print("\n" + "🚀" + " 开始精细化训练 (Tversky + Augmentation) " + "🚀")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img, mask, _ in pbar:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                pred = model(img)
                # 方案三：复合加权损失，Tversky 占大头
                loss = 0.3 * crit_bce(pred, mask) + 0.7 * crit_tversky(pred, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 验证阶段
        val_dice, val_iou = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        scheduler.step(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.output_path, "best_model_v2_pro.pth"))
            print(f">>> 💾 最佳权重已保存 (Dice: {val_dice:.4f})")

    # --- 最终测试 ---
    print("\n" + "评估最终测试集表现...")
    model.load_state_dict(torch.load(os.path.join(args.output_path, "best_model_v2_pro.pth")))
    test_dice, test_iou = evaluate(model, test_loader, device)
    print(f"\n>>>> FINAL TEST RESULTS <<<<")
    print(f"Mean Test Dice: {test_dice:.4f}")
    print(f"Mean Test IoU:  {test_iou:.4f}")
    print("="*40)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./results_v2_pro')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())