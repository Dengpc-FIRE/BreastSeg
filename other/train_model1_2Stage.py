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

# 确保 model1.py 在路径中
from model1 import HybridTemporalNet

# ==========================================
# 0. 固定随机种子与环境
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
# 1. 第一阶段：伪标签生成算法
# ==========================================
def generate_pseudo_roi(image_v0, lesion_gt):
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
        kernel = np.ones((31, 31), np.uint8)
        roi_mask = cv2.dilate(roi_mask, kernel, iterations=2)
    return roi_mask

# ==========================================
# 2. 级联数据类
# ==========================================
class CascadedDataset(Dataset):
    def __init__(self, root_dir, is_stage1=True, img_size=256):
        self.data_dir = os.path.join(root_dir, 'data')
        self.gt_dir = os.path.join(root_dir, 'GT')
        self.ids = [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        self.is_stage1 = is_stage1
        self.img_size = img_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fname = self.ids[i]
        data = np.load(os.path.join(self.data_dir, fname))
        mask_path = os.path.join(self.gt_dir, fname.replace('.npy', '.png'))
        lesion_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        lesion_gt = cv2.resize(lesion_gt, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        lesion_gt = (lesion_gt > 127).astype(np.float32)

        if self.is_stage1:
            v0 = data[:, :, 0].astype(np.float32) / 255.0
            roi_label = generate_pseudo_roi(v0, lesion_gt)
            return torch.from_numpy(v0).unsqueeze(0), torch.from_numpy(roi_label).unsqueeze(0).float(), fname
        else:
            img = data.transpose(2, 0, 1).astype(np.float32) / 255.0
            return torch.from_numpy(img), torch.from_numpy(lesion_gt).unsqueeze(0), fname

# ==========================================
# 3. 核心算法：评估函数 (保持与之前一致)
# ==========================================
def evaluate(model, loader, device, desc="Evaluating"):
    model.eval()
    dice_scores, iou_scores = [], []
    with torch.no_grad():
        for img, mask, _ in tqdm(loader, desc=desc, leave=False):
            img, mask = img.to(device), mask.to(device)
            with autocast():
                preds = model(img)
            preds = (torch.sigmoid(preds) > 0.5).float()
            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), mask.long(), mode='binary')
            dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            dice_scores.append(dice.item()); iou_scores.append(iou.item())
    return np.mean(dice_scores), np.mean(iou_scores)

# ==========================================
# 4. 级联推理逻辑
# ==========================================
def cascaded_predict(full_9ch, model_roi, model_seg, device):
    model_roi.eval(); model_seg.eval()
    v0 = full_9ch[0:1].unsqueeze(0).to(device)
    with torch.no_grad():
        roi_mask = (torch.sigmoid(model_roi(v0)) > 0.5).cpu().numpy()[0, 0]
        coords = np.argwhere(roi_mask)
        if coords.size == 0: return np.zeros((256, 256))
        y1, x1 = coords.min(axis=0); y2, x2 = coords.max(axis=0)
        y1, y2, x1, x2 = max(0,y1-15), min(256,y2+15), max(0,x1-15), min(256,x2+15)
        cropped = full_9ch[:, y1:y2, x1:x2].numpy()
        input_seg = np.zeros((9, 256, 256), dtype=np.float32)
        for c in range(9): input_seg[c] = cv2.resize(cropped[c], (256, 256))
        seg_out = torch.sigmoid(model_seg(torch.from_numpy(input_seg).unsqueeze(0).to(device))).cpu().numpy()[0, 0]
        final_mask = np.zeros((256, 256), dtype=np.float32)
        final_mask[y1:y2, x1:x2] = cv2.resize(seg_out, (x2-x1, y2-y1))
    return final_mask

# ==========================================
# 5. 主训练流水线
# ==========================================
def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_path, exist_ok=True)
    scaler = GradScaler()

    # --- Stage 1: ROI 定位训练 ---
    print("\n" + "="*20 + " Stage 1: ROI Training " + "="*20)
    train_loader1 = DataLoader(CascadedDataset(args.train_path, True), batch_size=args.batch_size, shuffle=True, num_workers=4)
    model_roi = smp.Unet(encoder_name='mobilenet_v2', in_channels=1, classes=1).to(device)
    opt1 = optim.Adam(model_roi.parameters(), lr=1e-3)
    crit1 = nn.BCEWithLogitsLoss()
    for e in range(1, args.epochs_roi + 1):
        model_roi.train()
        for img, mask, _ in tqdm(train_loader1, desc=f"ROI Epoch {e}"):
            img, mask = img.to(device), mask.to(device)
            opt1.zero_grad()
            with autocast(): loss = crit1(model_roi(img), mask)
            scaler.scale(loss).backward(); scaler.step(opt1); scaler.update()
    torch.save(model_roi.state_dict(), os.path.join(args.output_path, "roi_model.pth"))

    # --- Stage 2: 完整功能分割训练 ---
    print("\n" + "="*20 + " Stage 2: Segmentation Training " + "="*20)
    train_loader2 = DataLoader(CascadedDataset(args.train_path, False), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader2 = DataLoader(CascadedDataset(args.val_path, False), batch_size=args.val_batch_size, shuffle=False)
    
    model_seg = HybridTemporalNet(base_ch=args.base_ch, depth=args.depth).to(device)
    opt2 = optim.AdamW(model_seg.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode='max', factor=0.5, patience=5)
    crit_bce = nn.BCEWithLogitsLoss()
    crit_dice = smp.losses.DiceLoss(mode='binary', from_logits=True)

    best_val_dice = 0.0
    epochs_no_improve = 0
    
    for epoch in range(1, args.epochs_seg + 1):
        model_seg.train()
        train_loss = 0
        pbar = tqdm(train_loader2, desc=f"Seg Epoch {epoch}/{args.epochs_seg}")
        for img, mask, _ in pbar:
            img, mask = img.to(device), mask.to(device)
            opt2.zero_grad()
            with autocast():
                pred = model_seg(img)
                loss = 0.5 * crit_bce(pred, mask) + 0.5 * crit_dice(pred, mask)
            scaler.scale(loss).backward(); scaler.step(opt2); scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 验证逻辑 (保持和之前一样)
        val_dice, val_iou = evaluate(model_seg, val_loader2, device, desc="Validating")
        print(f"Epoch {epoch} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        scheduler.step(val_dice)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save(model_seg.state_dict(), os.path.join(args.output_path, "best_seg_model.pth"))
            print(">>> 💾 Best Segmentation Model Saved!")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= args.early_stop:
            print(f"Early stopping at epoch {epoch}"); break

    # --- 最终级联测试 ---
    print("\n" + "="*20 + " Final Cascaded Test " + "="*20)
    test_ds = CascadedDataset(args.test_path, False)
    model_seg.load_state_dict(torch.load(os.path.join(args.output_path, "best_seg_model.pth")))
    dice_list = []
    for i in range(len(test_ds)):
        full_9ch, lesion_gt, _ = test_ds[i]
        pred_mask = cascaded_predict(full_9ch, model_roi, model_seg, device)
        pred_bin = (pred_mask > 0.5).astype(np.float32)
        gt_bin = lesion_gt.numpy()[0]
        dice = (2. * (pred_bin * gt_bin).sum()) / (pred_bin.sum() + gt_bin.sum() + 1e-7)
        dice_list.append(dice)
    print(f"\n>>>> FINAL CASCADED TEST DICE: {np.mean(dice_list):.4f} <<<<")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./processed_9ch_vibrant_label/train')
    parser.add_argument('--val_path', type=str, default='./processed_9ch_vibrant_label/val')
    parser.add_argument('--test_path', type=str, default='./processed_9ch_vibrant_label/test')
    parser.add_argument('--output_path', type=str, default='./results_cascade_final')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--epochs_roi', type=int, default=10)
    parser.add_argument('--epochs_seg', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())