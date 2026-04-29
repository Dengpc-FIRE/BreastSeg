import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.amp import autocast, GradScaler # 浣跨敤鏈€鏂扮殑 AMP 鎺ュ彛
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from model.swinhr_v7 import SwinHR

# ==========================================
# 1. Dataset (閽堝 9 閫氶亾鏁版嵁鐨勫姞杞介€昏緫)
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
            print(f"[Warning] 璺緞鏈壘鍒? {data_dir}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        file_name = self.ids[i]
        data_path = os.path.join(self.data_dir, file_name)
        data = np.load(data_path) 
        
        # 褰掍竴鍖栧苟杞疆涓?[C, H, W]
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
# 2. 妯″瀷鍒濆鍖?
# ==========================================
def get_model():
    model = SwinHR(
        in_channels=1,        # VIBRANT 瑙ｅ墫娴?
        attn_channels=8,      # SUB 1-8 琛€娴佹祦
        out_channels=1,       
        spatial_dims=2        
    )
    return model

# ==========================================
# 3. 璇勪及/娴嬭瘯鍑芥暟 (鍏抽敭锛氶渶澶勭悊鍙岃緭鍑?
# ==========================================
def evaluate(model, loader, device, desc="Validation"):
    model.eval()
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for img, mask, _ in tqdm(loader, desc=desc, leave=False):
            img, mask = img.to(device), mask.to(device)
            with autocast('cuda'):
                # 璇勪及鏃跺彧鍙栫涓€涓富杈撳嚭锛屽拷鐣ヨ娴佸垎鏀娴?
                preds_main, _ = model(img)
            
            preds = (torch.sigmoid(preds_main) > 0.5).float()
            
            # 璁＄畻鎸囨爣
            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), mask.long(), mode='binary')
            dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
    
    return np.mean(dice_scores), np.mean(iou_scores)

# ==========================================
# 4. 璁粌涓绘祦绋?
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
    
    # 鏁版嵁鍔犺浇
    train_ds = BreastDM9ChDataset(os.path.join(args.train_path, 'data'), os.path.join(args.train_path, 'GT'))
    val_ds = BreastDM9ChDataset(os.path.join(args.val_path, 'data'), os.path.join(args.val_path, 'GT'))
    test_ds = BreastDM9ChDataset(os.path.join(args.test_path, 'data'), os.path.join(args.test_path, 'GT'))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 鍒濆鍖?
    model = get_model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
    scaler = GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_dice = 0.0

    print(f"璁粌寮€濮? {len(train_ds)} 鏍锋湰, 鐩爣 SOTA: 83.2%")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        aux_weight = 0.4 # 杈呭姪鎹熷け鏉冮噸锛屽缓璁湪 0.3-0.5 涔嬮棿
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img, mask, _ in pbar:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                # 1. 鎺ユ敹涓昏緭鍑哄拰杈呭姪杈撳嚭
                preds_main, preds_aux = model(img)
                
                # 2. 璁＄畻鍙岄噸鎹熷け
                loss_main = loss_fn(preds_main, mask)
                loss_aux = loss_fn(preds_aux, mask)
                
                # 3. 鑱斿悎浼樺寲锛氬紩瀵艰娴佸垎鏀浼氭湁鐗╃悊鎰忎箟鐨勭壒寰?
                loss = loss_main + aux_weight * loss_aux
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Main': f"{loss_main.item():.4f}", 'Aux': f"{loss_aux.item():.4f}"})
        
        scheduler.step()
        
        # 楠岃瘉
        val_dice, val_iou = evaluate(model, val_loader, device, desc="Validating")
        print(f"Epoch {epoch} | Loss: {epoch_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.output_path, 'best_model.pth'))
            print(f">>> 馃捑 鍙戠幇鏇翠紭妯″瀷! (Dice: {best_dice:.4f})")

    # 鏈€缁堟祴璇?
    print("\n璁粌缁撴潫锛屽姞杞芥渶浣虫潈閲嶈繘琛屾祴璇?..")
    best_model_path = os.path.join(args.output_path, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        test_dice, test_iou = evaluate(model, test_loader, device, desc="Final Testing")
        print(f"\n>>>> 鏈€缁堟祴璇曠粨鏋?<<<<")
        print(f"Test Dice: {test_dice:.4f}")
        print(f"Test IoU:  {test_iou:.4f}")
        print("==========================")

if __name__ == '__main__':
    main()
