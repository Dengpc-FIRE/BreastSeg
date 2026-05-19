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
from torch.amp import autocast, GradScaler
import argparse
from importlib import import_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_config import (  # noqa: E402
    apply_config_to_args,
    build_loss_from_config,
    build_model_from_config,
    load_config,
    resolve_config_path,
)

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

def get_model(model_name):
    module = import_module(f"model.{model_name}")
    SwinHR = getattr(module, "SwinHR")
    model = SwinHR(
        # img_size=(256, 256), 
        in_channels=1,        
        attn_channels=8,      
        out_channels=1,       
        spatial_dims=2        
    )
    return model


def boundary_target(mask, kernel_size=3):
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


def boundary_loss(logits, mask):
    device_type = "cuda" if logits.is_cuda else "cpu"
    with autocast(device_type, enabled=False):
        logits = logits.float()
        mask = mask.float()
        pred_boundary = boundary_target(torch.sigmoid(logits))
        gt_boundary = boundary_target(mask)
        return F.binary_cross_entropy(pred_boundary.clamp(1e-4, 1 - 1e-4), gt_boundary)


def unpack_model_output(output):
    if isinstance(output, dict):
        if "seg_logits" in output:
            return output["seg_logits"], output
        return output["logits"], output
    if isinstance(output, (tuple, list)):
        return output[0], {"extra": output[1:]}
    return output, {}


def approximate_signed_distance(mask, steps=16):
    mask = mask.float()
    inside = mask.clone()
    outside = 1.0 - mask
    dist_in = torch.zeros_like(mask)
    dist_out = torch.zeros_like(mask)

    cur_in = inside
    cur_out = outside
    for step in range(1, steps + 1):
        eroded_in = -F.max_pool2d(-cur_in, kernel_size=3, stride=1, padding=1)
        eroded_out = -F.max_pool2d(-cur_out, kernel_size=3, stride=1, padding=1)
        shell_in = (cur_in - eroded_in).clamp(0, 1)
        shell_out = (cur_out - eroded_out).clamp(0, 1)
        dist_in = torch.where((shell_in > 0) & (dist_in == 0), torch.full_like(dist_in, step), dist_in)
        dist_out = torch.where((shell_out > 0) & (dist_out == 0), torch.full_like(dist_out, step), dist_out)
        cur_in = eroded_in
        cur_out = eroded_out

    dist_in = torch.where((inside > 0) & (dist_in == 0), torch.full_like(dist_in, steps), dist_in)
    dist_out = torch.where((outside > 0) & (dist_out == 0), torch.full_like(dist_out, steps), dist_out)
    return ((dist_in - dist_out) / float(steps)).clamp(-1, 1)


def sdf_loss(output_info, mask):
    if "sdf" not in output_info:
        return mask.new_tensor(0.0)
    sdf_pred = torch.tanh(output_info["sdf"].float())
    with autocast("cuda" if mask.is_cuda else "cpu", enabled=False):
        target = approximate_signed_distance(mask.float(), steps=16)
        return F.smooth_l1_loss(sdf_pred.float(), target)


def hard_negative_prediction_loss(logits, images, mask):
    response = images[:, 1:].detach().abs().amax(dim=1, keepdim=True).float()
    response = (response - response.amin(dim=(-2, -1), keepdim=True)) / (
        response.amax(dim=(-2, -1), keepdim=True) - response.amin(dim=(-2, -1), keepdim=True) + 1e-6
    )
    hard_neg = (response > 0.6).float() * (1.0 - mask.float())
    if hard_neg.sum() < 1:
        return logits.new_tensor(0.0)
    probs = torch.sigmoid(logits.float())
    return (probs * hard_neg).sum() / hard_neg.sum()


def hard_negative_separability_loss(model, images, mask, margin=0.25):
    aux = getattr(model, "aux", None)
    if not aux or "separability_feature" not in aux:
        return images.new_tensor(0.0)

    features = F.normalize(aux["separability_feature"].float(), dim=1)
    response = aux.get("subtraction_response", images[:, 1:].detach().abs().amax(dim=1, keepdim=True)).float()
    mask_small = F.interpolate(mask, size=features.shape[-2:], mode="nearest")
    response_small = F.interpolate(response, size=features.shape[-2:], mode="bilinear", align_corners=False)

    losses = []
    for b in range(features.shape[0]):
        feat = features[b].flatten(1).transpose(0, 1)
        pos = mask_small[b, 0].flatten() > 0.5
        neg = ~pos
        if pos.sum() < 2 or neg.sum() < 2:
            continue

        neg_response = response_small[b, 0].flatten()[neg]
        threshold = torch.quantile(neg_response, 0.75)
        hard_neg = neg.clone()
        hard_neg[neg] = neg_response >= threshold
        if hard_neg.sum() < 2:
            hard_neg = neg

        pos_proto = F.normalize(feat[pos].mean(dim=0, keepdim=True), dim=1)
        pos_sim = (feat[pos] * pos_proto).sum(dim=1).mean()
        hard_sim = (feat[hard_neg] * pos_proto).sum(dim=1).mean()
        losses.append((1.0 - pos_sim) + F.relu(hard_sim + margin))

    if not losses:
        return images.new_tensor(0.0)
    return torch.stack(losses).mean()

# ==========================================
# 3. 评估/测试通用函数
# ==========================================
def evaluate(model, loader, device, desc="Validation"):
    model.eval()
    dice_scores = []
    # 也可以加 IoU
    iou_scores = []
    sensitivity_scores = []
    precision_scores = []
    
    with torch.no_grad():
        for img, mask, _ in tqdm(loader, desc=desc, leave=False):
            img, mask = img.to(device), mask.to(device)
            with autocast('cuda', enabled=device.type == 'cuda'):
                model_output = model(img)
                preds, _ = unpack_model_output(model_output)
            
            # 预测二值化
            preds = (torch.sigmoid(preds) > 0.5).float()
            
            # 计算指标
            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), mask.long(), mode='binary')
            
            # Dice (F1)
            dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
            # IoU
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            sensitivity = (tp.float() / (tp.float() + fn.float() + 1e-7)).mean()
            precision = (tp.float() / (tp.float() + fp.float() + 1e-7)).mean()
            
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
            sensitivity_scores.append(sensitivity.item())
            precision_scores.append(precision.item())
    
    return np.mean(dice_scores), np.mean(iou_scores), np.mean(sensitivity_scores), np.mean(precision_scores)

# ==========================================
# 4. 训练主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Optional YAML config. If omitted, the legacy --model_name path is used.')
    # 【修改点 1】默认路径改为新的 'processed_9ch_vibrant_label'
    parser.add_argument('--train_path', type=str, default='./processed_9ch_vibrant_label/train')
    parser.add_argument('--val_path', type=str, default='./processed_9ch_vibrant_label/val')
    # 【修改点 2】新增测试集路径参数
    parser.add_argument('--test_path', type=str, default='./processed_9ch_vibrant_label/test')
    
    parser.add_argument('--output_path', type=str, default='./results_unet_9ch')
    parser.add_argument('--model_name', type=str, default='swinhr_v9',
                        help='SwinHR model file under model/, e.g. swinhr_v7, swinhr_v8, swinhr_v9.')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4) # Windows下如果不稳定可以改为0
    parser.add_argument('--bce_weight', type=float, default=0.0)
    parser.add_argument('--boundary_weight', type=float, default=0.0)
    parser.add_argument('--separability_weight', type=float, default=0.0)
    parser.add_argument('--hardneg_weight', type=float, default=0.0)
    parser.add_argument('--sdf_weight', type=float, default=0.0)
    parser.add_argument('--disable_kinetic_maps', action='store_true')
    parser.add_argument('--disable_subtraction_guided_fusion', action='store_true')
    parser.add_argument('--disable_boundary_head', action='store_true')
    parser.add_argument('--disable_kinetic_loss', action='store_true')
    parser.add_argument('--disable_pseudo_kinetic_maps', action='store_true')
    parser.add_argument('--disable_pixelwise_phase_attention', action='store_true')
    parser.add_argument('--disable_uncertainty_refinement', action='store_true')
    parser.add_argument('--disable_uncertainty_head', action='store_true')
    parser.add_argument('--disable_attention_smooth_loss', action='store_true')
    args = parser.parse_args()
    config = load_config(resolve_config_path(args.config))
    if config:
        config.setdefault("ablation", {})
        for key in (
            "disable_kinetic_maps",
            "disable_subtraction_guided_fusion",
            "disable_boundary_head",
            "disable_kinetic_loss",
            "disable_pseudo_kinetic_maps",
            "disable_pixelwise_phase_attention",
            "disable_uncertainty_refinement",
            "disable_uncertainty_head",
            "disable_attention_smooth_loss",
        ):
            if getattr(args, key, False):
                config["ablation"][key] = True
        args = apply_config_to_args(args, config)

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
    if config:
        print(f"Initializing configured model from {args.config}...")
        model = build_model_from_config(config).to(device)
    else:
        print("Initializing legacy SwinHR model...")
        model = get_model(args.model_name).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
    configured_loss_fn = build_loss_from_config(config) if config else None
    bce_loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda', enabled=device.type == 'cuda')
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
            with autocast('cuda', enabled=device.type == 'cuda'):
                model_output = model(img)
                preds, output_info = unpack_model_output(model_output)
                if configured_loss_fn is not None:
                    loss = configured_loss_fn(model_output, mask, images=img)
                else:
                    dice_loss = loss_fn(preds, mask)
                    loss = dice_loss
                    if args.bce_weight > 0:
                        loss = loss + args.bce_weight * bce_loss_fn(preds, mask)
                    if args.boundary_weight > 0:
                        loss = loss + args.boundary_weight * boundary_loss(preds, mask)
                    if args.separability_weight > 0:
                        loss = loss + args.separability_weight * hard_negative_separability_loss(model, img, mask)
                    if args.hardneg_weight > 0:
                        loss = loss + args.hardneg_weight * hard_negative_prediction_loss(preds, img, mask)
                    if args.sdf_weight > 0:
                        loss = loss + args.sdf_weight * sdf_loss(output_info, mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        
        # --- 验证 ---
        val_dice, val_iou, val_sens, val_prec = evaluate(model, val_loader, device, desc="Validating")
        print(
            f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | "
            f"Val Sens: {val_sens:.4f} | Val Prec: {val_prec:.4f}"
        )
        
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
        
        test_dice, test_iou, test_sens, test_prec = evaluate(model, test_loader, device, desc="Testing")
        
        print(f"\n>>>> FINAL TEST RESULTS <<<<")
        print(f"Test Set Size: {len(test_ds)}")
        print(f"Test Dice Score: {test_dice:.4f}")
        print(f"Test IoU Score:  {test_iou:.4f}")
        print(f"Test Sensitivity: {test_sens:.4f}")
        print(f"Test Precision:   {test_prec:.4f}")
        print("=======================================")
    else:
        print("Error: Best model file not found!")

if __name__ == '__main__':
    main()

