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

# 训练入口同时兼容旧 2D baseline 和新的 DCE 研究模型。
# 不传 --config 时，保持旧的 --model_name / SwinHR 加载方式。
# 传 --config 时，模型、损失、数据集都由 YAML 配置选择。
# SG-KTFNet、KPTA-Net、KPR-Net、SA-KPTA-Net 都通过该配置式入口接入。
# 这样新增研究模型不会破坏原始 baseline 训练流程。
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
    """旧版 2D BreastDM 数据集，同时带 2.5D 防御性 fallback。

    旧数据通常保存为 [H,W,C]，返回给模型时转成 [C,H,W]。
    方案 D 的数据保存为 [K,T,H,W]。
    如果 2.5D 数据意外进入该旧 Dataset，也会直接返回原张量，避免 transpose 报错。
    """

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
        """返回当前 split 中的切片样本数量。"""
        return len(self.ids)

    def __getitem__(self, i):
        """读取一个 image/mask 样本。

        2D 数据在这里 /255，因为它只是普通多通道输入。
        2.5D 数据保留原始强度，因为 SA-KPTA-Net 必须先构造 pseudo-kinetic maps。
        """
        # 当前样本文件名。
        file_name = self.ids[i]
        # 拼出 .npy 路径。
        data_path = os.path.join(self.data_dir, file_name)
        # 读取 numpy 数据。
        data = np.load(data_path) 
        
        # 你的 NPY 是 (256, 256, 9) -> 转置为 (9, 256, 256) 给 PyTorch
        if data.ndim == 4:
            # 4D 表示 2.5D 数据：[K,T,H,W]，不能做 HWC->CHW 转置。
            image = torch.from_numpy(data.astype(np.float32))  # [K, T, H, W]
        elif data.ndim == 3:
            # 3D 表示旧 2D 数据：[H,W,C]。
            image = data.astype(np.float32) / 255.0
            # 转成 PyTorch 常用格式：[C,H,W]。
            image = torch.from_numpy(image.transpose(2, 0, 1)) # [C, H, W]
        else:
            # 其他维度说明预处理数据格式不对，直接报错。
            raise ValueError(f"Unsupported .npy shape {data.shape} for {file_name}")

        # mask 文件名与 data 同名，只是扩展名从 .npy 换成 .png。
        mask_name = file_name.replace('.npy', '.png')
        # 拼出 GT mask 路径。
        gt_path = os.path.join(self.gt_dir, mask_name)
        
        if os.path.exists(gt_path):
            # 读取灰度 mask。
            mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[0] != self.img_size:
                    # mask 使用最近邻 resize，避免产生中间类别值。
                    mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            else:
                # 读取失败时用全 0 mask 防止崩溃。
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            # GT 文件缺失时用全 0 mask。
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
        # mask 归一化到 [0,1]。
        mask = mask.astype(np.float32) / 255.0
        # 二值化 mask。
        mask[mask > 0.5] = 1.0
        # 增加 channel 维度：[1,H,W]。
        mask_tensor = torch.from_numpy(np.expand_dims(mask, axis=0))

        return image, mask_tensor, file_name


class BreastDM25DDataset(Dataset):
    """processed_25d_dce 专用 Dataset。

    每个样本：
        image: [K,T,H,W]，K 是相邻切片，T 是 DCE phase。
        mask : [1,H,W]，只监督中心切片。

    该 Dataset 对应 SA-KPTA-Net / KPTA-2.5DNet。
    """

    def __init__(self, data_dir, gt_dir, img_size=256):
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        if os.path.exists(data_dir):
            self.ids = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        else:
            self.ids = []
            print(f"[Warning] Directory not found: {data_dir}")

    def __len__(self):
        """返回中心切片样本数量。"""
        return len(self.ids)

    def __getitem__(self, i):
        """读取一个 2.5D 样本，不在 Dataset 内做强度归一化。"""
        # 文件名。
        file_name = self.ids[i]
        # 读取 [K,T,H,W]。
        data = np.load(os.path.join(self.data_dir, file_name))
        if data.ndim != 4:
            # 方案 D 必须是 4D 输入。
            raise ValueError(f"Expect 2.5D .npy with shape [K,T,H,W], got {data.shape} for {file_name}")
        # 保留原始强度，模型内部先构造 kinetic maps 再归一化。
        image = torch.from_numpy(data.astype(np.float32))

        # mask 文件与 data 同名。
        mask_name = file_name.replace('.npy', '.png')
        gt_path = os.path.join(self.gt_dir, mask_name)
        if os.path.exists(gt_path):
            # 读取中心切片 GT。
            mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # 读取失败则使用全 0 mask。
                mask = np.zeros((data.shape[-2], data.shape[-1]), dtype=np.uint8)
        else:
            # 文件缺失则使用全 0 mask。
            mask = np.zeros((data.shape[-2], data.shape[-1]), dtype=np.uint8)
        if mask.shape != (data.shape[-2], data.shape[-1]):
            # mask 尺寸不一致时，用最近邻调整到图像尺寸。
            mask = cv2.resize(mask, (data.shape[-1], data.shape[-2]), interpolation=cv2.INTER_NEAREST)
        # mask 归一化到 [0,1]。
        mask = mask.astype(np.float32) / 255.0
        # 二值化。
        mask[mask > 0.5] = 1.0
        # 返回 image、mask、文件名。
        return image, torch.from_numpy(np.expand_dims(mask, axis=0)).float(), file_name


def build_dataset(split_path, dataset_type="breastdm_2d", img_size=256):
    """根据配置和样本 shape 自动选择 2D 或 2.5D Dataset。

    优先使用 YAML 的 dataset.type。
    同时检查第一个 .npy 的 shape。
    如果发现是 4D，即 [K,T,H,W]，则自动使用 BreastDM25DDataset。
    """
    # data 子目录。
    data_dir = os.path.join(split_path, 'data')
    # GT 子目录。
    gt_dir = os.path.join(split_path, 'GT')
    # sample_shape 用于打印和自动识别数据格式。
    sample_shape = None
    if os.path.exists(data_dir):
        # 找到该 split 下的 .npy 样本。
        sample_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        if sample_files:
            # mmap_mode='r' 只读取 shape，不把大数组完整加载进内存。
            sample_shape = np.load(os.path.join(data_dir, sample_files[0]), mmap_mode='r').shape
    if dataset_type in {"breastdm_25d", "25d", "kpta_25d"} or (sample_shape is not None and len(sample_shape) == 4):
        # 2.5D 数据路径。
        print(f"[Dataset] BreastDM25DDataset split={split_path}, sample_shape={sample_shape}")
        return BreastDM25DDataset(data_dir, gt_dir, img_size=img_size)
    # 默认使用旧 2D Dataset。
    print(f"[Dataset] BreastDM9ChDataset split={split_path}, sample_shape={sample_shape}")
    return BreastDM9ChDataset(data_dir, gt_dir, img_size=img_size)

# ==========================================
# 2. 模型：标准 U-Net 但改为 9 通道输入
# ==========================================

def get_model(model_name):
    """根据 model/{model_name}.py 初始化旧版 baseline 模型。

    新研究模型通过 YAML 和 build_model_from_config 初始化。
    保留该函数是为了让原始 baseline 命令仍然可运行。
    """
    # 动态导入 model 文件。
    module = import_module(f"model.{model_name}")
    # 旧模型文件通常暴露 SwinHR 类。
    SwinHR = getattr(module, "SwinHR")
    # 保持原始构造参数，避免破坏旧 baseline。
    model = SwinHR(
        # img_size=(256, 256), 
        in_channels=1,        
        attn_channels=8,      
        out_channels=1,       
        spatial_dims=2        
    )
    return model


def boundary_target(mask, kernel_size=3):
    """只用 max_pool2d 构造 2D 形态学边界。"""
    # padding 保证输出尺寸不变。
    pad = kernel_size // 2
    # dilation：局部最大池化。
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    # erosion：对负 mask 做最大池化再取负。
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    # boundary = dilation - erosion。
    return (dilated - eroded).clamp(0, 1)


def boundary_loss(logits, mask):
    """旧实验使用的边界辅助损失。"""
    # 根据 logits 所在设备选择 autocast 类型。
    device_type = "cuda" if logits.is_cuda else "cpu"
    with autocast(device_type, enabled=False):
        # 边界 loss 用 float32 计算更稳定。
        logits = logits.float()
        mask = mask.float()
        # 从预测概率图中构造预测边界。
        pred_boundary = boundary_target(torch.sigmoid(logits))
        # 从 GT mask 中构造真实边界。
        gt_boundary = boundary_target(mask)
        # BCE 输入需要避免 0/1 极值。
        return F.binary_cross_entropy(pred_boundary.clamp(1e-4, 1 - 1e-4), gt_boundary)


def unpack_model_output(output):
    """从 tensor/tuple/dict 输出中提取分割 logits。"""
    if isinstance(output, dict):
        # 新模型优先使用 seg_logits。
        if "seg_logits" in output:
            return output["seg_logits"], output
        # 兼容 logits 字段。
        return output["logits"], output
    if isinstance(output, (tuple, list)):
        # tuple/list 默认第一个元素是 logits。
        return output[0], {"extra": output[1:]}
    # 普通 tensor 直接返回。
    return output, {}


def approximate_signed_distance(mask, steps=16):
    """用迭代 2D erosion 近似 signed distance map。"""
    # mask 内部为正区域。
    mask = mask.float()
    inside = mask.clone()
    # mask 外部为负区域。
    outside = 1.0 - mask
    # dist_in 存储前景内部到边界的近似距离。
    dist_in = torch.zeros_like(mask)
    # dist_out 存储背景到边界的近似距离。
    dist_out = torch.zeros_like(mask)

    # 当前待腐蚀的前景区域。
    cur_in = inside
    # 当前待腐蚀的背景区域。
    cur_out = outside
    for step in range(1, steps + 1):
        # 前景 erosion。
        eroded_in = -F.max_pool2d(-cur_in, kernel_size=3, stride=1, padding=1)
        # 背景 erosion。
        eroded_out = -F.max_pool2d(-cur_out, kernel_size=3, stride=1, padding=1)
        # 当前 step 被剥离出来的前景 shell。
        shell_in = (cur_in - eroded_in).clamp(0, 1)
        # 当前 step 被剥离出来的背景 shell。
        shell_out = (cur_out - eroded_out).clamp(0, 1)
        # 只给第一次出现的 shell 写入距离。
        dist_in = torch.where((shell_in > 0) & (dist_in == 0), torch.full_like(dist_in, step), dist_in)
        dist_out = torch.where((shell_out > 0) & (dist_out == 0), torch.full_like(dist_out, step), dist_out)
        # 更新 erosion 后区域。
        cur_in = eroded_in
        cur_out = eroded_out

    # 对未被完全腐蚀到的区域赋最大距离。
    dist_in = torch.where((inside > 0) & (dist_in == 0), torch.full_like(dist_in, steps), dist_in)
    dist_out = torch.where((outside > 0) & (dist_out == 0), torch.full_like(dist_out, steps), dist_out)
    # 内部为正，外部为负，并归一化到 [-1,1]。
    return ((dist_in - dist_out) / float(steps)).clamp(-1, 1)


def sdf_loss(output_info, mask):
    """如果模型输出 sdf，则计算可选 SDF 辅助损失。"""
    if "sdf" not in output_info:
        # 没有 sdf 输出时跳过。
        return mask.new_tensor(0.0)
    # tanh 把预测 SDF 限制到 [-1,1]。
    sdf_pred = torch.tanh(output_info["sdf"].float())
    with autocast("cuda" if mask.is_cuda else "cpu", enabled=False):
        target = approximate_signed_distance(mask.float(), steps=16)
        return F.smooth_l1_loss(sdf_pred.float(), target)


def hard_negative_prediction_loss(logits, images, mask):
    """惩罚高增强背景区域上的假阳性预测。"""
    # 使用非第 0 通道的最大响应近似增强强度。
    response = images[:, 1:].detach().abs().amax(dim=1, keepdim=True).float()
    # 将响应归一化到 [0,1]。
    response = (response - response.amin(dim=(-2, -1), keepdim=True)) / (
        response.amax(dim=(-2, -1), keepdim=True) - response.amin(dim=(-2, -1), keepdim=True) + 1e-6
    )
    # 高响应且非肿瘤区域作为 hard negative。
    hard_neg = (response > 0.6).float() * (1.0 - mask.float())
    if hard_neg.sum() < 1:
        # 没有 hard negative 时跳过。
        return logits.new_tensor(0.0)
    # 预测概率。
    probs = torch.sigmoid(logits.float())
    # 惩罚 hard negative 区域内的预测概率。
    return (probs * hard_neg).sum() / hard_neg.sum()


def hard_negative_separability_loss(model, images, mask, margin=0.25):
    """用于增强假阳性区域的特征级分离损失。"""
    # 旧模型可把中间特征放在 model.aux 中。
    aux = getattr(model, "aux", None)
    if not aux or "separability_feature" not in aux:
        # 没有对应特征时跳过。
        return images.new_tensor(0.0)

    # 对特征做 L2 normalize，方便用 cosine similarity。
    features = F.normalize(aux["separability_feature"].float(), dim=1)
    # 获取增强响应图，若模型没提供则从输入通道估计。
    response = aux.get("subtraction_response", images[:, 1:].detach().abs().amax(dim=1, keepdim=True)).float()
    # 将 mask resize 到特征图大小。
    mask_small = F.interpolate(mask, size=features.shape[-2:], mode="nearest")
    # 将响应图 resize 到特征图大小。
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
    """只使用最终分割 logits 做评估，保证不同模型之间公平比较。"""
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
    """主训练流程：解析参数、构建数据/模型/损失、训练、验证和测试。"""
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
    parser.add_argument('--epochs', type=int, default=2)
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
    parser.add_argument('--disable_phase_dropout', action='store_true')
    parser.add_argument('--disable_kinetic_prior_encoder', action='store_true')
    parser.add_argument('--disable_kinetic_fusion', action='store_true')
    parser.add_argument('--disable_temporal_contrastive_loss', action='store_true')
    parser.add_argument('--disable_slice_context', action='store_true')
    parser.add_argument('--disable_transformer_bottleneck', action='store_true')
    args = parser.parse_args()
    config = load_config(resolve_config_path(args.config))
    if config:
        # 命令行消融开关优先级高于 YAML，便于不改配置文件直接跑消融。
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
            "disable_phase_dropout",
            "disable_kinetic_prior_encoder",
            "disable_kinetic_fusion",
            "disable_temporal_contrastive_loss",
            "disable_slice_context",
            "disable_transformer_bottleneck",
        ):
            if getattr(args, key, False):
                config["ablation"][key] = True
        args = apply_config_to_args(args, config)

    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 准备数据 ---
    print("Loading Datasets...")
    # 对 SA-KPTA-Net，这里应选择 BreastDM25DDataset，输出 [B,K,T,H,W]。
    dataset_cfg = config.get("dataset", {}) if config else {}
    # 读取数据集类型。
    dataset_type = dataset_cfg.get("type", "breastdm_2d")
    # 读取图像尺寸。
    img_size = int(dataset_cfg.get("img_size", 256))
    # 构建训练集。
    train_ds = build_dataset(args.train_path, dataset_type=dataset_type, img_size=img_size)
    # 构建验证集。
    val_ds = build_dataset(args.val_path, dataset_type=dataset_type, img_size=img_size)
    # 构建测试集。
    test_ds = build_dataset(args.test_path, dataset_type=dataset_type, img_size=img_size)
    
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
        # 配置式模型根据 model.name 构建，可返回包含辅助分支的 dict。
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
    best_test_records = []

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
                # 新模型可能返回 dict；旧模型返回 tensor。
                # unpack_model_output 始终取出最终分割 logits。
                model_output = model(img)
                preds, output_info = unpack_model_output(model_output)
                if configured_loss_fn is not None:
                    # 配置式损失可读取 boundary/uncertainty/attention 等辅助输出。
                    loss = configured_loss_fn(model_output, mask, images=img)
                else:
                    # 旧 baseline 的 fallback loss 路径。
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

            if not torch.isfinite(loss):
                # 遇到 NaN/Inf 直接报错，避免误显示 Train Loss: 0.0000。
                # 同时打印输入、mask、预测范围，方便定位数值问题。
                with torch.no_grad():
                    pred_min = preds.detach().float().amin().item()
                    pred_max = preds.detach().float().amax().item()
                    mask_sum = mask.detach().float().sum().item()
                    img_min = img.detach().float().amin().item()
                    img_max = img.detach().float().amax().item()
                raise FloatingPointError(
                    "Non-finite training loss detected. "
                    f"loss={loss.detach().float().item()}, "
                    f"pred_range=({pred_min:.4g},{pred_max:.4g}), "
                    f"img_range=({img_min:.4g},{img_max:.4g}), "
                    f"mask_sum={mask_sum:.4g}"
                )
            
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
            # 验证集 Dice 刷新 best，说明当前权重是新的 best-on-val 模型。
            best_dice = val_dice
            # 保存 best-on-val 权重；最终测试会重新加载这个文件，而不是用最后一轮权重。
            torch.save(model.state_dict(), os.path.join(args.output_path, 'best_model.pth'))
            print(f">>> 💾 Best Model Saved! (Dice: {best_dice:.4f})")
            # 立刻在测试集上做一次无梯度评估；evaluate 内部使用 torch.no_grad() 和 model.eval()。
            # 这里只读测试指标，不 backward、不 optimizer.step，因此不会污染训练。
            test_dice, test_iou, test_sens, test_prec = evaluate(model, test_loader, device, desc="Testing Best")
            # 记录每一次 best-on-val 对应的测试集结果，方便训练结束后汇总。
            best_test_records.append(
                {
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "test_dice": test_dice,
                    "test_iou": test_iou,
                    "test_sens": test_sens,
                    "test_prec": test_prec,
                }
            )
            # 按要求用 [test_dice] 标识测试结果，便于从日志中 grep。
            print(
                f"[test_dice] Epoch {epoch} | Val Dice: {val_dice:.4f} | "
                f"Test Dice: {test_dice:.4f} | Test IoU: {test_iou:.4f} | "
                f"Test Sens: {test_sens:.4f} | Test Prec: {test_prec:.4f}"
            )

    # ==========================================
    # 【修改点 4】训练结束后，加载最佳模型进行测试
    # ==========================================
    print("\n=======================================")
    print("Training Finished. Starting Testing...")
    print("=======================================")
    
    best_model_path = os.path.join(args.output_path, 'best_model.pth')
    if os.path.exists(best_model_path):
        # 最终测试前重新加载验证集最优模型，明确不使用最后一轮模型。
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded Best Model from {best_model_path}")
        
        test_dice, test_iou, test_sens, test_prec = evaluate(model, test_loader, device, desc="Testing")
        
        print(f"\n>>>> FINAL TEST RESULTS <<<<")
        print(f"Test Set Size: {len(test_ds)}")
        print(f"Test Dice Score: {test_dice:.4f}")
        print(f"Test IoU Score:  {test_iou:.4f}")
        print(f"Test Sensitivity: {test_sens:.4f}")
        print(f"Test Precision:   {test_prec:.4f}")
        if best_test_records:
            # 汇总训练过程中每次刷新验证集 best 时的测试集结果。
            print("\n>>>> BEST-ON-VAL TEST HISTORY <<<<")
            for record in best_test_records:
                print(
                    f"[test_dice] Epoch {record['epoch']} | "
                    f"Val Dice: {record['val_dice']:.4f} | "
                    f"Test Dice: {record['test_dice']:.4f} | "
                    f"Test IoU: {record['test_iou']:.4f} | "
                    f"Test Sens: {record['test_sens']:.4f} | "
                    f"Test Prec: {record['test_prec']:.4f}"
                )
        print("=======================================")
    else:
        print("Error: Best model file not found!")

if __name__ == '__main__':
    main()

