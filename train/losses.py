"""多种 DCE-MRI 分割模型共用的损失函数。

本文件服务于 SG-KTFNet、KPTA-Net、KPR-Net 和 SA-KPTA-Net。
训练脚本既要兼容旧模型的 tensor 输出，也要兼容新模型的 dict 输出。
dict 输出可以包含：
    seg_logits：最终分割 logits。
    boundary_logits：边界辅助分支输出。
    uncertainty_logits：不确定性辅助分支输出。
    kinetic_maps：增强动力学先验图。
    attention_maps：像素级 phase attention map。
    contrastive_embeddings：时序对比学习特征。

方案 D / SA-KPTA-Net 的主要损失：
    L = L_seg
        + lambda_boundary * L_boundary
        + lambda_uncertainty * L_uncertainty
        + lambda_attention_smooth * L_attention_smooth

其中：
    L_seg：Dice + BCE，用于中心切片肿瘤分割。
    L_boundary：边界监督，用于提高轮廓质量。
    L_uncertainty：让边界/错误区域具有更高不确定性。
    L_attention_smooth：约束 phase attention 空间上更平滑。
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dce_kinetic_utils import boundary_target_2d


def unpack_model_output(output) -> Tuple[torch.Tensor, Dict]:
    """从模型输出中取出 segmentation logits 和辅助信息字典。

    旧模型通常直接返回 tensor。
    新模型通常返回 dict，例如 {"seg_logits": ..., "boundary_logits": ...}。
    该函数让训练和验证代码不用关心具体模型输出格式。
    """
    if isinstance(output, dict):
        # 新模型优先使用 seg_logits 作为最终分割输出。
        if "seg_logits" in output:
            return output["seg_logits"], output
        # 部分旧/兼容模型可能使用 logits 字段。
        if "logits" in output:
            return output["logits"], output
        raise KeyError("Dictionary model output must contain 'seg_logits' or 'logits'.")
    # tuple/list 输出时默认第一个元素是 logits，其余作为 extra 信息保留。
    if isinstance(output, (tuple, list)):
        return output[0], {"extra": output[1:]}
    # tensor 输出时直接作为 logits。
    return output, {}


class DiceBCELoss(nn.Module):
    """二分类分割常用的 Dice + BCE 损失。

    Dice 主要解决肿瘤区域小、前景背景极不平衡的问题。
    BCE 提供逐像素监督，有助于训练初期稳定优化。
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        # BCE 权重；为 0 时退化成纯 Dice。
        self.bce_weight = bce_weight
        # smooth 防止空 mask 或小区域时除零。
        self.smooth = smooth
        # PyTorch 原生 BCEWithLogitsLoss，输入必须是 logits。
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """根据 raw logits 和二值 mask 计算 Dice + 可选 BCE。"""
        # 显式转 float，避免 AMP 或整型 mask 引发类型问题。
        logits = logits.float()
        target = target.float()
        # Dice 需要概率图，因此先 sigmoid。
        probs = torch.sigmoid(logits)
        # 对 C/H/W 求和；这里 C=1。
        dims = (1, 2, 3)
        # soft intersection。
        intersection = (probs * target).sum(dim=dims)
        # Dice 分母。
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        # 计算 batch 平均 Dice loss。
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (denom + self.smooth)).mean()
        if self.bce_weight <= 0:
            # 不使用 BCE 时直接返回 Dice。
            return dice
        # 总损失 = Dice + bce_weight * BCE。
        return dice + self.bce_weight * self.bce(logits, target)


class KineticConsistencyLoss(nn.Module):
    """增强动力学一致性损失。

    医学先验：肿瘤区域通常比周围组织具有更明显的 DCE 增强响应。
    该损失比较肿瘤内部 enhancement 与 peritumoral ring 的 enhancement。
    它是轻量约束，不应过强，因为血管也可能强增强。
    """

    def __init__(
        self,
        margin: float = 0.05,
        ring_size: int = 5,
        eps: float = 1e-6,
        use_predicted_mask: bool = False,
    ) -> None:
        super().__init__()
        # margin 表示希望肿瘤增强比周围至少高多少。
        self.margin = margin
        # ring_size 控制肿瘤外扩环区域大小。
        self.ring_size = ring_size
        # eps 防止除零。
        self.eps = eps
        # 默认用 GT mask 更稳定；可选用预测 mask。
        self.use_predicted_mask = use_predicted_mask

    def forward(
        self,
        seg_logits: torch.Tensor,
        kinetic_maps: Optional[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """计算肿瘤内部增强和周围环状区域增强之间的 margin loss。"""
        if kinetic_maps is None or kinetic_maps.numel() == 0:
            # 没有 kinetic map 时跳过该损失。
            return seg_logits.new_tensor(0.0)

        if self.use_predicted_mask:
            # 可选：用预测 mask 构造 tumor 区域。
            tumor = (torch.sigmoid(seg_logits.detach()) > 0.5).float()
        else:
            # 默认：用 GT mask，训练更稳定。
            tumor = target.float().clamp(0, 1)

        if tumor.sum() < 1:
            # 空肿瘤样本跳过 kinetic consistency。
            return seg_logits.new_tensor(0.0)

        # max_pool2d 实现 dilation，再减去原 tumor 得到 peritumoral ring。
        ring = (F.max_pool2d(tumor, kernel_size=self._kernel_size(), stride=1, padding=self._kernel_size() // 2) - tumor).clamp(0, 1)
        if ring.sum() < 1:
            # 如果 ring 为空，则跳过。
            return seg_logits.new_tensor(0.0)

        # 取绝对增强强度，兼容正负 subtraction。
        maps = kinetic_maps.float().abs()
        # 肿瘤区域像素数。
        tumor_sum = tumor.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        # 周围 ring 像素数。
        ring_sum = ring.sum(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        # 肿瘤内部平均增强。
        e_in = (maps * tumor).sum(dim=(2, 3), keepdim=True) / tumor_sum
        # 周围 ring 平均增强。
        e_out = (maps * ring).sum(dim=(2, 3), keepdim=True) / ring_sum
        # 若 e_in 没有比 e_out 高 margin，则产生惩罚。
        loss = F.relu(self.margin - e_in + e_out)
        # kinetic loss 可以安全转零，因为它本身是可选辅助项。
        return torch.nan_to_num(loss.mean(), nan=0.0, posinf=0.0, neginf=0.0)

    def _kernel_size(self) -> int:
        """把配置中的 ring size 转成 odd kernel size。"""
        kernel = max(3, int(self.ring_size))
        if kernel % 2 == 0:
            kernel += 1
        return kernel


class SGKTFNetLoss(nn.Module):
    """SG-KTFNet 的组合损失。

    包含：
        主分割损失。
        可选边界辅助损失。
        可选动力学一致性损失。

    消融开关会把对应 lambda 置 0，训练脚本不需要改。
    """

    def __init__(
        self,
        lambda_boundary: float = 0.2,
        lambda_kinetic: float = 0.1,
        kinetic_margin: float = 0.05,
        boundary_thickness: int = 3,
        peritumor_ring_size: int = 5,
        kinetic_eps: float = 1e-6,
        bce_weight: float = 0.5,
        ablation: Optional[Dict] = None,
        **_: Dict,
    ) -> None:
        super().__init__()
        # ablation 为空时使用默认配置。
        ablation = ablation or {}
        # 主分割损失。
        self.seg_loss = DiceBCELoss(bce_weight=bce_weight)
        # 边界分支被关闭时，边界损失权重置零。
        self.lambda_boundary = 0.0 if ablation.get("disable_boundary_head", False) else lambda_boundary
        # kinetic loss 被关闭时，动力学损失权重置零。
        self.lambda_kinetic = 0.0 if ablation.get("disable_kinetic_loss", False) else lambda_kinetic
        # 生成边界 target 的厚度。
        self.boundary_thickness = boundary_thickness
        # 边界分支使用 BCEWithLogitsLoss。
        self.boundary_bce = nn.BCEWithLogitsLoss()
        # 动力学一致性损失模块。
        self.kinetic_loss = KineticConsistencyLoss(
            margin=kinetic_margin,
            ring_size=peritumor_ring_size,
            eps=kinetic_eps,
        )

    def forward(self, output, target: torch.Tensor, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """从 tensor 或 dict 输出计算 SG-KTFNet loss。"""
        # 取出最终分割 logits 和辅助输出字典。
        seg_logits, info = unpack_model_output(output)
        # 先计算主分割损失。
        loss = self.seg_loss(seg_logits, target)

        if self.lambda_boundary > 0 and "boundary_logits" in info:
            # 读取边界分支 logits。
            boundary_logits = info["boundary_logits"]
            # 使用 2D morphological gradient 生成边界标签。
            boundary_target = boundary_target_2d(target, thickness=self.boundary_thickness)
            # 加入边界 BCE 损失。
            loss = loss + self.lambda_boundary * self.boundary_bce(boundary_logits.float(), boundary_target.float())

        if self.lambda_kinetic > 0:
            # 加入肿瘤内部与周围区域的增强一致性损失。
            loss = loss + self.lambda_kinetic * self.kinetic_loss(
                seg_logits,
                info.get("kinetic_maps"),
                target,
            )
        return loss


def attention_smoothness_loss(attention_maps) -> torch.Tensor:
    """phase attention map 的总变分平滑损失。

    直觉：同一组织区域的相邻像素通常应依赖相似的 DCE phase。
    该损失抑制 attention map 高频噪声，同时保留像素级自适应能力。
    """
    if not attention_maps:
        # 没有 attention map 时返回 CPU 零张量，调用方会避免使用。
        return torch.tensor(0.0)
    losses = []
    for attention in attention_maps:
        if attention.shape[1] <= 1:
            # 单 phase 没有时相选择意义，跳过。
            continue
        # y 方向相邻像素差异。
        dy = (attention[..., 1:, :] - attention[..., :-1, :]).abs().mean()
        # x 方向相邻像素差异。
        dx = (attention[..., :, 1:] - attention[..., :, :-1]).abs().mean()
        # 累计该 attention map 的 TV loss。
        losses.append(dy + dx)
    if not losses:
        # 若所有 map 都被跳过，则返回和 attention 同设备的零。
        ref = attention_maps[0]
        return ref.new_tensor(0.0)
    # 多个 attention map 时求平均。
    return torch.nan_to_num(torch.stack(losses).mean(), nan=0.0, posinf=0.0, neginf=0.0)


class KPTANetLoss(nn.Module):
    """KPTA-Net 和 SA-KPTA-Net 共用的组合损失。

    监督内容：
        segmentation logits：Dice+BCE。
        boundary logits：2D 边界 target。
        uncertainty logits：边界 target + 分割误差 target。
        phase attention maps：空间平滑约束。

    方案 D 复用该损失，因为它和方案 B 拥有同样的辅助输出头。
    """

    def __init__(
        self,
        lambda_boundary: float = 0.2,
        lambda_uncertainty: float = 0.1,
        lambda_attention_smooth: float = 0.01,
        boundary_thickness: int = 3,
        bce_weight: float = 0.5,
        ablation: Optional[Dict] = None,
        **_: Dict,
    ) -> None:
        super().__init__()
        # 读取消融配置。
        ablation = ablation or {}
        # 主分割损失。
        self.seg_loss = DiceBCELoss(bce_weight=bce_weight)
        # boundary head 关闭时，边界损失不参与训练。
        self.lambda_boundary = 0.0 if ablation.get("disable_boundary_head", False) else lambda_boundary
        # uncertainty head 关闭时，不确定性损失不参与训练。
        self.lambda_uncertainty = 0.0 if ablation.get("disable_uncertainty_head", False) else lambda_uncertainty
        # attention smoothness 关闭时，TV loss 不参与训练。
        self.lambda_attention_smooth = (
            0.0 if ablation.get("disable_attention_smooth_loss", False) else lambda_attention_smooth
        )
        # 边界 target 厚度。
        self.boundary_thickness = boundary_thickness
        # 边界和不确定性都使用 BCEWithLogitsLoss。
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, output, target: torch.Tensor, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算 KPTA 风格的多头分割损失。"""
        # 解包最终分割 logits 和辅助输出。
        seg_logits, info = unpack_model_output(output)
        # mask 转 float。
        target = target.float()
        # 主分割损失。
        loss = self.seg_loss(seg_logits, target)
        # 由 GT mask 生成边界监督 target。
        boundary_target = boundary_target_2d(target, thickness=self.boundary_thickness)

        if self.lambda_boundary > 0 and "boundary_logits" in info:
            # 边界辅助分支监督肿瘤轮廓。
            loss = loss + self.lambda_boundary * self.bce(info["boundary_logits"].float(), boundary_target)

        if self.lambda_uncertainty > 0 and "uncertainty_logits" in info:
            with torch.no_grad():
                # 分割误差越大，说明该像素越不确定。
                error = (torch.sigmoid(seg_logits.detach()) - target).abs()
                # 不确定性 target = 边界区域 + 当前分割错误区域。
                uncertainty_target = (boundary_target + error).clamp(0, 1)
            # 用 BCE 监督 uncertainty_logits。
            loss = loss + self.lambda_uncertainty * self.bce(info["uncertainty_logits"].float(), uncertainty_target)

        if self.lambda_attention_smooth > 0:
            # 从输出字典读取 phase attention maps。
            attention_maps = info.get("attention_maps", [])
            # 没有 attention map 时平滑损失为 0。
            smooth = attention_smoothness_loss(attention_maps) if attention_maps else seg_logits.new_tensor(0.0)
            # 加入 attention TV 平滑项。
            loss = loss + self.lambda_attention_smooth * smooth

        return loss


class KPTA25DNetLoss(KPTANetLoss):
    """SA-KPTA-Net 的损失。

    方案 D 在模型结构上加入 2.5D slice-aware encoder，
    但输出头仍然是 segmentation/boundary/uncertainty/attention，
    因此直接继承 KPTANetLoss。
    """

    pass


class TemporalContrastiveLoss(nn.Module):
    """KPR-Net 使用的 InfoNCE 风格时序对比损失。

    正样本：同一样本、同一肿瘤在不同 DCE phase 的特征。
    负样本：背景特征，或者 batch 中其他样本的肿瘤特征。
    若肿瘤像素太少或可用 phase 太少，则安全跳过。
    """

    def __init__(self, temperature: float = 0.1, min_tumor_pixels: int = 20) -> None:
        super().__init__()
        # temperature 控制 softmax 分布锐度。
        self.temperature = temperature
        # 肿瘤像素少于该阈值时跳过，避免噪声原型。
        self.min_tumor_pixels = min_tumor_pixels

    def forward(self, embeddings, target: torch.Tensor) -> torch.Tensor:
        """根据 phase feature embeddings 计算时序对比损失。"""
        if not embeddings or "phase_features" not in embeddings:
            # 没有 embeddings 时跳过。
            return target.new_tensor(0.0)
        # phase_features: [B,T,C,H,W]。
        phase_features = embeddings["phase_features"].float()
        # phase_mask 表示哪些 phase 可用。
        phase_mask = embeddings.get("phase_mask")
        if phase_features.ndim != 5 or phase_features.shape[1] <= 1:
            # 只有一个 phase 时没有时序对比意义。
            return phase_features.new_tensor(0.0)
        if phase_mask is None:
            # 未提供 phase_mask 时，默认所有 phase 可用。
            phase_mask = phase_features.new_ones((phase_features.shape[0], phase_features.shape[1]))
        phase_mask = phase_mask.to(device=phase_features.device, dtype=phase_features.dtype)
        # 把 GT mask resize 到 feature map 尺寸。
        target_small = F.interpolate(target.float(), size=phase_features.shape[-2:], mode="nearest")

        # 存放 tumor/background pooled embeddings。
        tumor_embeds = []
        bg_embeds = []
        sample_ids = []
        for b in range(phase_features.shape[0]):
            # 当前样本的 tumor mask。
            tumor = target_small[b : b + 1]
            # 当前样本背景 mask。
            bg = 1.0 - tumor
            if tumor.sum() < self.min_tumor_pixels or bg.sum() < self.min_tumor_pixels:
                # 前景或背景太少时跳过该样本。
                continue
            # 找出当前样本可用 phase。
            available = torch.nonzero(phase_mask[b] > 0.5, as_tuple=False).flatten()
            if available.numel() <= 1:
                # 可用 phase 少于 2，无法构造跨 phase 正样本。
                continue
            for t in available.tolist():
                # 当前 phase 的特征。
                feat = phase_features[b, t]
                # 池化 tumor prototype。
                tumor_embeds.append(self._masked_pool(feat, tumor[0]))
                # 池化 background prototype。
                bg_embeds.append(self._masked_pool(feat, bg[0]))
                # 记录样本 id，用于判断正负样本关系。
                sample_ids.append(b)

        if len(tumor_embeds) < 2 or len(bg_embeds) < 1:
            # 没有足够 embedding 时跳过。
            return phase_features.new_tensor(0.0)

        # L2 normalize 后使用余弦相似度。
        tumor_embeds = F.normalize(torch.stack(tumor_embeds), dim=1)
        bg_embeds = F.normalize(torch.stack(bg_embeds), dim=1)
        # sample id 张量用于构建正样本 mask。
        sample_ids_t = torch.tensor(sample_ids, device=phase_features.device)
        losses = []
        for idx in range(tumor_embeds.shape[0]):
            # 同一样本的其他 phase tumor embedding 是正样本。
            pos_mask = (sample_ids_t == sample_ids_t[idx])
            pos_mask[idx] = False
            if pos_mask.sum() < 1:
                continue
            # 正样本 logits。
            pos_logits = torch.matmul(tumor_embeds[idx : idx + 1], tumor_embeds[pos_mask].T).flatten() / self.temperature
            # 背景 prototype 是负样本。
            neg_logits_bg = torch.matmul(tumor_embeds[idx : idx + 1], bg_embeds.T).flatten() / self.temperature
            # 其他样本的 tumor embedding 也可作为负样本。
            other_tumor = sample_ids_t != sample_ids_t[idx]
            neg_logits = neg_logits_bg
            if other_tumor.any():
                neg_logits_other = torch.matmul(tumor_embeds[idx : idx + 1], tumor_embeds[other_tumor].T).flatten() / self.temperature
                neg_logits = torch.cat([neg_logits, neg_logits_other], dim=0)
            # InfoNCE 分子：所有正样本 logsumexp。
            numerator = torch.logsumexp(pos_logits, dim=0)
            # InfoNCE 分母：正样本 + 负样本。
            denominator = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=0), dim=0)
            # 加入当前 anchor 的对比损失。
            losses.append(-(numerator - denominator))

        if not losses:
            return phase_features.new_tensor(0.0)
        # 返回所有 anchor 的平均对比损失。
        return torch.nan_to_num(torch.stack(losses).mean(), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _masked_pool(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """在二值 mask 内对 feature map 做平均池化。"""
        # denom 至少为 1，避免空 mask 除零。
        denom = mask.sum().clamp_min(1.0)
        # feat: [C,H,W]，mask: [1,H,W] 或 [H,W]，广播后求区域均值。
        return (feat * mask).sum(dim=(1, 2)) / denom


class KPRNetLoss(nn.Module):
    """KPR-Net 的组合损失。

    KPR-Net 关注缺失 phase 鲁棒性，因此组合：
        主分割损失。
        可选 kinetic consistency。
        可选 temporal contrastive loss。
    """

    def __init__(
        self,
        lambda_contrastive: float = 0.1,
        lambda_kinetic: float = 0.05,
        contrastive_temperature: float = 0.1,
        min_tumor_pixels: int = 20,
        kinetic_margin: float = 0.05,
        peritumor_ring_size: int = 5,
        kinetic_eps: float = 1e-6,
        bce_weight: float = 0.5,
        ablation: Optional[Dict] = None,
        **_: Dict,
    ) -> None:
        super().__init__()
        # 读取消融配置。
        ablation = ablation or {}
        # 主分割损失。
        self.seg_loss = DiceBCELoss(bce_weight=bce_weight)
        # 对比学习损失权重。
        self.lambda_contrastive = (
            0.0 if ablation.get("disable_temporal_contrastive_loss", False) else lambda_contrastive
        )
        # kinetic consistency 权重。
        self.lambda_kinetic = 0.0 if ablation.get("disable_kinetic_loss", False) else lambda_kinetic
        # 动力学一致性损失。
        self.kinetic_loss = KineticConsistencyLoss(
            margin=kinetic_margin,
            ring_size=peritumor_ring_size,
            eps=kinetic_eps,
        )
        # 时序对比学习损失。
        self.temporal_contrastive = TemporalContrastiveLoss(
            temperature=contrastive_temperature,
            min_tumor_pixels=min_tumor_pixels,
        )

    def forward(self, output, target: torch.Tensor, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算 KPR-Net loss，并在辅助信息不可用时安全跳过。"""
        # 解包分割 logits 和辅助输出。
        seg_logits, info = unpack_model_output(output)
        # 主分割损失。
        loss = self.seg_loss(seg_logits, target)
        if self.lambda_kinetic > 0:
            # 加入 kinetic consistency。
            loss = loss + self.lambda_kinetic * self.kinetic_loss(seg_logits, info.get("kinetic_maps"), target)
        if self.lambda_contrastive > 0:
            # 加入 temporal contrastive loss。
            loss = loss + self.lambda_contrastive * self.temporal_contrastive(
                info.get("contrastive_embeddings"),
                target,
            )
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
