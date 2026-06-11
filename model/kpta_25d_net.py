"""SA-KPTA-Net / KPTA-2.5DNet 模型实现。

本文件实现方案 D：面向乳腺 DCE-MRI 肿瘤分割的 2.5D KPTA-Net。
核心思想是不要一开始把所有相邻切片和所有 DCE 时相直接压平成普通通道，
而是显式保留两个医学上有意义的维度：

    K：相邻切片维度，例如 z-1、z、z+1，用来建模解剖连续性。
    T：DCE 时相维度，例如 pre、post1..post8、sub1..sub8，用来建模增强时序。

因此输入张量是 [B,K,T,H,W]：
    B 是 batch size。
    K 是相邻切片数量，默认 K=3，即 [z-1, z, z+1]。
    T 是 DCE 多时相数量，默认 T=17，即 1 个 pre + 8 个 post + 8 个 sub。
    H/W 是二维切片空间尺寸。

注意：当前默认输入不是“17 个切片”，而是“3 个相邻切片 × 17 个 DCE phase”。
也就是单个样本维度为 [3,17,H,W]，DataLoader 拼 batch 后是 [B,3,17,H,W]。

整体结构：
    原始输入                 ：【B,K,T,H,W】
        -> kinetic builder   ：【B,K,T,H,W】->【B,K,M,H,W】
        -> Slice-wise CNN    ：【B,K,T,H,W】->【B,K,T,C,H,W】
        -> Slice Context     ：【B,K,T,C,H,W】->【B,T,C,H,W】
        -> Kinetic Branch    ：【B,K,M,H,W】->【B,C,H,W】
        -> Phase Attention   ：【B,T,C,H,W】+【B,C,H,W】->【B,C,H,W】
        -> Hybrid Encoder    ：【B,C,H,W】-> 多尺度 skip features
        -> U-Net Decoder     ：多尺度 skip features ->【B,C,H,W】
        -> Boundary Refine   ：【B,C,H,W】->【B,C,H,W】
        -> 输出中心切片 mask ：【B,1,H,W】
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F

from .kpta_blocks import ConvBlock, UpBlock
from .kpta_utils import PhaseIndices


class KineticMapBuilder25D(nn.Module):
    """从原始 2.5D DCE 输入构造保留切片维度的伪动力学图。

    这里最重要的是计算顺序：
        1. 先读取原始 pre/post/sub 强度。
        2. 先构造 SUB、PE、ME、STD 等 kinetic maps。
        3. 再进行统一归一化。

    这样做是为了避免“先逐 phase 归一化”破坏 DCE 增强幅度关系。

    输入：
        x: [B,K,T,H,W]。

    输出：
        kinetic_maps: [B,K,M,H,W]。
        M 是 kinetic map 的通道数。
        K 维度被保留，方便 kinetic branch 也使用相邻切片信息。
    """

    def __init__(
        self,
        phase_indices: Optional[Dict] = None,
        kinetic_maps: Optional[Sequence[str]] = None,
        eps: float = 1e-6,
        clip_value: Optional[float] = None,
        normalize_after_kinetic: bool = True,
        normalize_kinetic: bool = True,
    ) -> None:
        super().__init__()
        self.phase_indices = PhaseIndices.from_config(phase_indices)
        self.kinetic_maps = list(
            [
                "sub_stack",
                "peak_enhancement",
                "mean_enhancement",
                "temporal_std",
                "early_enhancement",
                "late_enhancement",
            ]
            if kinetic_maps is None
            else kinetic_maps
        )
        self.eps = eps
        self.clip_value = clip_value
        self.normalize_after_kinetic = normalize_after_kinetic
        self.normalize_kinetic = normalize_kinetic

    @property
    def expected_channels(self) -> int:
        """计算 kinetic branch 需要接收的 kinetic map 通道数。

        有些 map 只产生 1 个通道，例如 peak_enhancement。
        有些 map 会按 phase 产生多个通道，例如 sub_stack。
        该值用于初始化 KineticPriorBranch25D 的第一层卷积。
        """
        # 对 sub_stack / relative_enhancement 来说，通道数等于动态 phase 数。
        n_sub = max(len(self.phase_indices.subtraction), len(self.phase_indices.post), 1)
        # 累计所有配置中 kinetic map 的输出通道数。
        channels = 0
        for name in self.kinetic_maps:
            if name in {"sub_stack", "relative_enhancement"}:
                channels += n_sub
            else:
                channels += 1
        return max(channels, 1)

    def split(self, x: torch.Tensor):
        """把 T 维度拆成 pre、post、subtraction 三组。

        BreastDM 17 通道默认布局：
            0      : VIBRANT / pre。
            1..8   : VIBRANT+C1..C8 / post。
            9..16  : SUB1..SUB8 / subtraction。

        稳健 fallback：
            如果 post 缺失但 sub 存在，则近似 post = pre + sub。
            如果 sub 缺失但 post 存在，则计算 sub = post - pre。
            如果两者都缺失，则使用零 subtraction，保证模型不崩溃。
        """
        # 从 T 维取出 pre-contrast：【B,K,T,H,W】->【B,K,1,H,W】。
        pre = x[:, :, self.phase_indices.pre : self.phase_indices.pre + 1]
        # 根据配置读取 post phase：【B,K,T,H,W】->【B,K,N_post,H,W】；如果索引越界，_select 会自动忽略。
        post = self._select(x, self.phase_indices.post)
        # 根据配置读取 subtraction phase：【B,K,T,H,W】->【B,K,N_sub,H,W】；BreastDM 中优先使用真实 SUB。
        sub = self._select(x, self.phase_indices.subtraction)
        # 没有 post 但有 sub 时，用 pre + sub 近似 post。
        if post is None and sub is not None:
            post = pre + sub
        # 没有 sub 但有 post 时，用 post - pre 构造 subtraction。
        if sub is None and post is not None:
            sub = post - pre
        # 如果 post/sub 都没有，退化成只看 pre，并构造零增强图。
        if post is None and sub is None:
            post = pre
            sub = torch.zeros_like(pre)
        # 到这里 post/sub 必须存在，否则说明 fallback 逻辑有问题。
        assert post is not None and sub is not None
        return pre, post, sub

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """在 kinetic maps 构造之后，对最终模型输入做统一归一化。

        归一化维度是每个样本的 K/T/H/W 整体，而不是每个 phase 单独归一化。
        这样能保留同一病例内部不同 phase 之间的增强强弱关系。
        """
        # 如果配置关闭归一化，则只转换成 float。
        if not self.normalize_after_kinetic:
            return x.float()
        # 转 float，避免 uint8/int 输入导致均值方差计算异常。
        x = x.float()
        # 每个样本独立计算均值，覆盖切片、phase 和空间维度。
        mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        # 每个样本独立计算标准差，unbiased=False 避免小张量警告。
        std = x.std(dim=(1, 2, 3, 4), keepdim=True, unbiased=False)
        # 加 eps 防止除零。
        return (x - mean) / (std + self.eps)

    def normalize_kinetic_maps(self, kinetic_maps: torch.Tensor) -> torch.Tensor:
        """对 kinetic maps 做每个样本级别的统一归一化。

        kinetic maps 可能包含 raw subtraction、STD、relative enhancement 等不同尺度。
        统一归一化可以稳定 kinetic branch 的训练，同时保留空间差异。
        """
        # 如果关闭 kinetic 归一化，仍然转为 float 以适配卷积。
        if not self.normalize_kinetic:
            return kinetic_maps.float()
        # 转 float 后计算每个样本所有 kinetic map 的整体统计量。
        kinetic_maps = kinetic_maps.float()
        # 均值覆盖 K/M/H/W，避免逐 map 归一化破坏 map 间相对强度。
        mean = kinetic_maps.mean(dim=(1, 2, 3, 4), keepdim=True)
        # 标准差同样覆盖 K/M/H/W。
        std = kinetic_maps.std(dim=(1, 2, 3, 4), keepdim=True, unbiased=False)
        # 输出归一化后的 kinetic maps。
        return (kinetic_maps - mean) / (std + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """根据配置构造伪动力学图。

        当前支持：
            sub_stack           : SUB1..SUBN，直接使用 BreastDM 提供的减影图。
            peak_enhancement    : 多个 SUB phase 的最大增强。
            mean_enhancement    : 多个 SUB phase 的平均增强。
            temporal_std        : post phase 的时间变化强度。
            early_enhancement   : 早期增强，即第一个 SUB。
            late_enhancement    : 晚期增强，即最后一个 SUB。
            relative_enhancement: SUB / abs(pre)，可选相对增强。
        """
        # 方案 D 只接受 [B,K,T,H,W]，不接受 3D 卷积格式。
        if x.ndim != 5:
            raise ValueError(f"Expected [B,K,T,H,W], got {tuple(x.shape)}")
        # 先拆出 pre/post/sub，后续所有 kinetic map 都基于这三组构造。
        pre, post, sub = self.split(x.float())
        # maps 用来收集不同类型的 kinetic map，最后在 M 维拼接。
        # 每个元素都保持 2.5D 结构：【B,K,m_i,H,W】。
        maps = []
        for name in self.kinetic_maps:
            if name == "sub_stack":
                # 直接使用 SUB 序列，保留每个增强 phase：【B,K,N_sub,H,W】。
                maps.append(sub)
            elif name == "peak_enhancement":
                # 峰值增强：沿 phase 维取最大值，【B,K,N_sub,H,W】->【B,K,1,H,W】。
                maps.append(sub.amax(dim=2, keepdim=True))
            elif name == "mean_enhancement":
                # 平均增强：沿 phase 维取平均值，【B,K,N_sub,H,W】->【B,K,1,H,W】。
                maps.append(sub.mean(dim=2, keepdim=True))
            elif name == "temporal_std":
                # 时间变化：post 序列标准差，【B,K,N_post,H,W】->【B,K,1,H,W】。
                maps.append(post.std(dim=2, keepdim=True, unbiased=False))
            elif name == "early_enhancement":
                # 早期增强：取第一个 SUB phase，【B,K,N_sub,H,W】->【B,K,1,H,W】。
                maps.append(sub[:, :, :1])
            elif name == "late_enhancement":
                # 晚期增强：取最后一个 SUB phase，【B,K,N_sub,H,W】->【B,K,1,H,W】。
                maps.append(sub[:, :, -1:])
            elif name == "relative_enhancement":
                # 相对增强：利用 pre 广播归一化 SUB，【B,K,N_sub,H,W】/【B,K,1,H,W】->【B,K,N_sub,H,W】。
                maps.append(sub / (pre.abs() + self.eps))
            else:
                raise ValueError(f"Unknown 2.5D kinetic map: {name}")
        # 如果 kinetic map 被 ablation 关闭，则给一个零通道占位：【B,K,1,H,W】，保证分支可运行。
        kinetic = torch.cat(maps, dim=2) if maps else x.new_zeros((x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4]))
        # 可选截断极端增强值，避免异常像素影响训练。
        if self.clip_value is not None and self.clip_value > 0:
            kinetic = kinetic.clamp(-self.clip_value, self.clip_value)
        # 先清理 NaN/Inf，再进入归一化。
        kinetic = torch.nan_to_num(kinetic, nan=0.0, posinf=0.0, neginf=0.0)
        # 输出归一化后的 kinetic maps。
        return torch.nan_to_num(self.normalize_kinetic_maps(kinetic), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _select(x: torch.Tensor, indices) -> Optional[torch.Tensor]:
        """按配置选择 phase，并忽略越界索引。"""
        # 过滤掉超过 T 维长度的 phase index，增强配置鲁棒性。
        valid = [idx for idx in indices if 0 <= idx < x.shape[2]]
        # 没有有效 phase 时返回 None，让上层 fallback。
        if not valid:
            return None
        # 在 T 维度上选择对应 phase。
        return x[:, :, valid]


class SliceWiseCNNStem(nn.Module):
    """逐切片、逐时相共享的浅层 CNN stem。

    这是方案 D 的第一个 2.5D 设计点：
        不把 K*T 直接压平成普通通道。
        而是对每一张单通道 DCE slice 使用同一个 CNN stem。
        输出仍显式保留 [B,K,T,C,H,W] 结构。
    """

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        # ConvBlock 来自 SG-KTFNet，包含 Conv/Norm/Activation，负责浅层纹理提取。
        self.stem = ConvBlock(1, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把 [B,K,T,H,W] 编码为 [B,K,T,C,H,W]。"""
        # 记录输入维度：【B,K,T,H,W】中的 B/K/T/H/W。
        b, k, t, h, w = x.shape
        # 把 B/K/T 合并成 batch 维，让共享 CNN 一次性处理所有单通道图：
        # 【B,K,T,H,W】->【B*K*T,1,H,W】。
        feat = self.stem(x.reshape(b * k * t, 1, h, w))
        # CNN stem 输出：【B*K*T,C,H,W】。
        # 恢复显式 2.5D 结构：【B*K*T,C,H,W】->【B,K,T,C,H,W】。
        return feat.reshape(b, k, t, feat.shape[1], feat.shape[2], feat.shape[3])


class SliceContextAggregation(nn.Module):
    """用像素级 slice attention 融合相邻切片。

    对每个 DCE phase，模型同时看到 z-1、z、z+1。
    模块学习 beta_k(x,y)，对 K 个切片做空间位置相关的加权融合。
    由于监督 mask 只属于中心切片，因此默认加入 center residual，避免邻近切片过度干扰。
    """

    def __init__(self, channels: int, disabled: bool = False, use_center_residual: bool = True) -> None:
        super().__init__()
        # disabled=True 用于消融实验：关闭 slice context，只取中心切片。
        self.disabled = disabled
        # center residual 用于保持中心切片主导地位。
        self.use_center_residual = use_center_residual
        # 1x1 conv 为每个位置、每个切片输出一个 attention score。
        self.score = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """融合 K 个切片，同时保留 T 个 DCE phase。"""
        # 输入 x:【B,K,T,C,H,W】；输出 aggregated:【B,T,C,H,W】。
        if self.disabled or x.shape[1] == 1:
            # 关闭 slice context 时，直接取中心切片作为输出：【B,K,T,C,H,W】->【B,T,C,H,W】。
            center = x[:, x.shape[1] // 2]
            # 为了可视化/调试，仍返回一个均匀 slice attention map：【B,K,T,1,H,W】。
            attn = x.new_ones((x.shape[0], x.shape[1], x.shape[2], 1, x.shape[-2], x.shape[-1])) / float(x.shape[1])
            return center, attn
        # 展开维度，准备对每个 slice/phase 的特征计算 score。
        b, k, t, c, h, w = x.shape
        # 对每个 slice/phase 的 C 维特征做 1x1 conv：
        # 【B,K,T,C,H,W】->【B*K*T,C,H,W】->【B*K*T,1,H,W】->【B,K,T,1,H,W】。
        scores = self.score(x.reshape(b * k * t, c, h, w)).reshape(b, k, t, 1, h, w)
        # 在 K 维做 softmax，得到每个像素位置对不同切片的权重：【B,K,T,1,H,W】。
        weights = torch.softmax(scores, dim=1)
        # 用 slice attention 对相邻切片特征加权求和：
        # 【B,K,T,C,H,W】*【B,K,T,1,H,W】-> sum(K) ->【B,T,C,H,W】。
        aggregated = (x * weights).sum(dim=1)
        if self.use_center_residual:
            # 取中心切片特征：【B,K,T,C,H,W】->【B,T,C,H,W】。
            center = x[:, k // 2]
            # 把聚合结果和中心切片平均，保证输出仍聚焦中心 slice 的 mask：【B,T,C,H,W】->【B,T,C,H,W】。
            aggregated = 0.5 * (aggregated + center)
        return aggregated, weights


def custom_max(x: torch.Tensor, dim, keepdim: bool = True) -> torch.Tensor:
    """按多个维度连续取最大值，兼容原始 CSAM 实现。"""
    out = x
    for axis in sorted(dim, reverse=True):
        out = torch.max(out, dim=axis, keepdim=keepdim)[0]
    return out


class PositionalAttentionModule(nn.Module):
    """CSAM 的空间位置注意力，在 K 个切片和 C 个通道上汇聚后生成 H/W 权重。"""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [K,C,H,W]。max/avg 后得到 [1,1,H,W]，再生成空间注意力。
        max_x = custom_max(x, dim=(0, 1), keepdim=True)
        avg_x = torch.mean(x, dim=(0, 1), keepdim=True)
        att = torch.cat((max_x, avg_x), dim=1)
        att = torch.sigmoid(self.conv(att))
        return x * att


class SemanticAttentionModule(nn.Module):
    """CSAM 的语义/通道注意力，对每个 phase 的 C 个通道重新标定。"""

    def __init__(self, in_features: int, reduction_rate: int = 16) -> None:
        super().__init__()
        hidden_dim = max(in_features // reduction_rate, 4)
        self.linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [K,C,H,W]。在 slice 和空间维度上汇聚，得到通道注意力 [1,C,1,1]。
        max_x = custom_max(x, dim=(0, 2, 3), keepdim=False).unsqueeze(0)
        avg_x = torch.mean(x, dim=(0, 2, 3), keepdim=False).unsqueeze(0)
        att = torch.sigmoid(self.linear(max_x) + self.linear(avg_x)).unsqueeze(-1).unsqueeze(-1)
        return x * att


class SliceAttentionModule(nn.Module):
    """CSAM 的切片注意力，对 K 个相邻切片做全局权重标定。"""

    def __init__(
        self,
        in_features: int,
        rate: float = 4.0,
        uncertainty: bool = False,
        rank: int = 5,
    ) -> None:
        super().__init__()
        self.uncertainty = uncertainty
        self.rank = rank
        hidden_dim = max(int(in_features * rate), 4)
        self.linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=in_features),
        )
        if uncertainty:
            self.non_linear = nn.ReLU()
            self.mean = nn.Linear(in_features=in_features, out_features=in_features)
            self.log_diag = nn.Linear(in_features=in_features, out_features=in_features)
            self.factor = nn.Linear(in_features=in_features, out_features=in_features * rank)
        self.last_attention_map = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [K,C,H,W]。在通道和空间维度汇聚，得到 K 个 slice 的注意力。
        num_slices = x.shape[0]
        max_x = custom_max(x, dim=(1, 2, 3), keepdim=False).unsqueeze(0)
        avg_x = torch.mean(x, dim=(1, 2, 3), keepdim=False).unsqueeze(0)
        att = self.linear(max_x) + self.linear(avg_x)
        if self.uncertainty:
            temp = self.non_linear(att)
            mean = self.mean(temp)
            diag = self.log_diag(temp).exp().clamp_min(1e-6)
            factor = self.factor(temp).view(1, -1, self.rank)
            dist = td.LowRankMultivariateNormal(loc=mean, cov_factor=factor, cov_diag=diag)
            # 训练时采样增强鲁棒性，推理时固定用均值，避免测试结果随机漂移。
            att = dist.rsample() if self.training else mean
        att = torch.sigmoid(att).view(num_slices, 1, 1, 1)
        self.last_attention_map = att.detach()
        return x * att


class CSAM(nn.Module):
    """Cross-Slice Attention Module，用于增强一组相邻切片特征 [K,C,H,W]。"""

    def __init__(
        self,
        num_slices: int,
        num_channels: int,
        semantic: bool = True,
        positional: bool = True,
        slice: bool = True,
        uncertainty: bool = False,
        rank: int = 5,
    ) -> None:
        super().__init__()
        self.semantic = semantic
        self.positional = positional
        self.slice = slice
        if semantic:
            self.semantic_att = SemanticAttentionModule(num_channels)
        if positional:
            self.positional_att = PositionalAttentionModule()
        if slice:
            self.slice_att = SliceAttentionModule(num_slices, uncertainty=uncertainty, rank=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.semantic:
            x = self.semantic_att(x)
        if self.positional:
            x = self.positional_att(x)
        if self.slice:
            x = self.slice_att(x)
        return x


class CSAMSliceAggregation(nn.Module):
    """使用 CSAM 对每个 batch、每个 DCE phase 的 K 个相邻切片做 cross-slice attention。

    输入 features: [B,K,T,C,H,W]。
    输出 out: [B,T,C,H,W]，可选择中心切片特征或 K 切片均值特征。
    """

    def __init__(
        self,
        num_slices: int,
        channels: int,
        aggregate: str = "center",
        disabled: bool = False,
        semantic: bool = True,
        positional: bool = True,
        slice_attention: bool = True,
        uncertainty: bool = False,
        rank: int = 5,
    ) -> None:
        super().__init__()
        if aggregate not in {"center", "mean"}:
            raise ValueError(f"Unknown CSAM aggregate mode: {aggregate}")
        self.num_slices = num_slices
        self.aggregate = aggregate
        self.disabled = disabled
        self.csam = CSAM(
            num_slices=num_slices,
            num_channels=channels,
            semantic=semantic,
            positional=positional,
            slice=slice_attention,
            uncertainty=uncertainty,
            rank=rank,
        )
        self.last_slice_attention_map = None

    def forward(self, features: torch.Tensor):
        if features.ndim != 6:
            raise ValueError(f"CSAMSliceAggregation expects [B,K,T,C,H,W], got {tuple(features.shape)}")
        b, k, t, c, h, w = features.shape
        if k != self.num_slices:
            raise ValueError(f"CSAM expected {self.num_slices} slices, got {k}")
        if self.disabled or k == 1:
            center = features[:, k // 2]
            attn = features.new_ones((b, k, t, 1, 1, 1)) / float(k)
            self.last_slice_attention_map = attn.detach()
            return center, attn

        out = features.new_empty((b, t, c, h, w))
        attn_maps = features.new_empty((b, k, t, 1, 1, 1))
        for batch_idx in range(b):
            for phase_idx in range(t):
                feat = features[batch_idx, :, phase_idx]
                enhanced = self.csam(feat)
                if self.aggregate == "center":
                    out[batch_idx, phase_idx] = enhanced[k // 2]
                else:
                    out[batch_idx, phase_idx] = enhanced.mean(dim=0)
                if self.csam.slice:
                    attn = self.csam.slice_att.last_attention_map
                    if attn is None:
                        attn = features.new_ones((k, 1, 1, 1)) / float(k)
                    attn_maps[batch_idx, :, phase_idx] = attn.to(device=features.device, dtype=features.dtype)
                else:
                    attn_maps[batch_idx, :, phase_idx] = features.new_ones((k, 1, 1, 1)) / float(k)
        self.last_slice_attention_map = attn_maps.detach()
        return out, attn_maps


class KineticPriorBranch25D(nn.Module):
    """编码伪动力学图，并融合其相邻切片上下文。

    该分支不直接预测分割，而是输出 kinetic feature。
    kinetic feature 会进入 phase attention，帮助模型在像素级选择更符合肿瘤增强模式的 phase。
    """

    def __init__(self, in_channels: int, out_channels: int, disabled_slice_context: bool = False) -> None:
        super().__init__()
        # 对每个切片的 M 个 kinetic map 做 2D CNN 编码。
        self.encoder = ConvBlock(in_channels, out_channels)
        # kinetic map 同样进行 slice context aggregation。
        self.slice_agg = SliceContextAggregation(out_channels, disabled=disabled_slice_context)

    def forward(self, kinetic_maps: torch.Tensor):
        """返回中心切片感知的 kinetic feature 和 kinetic slice attention。"""
        # kinetic_maps:【B,K,M,H,W】，M 是 kinetic map 通道数。
        b, k, m, h, w = kinetic_maps.shape
        # 合并 B/K，用 CNN 编码每个切片的 kinetic maps：
        # 【B,K,M,H,W】->【B*K,M,H,W】->【B*K,C,H,W】。
        feat = self.encoder(kinetic_maps.reshape(b * k, m, h, w))
        # 恢复为带伪 phase 维的结构：【B*K,C,H,W】->【B,K,1,C,H,W】。
        # 这里 T=1 是为了复用 SliceContextAggregation 的 slice 融合逻辑。
        feat = feat.reshape(b, k, 1, feat.shape[1], feat.shape[2], feat.shape[3])
        # 聚合 K 个切片上的 kinetic feature：【B,K,1,C,H,W】->【B,1,C,H,W】。
        aggregated, attn = self.slice_agg(feat)
        # 去掉伪 phase 维，输出动力学先验特征：【B,1,C,H,W】->【B,C,H,W】。
        return aggregated[:, 0], attn


class PixelWisePhaseAttention25D(nn.Module):
    """动力学感知的像素级 DCE phase attention。

    核心公式：
        F_fused(x,y) = sum_t alpha_t(x,y) * F_t(x,y)

    每个 phase 的 score 由该 phase 的图像特征和 kinetic prior feature 共同决定。
    由于 score 网络在 phase 间共享，因此比固定 concat 更鲁棒，也避免 T 变化时 shape mismatch。
    """

    def __init__(self, channels: int, num_phases: int, disabled: bool = False) -> None:
        super().__init__()
        # disabled=True 用于消融：关闭像素级 phase attention，退化为均值融合。
        self.disabled = disabled
        # 记录配置中的 phase 数，主要用于调试和输入一致性检查。
        self.num_phases = num_phases
        # score 网络输入为 [phase_feature, kinetic_feature]，输出该 phase 的逐像素 score。
        self.score = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(
        self,
        phase_feats: torch.Tensor,
        sub_maps: Optional[torch.Tensor] = None,
        kinetic_feat: Optional[torch.Tensor] = None,
    ):
        """把 [B,T,C,H,W] phase feature 融合为 [B,C,H,W]。"""
        # phase_feats 表示每个 DCE phase 的中心切片上下文特征：【B,T,C,H,W】。
        b, t, c, h, w = phase_feats.shape
        if self.disabled or t == 1:
            # 关闭 attention 或只有一个 phase 时，使用均匀权重：【B,T,1,H,W】。
            attn = phase_feats.new_full((b, t, 1, h, w), 1.0 / float(t))
            # 均值融合 phase：【B,T,C,H,W】->【B,C,H,W】。
            return phase_feats.mean(dim=1), attn, {"feature_score": None}
        if kinetic_feat is None:
            kinetic_feat = phase_feats.mean(dim=1)
        # 将 kinetic feature 扩展到每个 phase，与 phase feature 对齐：
        # 【B,C,H,W】->【B,1,C,H,W】->【B,T,C,H,W】。
        kinetic = kinetic_feat.unsqueeze(1).expand(-1, t, -1, -1, -1)
        # 拼接 phase feature 与 kinetic feature：
        # 【B,T,C,H,W】+【B,T,C,H,W】->【B,T,2C,H,W】->【B*T,2C,H,W】。
        logits = self.score(torch.cat([phase_feats, kinetic], dim=2).reshape(b * t, c * 2, h, w))
        # 恢复 attention logits 维度：【B*T,1,H,W】->【B,T,H,W】。
        logits = logits.reshape(b, t, 1, h, w).squeeze(2)
        # 在 T 维 softmax，保证同一像素所有 phase 权重和为 1：【B,T,H,W】->【B,T,1,H,W】。
        attn = torch.softmax(logits, dim=1).unsqueeze(2)
        # 加权融合所有 phase 的特征：【B,T,C,H,W】*【B,T,1,H,W】-> sum(T) ->【B,C,H,W】。
        return (phase_feats * attn).sum(dim=1), attn, {"feature_score": logits.detach()}


class PhaseDifferenceWeightingAttention(nn.Module):
    """Phase Difference Weighting Attention（PDWA）。

    该模块借鉴 LesiOnTime 的 temporal difference weighting 思想，但迁移到单次
    DCE-MRI 多增强时相：用 raw phase feature、对应 subtraction/enhancement map
    和整体 kinetic prior feature 共同生成像素级 phase attention，并用 phase-fused
    feature 与 kinetic feature 的差异进一步增强变化显著区域。
    """

    def __init__(
        self,
        channels: int,
        num_phases: int,
        use_enhancement_prior: bool = True,
        use_kinetic_bias: bool = True,
        use_difference_refinement: bool = True,
        residual_mean_weight: float = 0.5,
        lambda_enh_init: float = 1.0,
        gamma_diff_init: float = 0.5,
        disabled: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_phases = num_phases
        self.use_enhancement_prior = use_enhancement_prior
        self.use_kinetic_bias = use_kinetic_bias
        self.use_difference_refinement = use_difference_refinement
        self.residual_mean_weight = float(residual_mean_weight)
        self.disabled = disabled

        self.feature_score = nn.Conv2d(channels, 1, kernel_size=1)
        hidden = max(channels // 4, 4)
        self.enhancement_score = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.kinetic_bias = nn.Conv2d(channels, 1, kernel_size=1)
        self.kinetic_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.diff_norm = nn.InstanceNorm2d(channels)
        self.diff_gate = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.lambda_enh = nn.Parameter(torch.tensor(float(lambda_enh_init)))
        self.gamma_diff = nn.Parameter(torch.tensor(float(gamma_diff_init)))

    def forward(
        self,
        phase_feats: torch.Tensor,
        sub_maps: Optional[torch.Tensor] = None,
        kinetic_feat: Optional[torch.Tensor] = None,
    ):
        """融合多时相特征。

        phase_feats: [B,T,C,H,W]
        sub_maps: [B,T_sub,H,W] 或 [B,K,T_sub,H,W]
        kinetic_feat: [B,C,H,W]
        """
        assert phase_feats.ndim == 5, f"phase_feats must be [B,T,C,H,W], got {tuple(phase_feats.shape)}"
        b, t, c, h, w = phase_feats.shape
        assert c == self.channels, f"PDWA expected {self.channels} channels, got {c}"

        if self.disabled:
            alpha = phase_feats.new_full((b, t, 1, h, w), 1.0 / float(t))
            fused = phase_feats.mean(dim=1)
            return fused, alpha, {
                "feature_score": None,
                "enhancement_score": None,
                "kinetic_bias": None,
                "diff_gate": None,
            }

        # 1) feature_score：每个 phase 的 raw feature 独立用 1x1 conv 产生像素级分数。
        flat_phase = phase_feats.reshape(b * t, c, h, w)
        feature_score = self.feature_score(flat_phase).reshape(b, t, 1, h, w)
        score = feature_score

        # 2) enhancement_score：用 BreastDM 的 C_sub_t / enhancement map 引导时相选择。
        enhancement_score = phase_feats.new_zeros((b, t, 1, h, w))
        if self.use_enhancement_prior and sub_maps is not None:
            sub = self._center_sub_maps(sub_maps)
            if sub.shape[-2:] != (h, w):
                sub = F.interpolate(sub, size=(h, w), mode="bilinear", align_corners=False)
            assert sub.shape[0] == b, f"sub_maps batch mismatch: {sub.shape[0]} vs {b}"
            t_sub = sub.shape[1]
            sub_flat = sub.reshape(b * t_sub, 1, h, w)
            enhancement_score = self.enhancement_score(sub_flat).reshape(b, t_sub, 1, h, w)
            enhancement_score = self._align_phase_count(enhancement_score, t)
            score = score + self.lambda_enh * enhancement_score

        # 3) kinetic_bias：整体动力学先验对所有 phase score 提供像素级偏置。
        kinetic_bias = phase_feats.new_zeros((b, 1, 1, h, w))
        if self.use_kinetic_bias and kinetic_feat is not None:
            kin = self._match_kinetic_shape(kinetic_feat, h, w)
            assert kin.shape[:2] == (b, c), f"kinetic_feat must be [B,C,H,W], got {tuple(kin.shape)}"
            kinetic_bias = self.kinetic_bias(kin).unsqueeze(1)
            score = score + kinetic_bias

        # 4) softmax 得到 phase attention，并加入均值残差防止 attention collapse。
        alpha = torch.softmax(score, dim=1)
        fused = (phase_feats * alpha).sum(dim=1)
        if self.residual_mean_weight != 0:
            fused = fused + self.residual_mean_weight * phase_feats.mean(dim=1)

        # 5) difference refinement：F_current=phase-fused raw feature，F_prior=kinetic prior feature。
        diff_gate = phase_feats.new_zeros((b, c, h, w))
        if self.use_difference_refinement and kinetic_feat is not None:
            kin = self._match_kinetic_shape(kinetic_feat, h, w)
            kin_proj = self.kinetic_proj(kin)
            diff = self.diff_norm(fused - kin_proj)
            diff_gate = torch.sigmoid(self.diff_gate(diff))
            fused = fused + self.gamma_diff * fused * diff_gate

        debug = {
            "feature_score": feature_score.detach(),
            "enhancement_score": enhancement_score.detach(),
            "kinetic_bias": kinetic_bias.detach(),
            "diff_gate": diff_gate.detach(),
            "lambda_enh": self.lambda_enh.detach(),
            "gamma_diff": self.gamma_diff.detach(),
        }
        return fused, alpha, debug

    @staticmethod
    def _center_sub_maps(sub_maps: torch.Tensor) -> torch.Tensor:
        if sub_maps.ndim == 5:
            return sub_maps[:, sub_maps.shape[1] // 2]
        if sub_maps.ndim == 4:
            return sub_maps
        raise ValueError(f"sub_maps must be [B,T,H,W] or [B,K,T,H,W], got {tuple(sub_maps.shape)}")

    @staticmethod
    def _align_phase_count(score: torch.Tensor, target_phases: int) -> torch.Tensor:
        b, t_sub, one, h, w = score.shape
        if t_sub == target_phases:
            return score
        if t_sub > target_phases:
            return score[:, :target_phases]
        pad = score.new_zeros((b, target_phases - t_sub, one, h, w))
        return torch.cat([pad, score], dim=1)

    @staticmethod
    def _match_kinetic_shape(kinetic_feat: torch.Tensor, height: int, width: int) -> torch.Tensor:
        assert kinetic_feat.ndim == 4, f"kinetic_feat must be [B,C,H,W], got {tuple(kinetic_feat.shape)}"
        if kinetic_feat.shape[-2:] == (height, width):
            return kinetic_feat
        return F.interpolate(kinetic_feat, size=(height, width), mode="bilinear", align_corners=False)


class KineticRawFusion(nn.Module):
    """在空间编码器之前显式注入动力学先验特征。
    
    动力学分支具有两个互补的作用：它引导相位注意力评分，
    并通过残差融合，作为额外的先验特征进入编码器。
    它增强了原始图像分支，而不是替换它。
    """

    def __init__(self, channels: int, disabled: bool = False) -> None:
        super().__init__()
        self.disabled = disabled
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, fused_phase: torch.Tensor, kinetic_feature: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            return fused_phase
        if kinetic_feature.shape[-2:] != fused_phase.shape[-2:]:
            kinetic_feature = F.interpolate(
                kinetic_feature,
                size=fused_phase.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        concat = torch.cat([fused_phase, kinetic_feature], dim=1)
        return fused_phase + self.fusion_conv(concat)


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """把 NHWC 特征图切分成不重叠的 Swin window。"""
    # 输入 x:【B,H,W,C】。
    b, h, w, c = x.shape
    # 先把 H/W 按 window_size 分组：
    # 【B,H,W,C】->【B,H/ws,ws,W/ws,ws,C】。
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    # 调整维度并展平成 window token：
    # 【B,H/ws,ws,W/ws,ws,C】->【B,H/ws,W/ws,ws,ws,C】->【B*num_windows,ws*ws,C】。
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)


def _window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int, b: int) -> torch.Tensor:
    """把 window token 还原为 NHWC 特征图。"""
    # 输入 windows:【B*num_windows,window_size*window_size,C】。
    # 先恢复 window 网格：【B*num_windows,ws*ws,C】->【B,H/ws,W/ws,ws,ws,C】。
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    # 按原始窗口排列恢复空间布局：【B,H/ws,W/ws,ws,ws,C】->【B,H,W,C】。
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowSelfAttention(nn.Module):
    """带相对位置偏置的窗口自注意力。

    这是轻量 Swin bottleneck 的核心。
    相比全局 Transformer，它只在局部窗口内计算注意力，更适合医学图像局部结构建模，
    同时相对位置偏置可以保留窗口内部的空间位置信息。
    """

    def __init__(self, channels: int, window_size: int = 7, num_heads: int = 4) -> None:
        super().__init__()
        # head 数不能超过通道数。
        heads = max(1, min(num_heads, channels))
        # 如果 channels 不能被 heads 整除，就递减 heads，保证 reshape 合法。
        while channels % heads != 0 and heads > 1:
            heads -= 1
        # 保存输入通道数。
        self.channels = channels
        # 保存窗口大小。
        self.window_size = window_size
        # 保存最终可用的 attention heads。
        self.num_heads = heads
        # 每个 head 的通道数。
        self.head_dim = channels // heads
        # 缩放因子，避免 qk 点积过大。
        self.scale = self.head_dim ** -0.5
        # qkv 线性投影，一次性生成 query/key/value。
        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        # attention 输出投影。
        self.proj = nn.Linear(channels, channels)

        # 下面构造 Swin 的相对位置索引表。
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        try:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        except TypeError:
            coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        # relative_position_index 不参与训练，只作为索引缓存。
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)
        # 每个相对位置、每个 head 学习一个 bias。
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), heads)
        )
        # 使用截断正态初始化相对位置偏置。
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """在每个局部窗口内执行多头自注意力。"""
        # 输入 x:【B*num_windows,window_area,C】，window_area=window_size*window_size。
        b_windows, n, c = x.shape
        # 生成 q/k/v：【Bw,N,C】->【Bw,N,3C】->【Bw,N,3,heads,head_dim】。
        qkv = self.qkv(x).reshape(b_windows, n, 3, self.num_heads, self.head_dim)
        # 调整为多头 attention 常用格式：【Bw,N,3,heads,head_dim】->【3,Bw,heads,N,head_dim】。
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # 分离 q、k、v。
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 计算 scaled dot-product attention logits：【Bw,heads,N,head_dim】x【Bw,heads,head_dim,N】->【Bw,heads,N,N】。
        attn = (q * self.scale) @ k.transpose(-2, -1)
        # 根据相对位置索引取出 bias。
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        # 将相对位置 bias 加到每个 head 的 attention logits 上。
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            # shifted window 需要 mask，防止 roll 后窗口边界错误连接。
            num_windows = mask.shape[0]
            attn = attn.view(b_windows // num_windows, num_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, n, n)
        # softmax 得到窗口内 token 权重。
        attn = torch.softmax(attn, dim=-1)
        # attention 加权 value，并合并多头：【Bw,heads,N,N】x【Bw,heads,N,head_dim】->【Bw,N,C】。
        out = (attn @ v).transpose(1, 2).reshape(b_windows, n, c)
        # 输出投影回 C 维。
        return self.proj(out)


class SwinWindowBlock(nn.Module):
    """一个 Swin 风格 block，可选 shifted window。

    bottleneck 中会交替使用普通窗口和 shifted window。
    shifted window 让相邻窗口发生信息交互，attention mask 防止 roll 后错误跨边界连接。
    """

    def __init__(
        self,
        channels: int,
        window_size: int = 7,
        num_heads: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        # 保存窗口大小。
        self.window_size = window_size
        # shift_size=0 表示普通窗口；>0 表示 shifted window。
        self.shift_size = shift_size
        # attention 前的 LayerNorm，Swin 使用 token 维归一化。
        self.norm1 = nn.LayerNorm(channels)
        # 局部窗口自注意力。
        self.attn = WindowSelfAttention(channels, window_size=window_size, num_heads=num_heads)
        # MLP 隐藏层通道数。
        hidden = int(channels * mlp_ratio)
        # MLP 前的 LayerNorm。
        self.norm2 = nn.LayerNorm(channels)
        # FFN/MLP，用于增强非线性表达。
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对 [B,C,H,W] 执行窗口注意力和 MLP 残差。"""
        # 记录输入尺寸：【B,C,H,W】。
        b, c, h, w = x.shape
        # 第一个残差分支。
        shortcut = x
        # Swin 的 LayerNorm/attention 更方便在 NHWC 上处理：【B,C,H,W】->【B,H,W,C】。
        x = x.permute(0, 2, 3, 1).contiguous()
        # 如果 H/W 不能整除 window_size，则补齐到可切窗口。
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        # 保存 padding 后尺寸。
        hp, wp = x.shape[1], x.shape[2]
        # 如果 feature map 太小，就不做 shift，避免无效滚动。
        shift = self.shift_size if min(hp, wp) > self.window_size else 0
        if shift > 0:
            # 负向 roll 实现 shifted window。
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
        # shifted window 需要 mask，普通窗口不需要。
        attn_mask = self._shifted_window_mask(hp, wp, x.device, x.dtype) if shift > 0 else None
        # 切分窗口并在窗口内做 attention：
        # 【B,Hp,Wp,C】->【B*num_windows,ws*ws,C】->【B*num_windows,ws*ws,C】。
        windows = _window_partition(self.norm1(x), self.window_size)
        windows = self.attn(windows, mask=attn_mask)
        # 把窗口重新拼回空间特征图：【B*num_windows,ws*ws,C】->【B,Hp,Wp,C】。
        x = _window_reverse(windows, self.window_size, hp, wp, b)
        if shift > 0:
            # 正向 roll 还原 shifted window 的空间位置。
            x = torch.roll(x, shifts=(shift, shift), dims=(1, 2))
        if pad_h or pad_w:
            # 移除 padding，恢复原始 H/W。
            x = x[:, :h, :w, :].contiguous()
        # attention 残差连接，恢复 NCHW：【B,H,W,C】->【B,C,H,W】。
        x = shortcut + x.permute(0, 3, 1, 2).contiguous()

        # 第二个残差分支：MLP。
        shortcut = x
        # MLP 在 NHWC token 通道上执行：【B,C,H,W】->【B,H,W,C】。
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.mlp(self.norm2(x))
        # MLP 输出再回到 NCHW：【B,H,W,C】->【B,C,H,W】。
        return shortcut + x.permute(0, 3, 1, 2).contiguous()

    def _shifted_window_mask(self, height: int, width: int, device, dtype) -> torch.Tensor:
        """构造 shifted window 的标准 Swin attention mask。"""
        # img_mask 为每个区域分配编号，用编号差异判断 token 是否可互相注意。
        img_mask = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                # 不同区域写入不同编号。
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1
        # 把编号图切成窗口。
        mask_windows = _window_partition(img_mask, self.window_size).squeeze(-1)
        # 编号不同的位置赋予大负数，使 softmax 后权重接近 0。
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


class LightweightSwinBottleneck(nn.Module):
    """只在最深层 encoder 使用的轻量 Swin bottleneck。

    整体网络仍然是 CNN 为主：
        浅层和中层用 CNN 提取边缘、纹理和局部结构。
        低分辨率 bottleneck 使用窗口注意力捕获更大范围语义。
    这样比全程 Transformer 更省显存，也更符合医学图像分割常用设计。
    """

    def __init__(
        self,
        channels: int,
        depth: int = 2,
        num_heads: int = 4,
        window_size: int = 7,
        disabled: bool = False,
    ) -> None:
        super().__init__()
        # disabled=True 用于消融实验：关闭 Swin bottleneck。
        self.disabled = disabled
        # window_size 至少为 2，避免无意义窗口。
        window_size = max(2, int(window_size))
        # 交替构造普通窗口和 shifted window block。
        self.blocks = nn.ModuleList(
            [
                SwinWindowBlock(
                    channels,
                    window_size=window_size,
                    num_heads=num_heads,
                    shift_size=0 if idx % 2 == 0 else window_size // 2,
                )
                for idx in range(max(1, depth))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 Swin bottleneck；若消融关闭则原样返回。"""
        if self.disabled:
            return x
        # 逐个 block 更新 bottleneck feature。
        for block in self.blocks:
            x = block(x)
        return x


class HybridEncoder25D(nn.Module):
    """CNN encoder + Swin window-attention bottleneck。

    encoder 输出多尺度 skip features，供 U-Net decoder 使用。
    前几级通过 CNN 和 max pooling 下采样，最深层 bottleneck 再执行 Swin window attention。
    """

    def __init__(
        self,
        channels: int,
        transformer_depth: int = 2,
        num_heads: int = 4,
        window_size: int = 7,
        disable_transformer: bool = False,
    ) -> None:
        super().__init__()
        # 第一层下采样后的 CNN stage。
        self.down1 = ConvBlock(channels, channels * 2)
        # 第二层下采样后的 CNN stage。
        self.down2 = ConvBlock(channels * 2, channels * 4)
        # 第三层下采样后的 CNN stage。
        self.down3 = ConvBlock(channels * 4, channels * 8)
        # 最深层 bottleneck 前的 CNN 特征提取。
        self.bottleneck = ConvBlock(channels * 8, channels * 16)
        # Swin bottleneck 负责低分辨率长程语义建模。
        self.transformer = LightweightSwinBottleneck(
            channels * 16,
            depth=transformer_depth,
            num_heads=num_heads,
            window_size=window_size,
            disabled=disable_transformer,
        )
        # 2D max pooling 用于 U-Net 风格下采样。
        self.pool = nn.MaxPool2d(2)

    def forward(self, x0: torch.Tensor) -> List[torch.Tensor]:
        """返回多尺度 skip features：[x0, x1, x2, x3, x4]。"""
        # x0 是 phase attention 融合后的最高分辨率特征：【B,C,H,W】。
        # pool 下采样后进入 stage1：【B,C,H,W】->【B,C,H/2,W/2】->【B,2C,H/2,W/2】。
        x1 = self.down1(self.pool(x0))
        # stage2 继续下采样，扩大感受野：【B,2C,H/2,W/2】->【B,4C,H/4,W/4】。
        x2 = self.down2(self.pool(x1))
        # stage3 是更低分辨率的语义特征：【B,4C,H/4,W/4】->【B,8C,H/8,W/8】。
        x3 = self.down3(self.pool(x2))
        # bottleneck CNN 输入继续下采样：【B,8C,H/8,W/8】->【B,16C,H/16,W/16】。
        x4 = self.bottleneck(self.pool(x3))
        # 在 bottleneck 上执行 Swin window attention，空间尺寸和通道数保持不变：【B,16C,H/16,W/16】->【B,16C,H/16,W/16】。
        x4 = self.transformer(x4)
        # 返回给 U-Net decoder 的 skip 列表：
        # [【B,C,H,W】, 【B,2C,H/2,W/2】, 【B,4C,H/4,W/4】, 【B,8C,H/8,W/8】, 【B,16C,H/16,W/16】]。
        return [x0, x1, x2, x3, x4]


class UncertaintyBoundaryRefinement25D(nn.Module):
    """用不确定性图控制边界细化强度。

    模型先预测 coarse mask。
    若某个像素 coarse probability 接近 0.5，则说明该位置不确定性高。
    不确定性越高，boundary feature 对 decoder feature 的修正越强。
    这样 refinement 主要作用在肿瘤边缘和模糊区域，而不是破坏置信度高的内部区域。
    """

    def __init__(self, channels: int, disabled: bool = False) -> None:
        super().__init__()
        # disabled=True 用于消融：关闭 uncertainty-guided refinement。
        self.disabled = disabled
        # 从 decoder feature 中提取边界相关特征。
        self.boundary_feature = ConvBlock(channels, channels)
        # 生成边界 gate，控制 boundary feature 的空间强弱。
        self.boundary_gate = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, decoder_feature: torch.Tensor, uncertainty_map: torch.Tensor):
        """返回 refined feature 和用于边界监督的 boundary feature。"""
        # decoder_feature 是 decoder 的最高分辨率输出：【B,C,H,W】。
        # 先从 decoder feature 提取边界特征：【B,C,H,W】->【B,C,H,W】。
        boundary_feature = self.boundary_feature(decoder_feature)
        if self.disabled:
            # 消融时不做 refinement，但仍返回 boundary_feature 供后续 head 使用。
            return decoder_feature, boundary_feature
        # uncertainty_map:【B,1,H,W】，boundary_gate(boundary_feature):【B,1,H,W】。
        # 两个 gate 会广播到 C 个通道，对 boundary_feature 做逐像素调制。
        # refined 维度保持不变：【B,C,H,W】+【B,1,H,W】*【B,1,H,W】*【B,C,H,W】->【B,C,H,W】。
        refined = decoder_feature + uncertainty_map * self.boundary_gate(boundary_feature) * boundary_feature
        return refined, boundary_feature


class KPTA25DNet(nn.Module):
    """完整的 Slice-aware 2.5D KPTA-Net。

    方案 D 的完整流程：
        1. 从原始 DCE phase 构造 pseudo-kinetic maps。
        2. kinetic maps 构造完成后再统一归一化输入。
        3. 对每个 slice、每个 phase 使用共享 CNN stem。
        4. 用 slice attention 聚合 z-1/z/z+1。
        5. 用 kinetic feature 引导 pixel-wise phase attention。
        6. 用 CNN encoder + Swin bottleneck 提取语义。
        7. 用 U-Net decoder 恢复分辨率。
        8. 用 coarse mask uncertainty 进行边界细化。
    """

    def __init__(
        self,
        in_phases: int = 17,
        num_slices: int = 3,
        base_channels: int = 32,
        phase_indices: Optional[Dict] = None,
        kinetic_maps: Optional[Sequence[str]] = None,
        normalize_after_kinetic: bool = True,
        normalize_kinetic: bool = True,
        slice_context_mode: str = "attention",
        slice_attention_type: str = "simple",
        csam_aggregate: str = "center",
        csam_uncertainty: bool = False,
        phase_attention_type: str = "pdwa",
        use_enhancement_prior: bool = True,
        use_kinetic_bias: bool = True,
        use_difference_refinement: bool = True,
        residual_mean_weight: float = 0.5,
        lambda_enh_init: float = 1.0,
        gamma_diff_init: float = 0.5,
        phase_attention: bool = True,
        use_uncertainty_head: bool = True,
        use_boundary_head: bool = True,
        use_uncertainty_refinement: bool = True,
        hybrid_encoder: Optional[Dict] = None,
        return_dict: bool = True,
        kinetic_eps: float = 1e-6,
        kinetic_clip_value: Optional[float] = None,
        ablation: Optional[Dict] = None,
        **_: Dict,
    ) -> None:
        super().__init__()
        # ablation 字典来自 YAML/CLI，用于关闭单个创新模块。
        ablation = ablation or {}
        # hybrid_encoder 存放 Swin bottleneck 的深度、head 数和窗口大小。
        hybrid_encoder = hybrid_encoder or {}
        # 记录输入 phase 数，forward 中会显式检查。
        self.in_phases = in_phases
        # 记录输入相邻切片数，默认 K=3。
        self.num_slices = num_slices
        # 控制推理时返回 dict 还是只返回 seg_logits。
        self.return_dict = return_dict
        if slice_attention_type not in {"simple", "csam"}:
            raise ValueError(f"slice_attention_type must be 'simple' or 'csam', got {slice_attention_type}")
        self.slice_attention_type = slice_attention_type
        if phase_attention_type not in {"pixelwise", "pdwa"}:
            raise ValueError(f"phase_attention_type must be 'pixelwise' or 'pdwa', got {phase_attention_type}")
        self.phase_attention_type = phase_attention_type
        self.phase_indices = PhaseIndices.from_config(phase_indices)
        # 以下 disable_* 都是消融实验开关。
        self.disable_kinetic_maps = bool(ablation.get("disable_kinetic_maps", False))
        self.disable_slice_context = bool(ablation.get("disable_slice_context", False)) or slice_context_mode == "none"
        self.disable_phase_attention = bool(ablation.get("disable_pixelwise_phase_attention", False)) or not phase_attention
        self.disable_kinetic_raw_fusion = bool(ablation.get("disable_kinetic_raw_fusion", False))
        self.disable_transformer = bool(ablation.get("disable_transformer_bottleneck", False))
        self.disable_boundary = bool(ablation.get("disable_boundary_head", False)) or not use_boundary_head
        self.disable_uncertainty = bool(ablation.get("disable_uncertainty_head", False)) or not use_uncertainty_head
        self.disable_refinement = bool(ablation.get("disable_uncertainty_refinement", False)) or not use_uncertainty_refinement

        # kinetic_builder 负责从原始输入中构造动力学先验图。
        self.kinetic_builder = KineticMapBuilder25D(
            phase_indices=phase_indices,
            kinetic_maps=[] if self.disable_kinetic_maps else kinetic_maps,
            eps=kinetic_eps,
            clip_value=kinetic_clip_value,
            normalize_after_kinetic=normalize_after_kinetic,
            normalize_kinetic=normalize_kinetic,
        )
        # slice_stem 对每个 slice/phase 共享提取浅层特征。
        self.slice_stem = SliceWiseCNNStem(base_channels)
        # slice_agg 融合相邻切片上下文。simple 保留原像素级 slice attention；
        # csam 使用语义/空间/切片联合注意力增强相邻切片，再取中心或均值聚合。
        if self.slice_attention_type == "csam":
            self.slice_agg = CSAMSliceAggregation(
                num_slices=num_slices,
                channels=base_channels,
                aggregate=csam_aggregate,
                disabled=self.disable_slice_context,
                uncertainty=csam_uncertainty,
            )
        else:
            self.slice_agg = SliceContextAggregation(base_channels, disabled=self.disable_slice_context)
        self.csam_enabled = self.slice_attention_type == "csam" and not self.disable_slice_context
        # kinetic_branch 编码 pseudo-kinetic maps。
        self.kinetic_branch = KineticPriorBranch25D(
            self.kinetic_builder.expected_channels,
            base_channels,
            disabled_slice_context=self.disable_slice_context,
        )
        # phase_attention 负责像素级 DCE phase 融合。pixelwise 保留原注意力；
        # pdwa 使用 phase difference weighting，把 C_sub_t 和 kinetic feature 纳入时相权重。
        if self.phase_attention_type == "pdwa":
            self.phase_attention = PhaseDifferenceWeightingAttention(
                base_channels,
                in_phases,
                use_enhancement_prior=use_enhancement_prior,
                use_kinetic_bias=use_kinetic_bias,
                use_difference_refinement=use_difference_refinement,
                residual_mean_weight=residual_mean_weight,
                lambda_enh_init=lambda_enh_init,
                gamma_diff_init=gamma_diff_init,
                disabled=self.disable_phase_attention,
            )
        else:
            self.phase_attention = PixelWisePhaseAttention25D(
                base_channels,
                in_phases,
                disabled=self.disable_phase_attention,
            )
        self.kinetic_raw_fusion = KineticRawFusion(base_channels, disabled=self.disable_kinetic_raw_fusion)
        # hybrid_encoder 是 CNN encoder + Swin bottleneck。
        self.hybrid_encoder = HybridEncoder25D(
            base_channels,
            transformer_depth=int(hybrid_encoder.get("transformer_depth", 2)),
            num_heads=int(hybrid_encoder.get("num_heads", 4)),
            window_size=int(hybrid_encoder.get("window_size", 7)),
            disable_transformer=self.disable_transformer,
        )
        # decoder 每一级的通道数。
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        # U-Net decoder：从 bottleneck 逐级上采样并融合 skip。
        self.up3 = UpBlock(channels[4], channels[3], channels[3])
        self.up2 = UpBlock(channels[3], channels[2], channels[2])
        self.up1 = UpBlock(channels[2], channels[1], channels[1])
        self.up0 = UpBlock(channels[1], channels[0], channels[0])
        # coarse_head 先产生粗分割，用于计算 uncertainty。
        self.coarse_head = nn.Conv2d(channels[0], 1, kernel_size=1)
        # refinement 使用 uncertainty_map 细化边界区域。
        self.refinement = UncertaintyBoundaryRefinement25D(channels[0], disabled=self.disable_refinement)
        # boundary_head 输出边界辅助监督 logits。
        self.boundary_head = nn.Conv2d(channels[0], 1, kernel_size=1)
        # seg_head 输出最终分割 logits。
        self.seg_head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor, return_dict: Optional[bool] = None):
        """执行一次 2.5D batch 前向传播。

        参数：
            x: [B,K,T,H,W]，例如 [B,3,17,256,256]。
               这里 K=3 是相邻切片数，不是 17；
               T=17 是 DCE phase/channel 数，即 pre + post1..8 + sub1..8。
            return_dict: True 时返回辅助输出；False 时只返回最终 seg_logits。
        """
        # 检查输入必须是 2.5D 格式。
        if x.ndim != 5:
            raise ValueError(f"KPTA25DNet expects [B,K,T,H,W], got {tuple(x.shape)}")
        # 检查 K 是否与配置一致，避免数据构建错误被静默吞掉。
        if x.shape[1] != self.num_slices:
            raise ValueError(f"KPTA25DNet expected {self.num_slices} slices, got {x.shape[1]}")
        # 检查 T 是否与配置一致，避免 phase attention 输入错位。
        if x.shape[2] != self.in_phases:
            raise ValueError(f"KPTA25DNet expected {self.in_phases} phases, got {x.shape[2]}")
        # 如果调用时没有显式指定，就使用模型初始化时的 return_dict。
        return_dict = self.return_dict if return_dict is None else return_dict

        # 先从原始输入构造 kinetic maps：
        # 【B,K,T,H,W】->【B,K,M,H,W】，例如【B,3,17,H,W】->【B,3,M,H,W】。
        kinetic_maps = self.kinetic_builder(x)
        # 再对图像输入做统一归一化，维度不变：【B,K,T,H,W】->【B,K,T,H,W】。
        x_norm = self.kinetic_builder.normalize_input(x)
        # 逐切片逐时相 CNN 编码：【B,K,T,H,W】->【B,K,T,C,H,W】。
        slice_phase_features = self.slice_stem(x_norm)
        # 聚合 K 个相邻切片，只保留 phase 维：【B,K,T,C,H,W】->【B,T,C,H,W】。
        context_features, slice_attention_maps = self.slice_agg(slice_phase_features)
        # 编码 kinetic maps，得到动力学先验特征：【B,K,M,H,W】->【B,C,H,W】。
        kinetic_feature, kinetic_slice_attention = self.kinetic_branch(kinetic_maps)
        # 用 kinetic feature 引导像素级 phase attention：
        # 【B,T,C,H,W】+【B,C,H,W】->【B,C,H,W】。
        sub_maps = self._select_phase_maps(x, self.phase_indices.subtraction)
        fused_phase, phase_attention_maps, phase_debug = self.phase_attention(
            context_features,
            sub_maps=sub_maps,
            kinetic_feat=kinetic_feature,
        )
        # Kinetic prior has two roles: guide phase attention and explicitly
        # enter the encoder through residual raw/kinetic fusion. It augments,
        # rather than replaces, the raw image branch.
        fused0 = self.kinetic_raw_fusion(fused_phase, kinetic_feature)

        # CNN encoder + Swin bottleneck 产生多尺度 skip features：
        # 【B,C,H,W】-> [【B,C,H,W】, 【B,2C,H/2,W/2】, ..., 【B,16C,H/16,W/16】]。
        skips = self.hybrid_encoder(fused0)
        # U-Net decoder 逐级上采样：
        # 【B,16C,H/16,W/16】+skip【B,8C,H/8,W/8】->【B,8C,H/8,W/8】。
        dec = self.up3(skips[4], skips[3])
        # 【B,8C,H/8,W/8】+skip【B,4C,H/4,W/4】->【B,4C,H/4,W/4】。
        dec = self.up2(dec, skips[2])
        # 【B,4C,H/4,W/4】+skip【B,2C,H/2,W/2】->【B,2C,H/2,W/2】。
        dec = self.up1(dec, skips[1])
        # 【B,2C,H/2,W/2】+skip【B,C,H,W】->【B,C,H,W】。
        dec = self.up0(dec, skips[0])

        # coarse logits 用于估计 uncertainty，不直接作为最终输出：【B,C,H,W】->【B,1,H,W】。
        coarse_logits = self.coarse_head(dec)
        # sigmoid 得到 coarse probability：【B,1,H,W】->【B,1,H,W】。
        coarse_prob = torch.sigmoid(coarse_logits.float())
        # p 越接近 0.5，不确定性越高；p 越接近 0/1，不确定性越低，维度保持【B,1,H,W】。
        uncertainty_prob = (1.0 - torch.abs(2.0 * coarse_prob - 1.0)).clamp(1e-3, 1.0 - 1e-3)
        # 转成 logits 便于 uncertainty loss 使用 BCEWithLogitsLoss：【B,1,H,W】->【B,1,H,W】。
        uncertainty_logits = torch.logit(uncertainty_prob).to(dtype=coarse_logits.dtype)
        # refinement 使用概率形式 uncertainty_map：【B,1,H,W】。
        uncertainty_map = uncertainty_prob.to(dtype=dec.dtype)
        if self.disable_uncertainty:
            # 消融 uncertainty head 时，将不确定性置零。
            uncertainty_logits = torch.zeros_like(uncertainty_logits)
            uncertainty_map = torch.zeros_like(uncertainty_map)
        # 使用 uncertainty_map 引导边界细化：【B,C,H,W】+【B,1,H,W】-> refined【B,C,H,W】。
        refined, boundary_feature = self.refinement(dec, uncertainty_map)
        # 预测边界 logits：【B,C,H,W】->【B,1,H,W】。
        boundary_logits = self.boundary_head(boundary_feature)
        if self.disable_boundary:
            # 消融 boundary head 时，输出零 boundary logits。
            boundary_logits = torch.zeros_like(boundary_logits)
        # 最终 segmentation logits：【B,C,H,W】->【B,1,H,W】。
        seg_logits = self.seg_head(refined)

        if not return_dict:
            # 推理时可只返回最终分割 logits。
            return seg_logits
        # 训练/可视化时返回完整字典，loss 可以读取辅助分支。
        return {
            "seg_logits": seg_logits,
            "logits": seg_logits,
            "coarse_logits": coarse_logits,
            "boundary_logits": boundary_logits,
            "uncertainty_logits": uncertainty_logits,
            "uncertainty_map": uncertainty_map,
            "kinetic_maps": kinetic_maps,
            "phase_attention": phase_attention_maps,
            "phase_attention_maps": [phase_attention_maps],
            "attention_maps": [phase_attention_maps],
            "slice_attention_maps": [slice_attention_maps, kinetic_slice_attention],
            "slice_attention_type": self.slice_attention_type,
            "csam_enabled": self.csam_enabled,
            "phase_attention_type": self.phase_attention_type,
            "pdwa_debug": phase_debug if self.phase_attention_type == "pdwa" else {},
            "debug": {
                "num_slices": x.shape[1],
                "num_phases": x.shape[2],
                "num_kinetic_maps": kinetic_maps.shape[2],
                "slice_attention_type": self.slice_attention_type,
                "csam_enabled": self.csam_enabled,
                "phase_attention_type": self.phase_attention_type,
                "sub_maps_shape": None if sub_maps is None else tuple(sub_maps.shape),
                "disable_kinetic_raw_fusion": self.disable_kinetic_raw_fusion,
                "fused_phase_shape": tuple(fused_phase.shape),
                "kinetic_feature_shape": tuple(kinetic_feature.shape),
                "fused0_shape": tuple(fused0.shape),
                "fused_phase_l2": fused_phase.detach().float().norm(dim=1).mean(),
                "kinetic_feature_l2": kinetic_feature.detach().float().norm(dim=1).mean(),
                "fused0_l2": fused0.detach().float().norm(dim=1).mean(),
            },
        }

    @staticmethod
    def _select_phase_maps(x: torch.Tensor, indices) -> Optional[torch.Tensor]:
        """从 2.5D 输入的 T 维读取指定 phase。

        输入 x: [B,K,T,H,W]；返回 [B,K,T_selected,H,W]。如果配置索引全越界，
        返回 None，让 PDWA 自动退化为无 enhancement prior 的稳定路径。
        """
        valid = [idx for idx in indices if 0 <= idx < x.shape[2]]
        if not valid:
            return None
        return x[:, :, valid]
