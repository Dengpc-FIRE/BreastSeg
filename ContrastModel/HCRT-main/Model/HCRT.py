import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, D, H, W = x.size()  # Adjust for 3D data
        mu = x.mean(1, keepdim=True)  # Mean across channels
        var = (x - mu).pow(2).mean(1, keepdim=True)  # Variance across channels
        y = (x - mu) / (var + eps).sqrt()  # Normalize
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1, 1) * y + bias.view(
            1, C, 1, 1, 1
        )  # Apply weight and bias
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, D, H, W = grad_output.size()  # Adjust for 3D data
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1, 1)  # Gradient with respect to output
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = (
            1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        )  # Gradient with respect to input
        return (
            gx,
            (grad_output * y).sum(dim=4).sum(dim=3).sum(dim=2),
            grad_output.sum(dim=4).sum(dim=3).sum(dim=2),
            None,
        )


class LayerNorm3d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm3d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class MLP1(nn.Module):

    def __init__(self, in_feat, h_feat=None, out_feat=None):
        super().__init__()

        self.fc1 = nn.Conv3d(
            in_channels=in_feat,
            out_channels=h_feat,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(
            in_channels=h_feat,
            out_channels=out_feat,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class RegionAttention(nn.Module):
    def __init__(self, n_feat, num_heads, bias=True):
        super(RegionAttention, self).__init__()
        self.num_heads = num_heads
        self.qkv_conv1x1 = nn.Conv3d(n_feat, n_feat * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(
            n_feat * 3,
            n_feat * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=n_feat * 3,
            bias=bias,
        )

        self.project_out = nn.Conv3d(n_feat, n_feat, kernel_size=1, bias=bias)

    def global_attention(self, q, k, v, mask):
        B, C, D, H, W = q.shape
        mask = repeat(mask, 'b l1 l2 -> b head l1 l2', head=self.num_heads)
        q = rearrange(q, "b (head c) d h w -> b head c (d h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) d h w -> b head c (d h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) d h w -> b head c (d h w)", head=self.num_heads)
        attn = (q @ k.transpose(-2, -1)) * (self.num_heads**-0.5)  # [b head c c]
        attn = attn + mask
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [b head c (d h w)]
        out = rearrange(out, "b head c (d h w) -> b (head c) d h w", d=D, h=H, w=W)

        return out

    def forward(self, x, mask):
        qkv = self.qkv_conv1x1(x)
        qkv = self.qkv_dwconv(qkv)
        q, k, v = qkv.chunk(3, dim=1)
        out = self.global_attention(q, k, v, mask)
        out = self.project_out(out)

        return out


class BasicLayer(nn.Module):
    def __init__(self, n_feat, num_heads, mlp_ratio=2):
        super(BasicLayer, self).__init__()

        self.norm_ra = LayerNorm3d(n_feat)
        self.ra = RegionAttention(n_feat, num_heads)
        self.lambda_ra = nn.Parameter(
            torch.zeros((1, n_feat, 1, 1, 1)), requires_grad=True
        )

        self.norm_ra_mlp = LayerNorm3d(n_feat)
        self.mlp_ra = MLP1(n_feat, int(n_feat * mlp_ratio), n_feat)

    def forward(self, x, mask):
        x_norm = self.norm_ra(x)
        x = x + self.lambda_ra * self.ra(x_norm, mask)
        x = self.mlp_ra(self.norm_ra_mlp(x)) + x

        return x


def conv_3d_NoDown(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm3d(out_dim),
        activation,
    )


class GhostModule3D(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=1.5, dw_size=3, stride=1):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        init_channels = int(init_channels)
        new_channels = int(new_channels)
        init_channels = max(1, init_channels)
        new_channels = max(init_channels, new_channels - new_channels % init_channels)

        self.primary_conv = nn.Sequential(
            nn.Conv3d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False
            ),
            nn.InstanceNorm3d(init_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv3d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.InstanceNorm3d(new_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :, :]

def conv_block_2_3d(in_dim, out_dim, activation):
    ghost_module = GhostModule3D(
        inp=in_dim, oup=out_dim, kernel_size=3, ratio=1.5, dw_size=3, stride=1
    )

    return nn.Sequential(
        ghost_module,
        nn.InstanceNorm3d(out_dim),
        activation,
        nn.Conv3d(
            out_dim, out_dim, kernel_size=3, stride=2, padding=1
        ),
        nn.InstanceNorm3d(out_dim),
        activation,
    )


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1
        ),
        nn.BatchNorm3d(out_dim),
        activation,
    )


def position_aware_correlation(features, value=-1000):
    b, c, d, h, w = features.shape
    device = features.device
    pos_d = torch.linspace(-1, 1, d, device=device).view(1, 1, d, 1, 1).expand(b, 1, d, h, w)
    pos_h = torch.linspace(-1, 1, h, device=device).view(1, 1, 1, h, 1).expand(b, 1, d, h, w)
    pos_w = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, 1, w).expand(b, 1, d, h, w)
    features_with_pos = torch.cat([features, pos_d, pos_h, pos_w], dim=1)
    features = features_with_pos.view(b, c + 3, -1)
    features = F.adaptive_avg_pool1d(features, 32)
    features = F.normalize(features, p=2, dim=1)
    correlation = torch.bmm(features.transpose(1, 2), features)
    mask = torch.sigmoid(correlation / 0.1)
    mask = (1 - mask) * value
    return mask

class HCRT(nn.Module):
    def __init__(
        self,
        inch=1,
        outch=2,
        downlayer=3,
        base_channeel=32,
        imgsize=[128, 128, 128],
        hidden_size=256,
    ):
        super().__init__()
        self.imgsize = imgsize
        self.bottlensize = [i // (2**downlayer) for i in imgsize]
        activation = nn.LeakyReLU(0.2, inplace=True)

        self.proxylayers = nn.ModuleList()
        self.proxylayers.append(conv_block_2_3d(inch, base_channeel, activation))
        self.proxylayers.append(
            conv_block_2_3d(base_channeel, base_channeel * 2, activation)
        )
        self.proxylayers.append(
            conv_block_2_3d(base_channeel * 2, hidden_size, activation)
        )

        self.proxylayers1 = nn.ModuleList()
        self.proxylayers1.append(conv_block_2_3d(inch, base_channeel, activation))
        self.proxylayers1.append(
            conv_block_2_3d(base_channeel, base_channeel * 2, activation)
        )
        self.proxylayers1.append(
            conv_block_2_3d(base_channeel * 2, hidden_size, activation)
        )

        self.trans_conv = conv_3d_NoDown(
            hidden_size, base_channeel * (2 ** (downlayer - 1)), activation
        )

        self.out = nn.Conv3d(base_channeel, outch, 3, 1, 1)
        self.trans_1 = conv_trans_block_3d(hidden_size, base_channeel * 2, activation)
        self.trans_2 = conv_trans_block_3d(base_channeel * 2, base_channeel, activation)
        self.trans_3 = conv_trans_block_3d(base_channeel, base_channeel, activation)

        self.trans = conv_trans_block_3d(128, 64, activation)

        self.block = nn.ModuleList()
        for i in range(4):
            self.block.append(BasicLayer(n_feat=128, num_heads=4))

    def forward(self, x):

        skip1 = self.proxylayers[0](x)
        skip2 = self.proxylayers[1](skip1)
        skip3 = self.proxylayers[2](skip2)

        # -----------------------
        l_skip1 = self.proxylayers1[0](x)
        l_skip2 = self.proxylayers1[1](l_skip1)
        l_skip3 = self.proxylayers1[2](l_skip2)
        mask = position_aware_correlation(l_skip3, -1000)
        # -----------------------

        x2 = self.trans_conv(skip3)
        for i in self.block:
            x2 = i(x2, mask)

        x2 = self.trans(x2)

        x2 = self.trans_2(skip2 + x2) + skip1

        x2 = self.trans_3(x2)
        x2 = self.out(x2)

        return x2


if __name__ == "__main__":
    from thop import profile

    model = HCRT(inch=2, base_channeel=32, imgsize=[48, 128, 128]).to("cuda:0")
    img = torch.randn((1, 2, 48, 128, 128)).to("cuda:0")
    flops, params = profile(model, (img,))
    print("flops: %.2f G, params: %.2f M" % (flops / 1e9, params / 1e6))
