"""
StarNet3 Neural Network Architecture.

Contains the progressive multi-scale U-Net architecture with Channel Attention Blocks (CAB),
Supervised Attention Modules (SAM), and Original Resolution Block (ORB) full-resolution refiner
for astronomical star removal.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def exists(x):
    return x is not None


class Residual(nn.Module):
    """Residual connection block."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim: int) -> nn.Sequential:
    """Upsample using bilinear interpolation followed by 3x3 convolution."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(dim, dim, kernel_size=3, padding=1),
    )


def Downsample(dim: int) -> nn.Conv2d:
    """Downsample using strided 4x4 convolution with padding 1."""
    return nn.Conv2d(dim, dim, 4, 2, 1)


class Block(nn.Module):
    """Convolutional block with GroupNorm and SiLU activation."""

    def __init__(self, dim_in: int, dim_out: int, groups: int = 8):
        super().__init__()
        g = groups
        while dim_out % g != 0 and g > 1:
            g -= 1
        self.proj = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(g, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, scale_shift=None) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    """Two-block residual unit with projection shortcut if channel count changes."""

    def __init__(self, dim_in: int, dim_out: int, *, groups: int = 8):
        super().__init__()
        self.block1 = Block(dim_in, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class PreNorm(nn.Module):
    """Pre-normalization using GroupNorm."""

    def __init__(self, dim: int, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(self.norm(x))


class Attention(nn.Module):
    """Multi-head self-attention operating over 2D feature maps."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        hidden = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        hidden = qkv[0].shape[1]
        d = hidden // self.heads

        q, k, v = [
            t.view(b, self.heads, d, h * w).permute(0, 1, 3, 2)
            for t in qkv
        ]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(b, hidden, h, w)
        return self.to_out(out)


class CAB(nn.Module):
    """Channel Attention Block with learnable PReLU activations."""

    def __init__(self, n_feat: int, kernel_size: int = 3, reduction: int = 4, bias: bool = False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=bias),
        )

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feat, n_feat // reduction, 1, padding=0, bias=bias),
            nn.PReLU(),
            nn.Conv2d(n_feat // reduction, n_feat, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        res = self.body(x)
        res = res * self.ca(res)
        return res + x


class ORB(nn.Module):
    """Original Resolution Block (MPRNet) composed of cascaded CABs."""

    def __init__(self, n_feat: int, kernel_size: int = 3, reduction: int = 4, bias: bool = False, num_cab: int = 2):
        super().__init__()
        modules_body = [CAB(n_feat, kernel_size, reduction, bias) for _ in range(num_cab)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2, bias=bias))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x) + x


class GaussianBlur2d(nn.Module):
    """Gaussian blur using depthwise separable 1-D convolutions."""

    def __init__(self, kernel_size: int, sigma: float):
        super().__init__()
        self.kernel_size = kernel_size
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        k1d = torch.exp(-0.5 * (x / sigma) ** 2)
        k1d = (k1d / k1d.sum()).unsqueeze(0)  # [1, kernel_size]
        self.register_buffer("kernel", k1d, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        pad = self.kernel_size // 2
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        c = x.shape[1]
        kh = self.kernel.unsqueeze(1).unsqueeze(2).repeat(c, 1, 1, 1)  # [c, 1, 1, ks]
        kv = self.kernel.unsqueeze(1).unsqueeze(3).repeat(c, 1, 1, 1)  # [c, 1, ks, 1]
        out = F.conv2d(padded, kh, groups=c)
        out = F.conv2d(out, kv, groups=c)
        return out


class Adapter(nn.Module):
    """Conv-PReLU-Conv feature adapter."""

    def __init__(self, in_ch: int, hidden_dim: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        p = kernel_size // 2
        self.conv_in = nn.Conv2d(in_ch, hidden_dim, kernel_size, padding=p)
        self.act = nn.PReLU()
        self.conv_out = nn.Conv2d(hidden_dim, out_ch, kernel_size, padding=p)

        for conv in (self.conv_in, self.conv_out):
            nn.init.normal_(conv.weight, std=0.02)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_out(self.act(self.conv_in(x)))


class ImageSAM(nn.Module):
    """Supervised Attention Module for cross-stage feature conditioning."""

    def __init__(
        self,
        image_channels: int = 3,
        hidden_dim: int = 32,
        kernel_size: int = 3,
        bias: bool = False,
    ):
        super().__init__()
        p = kernel_size // 2
        self.img_to_feat = nn.Conv2d(image_channels, hidden_dim, kernel_size, padding=p, bias=bias)
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=p, bias=bias)
        self.conv2 = nn.Conv2d(hidden_dim, image_channels, kernel_size, padding=p, bias=bias)
        self.conv3 = nn.Conv2d(image_channels, hidden_dim, kernel_size, padding=p, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.normal_(self.conv2.weight, std=0.025)
            if self.conv2.bias is not None:
                nn.init.zeros_(self.conv2.bias)
            nn.init.normal_(self.conv3.weight, std=0.025)
            if self.conv3.bias is not None:
                nn.init.zeros_(self.conv3.bias)
            nn.init.normal_(self.img_to_feat.weight, std=0.025)
            nn.init.normal_(self.conv1.weight, std=0.025)

    def forward(self, stage_out: Tensor, x_img: Tensor) -> Tuple[Tensor, Tensor]:
        feat = self.img_to_feat(stage_out)
        x1 = self.conv1(feat)
        img = self.conv2(feat) + x_img
        gate = torch.sigmoid(self.conv3(img))
        x1 = x1 * gate + feat
        return x1, img


class BranchProjection(nn.Module):
    """Project one source branch to a uniform width with learned scalar gating."""

    def __init__(self, in_ch: int, out_ch: int, gate_init: float = 0.0):
        super().__init__()
        num_groups = min(4, out_ch)
        while out_ch % num_groups != 0:
            num_groups -= 1
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(),
        )
        self.gate_logit = nn.Parameter(torch.tensor(gate_init))

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x) * self.gate_logit.sigmoid()


class ResidualFusionBlock(nn.Module):
    """Conv-Norm-SiLU-Conv residual block for cross-source feature fusion."""

    def __init__(self, channels: int):
        super().__init__()
        num_groups = min(8, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class FullResORBRefiner(nn.Module):
    """Full-resolution ORB-based refinement head with typed fusion."""

    def __init__(
        self,
        branch_specs: List[Tuple[int, float]],
        image_channels: int = 3,
        n_feat: int = 64,
        branch_width: int = 8,
        num_fusion_blocks: int = 2,
        num_orb: int = 4,
        orb_num_cab: int = 2,
        reduction: int = 4,
        output_init_std: float = 0.1,
        output_activation: str = "none",
        mode: str = "residual",
        output_residual_scale: float = 1.0,
    ):
        super().__init__()
        assert mode in ("direct", "residual"), f"Invalid refiner mode: {mode}"
        self.output_activation = output_activation
        self.mode = mode
        self.output_residual_scale = output_residual_scale

        self.branch_projs = nn.ModuleList(
            [BranchProjection(in_ch, branch_width, gate_init=gate_init) for in_ch, gate_init in branch_specs]
        )

        fused_ch = len(branch_specs) * branch_width
        self.fusion_blocks = nn.Sequential(*[ResidualFusionBlock(fused_ch) for _ in range(num_fusion_blocks)])
        self.fuse_to_feat = nn.Conv2d(fused_ch, n_feat, 1)

        self.orbs = nn.Sequential(
            *[
                ORB(
                    n_feat=n_feat,
                    kernel_size=3,
                    reduction=reduction,
                    bias=False,
                    num_cab=orb_num_cab,
                )
                for _ in range(num_orb)
            ]
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(n_feat, image_channels, kernel_size=3, padding=1),
        )

        final_conv = self.out_proj[-1]
        if output_init_std == 0:
            nn.init.zeros_(final_conv.weight)
        else:
            nn.init.normal_(final_conv.weight, std=output_init_std)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    def _apply_output_activation(self, x: Tensor) -> Tensor:
        if self.output_activation == "tanh":
            return torch.tanh(x)
        if self.output_activation == "clamp":
            return x.clamp(-1, 1)
        return x

    def forward(self, branches: List[Tensor], inp: Optional[Tensor] = None) -> Tensor:
        if self.mode == "residual" and inp is None:
            raise ValueError("inp is required for residual refiner mode")

        projected = [proj(b) for proj, b in zip(self.branch_projs, branches)]
        x = torch.cat(projected, dim=1)
        x = self.fusion_blocks(x)
        x = self.fuse_to_feat(x)
        x = self.orbs(x)
        out = self._apply_output_activation(self.out_proj(x))

        if self.mode == "residual":
            out = inp + out * self.output_residual_scale

        return out


class UNetWithCABSkips(nn.Module):
    """UNet with Channel Attention Blocks (CAB) applied to encoder skip connections."""

    def __init__(
        self,
        dim: int,
        num_inputs: int = 1,
        num_outputs: int = 1,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 3,
        resnet_block_groups: int = 8,
        cab_reduction: int = 4,
        input_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.in_channels = input_channels if input_channels is not None else channels * num_inputs
        self.out_channels = channels * num_outputs

        init_dim = dim // 3 * 2
        self.init_conv = nn.Conv2d(self.in_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block = partial(ResnetBlock, groups=resnet_block_groups)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.skip_cabs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            module = nn.ModuleList(
                [
                    block(dim_in, dim_out),
                    block(dim_out, dim_out),
                    Downsample(dim_out) if not is_last else nn.Identity(),
                ]
            )
            self.downs.append(module)

            if not is_last:
                cab = CAB(n_feat=dim_out, kernel_size=3, reduction=cab_reduction, bias=False)
                self._init_cab_small(cab)
                self.skip_cabs.append(cab)

        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            module = nn.ModuleList(
                [
                    block(dim_out * 2, dim_in),
                    block(dim_in, dim_in),
                    Upsample(dim_in) if not is_last else nn.Identity(),
                ]
            )
            self.ups.append(module)

        self.final_conv = nn.Sequential(block(dim, dim), nn.Conv2d(dim, self.out_channels, 1))

    def _init_cab_small(self, cab: CAB):
        with torch.no_grad():
            for module in cab.body.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            for module in cab.ca.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, *inputs: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        inp = torch.cat(inputs, dim=1)
        x = self.init_conv(inp)
        h = []

        for block1, block2, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for idx, (block1, block2, upsample) in enumerate(self.ups):
            skip_feat = h.pop()
            if idx > 0:
                cab_idx = len(self.skip_cabs) - idx
                skip_feat = self.skip_cabs[cab_idx](skip_feat)
            x = torch.cat((x, skip_feat), dim=1)
            x = block1(x)
            x = block2(x)
            x = upsample(x)

        x = self.final_conv(x)
        return x


class SimpleImageUNetWithCABSkipsAndRef(nn.Module):
    """UNet with CAB skips and explicit mask reference."""

    def __init__(
        self,
        dim: int,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        image_channels: int = 3,
        resnet_block_groups: int = 8,
        cab_reduction: int = 4,
        input_channels: Optional[int] = None,
        output_init_std: float = 0.1,
        output_activation: str = "none",
        output_mode: str = "direct",
        output_residual_scale: float = 1.0,
    ):
        super().__init__()
        self.image_channels = image_channels
        self.output_activation = output_activation
        self.output_mode = output_mode
        self.output_residual_scale = output_residual_scale

        self.unet = UNetWithCABSkips(
            dim=dim,
            dim_mults=dim_mults,
            channels=image_channels,
            resnet_block_groups=resnet_block_groups,
            cab_reduction=cab_reduction,
            input_channels=input_channels,
        )

        final_conv = self.unet.final_conv[1]
        if isinstance(final_conv, nn.Conv2d):
            if output_init_std == 0:
                nn.init.zeros_(final_conv.weight)
            else:
                nn.init.normal_(final_conv.weight, std=output_init_std)
            if final_conv.bias is not None:
                nn.init.zeros_(final_conv.bias)

    def _apply_output_activation(self, x: Tensor) -> Tensor:
        if self.output_activation == "tanh":
            return torch.tanh(x)
        if self.output_activation == "clamp":
            return x.clamp(-1, 1)
        return x

    def forward(self, inp: Tensor, *, mask_ref: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if mask_ref is None:
            mask_ref = inp

        render = self._apply_output_activation(self.unet(inp))
        if self.output_mode == "residual":
            render = mask_ref + render * self.output_residual_scale

        mask = (mask_ref - render).abs().mean(dim=1, keepdim=True).clamp(0, 1)
        return render, mask


class ProgressiveMultiScaleUNetWithRefiner(nn.Module):
    """
    Progressive multi-scale UNet with CAB skip connections, SAM cross-stage supervision,
    and full-resolution ORB refinement for StarNet3 star removal.

    Parameters
    ----------
    dim : int, default 24
        Base channel dimension.
    dim_mults : tuple of int, default (1, 2, 4, 8, 16)
        Channel multipliers for UNet depth.
    image_channels : int, default 3
        Number of image channels.
    resnet_block_groups : int, default 8
        GroupNorm groups for Resnet blocks.
    unet_scales : tuple of int, default (2, 16)
        Downsampling scale factors per stage.
    scale_progression : str, default 'down'
        Progression direction: 'down' (fine-to-coarse) or 'up' (coarse-to-fine).
    reuse_unet_weights : bool, default True
        Whether stages share weights.
    refiner_n_feat : int, default 64
        Feature dimension for ORB refiner.
    refiner_num_orb : int, default 4
        Number of ORB blocks in refiner.
    refiner_orb_num_cab : int, default 2
        Number of CAB blocks per ORB.
    refiner_reduction : int, default 4
        Channel attention reduction ratio.
    refiner_branch_width : int, default 8
        Per-source branch projection channels.
    refiner_num_fusion_blocks : int, default 2
        Number of post-concat residual fusion blocks.
    refiner_output_init_std : float, default 0.1
        Initialization standard deviation for refiner head.
    adapter_hidden_dim : int, default 32
        Hidden dimension in input/guide adapters.
    cab_reduction : int, default 4
        Reduction ratio in CAB skip connections.
    sam_hidden_dim : int, default 32
        Hidden dimension in SAM modules.
    refiner_output_activation : str, default 'none'
        Activation for refiner output ('tanh', 'clamp', 'none').
    refiner_mode : str, default 'residual'
        Refiner operation mode ('direct' or 'residual').
    refiner_output_residual_scale : float, default 1.0
        Multiplier for residual addition.
    stage_output_init_std : float, default 0.1
        Initialization standard deviation for stage outputs.
    stage_output_activation : str, default 'none'
        Activation for stage outputs.
    stage_output_mode : str, default 'direct'
        Output mode for stages ('direct' or 'residual').
    stage_output_residual_scale : float, default 1.0
        Stage residual scale.
    stage_input_channels : int or None, default 8
        Input channels to stage UNets (after adapter).
    use_pos_guide : bool, default False
        Whether model accepts positive star guide mask.
    use_neg_guide : bool, default False
        Whether model accepts negative guide mask.
    guide_blur_sigma : float, default 10.0
        Sigma for Gaussian blur on guide masks.
    internal_normalize : bool, default True
        Whether model maps [0, 1] inputs to [-1, 1] internally.
    """

    def __init__(
        self,
        dim: int = 24,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8, 16),
        image_channels: int = 3,
        resnet_block_groups: int = 8,
        unet_scales: Tuple[int, ...] = (2, 16),
        scale_progression: str = "down",
        reuse_unet_weights: bool = True,
        refiner_n_feat: int = 64,
        refiner_num_orb: int = 4,
        refiner_orb_num_cab: int = 2,
        refiner_reduction: int = 4,
        refiner_branch_width: int = 8,
        refiner_num_fusion_blocks: int = 2,
        refiner_output_init_std: float = 0.1,
        adapter_hidden_dim: int = 32,
        cab_reduction: int = 4,
        sam_hidden_dim: int = 32,
        refiner_output_activation: str = "none",
        refiner_mode: str = "residual",
        refiner_output_residual_scale: float = 1.0,
        stage_output_init_std: float = 0.1,
        stage_output_activation: str = "none",
        stage_output_mode: str = "direct",
        stage_output_residual_scale: float = 1.0,
        stage_input_channels: Optional[int] = 8,
        use_pos_guide: bool = False,
        use_neg_guide: bool = False,
        guide_blur_sigma: float = 10.0,
        internal_normalize: bool = True,
    ):
        super().__init__()

        for s in unet_scales:
            if s < 1 or (s & (s - 1)) != 0:
                raise ValueError(f"unet_scales must be powers of 2, got {s}")

        if scale_progression not in {"down", "up"}:
            raise ValueError(f"scale_progression must be 'down' or 'up', got {scale_progression!r}")

        self.unet_scales = sorted(set(unet_scales), reverse=(scale_progression == "up"))
        self.num_stages = len(self.unet_scales)

        self.image_channels = image_channels
        self.stage_input_channels = stage_input_channels if stage_input_channels is not None else image_channels
        self.reuse_unet_weights = reuse_unet_weights
        self.sam_hidden_dim = sam_hidden_dim
        self.internal_normalize = internal_normalize

        self.use_pos_guide = use_pos_guide
        self.use_neg_guide = use_neg_guide
        self.guide_channels = int(use_pos_guide) + int(use_neg_guide)
        self.has_guides = self.guide_channels > 0

        unet_kwargs = dict(
            dim=dim,
            dim_mults=dim_mults,
            image_channels=image_channels,
            resnet_block_groups=resnet_block_groups,
            cab_reduction=cab_reduction,
            output_init_std=stage_output_init_std,
            output_activation=stage_output_activation,
            output_mode=stage_output_mode,
            output_residual_scale=stage_output_residual_scale,
            input_channels=self.stage_input_channels if self.stage_input_channels != image_channels else None,
        )

        if reuse_unet_weights:
            self.shared_unet = SimpleImageUNetWithCABSkipsAndRef(**unet_kwargs)
        else:
            self.stage_nets = nn.ModuleList(
                [SimpleImageUNetWithCABSkipsAndRef(**unet_kwargs) for _ in range(self.num_stages)]
            )

        self.sam_modules = nn.ModuleList(
            [
                ImageSAM(image_channels=image_channels, hidden_dim=sam_hidden_dim)
                for _ in range(self.num_stages - 1)
            ]
        )

        if self.has_guides:
            guide_adapter_dim = adapter_hidden_dim
            ks = int(6 * guide_blur_sigma + 1) | 1
            self.guide_blur = GaussianBlur2d(kernel_size=ks, sigma=guide_blur_sigma)
            self.guide_adapters = nn.ModuleList(
                [
                    Adapter(self.guide_channels, guide_adapter_dim, guide_adapter_dim, kernel_size=3)
                    for _ in range(self.num_stages)
                ]
            )
            self.guide_adapter_dim = guide_adapter_dim
        else:
            guide_adapter_dim = 0

        first_in_ch = 2 * image_channels + guide_adapter_dim
        other_in_ch = 2 * image_channels + sam_hidden_dim + image_channels + guide_adapter_dim
        sin_ch = self.stage_input_channels

        self.first_stage_adapter = Adapter(first_in_ch, adapter_hidden_dim, sin_ch, 3)
        self.input_adapters = nn.ModuleList(
            [Adapter(other_in_ch, adapter_hidden_dim, sin_ch, 3) for _ in range(self.num_stages - 1)]
        )

        branch_specs: List[Tuple[int, float]] = [
            (image_channels, 0.0),
            (self.num_stages * image_channels, 0.0),
            (self.num_stages * 1, -2.0),
        ]
        if self.num_stages > 1:
            branch_specs.append(((self.num_stages - 1) * image_channels, -1.0))
        if self.has_guides:
            branch_specs.append((guide_adapter_dim, -1.0))

        self.fullres_refiner = FullResORBRefiner(
            branch_specs=branch_specs,
            image_channels=image_channels,
            n_feat=refiner_n_feat,
            branch_width=refiner_branch_width,
            num_fusion_blocks=refiner_num_fusion_blocks,
            num_orb=refiner_num_orb,
            orb_num_cab=refiner_orb_num_cab,
            reduction=refiner_reduction,
            output_init_std=refiner_output_init_std,
            output_activation=refiner_output_activation,
            mode=refiner_mode,
            output_residual_scale=refiner_output_residual_scale,
        )

    def _prepare_guides(
        self,
        pos_guide: Optional[Tensor],
        neg_guide: Optional[Tensor],
    ) -> Optional[Tensor]:
        if not self.has_guides:
            return None

        parts: List[Tensor] = []
        if self.use_pos_guide:
            if pos_guide is None:
                raise ValueError("pos_guide was not provided, but model was instantiated with use_pos_guide=True")
            parts.append(pos_guide)
        if self.use_neg_guide:
            if neg_guide is None:
                raise ValueError("neg_guide was not provided, but model was instantiated with use_neg_guide=True")
            parts.append(neg_guide)

        guides = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.guide_blur(guides)

    @staticmethod
    def _resize_bilinear(x: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _downsample_area(x: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="area")

    @staticmethod
    def _resize(x: Tensor, size: Tuple[int, int]) -> Tensor:
        if size[0] >= x.shape[-2] and size[1] >= x.shape[-1]:
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return F.interpolate(x, size=size, mode="area")

    def _downsample_multi(self, inp: Tensor, target_size: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
        area = self._downsample_area(inp, target_size)
        bilinear = self._resize_bilinear(inp, target_size)
        return area, torch.cat([area, bilinear], dim=1)

    def forward(
        self,
        inp: Tensor,
        pos_guide: Optional[Tensor] = None,
        neg_guide: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tuple[Tensor, Tensor, Tensor]]]:
        """
        Forward pass with optional guide masks.

        Parameters
        ----------
        inp : Tensor
            Input image [B, C, H, W] in [0, 1].
        pos_guide : Tensor, optional
            Positive star guide mask [B, 1, H, W] in [0, 1].
        neg_guide : Tensor, optional
            Negative protected guide mask [B, 1, H, W] in [0, 1].

        Returns
        -------
        output : Tensor
            Starless image [B, C, H, W].
        mask : Tensor
            Star removal mask [B, 1, H, W].
        scales : list
            Multi-scale intermediate outputs.
        """
        if self.has_guides and self.use_pos_guide and pos_guide is None:
            # Auto-fill zero guide if expected but omitted
            pos_guide = torch.zeros(
                (inp.shape[0], 1, inp.shape[2], inp.shape[3]),
                dtype=inp.dtype,
                device=inp.device,
            )
        if self.has_guides and self.use_neg_guide and neg_guide is None:
            neg_guide = torch.zeros(
                (inp.shape[0], 1, inp.shape[2], inp.shape[3]),
                dtype=inp.dtype,
                device=inp.device,
            )

        if self.internal_normalize:
            inp = inp * 2 - 1

        _, _, H, W = inp.shape
        full_res = (H, W)

        guides = self._prepare_guides(pos_guide, neg_guide)

        scales_out: List[Tuple[Tensor, Tensor, Tensor]] = []
        sam_imgs: List[Tensor] = []

        prev_stage_out: Tensor = torch.empty(0, device=inp.device)
        prev_sam_feats: Tensor = torch.empty(0, device=inp.device)

        for stage_idx, scale_factor in enumerate(self.unet_scales):
            target_size = (H // scale_factor, W // scale_factor)
            ref_img, inp_dual = self._downsample_multi(inp, target_size)

            parts: List[Tensor] = [inp_dual]
            if stage_idx > 0:
                parts.append(self._resize(prev_sam_feats, target_size))
                parts.append(self._resize(prev_stage_out, target_size))
            if guides is not None:
                parts.append(self.guide_adapters[stage_idx](self._downsample_area(guides, target_size)))

            adapter = self.first_stage_adapter if stage_idx == 0 else self.input_adapters[stage_idx - 1]
            stage_in = adapter(torch.cat(parts, dim=1))

            net = self.shared_unet if self.reuse_unet_weights else self.stage_nets[stage_idx]
            stage_out, stage_mask = net(stage_in, mask_ref=ref_img)
            scales_out.append((ref_img, stage_out, stage_mask))

            if stage_idx < self.num_stages - 1:
                prev_sam_feats, sam_img = self.sam_modules[stage_idx](stage_out, ref_img)
                sam_imgs.append(sam_img)

            prev_stage_out = stage_out

        branches: List[Tensor] = [
            inp,
            torch.cat([self._resize_bilinear(out, full_res) for _, out, _ in scales_out], dim=1),
            torch.cat([self._resize_bilinear(mask, full_res) for _, _, mask in scales_out], dim=1),
        ]
        if self.num_stages > 1:
            branches.append(torch.cat([self._resize_bilinear(img, full_res) for img in sam_imgs], dim=1))
        if guides is not None:
            branches.append(self._resize_bilinear(self.guide_adapters[0](guides), full_res))

        final_out = self.fullres_refiner(branches, inp=inp)
        final_mask = (final_out - inp).abs().mean(dim=1, keepdim=True).clamp(0, 1)

        if self.internal_normalize:
            final_out = (final_out + 1) / 2
            scales_out = [((ref + 1) / 2, (out + 1) / 2, mask) for ref, out, mask in scales_out]

        return final_out, final_mask, scales_out


StarNet3Model = ProgressiveMultiScaleUNetWithRefiner
