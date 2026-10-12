"""Self-contained Swin-UNet for retinal vessel segmentation.

The module follows the workflow in the supplied Swin-Unet DRIVE notebook:
patch embedding, hierarchical shifted-window encoder, U-shaped decoder with
skip connections, and a single sigmoid-ready output map.  It is implemented
in PyTorch so the same code runs locally and in Google Colab without relying
on the notebook's machine-specific Kaggle paths.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as functional
from torch import nn


def _pair(value: int | Iterable[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    first, second = tuple(value)
    return int(first), int(second)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Convert [B,H,W,C] features into [num_windows*B,window^2,C]."""
    batch, height, width, channels = x.shape
    x = x.reshape(batch, height // window_size, window_size, width // window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int, batch: int) -> torch.Tensor:
    """Restore [B,H,W,C] features from non-overlapping window tokens."""
    channels = windows.shape[-1]
    x = windows.reshape(batch, height // window_size, width // window_size, window_size, window_size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"Embedding dimension {dim} must divide evenly across {num_heads} heads.")
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        coordinates = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        flattened = coordinates.flatten(1)
        relative = flattened[:, :, None] - flattened[:, None, :]
        relative[0] += window_size - 1
        relative[1] += window_size - 1
        relative[0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative.sum(0), persistent=False)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query * self.scale) @ key.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(tokens, tokens, self.num_heads).permute(2, 0, 1)
        attention = attention + bias.unsqueeze(0)
        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.reshape(batch_windows // num_windows, num_windows, self.num_heads, tokens, tokens)
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.reshape(-1, self.num_heads, tokens, tokens)
        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj_drop(self.proj(x))


class SwinBlock(nn.Module):
    """Shifted-window transformer block with padded-window support."""

    def __init__(self, dim: int, num_heads: int, window_size: int, shift: bool, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, num_heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)

    def _attention_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor | None:
        shift_size = self.window_size // 2
        if not self.shift or min(height, width) <= self.window_size:
            return None
        mask = torch.zeros((1, height, width, 1), device=device)
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -shift_size), slice(-shift_size, None))
        index = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                mask[:, h_slice, w_slice, :] = index
                index += 1
        windows = window_partition(mask, self.window_size).squeeze(-1)
        attention_mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return attention_mask.masked_fill(attention_mask != 0, -100.0).masked_fill(attention_mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, original_h, original_w, channels = x.shape
        residual = x
        x = self.norm1(x)
        pad_h = (self.window_size - original_h % self.window_size) % self.window_size
        pad_w = (self.window_size - original_w % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = functional.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
        height, width = x.shape[1:3]
        shift_size = self.window_size // 2 if self.shift and min(height, width) > self.window_size else 0
        if shift_size:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        windows = window_partition(x, self.window_size)
        windows = self.attention(windows, self._attention_mask(height, width, x.device))
        x = window_reverse(windows, self.window_size, height, width, batch)
        if shift_size:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
        x = x[:, :original_h, :original_w, :]
        x = residual + x
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        if height % 2 or width % 2:
            x = functional.pad(x.permute(0, 3, 1, 2), (0, width % 2, 0, height % 2)).permute(0, 2, 3, 1)
        x0, x1 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :]
        x2, x3 = x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = torch.cat((x0, x1, x2, x3), dim=-1)
        return self.reduction(self.norm(x))


class SwinStage(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, window_size: int, downsample: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, heads, window_size, shift=index % 2 == 1) for index in range(depth)])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        return x, self.downsample(x) if self.downsample else x


class PatchExpand(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class SwinUNet(nn.Module):
    """Hierarchical shifted-window U-Net for binary vessel segmentation."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 24, depths: tuple[int, ...] = (1, 1, 2, 1), num_heads: tuple[int, ...] = (3, 6, 12, 24), window_size: int = 4, patch_size: int = 4) -> None:
        super().__init__()
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("This Swin-UNet uses exactly four encoder stages.")
        self.config = {"in_channels": in_channels, "embed_dim": embed_dim, "depths": list(depths), "num_heads": list(num_heads), "window_size": window_size, "patch_size": patch_size}
        self.patch_size = _pair(patch_size)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        dimensions = [embed_dim * (2**index) for index in range(4)]
        self.encoder = nn.ModuleList([SwinStage(dimensions[index], depths[index], num_heads[index], window_size, downsample=index < 3) for index in range(4)])
        self.up3 = PatchExpand(dimensions[3], dimensions[2])
        self.fuse3 = nn.Linear(dimensions[2] * 2, dimensions[2])
        self.dec3 = SwinStage(dimensions[2], max(1, depths[2] // 2), num_heads[2], window_size, downsample=False)
        self.up2 = PatchExpand(dimensions[2], dimensions[1])
        self.fuse2 = nn.Linear(dimensions[1] * 2, dimensions[1])
        self.dec2 = SwinStage(dimensions[1], max(1, depths[1] // 2), num_heads[1], window_size, downsample=False)
        self.up1 = PatchExpand(dimensions[1], dimensions[0])
        self.fuse1 = nn.Linear(dimensions[0] * 2, dimensions[0])
        self.dec1 = SwinStage(dimensions[0], max(1, depths[0] // 2), num_heads[0], window_size, downsample=False)
        self.final_up = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.head = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _align(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if source.shape[1:3] == reference.shape[1:3]:
            return source
        return functional.interpolate(source.permute(0, 3, 1, 2), size=reference.shape[1:3], mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        skips: list[torch.Tensor] = []
        for stage in self.encoder:
            skip, x = stage(x)
            skips.append(skip)
        x = self._align(self.up3(skips[3]), skips[2])
        x, _ = self.dec3(self.fuse3(torch.cat((x, skips[2]), dim=-1)))
        x = self._align(self.up2(x), skips[1])
        x, _ = self.dec2(self.fuse2(torch.cat((x, skips[1]), dim=-1)))
        x = self._align(self.up1(x), skips[0])
        x, _ = self.dec1(self.fuse1(torch.cat((x, skips[0]), dim=-1)))
        x = self.head(self.final_up(x.permute(0, 3, 1, 2)))
        return functional.interpolate(x, size=original_size, mode="bilinear", align_corners=False)
