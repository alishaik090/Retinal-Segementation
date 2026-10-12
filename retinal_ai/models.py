"""Torch models used by the reproducible training and inference paths."""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights, efficientnet_b0, resnet18

from .config import ARTIFACTS_DIR
from .swin_unet import SwinUNet


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class VesselUNet(nn.Module):
    """Compact U-Net suited to the small DRIVE training partition."""

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(3, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(b * 8, b * 16)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = ConvBlock(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ConvBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class ResNet18UNet(nn.Module):
    """U-Net decoder on a ResNet-18 encoder; useful for small labelled sets."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.out = nn.Conv2d(32, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        s = self.stem(x)                    # 1/2, 64 channels
        e1 = self.layer1(self.maxpool(s))   # 1/4, 64 channels
        e2 = self.layer2(e1)                # 1/8, 128 channels
        e3 = self.layer3(e2)                # 1/16, 256 channels
        e4 = self.layer4(e3)                # 1/32, 512 channels
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s], dim=1))
        return self.out(self.up0(d1))


def build_segmenter(base_channels: int = 24, architecture: str = "unet", pretrained: bool = False, **kwargs) -> nn.Module:
    if architecture == "unet":
        return VesselUNet(base_channels=base_channels)
    if architecture == "resnet18_unet":
        # Keep public pretrained weights inside the project rather than a
        # user-profile cache, which also makes the project portable.
        torch.hub.set_dir(str(ARTIFACTS_DIR / "hub"))
        return ResNet18UNet(pretrained=pretrained)
    if architecture == "swin_unet":
        return SwinUNet(
            in_channels=int(kwargs.get("in_channels", 3)),
            embed_dim=int(kwargs.get("embed_dim", base_channels)),
            depths=tuple(kwargs.get("depths", (1, 1, 2, 1))),
            num_heads=tuple(kwargs.get("num_heads", (3, 6, 12, 24))),
            window_size=int(kwargs.get("window_size", 4)),
            patch_size=int(kwargs.get("patch_size", 4)),
        )
    raise ValueError(f"Unknown segmentation architecture: {architecture}")


def build_classifier(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    """EfficientNet-B0 with a disease-classification head.

    Pretrained ImageNet weights are optional so this project can run in an
    offline environment. When enabled torchvision retrieves or uses its cache.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
