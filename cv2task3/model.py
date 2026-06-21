"""
Hand-written U-Net from scratch. No pre-trained weights.
Architecture:
  Encoder: 4 down-blocks (Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU → MaxPool2x2)
  Bottleneck: Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU
  Decoder: 4 up-blocks (UpConv2x2 → Concat → Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU)
  Output: Conv1x1 → Softmax (handled in loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 → BN → ReLU) × 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Down-sampling: MaxPool2x2 → DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Up-sampling: ConvTranspose2x2 (halves channels, doubles spatial)
       → Concat skip → DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # (N, in_ch//2, H*2, W*2)
        # Handle spatial size mismatch (padding if needed)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)  # Skip connection → (N, in_ch, H, W)
        return self.conv(x)             # → (N, out_ch, H, W)


class OutConv(nn.Module):
    """Final 1×1 convolution to produce logits"""

    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, base_channels=64):
        """
        Args:
            n_channels:  Number of input channels (RGB = 3)
            n_classes:   Number of output classes (foreground/background/boundary = 3)
            base_channels: Number of channels in first layer (doubled each down)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, base_channels)              # 64
        self.down1 = Down(base_channels, base_channels * 2)           # 128
        self.down2 = Down(base_channels * 2, base_channels * 4)       # 256
        self.down3 = Down(base_channels * 4, base_channels * 8)       # 512
        self.down4 = Down(base_channels * 8, base_channels * 16)      # 1024 (bottleneck)

        # Decoder:  Up(in_channels, out_channels)
        #   up conv:  in_channels → in_channels//2
        #   concat:   (in_channels//2 + in_channels//2) → in_channels
        #   double_conv: in_channels → out_channels
        self.up1 = Up(base_channels * 16, base_channels * 8)   # 1024→512
        self.up2 = Up(base_channels * 8, base_channels * 4)    # 512→256
        self.up3 = Up(base_channels * 4, base_channels * 2)    # 256→128
        self.up4 = Up(base_channels * 2, base_channels)        # 128→64

        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)        # 256→256,  64ch
        x2 = self.down1(x1)     # 128→128, 128ch
        x3 = self.down2(x2)     # 64→64,   256ch
        x4 = self.down3(x3)     # 32→32,   512ch
        x5 = self.down4(x4)     # 16→16,  1024ch

        # Decoder
        x = self.up1(x5, x4)    # 32→32,   512ch
        x = self.up2(x, x3)     # 64→64,   256ch
        x = self.up3(x, x2)     # 128→128, 128ch
        x = self.up4(x, x1)     # 256→256,  64ch
        logits = self.outc(x)   # 256×256×3
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = UNet(n_channels=3, n_classes=3)
    x = torch.randn(4, 3, 256, 256)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total params: {count_parameters(model):,}")
    assert out.shape == (4, 3, 256, 256), f"Expected (4, 3, 256, 256), got {out.shape}"
    print("Model check passed ✓")
