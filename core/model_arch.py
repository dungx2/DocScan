"""
model_arch.py  (FIXED — v2)
-------------
Fixes vs original:

  FIX 2 — Flow head scaled for displacement output.
    BEFORE: flow_head output = absolute normalized coords in [-1, 1].
            Last conv zero-initialized → tanh(0) = 0 → all pixels sampled
            from (0, 0) = center of image. Identity mapping requires spatially
            varying outputs that are very different from zero, making it hard
            to learn.
    AFTER:  flow_head output = displacement from identity, typically small
            values near 0. Zero-init → identity transform = correct start.
            A scale factor (MAX_DISP) controls the maximum allowed displacement
            per pixel (default 0.5 = half the image width/height).

    The DilatedBottleneck padding values are also corrected here so that
    dilation=2 and dilation=4 produce same-size feature maps (same as original
    but with explicit padding values matching the _conv_bn_relu multiplier).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Maximum displacement as a fraction of the image side.
# tanh output × MAX_DISP gives the actual pixel displacement in [-MAX_DISP, MAX_DISP].
# 0.5 = up to half the image width/height, which covers realistic page curves.
MAX_DISP = 0.5


def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3,
                  stride: int = 1, padding: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                  padding=padding * dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _dw_sep_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            _conv_bn_relu(in_ch, out_ch),
            _conv_bn_relu(out_ch, out_ch),
        )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return self.pool(feat), feat


class DilatedBottleneck(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        inner = ch // 3
        self.d1 = _conv_bn_relu(ch, inner, dilation=1, padding=1)
        self.d2 = _conv_bn_relu(ch, inner, dilation=2, padding=1)  # actual pad = 2
        self.d4 = _conv_bn_relu(ch, inner, dilation=4, padding=1)  # actual pad = 4
        self.proj = _conv_bn_relu(inner * 3, ch, kernel=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([self.d1(x), self.d2(x), self.d4(x)], dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            _dw_sep_conv(in_ch // 2 + skip_ch, out_ch),
            _dw_sep_conv(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class LightDewarpNet(nn.Module):
    """
    Lightweight document dewarping U-Net (~12 M params).

    Input:  (B, 3, 448, 448) RGB float32 in [0, 1]
    Output: (B, 2, 448, 448) flow displacement scaled by MAX_DISP

    The output is a displacement map, NOT absolute sampling coordinates.
    At zero init: output = 0 = no displacement = identity transform.
    Use with USE_DISPLACEMENT=True in dewarp.py.
    """

    INPUT_SIZE = (448, 448)

    def __init__(self, base_ch: int = 64):
        super().__init__()

        self.enc1 = EncoderBlock(3,         base_ch)
        self.enc2 = EncoderBlock(base_ch,   base_ch * 2)
        self.enc3 = EncoderBlock(base_ch*2, base_ch * 4)
        self.enc4 = EncoderBlock(base_ch*4, base_ch * 8)

        self.bottleneck = nn.Sequential(
            _conv_bn_relu(base_ch*8, base_ch*8),
            DilatedBottleneck(base_ch*8),
            _conv_bn_relu(base_ch*8, base_ch*8),
        )

        self.dec4 = DecoderBlock(base_ch*8, base_ch*8, base_ch*4)
        self.dec3 = DecoderBlock(base_ch*4, base_ch*4, base_ch*2)
        self.dec2 = DecoderBlock(base_ch*2, base_ch*2, base_ch)
        self.dec1 = DecoderBlock(base_ch,   base_ch,   base_ch//2)

        # FIX 2: output displacement scaled by MAX_DISP
        # tanh → (-1, 1) → multiply by MAX_DISP → (-MAX_DISP, MAX_DISP)
        # Small random init last conv → output starts with small non-zero flow
        self.flow_head = nn.Sequential(
            nn.Conv2d(base_ch//2, base_ch//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch//2, 2, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

        # Small random init last conv → output starts with small non-zero flow
        nn.init.normal_(self.flow_head[-2].weight, mean=0.0, std=0.001)
        nn.init.constant_(self.flow_head[-2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1_pool, x1_skip = self.enc1(x)
        x2_pool, x2_skip = self.enc2(x1_pool)
        x3_pool, x3_skip = self.enc3(x2_pool)
        x4_pool, x4_skip = self.enc4(x3_pool)

        b  = self.bottleneck(x4_pool)
        d4 = self.dec4(b,  x4_skip)
        d3 = self.dec3(d4, x3_skip)
        d2 = self.dec2(d3, x2_skip)
        d1 = self.dec1(d2, x1_skip)

        # FIX 2: scale tanh output so displacement is bounded by MAX_DISP
        return self.flow_head(d1) * MAX_DISP

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_vram_mb(self, batch_size: int = 1) -> float:
        param_mb = self.count_parameters() * 4 / 1024**2
        activation_mb = batch_size * 448 * 448 * 512 * 4 * 6 / 1024**2
        return param_mb + activation_mb


def make_identity_flow(batch_size: int, height: int, width: int,
                       device: torch.device) -> torch.Tensor:
    ys = torch.linspace(-1, 1, height, device=device)
    xs = torch.linspace(-1, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    identity = torch.stack([grid_x, grid_y], dim=0)
    return identity.unsqueeze(0).expand(batch_size, -1, -1, -1)


if __name__ == "__main__":
    model = LightDewarpNet()
    print(f"Parameters : {model.count_parameters() / 1e6:.2f} M")
    print(f"VRAM (B=1) : ~{model.estimate_vram_mb(1):.0f} MB")
    print(f"VRAM (B=4) : ~{model.estimate_vram_mb(4):.0f} MB")
    dummy = torch.randn(1, 3, 448, 448)
    flow  = model(dummy)
    print(f"Output shape : {flow.shape}")
    print(f"Flow range   : [{flow.min():.4f}, {flow.max():.4f}]")
    print(f"(near 0 at init = correct — identity transform)")
