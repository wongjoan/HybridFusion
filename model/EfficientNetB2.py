import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# ==============================================================================
# 1. MODULES (Fusion & Edge — unchanged from original)
# ==============================================================================

class Conv(nn.Module):
    """Standard Conv-BN-SiLU"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        pad = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, pad, groups=g, dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.channel_sig  = nn.Sigmoid()
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sig  = nn.Sigmoid()

    def forward(self, x):
        x = x * self.channel_sig(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s = torch.max(x,  dim=1, keepdim=True)[0]
        return x * self.spatial_sig(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))


class CrossModalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key   = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels,      1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.pool  = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, rgb_feat, ir_feat):
        if rgb_feat.shape[2:] != ir_feat.shape[2:]:
            ir_feat = F.interpolate(ir_feat, size=rgb_feat.shape[2:],
                                    mode='bilinear', align_corners=False)
        B, C, H, W = rgb_feat.shape
        q    = self.query(rgb_feat).view(B, -1, H * W).permute(0, 2, 1)
        ir_p = self.pool(ir_feat)
        k    = self.key(ir_p).view(B, -1, (H // 2) * (W // 2))
        v    = self.value(ir_p).view(B, C, (H // 2) * (W // 2))
        att  = F.softmax(torch.bmm(q, k) / (C // 8) ** 0.5, dim=-1)
        out  = rgb_feat + self.gamma * torch.bmm(v, att.permute(0, 2, 1)).view(B, C, H, W)
        return out, ir_feat


class AddFusionBlock(nn.Module):
    def __init__(self, c_rgb, c_ir, c_out):
        super().__init__()
        self.proj_rgb  = Conv(c_rgb, c_out, 1)
        self.proj_ir   = Conv(c_ir,  c_out, 1)
        self.cross_att = CrossModalAttention(c_out)
        self.cbam      = CBAM(c_out)
        self.refine    = Conv(c_out, c_out, 3)

    def forward(self, rgb, ir):
        if rgb.shape[2:] != ir.shape[2:]:
            ir = F.interpolate(ir, size=rgb.shape[2:], mode='bilinear', align_corners=False)
        rgb_att, ir_att = self.cross_att(self.proj_rgb(rgb), self.proj_ir(ir))
        fused = rgb_att + ir_att
        return self.refine(self.cbam(fused)) + fused


class EdgeEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encode(x)


# ==============================================================================
# 2. EFFICIENTNET-B2 BACKBONES (torchvision, pretrained ImageNet)
#
#    Verified spatial profile at 640px input:
#      features[0]: 32ch  320×320  (stem)
#      features[1]: 16ch  320×320
#      features[2]: 24ch  160×160
#      features[3]: 48ch   80×80   ← P3 (stride-8)
#      features[4]: 88ch   40×40
#      features[5]:120ch   40×40   ← P4 (stride-16)
#      features[6]:208ch   20×20
#      features[7]:352ch   20×20   ← P5 (stride-32)
#      features[8]:1408ch  20×20   ← classifier stem, NOT used
#
#    RGB stream: full B2 (width_mult=1.0) — ImageNet pretrained
#    IR  stream: same B2 structure, stem adapted to 1-channel input
#                by summing pretrained RGB weights (preserves learned filters)
#                IR width multiplier defaults to 0.75 (same as original design)
#                but uses B2 MBConv stages, not B0.
# ==============================================================================

class EfficientNetB2Backbone(nn.Module):
    """
    EfficientNet-B2 feature extractor.
    Extracts three pyramid levels at stride 8 / 16 / 32.
    in_channels=3 for RGB (full pretrained), in_channels=1 for IR (adapted).
    """
    def __init__(self, in_channels=3):
        super().__init__()
        base = torchvision.models.efficientnet_b2(weights='DEFAULT')

        # 1-channel adaptation for IR stream
        if in_channels == 1:
            orig = base.features[0][0]
            new_stem = nn.Conv2d(1, 32, 3, 2, 1, bias=False)
            with torch.no_grad():
                new_stem.weight[:] = orig.weight.sum(dim=1, keepdim=True)
            base.features[0][0] = new_stem

        # Slice out only the feature extraction stages (drop classifier stem)
        self.stem     = base.features[0:3]   # 24ch  @ 160×160
        self.layer_p3 = base.features[3]     # 48ch  @  80×80
        self.layer_p4 = base.features[4:6]   # 120ch @  40×40
        self.layer_p5 = base.features[6:8]   # 352ch @  20×20

    def forward(self, x):
        x  = self.stem(x)
        p3 = self.layer_p3(x)
        p4 = self.layer_p4(p3)
        p5 = self.layer_p5(p4)
        return p3, p4, p5
        # Output channels: p3=48, p4=120, p5=352


# ==============================================================================
# 3. SEMANTIC HEAD
#    Channel dims updated for B2:
#      RGB: P3=48,  P4=120, P5=352
#      IR:  P3=48,  P4=120, P5=352  (same structure, 1-ch adapted)
# ==============================================================================

class SemanticHead(nn.Module):
    def __init__(self, num_classes=1,
                 rgb_channels=(48, 120, 352),
                 ir_channels=(48, 120, 352)):
        super().__init__()
        self.edge_rgb = EdgeEncoder(3, 64)
        self.edge_ir  = EdgeEncoder(1, 64)

        # Fusion blocks — output channels match original design
        self.fuse_p3 = AddFusionBlock(rgb_channels[0], ir_channels[0], 128)
        self.fuse_p4 = AddFusionBlock(rgb_channels[1], ir_channels[1], 256)
        self.fuse_p5 = AddFusionBlock(rgb_channels[2], ir_channels[2], 512)

        # Top-down channel reductions
        self.reduce_p5       = Conv(512, 256, 1)   # P5 512 → 256
        self.reduce_p4_to_p3 = Conv(256, 128, 1)   # P4 256 → 128 to match f3

        # Project each level to 64ch for final concat
        self.to_64_p3 = Conv(128, 64, 1)
        self.to_64_p4 = Conv(256, 64, 1)
        self.to_64_p5 = Conv(256, 64, 1)

        # Final head: concat 5×64 = 320ch → output
        self.final_block = nn.Sequential(
            nn.Conv2d(320, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        self.aux_head_p4 = nn.Conv2d(256, num_classes, 1)
        self.aux_head_p5 = nn.Conv2d(256, num_classes, 1)

    def forward(self, rgb_feats, ir_feats, raw_rgb, raw_ir):
        r3, r4, r5 = rgb_feats
        i3, i4, i5 = ir_feats

        # Fusion at each scale
        f3 = self.fuse_p3(r3, i3)   # 128ch @ 80×80
        f4 = self.fuse_p4(r4, i4)   # 256ch @ 40×40
        f5 = self.fuse_p5(r5, i5)   # 512ch @ 20×20

        # Top-down FPN
        p5_red = self.reduce_p5(f5)                                              # 256ch @ 20×20
        p4_enh = f4 + F.interpolate(p5_red, scale_factor=2,
                                     mode='bilinear', align_corners=False)        # 256ch @ 40×40
        p3_up  = F.interpolate(p4_enh, scale_factor=2,
                                mode='bilinear', align_corners=False)             # 256ch @ 80×80
        m3     = f3 + self.reduce_p4_to_p3(p3_up)                               # 128ch @ 80×80

        # Project all levels to 64ch and align to P3 spatial size
        m3_64 = self.to_64_p3(m3)
        m4_64 = F.interpolate(self.to_64_p4(p4_enh), size=m3_64.shape[2:],
                               mode='bilinear', align_corners=False)
        m5_64 = F.interpolate(self.to_64_p5(p5_red), size=m3_64.shape[2:],
                               mode='bilinear', align_corners=False)

        # Edge features aligned to P3
        er = F.interpolate(
            F.avg_pool2d(self.edge_rgb(raw_rgb), 2, 2),
            size=m3_64.shape[2:], mode='bilinear', align_corners=False)
        ei = F.interpolate(
            F.avg_pool2d(self.edge_ir(raw_ir), 2, 2),
            size=m3_64.shape[2:], mode='bilinear', align_corners=False)

        # Concat → head → upsample to 640×640
        concat = torch.cat([m3_64, m4_64, m5_64, er, ei], dim=1)  # 320ch
        out = self.final_block(concat)
        out = F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=False)

        if self.training:
            aux_p4 = F.interpolate(self.aux_head_p4(p4_enh),
                                    size=out.shape[2:], mode='bilinear', align_corners=False)
            aux_p5 = F.interpolate(self.aux_head_p5(p5_red),
                                    size=out.shape[2:], mode='bilinear', align_corners=False)
            return out, aux_p4, aux_p5

        return out


# ==============================================================================
# 4. EFFICIENTNETFIRE (top-level model)
# ==============================================================================

class EfficientNetFire_b2(nn.Module):
    """
    Dual-stream EfficientNet-B2 wildfire segmentation model.
    RGB stream: EfficientNet-B2, full width, ImageNet pretrained.
    IR  stream: EfficientNet-B2, 1-channel adapted, ImageNet pretrained.
    Both streams produce P3/P4/P5 at 48/120/352 channels.
    """
    def __init__(self, num_classes=1):
        super().__init__()
        self.rgb_backbone = EfficientNetB2Backbone(in_channels=3)
        self.ir_backbone  = EfficientNetB2Backbone(in_channels=1)
        # B2 output channels are fixed: 48, 120, 352
        self.head = SemanticHead(
            num_classes=num_classes,
            rgb_channels=(48, 120, 352),
            ir_channels=(48, 120, 352),
        )

    def forward(self, rgb, ir):
        rgb_feats = self.rgb_backbone(rgb)
        ir_feats  = self.ir_backbone(ir)
        return self.head(rgb_feats, ir_feats, rgb, ir)


# ==============================================================================
# MAIN — forward pass test
# ==============================================================================
if __name__ == "__main__":
    model = EfficientNetFire_b2(num_classes=1)
    model.eval()

    x_rgb = torch.randn(2, 3, 640, 640)
    x_ir  = torch.randn(2, 1, 640, 640)

    with torch.no_grad():
        out = model(x_rgb, x_ir)

    print(f"Output shape: {out.shape}")
    print(f"Expected:     torch.Size([2, 1, 640, 640])")

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total:.1f}M")