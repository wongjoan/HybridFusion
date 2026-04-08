import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from ultralytics.nn.modules.block import C3k2, A2C2f, AAttn
from ultralytics.nn.modules.conv import Conv

class InputEdgeEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels // 2, k=3, s=2)
        self.conv2 = Conv(out_channels // 2, out_channels, k=3, s=2)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class YOLOv12Backbone(nn.Module):
    """YOLOv12n Backbone"""
    def __init__(self, input_channels=3):
        super().__init__()
        self.l0 = Conv(input_channels, 16, 3, 2)
        self.l1 = Conv(16, 32, 3, 2)
        self.l2 = C3k2(32, 64, n=1, shortcut=True)
        self.l3 = Conv(64, 64, 3, 2)
        self.l4 = C3k2(64, 128, n=2, shortcut=True)
        self.l5 = Conv(128, 128, 3, 2)
        self.l6 = A2C2f(128, 256, n=2, shortcut=True)
        self.l7 = Conv(256, 256, 3, 2)
        self.l8 = A2C2f(256, 256, n=1, shortcut=True)

    def forward(self, x):
        x  = self.l2(self.l1(self.l0(x)))
        p3 = self.l4(self.l3(x))    # 128ch
        p4 = self.l6(self.l5(p3))   # 256ch
        p5 = self.l8(self.l7(p4))   # 256ch
        return p3, p4, p5

class SpatialGateFusion(nn.Module):
    """Fuses RGB and IR features at a given scale using a spatial gate."""
    def __init__(self, channels, use_cbam=True):
        super().__init__()
        # Gate: concat(rgb, ir) -> sigmoid gate map
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        # Output refinement conv
        self.refine = Conv(channels, channels, 3)
        # Optional CBAM
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(channels)

    def forward(self, rgb, ir):
        # Spatial gate — learned per pixel, per channel
        g = self.gate(torch.cat([rgb, ir], dim=1))   # [B, C, H, W]
        fused = g * rgb + (1 - g) * ir
        if self.use_cbam:
            fused = self.cbam(fused)
        return self.refine(fused) + fused             # residual


class CBAM(nn.Module):
    """Convolutional Block Attention Module — channel then spatial gate."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.channel_sig  = nn.Sigmoid()
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.spatial_sig  = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        ca = self.channel_sig(
            self.channel_mlp(self.avg_pool(x)) +
            self.channel_mlp(self.max_pool(x))
        )
        x = x * ca
        # Spatial attention
        sa = self.spatial_sig(self.spatial_conv(
            torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        ))
        return x * sa
    
# ==============================================================================
# 2. EFFICIENTNET-B2 BACKBONE (For IR)
# ==============================================================================
class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-B2 backbone for IR (1-channel) input.
    B2 feature map stages at 640px input (verified from runtime):
      features[0]    stem Conv              ->  32ch,  160x160
      features[1]    MBConv stage 1 s=1     ->  16ch,  160x160
      features[2]    MBConv stage 2 s=2     ->  24ch,   80x80
      features[3]    MBConv stage 3 s=2     ->  48ch,   40x40  -- stride mismatch vs YOLO P3
      features[4]    MBConv stage 4 s=2     ->  88ch,   20x20
      features[5]    MBConv stage 5 s=1     -> 120ch,   20x20
      features[6]    MBConv stage 6 s=2     -> 208ch,   10x10
      features[7]    MBConv stage 7 s=1     -> 352ch,   10x10
      features[8]    classifier stem Conv   -> 1408ch  <-- DO NOT include

    We align to YOLO spatial scales (80/40/20) by choosing:
      stem     = features[0:3]  -> 24ch,  80x80
      P3       = features[3]    -> 48ch,  40x40  -- extra downsample, handled by proj
      P4       = features[4:6]  -> 120ch, 20x20
      P5       = features[6:8]  -> 352ch, 10x10  -- upsampled to 20x20 before fusion

    NOTE: B2 spatial strides don't align perfectly with YOLO. P5 is at 10x10 so
    it needs one extra upsample step in the fusion head to match YOLO's 20x20 P5.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        base = torchvision.models.efficientnet_b2(weights='DEFAULT')

        # 1-Channel Adaptation
        if in_channels == 1:
            original_layer = base.features[0][0]
            new_layer = nn.Conv2d(1, 32, 3, 2, 1, bias=False)
            with torch.no_grad():
                new_layer.weight[:] = original_layer.weight.sum(dim=1, keepdim=True)
            base.features[0][0] = new_layer

        self.stem      = base.features[0:3]   # -> 24ch,  80x80
        self.layer_p3  = base.features[3]     # -> 48ch,  40x40
        self.layer_p4  = base.features[4:6]   # -> 120ch, 20x20
        self.layer_p5  = base.features[6:8]   # -> 352ch, 10x10

    def forward(self, x):
        x  = self.stem(x)
        p3 = self.layer_p3(x)
        p4 = self.layer_p4(p3)
        p5 = self.layer_p5(p4)
        return p3, p4, p5

# ==============================================================================
# 3. HYBRID MODEL (YOLOv12n RGB + EfficientNet-B2 IR)
# ==============================================================================
class HybridModel_nb2_(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # --- Backbones ---
        self.rgb_backbone = YOLOv12Backbone(3)      # YOLOv12n: P3=128, P4=256, P5=256
        self.ir_backbone  = EfficientNetBackbone(1) # B2:       P3=48,  P4=120, P5=352

        # --- Projectors (B2 -> YOLOv12n) ---
        # B2 outputs: P3=48ch@40x40, P4=120ch@20x20, P5=352ch@10x10
        # YOLO outputs: P3=128ch@80x80, P4=256ch@40x40, P5=256ch@20x20
        self.proj_ir_p3 = Conv(48,  128, 1)   # 48  -> 128
        self.proj_ir_p4 = Conv(120, 256, 1)   # 120 -> 256
        self.proj_ir_p5 = Conv(352, 256, 1)   # 352 -> 256

        self.fuse_p3 = SpatialGateFusion(128, use_cbam=True)
        self.fuse_p4 = SpatialGateFusion(256, use_cbam=True)
        self.fuse_p5 = SpatialGateFusion(256, use_cbam=True)
        self.reduce_p4 = Conv(256, 128, 1)
        self.reduce_p5 = Conv(256, 128, 1)

        # --- Edge Encoders ---
        self.edge_rgb  = InputEdgeEncoder(in_channels=3, out_channels=64)
        self.edge_ir   = InputEdgeEncoder(in_channels=1, out_channels=64)
        self.edge_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.up      = nn.Upsample(scale_factor=2, mode='nearest')

        # --- Segmentation Head ---
        # f3=128, e_rgb=64, e_ir=64 -> concat=256
        self.final_conv1  = Conv(256, 128, 3)
        self.final_conv2  = nn.Conv2d(128, num_classes, 1)
        self.aux_head_p4  = nn.Conv2d(128, num_classes, 1)
        self.aux_head_p5  = nn.Conv2d(128, num_classes, 1)

    def forward(self, rgb, ir):
        e_rgb = self.edge_pool(self.edge_rgb(rgb))  # 64ch @ 80x80
        e_ir  = self.edge_pool(self.edge_ir(ir))    # 64ch @ 80x80

        # 1. Extract features
        r3, r4, r5 = self.rgb_backbone(rgb)  # 128@80, 256@40, 256@20
        i3, i4, i5 = self.ir_backbone(ir)    # 48@40,  120@20, 352@10

        # 2. Project IR channels to match YOLO widths
        i3 = self.proj_ir_p3(i3)  # 48  -> 128  (still @40x40)
        i4 = self.proj_ir_p4(i4)  # 120 -> 256  (still @20x20)
        i5 = self.proj_ir_p5(i5)  # 352 -> 256  (still @10x10)

        # 3. Align IR spatial resolution to YOLO scales dynamically.
        # This is robust to backbone stride differences across torchvision versions.
        i3 = F.interpolate(i3, size=r3.shape[2:], mode='bilinear', align_corners=False)
        i4 = F.interpolate(i4, size=r4.shape[2:], mode='bilinear', align_corners=False)
        i5 = F.interpolate(i5, size=r5.shape[2:], mode='bilinear', align_corners=False)

        # 4. Fusion (Learned Weighted Sum — alpha learned per scale)
        f5 = self.reduce_p5(self.fuse_p5(r5, i5))                          # 256→128 @ 20²
        f4 = self.reduce_p4(self.fuse_p4(r4, i4)) + self.up(f5)            # 256→128 @ 40²
        f3 = self.fuse_p3(r3, i3) + self.up(f4)                            # 128 @ 80²

        # 5. Output Head
        concat_feats = torch.cat([f3, e_rgb, e_ir], dim=1)  # 128+64+64=256 @ 80x80
        x   = self.final_conv1(concat_feats)
        out = self.final_conv2(x)

        if self.training:
            aux_p4 = F.interpolate(self.aux_head_p4(f4), size=out.shape[2:], mode='bilinear', align_corners=False)
            aux_p5 = F.interpolate(self.aux_head_p5(f5), size=out.shape[2:], mode='bilinear', align_corners=False)
            return out, aux_p4, aux_p5

        return out

# ==============================================================================
# 4. WEIGHT LOADING (YOLOv12n pretrained weights -> RGB backbone)
# ==============================================================================
def load_yolo_weights(model, path):
    print(f"Loading YOLO weights from {path}...")
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt
    except Exception as e:
        print(f"Failed to load YOLO weights: {e}")
        return model

    layer_map = {
        'model.0.': 'l0.', 'model.1.': 'l1.', 'model.2.': 'l2.',
        'model.3.': 'l3.', 'model.4.': 'l4.', 'model.5.': 'l5.',
        'model.6.': 'l6.', 'model.7.': 'l7.', 'model.8.': 'l8.'
    }

    new_sd   = {}
    model_sd = model.state_dict()

    for k, v in sd.items():
        matched_prefix = None
        for old_p, new_p in layer_map.items():
            if k.startswith(old_p):
                matched_prefix = (old_p, new_p)
                break

        if matched_prefix:
            core_name = k.replace(matched_prefix[0], matched_prefix[1])
            dest_k    = f"rgb_backbone.{core_name}"

            if 'attn.qk' in dest_k or 'attn.v' in dest_k:
                parent = dest_k.rsplit('.', 2)[0]
                sub    = dest_k.rsplit('.', 2)[1]
                p_name = dest_k.rsplit('.', 1)[1]
                try:
                    mod = model
                    for part in parent.split('.'): mod = getattr(mod, part)
                    if isinstance(mod, AAttn):
                        if getattr(mod, sub) is None:
                            oc, ic, kh, kw = v.shape
                            l = nn.Conv2d(ic, oc, (kh, kw), bias=False).to(v.device)
                            setattr(mod, sub, l)
                            mod.qkv = nn.Identity()
                        target = getattr(mod, sub)
                        if p_name == 'weight': target.weight.data = v
                        elif p_name == 'bias':
                            if target.bias is None: target.bias = nn.Parameter(v)
                            else: target.bias.data = v
                        continue
                except: pass

            if dest_k in model_sd and model_sd[dest_k].shape == v.shape:
                new_sd[dest_k] = v

    model.load_state_dict(new_sd, strict=False)
    print(f"YOLO weights loaded. EfficientNet-B2 ImageNet weights loaded automatically.")
    return model

# ==============================================================================
# MAIN — forward pass test
# ==============================================================================
if __name__ == "__main__":
    YOLO_PATH = r"C:\Users\User\WildfireDetection\willdfire_seg_yolov12\multimodal\yolov12n-seg.pt"

    model = HybridModel_nb2_(num_classes=1)

    if os.path.exists(YOLO_PATH):
        model = load_yolo_weights(model, YOLO_PATH)
    else:
        print("YOLO weights not found. Using random init for RGB backbone.")

    print("\nRunning forward pass test...")
    x_rgb = torch.randn(2, 3, 640, 640)
    x_ir  = torch.randn(2, 1, 640, 640)
    model.eval()
    try:
        y = model(x_rgb, x_ir)
        print(f"Success! Output shape: {y.shape}")
        print("Architecture: RGB=YOLOv12n, IR=EfficientNet-B2, Learned Fusion")
    except Exception as e:
        print(f"Error: {e}")