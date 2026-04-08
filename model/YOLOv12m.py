import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ==============================================================================
# 1. YOLOv12 BLOCKS — imported directly from Ultralytics
#    Requires: pip install ultralytics
#    Conv   — standard Conv-BN-SiLU block
#    C3k2   — cross-stage partial block (YOLO11/YOLOv12)
#    A2C2f  — area-attention C2f block (YOLOv12-specific, wraps AAttn)
#    AAttn  — area attention module (YOLOv12-specific, needed for weight loading)
# ==============================================================================
from ultralytics.nn.modules.block import C3k2, A2C2f, AAttn
from ultralytics.nn.modules.conv import Conv

# ==============================================================================
# 2. EDGE ENCODER (custom — not part of Ultralytics)
# ==============================================================================
class InputEdgeEncoder(nn.Module):
    """Extracts high-frequency edge features from raw inputs (4x downsample)."""
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels // 2, k=3, s=2)
        self.conv2 = Conv(out_channels // 2, out_channels, k=3, s=2)

    def forward(self, x):
        return self.conv2(self.conv1(x))

# ==============================================================================
# 3. YOLOv12m BACKBONE
#    Medium-scale channel config (width=0.5 of full model):
#      l0: Conv  3  -> 64   stride 2   (320x320)
#      l1: Conv  64 -> 128  stride 2   (160x160)
#      l2: C3k2  128 -> 256  n=1
#      l3: Conv  256 -> 256  stride 2  (80x80)  -> P3
#      l4: C3k2  256 -> 512  n=1
#      l5: Conv  512 -> 512  stride 2  (40x40)  -> P4
#      l6: A2C2f 512 -> 512  n=2
#      l7: Conv  512 -> 512  stride 2  (20x20)  -> P5
#      l8: A2C2f 512 -> 512  n=2
#    Output channels: P3=512, P4=512, P5=512
#
#    For IR (1-channel input): same structure, input_channels=1.
#    The stem conv weight is adapted at load time by summing pretrained
#    3-channel weights into a single channel.
# ==============================================================================
class YOLOv12Backbone(nn.Module):
    """YOLOv12m backbone. Pass input_channels=1 for IR stream."""
    def __init__(self, input_channels=3):
        super().__init__()
        self.l0 = Conv(input_channels, 64,  3, 2)
        self.l1 = Conv(64,  128, 3, 2)
        self.l2 = C3k2(128, 256, n=1, shortcut=False)
        self.l3 = Conv(256, 256, 3, 2)
        self.l4 = C3k2(256, 512, n=1, shortcut=False)
        self.l5 = Conv(512, 512, 3, 2)
        self.l6 = A2C2f(512, 512, n=2, shortcut=True)
        self.l7 = Conv(512, 512, 3, 2)
        self.l8 = A2C2f(512, 512, n=2, shortcut=True)

    def forward(self, x):
        x  = self.l2(self.l1(self.l0(x)))
        p3 = self.l4(self.l3(x))    # 512ch @ 1/8
        p4 = self.l6(self.l5(p3))   # 512ch @ 1/16
        p5 = self.l8(self.l7(p4))   # 512ch @ 1/32
        return p3, p4, p5

# ==============================================================================
# 4. DUAL MODAL YOLO
# ==============================================================================
class YOLOv12m(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # --- 1. Dual Backbones (weight-isolated) ---
        self.rgb_backbone = YOLOv12Backbone(input_channels=3)
        self.ir_backbone  = YOLOv12Backbone(input_channels=1)

        # --- 2. Edge Encoders ---
        self.edge_rgb  = InputEdgeEncoder(in_channels=3, out_channels=64)
        self.edge_ir   = InputEdgeEncoder(in_channels=1, out_channels=64)
        self.edge_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # --- 3. FPN channel reduction (512 -> 256) ---
        self.reduce_p5 = Conv(512, 256, 1)
        self.reduce_p4 = Conv(512, 256, 1)
        self.reduce_p3 = Conv(512, 256, 1)

        self.up2x_nearest = nn.Upsample(scale_factor=2, mode='nearest')

        # --- 4. Multi-scale alignment projections (256 -> 64) ---
        self.m5_proj = Conv(256, 64, 1)
        self.m4_proj = Conv(256, 64, 1)
        self.m3_proj = Conv(256, 64, 1)

        self.up4x_bilin = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.up2x_bilin = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # --- 5. Segmentation head ---
        # concat: m3(64) + m4_up(64) + m5_up(64) + e_rgb(64) + e_ir(64) = 320ch
        self.final_conv1 = Conv(320, 256, k=3, s=1)
        self.final_bn    = nn.BatchNorm2d(256)
        self.final_relu  = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.aux_head_p4 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.aux_head_p5 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, rgb, ir):
        # Edge features (64ch @ 1/8)
        e_rgb = self.edge_pool(self.edge_rgb(rgb))
        e_ir  = self.edge_pool(self.edge_ir(ir))

        # Backbone features
        r3, r4, r5 = self.rgb_backbone(rgb)  # 512ch @ 1/8, 1/16, 1/32
        i3, i4, i5 = self.ir_backbone(ir)    # 512ch @ 1/8, 1/16, 1/32

        # Element-wise fusion
        f5 = r5 + i5
        f4 = r4 + i4
        f3 = r3 + i3

        # Top-down FPN
        p5_red = self.reduce_p5(f5)                          # 256ch @ 1/32
        p4_red = self.reduce_p4(f4)                          # 256ch @ 1/16
        p3_red = self.reduce_p3(f3)                          # 256ch @ 1/8

        p4_enh = p4_red + self.up2x_nearest(p5_red)          # 256ch @ 1/16
        p3_enh = p3_red + self.up2x_nearest(p4_enh)          # 256ch @ 1/8

        # Multi-scale alignment to 1/8
        m5_up       = self.up4x_bilin(self.m5_proj(p5_red))  # 64ch @ 1/8
        m4_up       = self.up2x_bilin(self.m4_proj(p4_enh))  # 64ch @ 1/8
        m3_aligned  = self.m3_proj(p3_enh)                   # 64ch @ 1/8

        # Concat + head
        concat_feats = torch.cat([m3_aligned, m4_up, m5_up, e_rgb, e_ir], dim=1)  # 320ch
        x   = self.final_conv1(concat_feats)    # 320 -> 256ch
        x   = self.final_relu(self.final_bn(x))
        out = self.final_conv2(x)               # 256 -> 1ch @ 1/8

        if self.training:
            aux_p4 = F.interpolate(self.aux_head_p4(p4_enh), size=out.shape[2:], mode='bilinear', align_corners=False)
            aux_p5 = F.interpolate(self.aux_head_p5(p5_red), size=out.shape[2:], mode='bilinear', align_corners=False)
            return out, aux_p4, aux_p5

        return out

# ==============================================================================
# 5. WEIGHT LOADER
#    Maps YOLOv12m checkpoint keys (model.0 … model.8) to
#    rgb_backbone.l0 … rgb_backbone.l8  and
#    ir_backbone.l0  … ir_backbone.l8
#    IR stem (l0.conv.weight) is adapted by summing 3-channel pretrained
#    weights into a single channel.
# ==============================================================================
def load_weights(model, path):
    print(f"Loading weights from {path}...")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = (ckpt['model'].state_dict()
          if 'model' in ckpt and hasattr(ckpt['model'], 'state_dict')
          else ckpt.get('model', ckpt))

    new_sd   = {}
    model_sd = model.state_dict()

    for k, v in sd.items():
        k = k.replace('model.model.', 'model.')

        # Map model.{i}.* -> l{i}.*
        nk = None
        for i in range(9):
            if k.startswith(f'model.{i}.'):
                nk = k.replace(f'model.{i}.', f'l{i}.')
                break
        if nk is None:
            continue  # skip neck/head layers

        for prefix in ('rgb_backbone', 'ir_backbone'):
            dest_k = f"{prefix}.{nk}"
            if dest_k not in model_sd:
                continue

            val = v
            # IR stem 1-channel adaptation
            if prefix == 'ir_backbone' and 'l0.conv.weight' in dest_k and v.shape[1] == 3:
                val = v.sum(dim=1, keepdim=True)

            if model_sd[dest_k].shape == val.shape:
                new_sd[dest_k] = val

    msg = model.load_state_dict(new_sd, strict=False)
    print(f"Loaded {len(new_sd)} tensors. "
          f"Missing (head/fusion layers expected): {len(msg.missing_keys)}")
    return model

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    CKPT_PATH = r"C:\Users\User\WildfireDetection\willdfire_seg_yolov12\multimodal\yolov12m-seg.pt"

    model = YOLOv12m(num_classes=1)
    if torch.cuda.is_available():
        model = model.cuda()

    if os.path.exists(CKPT_PATH):
        model = load_weights(model, CKPT_PATH)
    else:
        print(f"Weights not found at {CKPT_PATH}. Using random init.")

    dummy_rgb = torch.randn(2, 3, 640, 640)
    dummy_ir  = torch.randn(2, 1, 640, 640)
    if torch.cuda.is_available():
        dummy_rgb, dummy_ir = dummy_rgb.cuda(), dummy_ir.cuda()

    model.eval()
    with torch.no_grad():
        out = model(dummy_rgb, dummy_ir)
    print(f"Success. Output shape: {out.shape}")