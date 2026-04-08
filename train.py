import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
import albumentations as A


# ==========================================
# IMPORT ALL MODELS (see paths below)
# ==========================================
# Model source files:
#   - model/YOLOv12n_EfficientNetB2.py
#   - model/EfficientNetB2.py
#   - model/YOLOv12m.py
#
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
# from multimodal.code.model.YOLOv12n_EfficientNetB2 import HybridModel_nb2_  # model/YOLOv12n_EfficientNetB2.py
# from multimodal.code.model.EfficientNetB2 import *  # model/EfficientNetB2.py
# from multimodal.code.model.YOLOv12m import *        # model/YOLOv12m.py

# To use a model, uncomment the relevant import above and reference the correct class.

# ==========================================
# 0. CONFIGURATION
# ==========================================
CONFIG = {
    "ROOT_DIR": r"", # Dataset path
    "PRETRAINED_PATH": r"",#Pretrained weights path (optional, can be left empty to train from scratch)
    "SAVE_DIR": r"", # Directory to save checkpoints 
    "IMG_SIZE": 640,
    "BATCH_SIZE": 2,
    "EPOCHS": 100,
    "LR0": 3e-4,
    "MIN_LR": 1e-6,
    "WEIGHT_DECAY": 1e-4,
    "PATIENCE": 20,
    "NUM_WORKERS": 2,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "WARMUP_EPOCHS": 3,
}

# ==========================================
# 1. DATASET
# ==========================================
class FireDataset(Dataset):
    def __init__(self, root, split='test', size=640):
        self.root = Path(root) / split
        if not self.root.exists():
            fallback = 'val' if split == 'test' else 'train'
            self.root = Path(root) / fallback
            if not self.root.exists():
                raise FileNotFoundError(f"❌ Data not found at {self.root}")
        self.size = size
        possible_rgb = [self.root / "rgb" / "images", self.root / "rgb", self.root / "images"]
        self.rgb_dir = next((d for d in possible_rgb if d.exists()), None)
        if not self.rgb_dir:
            raise FileNotFoundError(f"❌ No RGB folder found in {self.root}")
        self.rgb_files = sorted(list(self.rgb_dir.glob("*.jpg")) + list(self.rgb_dir.glob("*.png")))
        self.augment = (split == 'train')
        if self.augment:
            self.geo_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(
                    scale=(0.9, 1.1),
                    rotate=(-10, 10),
                    translate_percent=(-0.05, 0.05),
                    shear=(-5, 5),
                    p=0.5
                ),
            ], additional_targets={'ir': 'image'})
            self.rgb_only_transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
                A.GaussNoise(p=0.2),
                A.CLAHE(p=0.2),
            ])
        else:
            self.geo_transform = None
            self.rgb_only_transform = None
    def __len__(self):
        return len(self.rgb_files)
    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        root_context = rgb_path.parent.parent.parent if "images" in str(rgb_path.parent) else rgb_path.parent.parent
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size))
        ir_candidates = [root_context / "ir" / "images" / rgb_path.name, root_context / "ir" / rgb_path.name]
        ir_path = next((c for c in ir_candidates if c.exists()), None)
        ir = cv2.imread(str(ir_path), 0) if ir_path else np.zeros((self.size, self.size), dtype=np.uint8)
        ir = cv2.resize(ir, (self.size, self.size))
        mask_candidates = [root_context / "masks" / f"{rgb_path.stem}_gt.png", root_context / "masks" / rgb_path.name]
        mask_path = next((c for c in mask_candidates if c.exists()), None)
        mask = cv2.imread(str(mask_path), 0) if mask_path else np.zeros((self.size, self.size), dtype=np.uint8)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment and self.geo_transform is not None:
            augmented = self.geo_transform(image=rgb, ir=ir, mask=mask)
            rgb, ir, mask = augmented['image'], augmented['ir'], augmented['mask']
        if self.augment:
            rgb = self.rgb_only_transform(image=rgb)["image"]
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        if len(ir.shape) == 2:
            ir = np.expand_dims(ir, axis=-1)
        ir_tensor = torch.from_numpy(ir).permute(2, 0, 1).float() / 255.0
        mask_tensor = (torch.from_numpy(mask).unsqueeze(0).float() > 127).float()
        return rgb_tensor, ir_tensor, mask_tensor, rgb_path.name, rgb

# ==========================================
# 2. LOSS FUNCTION
# ==========================================
class SegmentationLoss(nn.Module):
    def __init__(self, dice_weight=0.5, tversky_weight=0.5, focal_weight=1.0, aux_weight=0.4, alpha=0.5, beta=0.5, gamma=2.0, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        self.aux_weight = aux_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    def focal_loss(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * bce
        top_k = int(0.7 * loss.numel())
        return loss.view(-1).topk(top_k)[0].mean()
    def dice_loss(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        intersection = (probs * targets).sum(dim=[1, 2, 3])
        union = probs.sum(dim=[1, 2, 3]) + targets.sum(dim=[1, 2, 3])
        dice = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice.mean()
    def tversky_loss(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        tp = (probs * targets).sum(dim=[1, 2, 3])
        fp = (probs * (1 - targets)).sum(dim=[1, 2, 3])
        fn = ((1 - probs) * targets).sum(dim=[1, 2, 3])
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - tversky).mean()
    def single_loss(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        tversky = self.tversky_loss(inputs, targets)
        return (self.focal_weight * focal) + (self.dice_weight * dice) + (self.tversky_weight * tversky)
    def forward(self, pred_primary, targets, pred_aux_list=None):
        loss = self.single_loss(pred_primary, targets)
        if pred_aux_list is not None and len(pred_aux_list) > 0:
            aux_loss = sum(self.single_loss(aux, targets) for aux in pred_aux_list)
            aux_loss = aux_loss / len(pred_aux_list)
            loss = loss + self.aux_weight * aux_loss
        return loss

def calculate_metrics(pred, target):
    pred   = pred.view(-1)
    target = target.view(-1)
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    epsilon   = 1e-6
    precision = tp / (tp + fp + epsilon)
    recall    = tp / (tp + fn + epsilon)
    f1        = 2 * (precision * recall) / (precision + recall + epsilon)
    iou_fire  = tp / (tp + fp + fn + epsilon)
    iou_bg    = tn / (tn + fp + fn + epsilon)
    miou      = (iou_fire + iou_bg) / 2.0
    acc       = (tp + tn) / (tp + tn + fp + fn + epsilon)
    return precision, recall, f1, iou_fire, iou_bg, miou, acc

def split_outputs(output):
    if isinstance(output, tuple):
        return output[0], list(output[1:])
    return output, []

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                path = Path(p)
                break
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path

def train():
    SAVE_DIR   = increment_path(CONFIG["SAVE_DIR"], exist_ok=False, mkdir=True)
    weights_dir = SAVE_DIR / "weights"
    weights_dir.mkdir(exist_ok=True)
    print(f"🚀 Training Started.")
    print(f"📊 Configuration: {CONFIG}")
    print(f"📂 Saving to: {SAVE_DIR}")
    model = HybridModel_nb2_().to(CONFIG["DEVICE"]) #Change this to switch models
    if CONFIG["PRETRAINED_PATH"] and os.path.exists(CONFIG["PRETRAINED_PATH"]):
        ckpt = torch.load(CONFIG["PRETRAINED_PATH"], map_location=CONFIG["DEVICE"])
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt, strict=False)
        print(f"✅ Loaded pretrained weights from: {CONFIG['PRETRAINED_PATH']}")
    else:
        print("⚠️ No pretrained checkpoint loaded. Training from scratch.")
    train_ds = FireDataset(CONFIG["ROOT_DIR"], 'train', CONFIG["IMG_SIZE"])
    val_ds   = FireDataset(CONFIG["ROOT_DIR"], 'test',  CONFIG["IMG_SIZE"])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                              num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    criterion = SegmentationLoss(
            focal_weight=1.0,
            dice_weight=0.5,
            tversky_weight=0.5,
            alpha=0.3,
            beta=0.7,
            gamma=2.0,
            smooth=1.0,
        )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["LR0"],
        weight_decay=CONFIG["WEIGHT_DECAY"],
    )
    scheduler = None
    best_miou = -1.0
    patience_counter = 0
    for epoch in range(1, CONFIG["EPOCHS"] + 1):
        if epoch <= CONFIG["WARMUP_EPOCHS"]:
            lr_step = CONFIG["LR0"] * (epoch / CONFIG["WARMUP_EPOCHS"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr_step
        if epoch == CONFIG["WARMUP_EPOCHS"] + 1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, CONFIG["EPOCHS"] - CONFIG["WARMUP_EPOCHS"]),
                eta_min=CONFIG["MIN_LR"],
            )
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['EPOCHS']}")
        for rgb, ir, mask, _, _ in pbar:
            rgb, ir, mask = rgb.to(CONFIG["DEVICE"]), ir.to(CONFIG["DEVICE"]), mask.to(CONFIG["DEVICE"])
            optimizer.zero_grad()
            out, aux_outs = split_outputs(model(rgb, ir))
            out = F.interpolate(out, size=mask.shape[2:], mode='bilinear', align_corners=False)
            aux_outs_up = [F.interpolate(a, size=mask.shape[2:], mode='bilinear', align_corners=False) for a in aux_outs]
            loss = criterion(out, mask, aux_outs_up)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=10.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.2e}"})
        if epoch > CONFIG["WARMUP_EPOCHS"] and scheduler is not None:
            scheduler.step()
        model.eval()
        iou_scores = []
        with torch.no_grad():
            for rgb, ir, mask, _, _ in val_loader:
                rgb, ir, mask = rgb.to(CONFIG["DEVICE"]), ir.to(CONFIG["DEVICE"]), mask.to(CONFIG["DEVICE"])
                out, _ = split_outputs(model(rgb, ir))
                out  = F.interpolate(out, size=mask.shape[2:], mode='bilinear', align_corners=False)
                pred = (torch.sigmoid(out) > 0.5).float()
                _, _, _, _, _, miou, _ = calculate_metrics(pred, mask)
                iou_scores.append(miou)
        current_miou = sum(iou_scores) / len(iou_scores)
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d} | LR: {curr_lr:.2e} | Loss: {train_loss/len(train_loader):.4f} | mIoU: {current_miou:.4f}")
        torch.save(model.state_dict(), weights_dir / "last.pt")
        if current_miou > best_miou:
            best_miou = current_miou
            torch.save(model.state_dict(), weights_dir / "best.pt")
            patience_counter = 0
            print(f"⭐ New Best mIoU: {best_miou:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{CONFIG['PATIENCE']} epoch(s)")
        if patience_counter >= CONFIG["PATIENCE"]:
            print(f"🛑 Early stopping triggered after {CONFIG['PATIENCE']} epochs without mIoU improvement.")
            break

if __name__ == "__main__":
    train()
