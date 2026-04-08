import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
#
# To use a model, uncomment the relevant import above and reference the correct class.

# ==========================================
# 0. CONFIGURATION
# ==========================================
CONFIG = {
    "ROOT_DIR": r"C:\Users\User\WildfireDetection\willdfire_seg_yolov12\dataset\RGB-T_wildfire_dataset",
    "PRETRAINED_PATH": r"",
    "SAVE_DIR": r"C:\Users\User\WildfireDetection\willdfire_seg_yolov12\multimodal\hybrid\rgbt\nb2\run",
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

def split_outputs(output):
    if isinstance(output, tuple):
        return output[0], list(output[1:])
    return output, []

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

def validate():
    SAVE_DIR   = increment_path(CONFIG["SAVE_DIR"], exist_ok=True, mkdir=True)
    weights_dir = SAVE_DIR / "weights"
    best_model_path = weights_dir / "best.pt"
    print(f"\n🔍 Running standalone validation...")
    print(f"📂 Loading model from: {best_model_path}")
    device = CONFIG["DEVICE"]
    model = HybridModel_nb2_().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    val_ds = FireDataset(CONFIG["ROOT_DIR"], 'test', CONFIG["IMG_SIZE"])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    vals = {'prec': [], 'rec': [], 'f1': [], 'iou_fire': [], 'iou_bg': [], 'miou': [], 'acc': []}
    print(" Calculating Final Metrics...")
    with torch.no_grad():
        for rgb, ir, mask, _, _ in tqdm(val_loader):
            rgb, ir, mask = rgb.to(device), ir.to(device), mask.to(device)
            out, _ = split_outputs(model(rgb, ir))
            out  = F.interpolate(out, size=mask.shape[2:], mode='bilinear', align_corners=False)
            pred = (torch.sigmoid(out) > 0.5).float()
            p, r, f1, iou_fire, iou_bg, miou, acc = calculate_metrics(pred, mask)
            vals['prec'].append(p); vals['rec'].append(r); vals['f1'].append(f1)
            vals['iou_fire'].append(iou_fire); vals['iou_bg'].append(iou_bg)
            vals['miou'].append(miou); vals['acc'].append(acc)
    avg = {k: sum(v) / len(v) for k, v in vals.items()}
    results_txt = (
        f"========================================\n"
        f"VALIDATION RESULTS\n"
        f"========================================\n"
        f"IoU Fire:       {avg['iou_fire']:.4f}\n"
        f"IoU Background: {avg['iou_bg']:.4f}\n"
        f"mIoU:           {avg['miou']:.4f}\n"
        f"F1-Score:  {avg['f1']:.4f}\n"
        f"Precision: {avg['prec']:.4f}\n"
        f"Recall:    {avg['rec']:.4f}\n"
        f"Accuracy:  {avg['acc']:.4f}\n"
        f"========================================\n"
    )
    print("\n" + results_txt)
    final_res_file = SAVE_DIR / "validation_results.txt"
    with open(final_res_file, "w", encoding="utf-8") as f:
        f.write(results_txt)
    print(f"📄 Results saved to: {final_res_file}")

if __name__ == "__main__":
    validate()
