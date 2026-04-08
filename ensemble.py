import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import re

from model.YOLOv12m import YOLOv12m
from model.YOLOv12n_EfficientNetB2 import HybridModel_nb2_
from model.EfficientNetB2 import EfficientNetFire_b2
from train_val import FireDataset

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "ROOT_DIR": r"", # Dataset path
    "MODEL_PATHS": {
        "m1": r"", # Path to YOLOv12m checkpoint
        "m2": r"",# Path to HybridModel_nb2_ checkpoint
        "m3": r"",# Path to EfficientNetFire_b2 checkpoint
    },
    "IMG_SIZE": 640,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    "THRESHOLDS": {
        "m1": 0.49,
        "m2": 0.49,
        "m3": 0.55
    },
    "WEIGHTS": {
        "m1": 0.2,
        "m2": 0.25,
        "m3": 0.55,
    },
    "ENSEMBLE_THRESHOLD": 0.45,

    "SAVE_DIR_BASE": r"C:\Users\User\WildfireDetection\willdfire_seg_yolov12\multimodal\max_voting\rgbt",
    "SAVE_LIMIT": 50,
    "BATCH_SIZE": 1,         
}

_w = CONFIG["WEIGHTS"]
assert abs(sum(_w.values()) - 1.0) < 1e-6, \
    f"Weights must sum to 1.0, got {sum(_w.values()):.4f}"

def calculate_metrics(pred, target):
    """
    Matches train_val.py calculate_metrics exactly.
    pred, target: [B, 1, H, W] float tensors with values 0 or 1.
    Returns scalar metrics for a single image.
    """
    pred   = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()

    epsilon   = 1e-6          # matches train_val.py (not 1e-7)
    precision = tp / (tp + fp + epsilon)
    recall    = tp / (tp + fn + epsilon)
    f1        = 2 * (precision * recall) / (precision + recall + epsilon)
    iou_fire  = tp / (tp + fp + fn + epsilon)
    iou_bg    = tn / (tn + fp + fn + epsilon)
    miou      = (iou_fire + iou_bg) / 2.0
    acc       = (tp + tn) / (tp + tn + fp + fn + epsilon)

    return precision, recall, f1, iou_fire, iou_bg, miou, acc


def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


# ==========================================
# UTILS
# ==========================================

def increment_path(path):
    if not os.path.exists(path):
        return path
    path = str(path)
    base  = path
    match = re.search(r"(\d+)$", base)
    if match:
        base = base[: match.start()]
        n    = int(match.group(1)) + 1
    else:
        n = 2
    while True:
        new_path = f"{base}{n}"
        if not os.path.exists(new_path):
            return new_path
        n += 1


def load_checkpoint_state_dict(path, device):
    if os.path.isdir(path):
        best_ckpt = os.path.join(path, "best.pt")
        last_ckpt = os.path.join(path, "last.pt")
        if os.path.isfile(best_ckpt):
            path = best_ckpt
        elif os.path.isfile(last_ckpt):
            path = last_ckpt
        else:
            raise FileNotFoundError(f"No checkpoint file found in directory: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt:
                sd = ckpt[key]
                return sd.state_dict() if hasattr(sd, "state_dict") else sd
    return ckpt


def load_partial_state_dict(model, state_dict):
    model_dict  = model.state_dict()
    compatible  = {k: v for k, v in state_dict.items()
                   if k in model_dict and model_dict[k].shape == v.shape}
    skipped     = len(state_dict) - len(compatible)
    model.load_state_dict(compatible, strict=False)
    return len(compatible), skipped


# ==========================================
# MODEL LOADING
# ==========================================

def load_models():
    print("=" * 50)
    print("Loading models...")
    device = CONFIG["DEVICE"]

    m1 = YOLOv12m(num_classes=1).to(device)
    m2 = HybridModel_nb2_(num_classes=1).to(device)
    m3 = EfficientNetFire_b2(num_classes=1).to(device)

    models_info = [
        ("YOLOv12m",         m1, CONFIG["MODEL_PATHS"]["m1"]),
        ("HybridModel_nb2_", m2, CONFIG["MODEL_PATHS"]["m2"]),
        ("EfficientNetFire", m3, CONFIG["MODEL_PATHS"]["m3"]),
    ]

    loaded_models = []
    for name, model, path in models_info:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found for {name}: {path}")

        sd = load_checkpoint_state_dict(path, device)
        n_loaded, n_skipped = load_partial_state_dict(model, sd)
        print(f"  [{name}] loaded: {n_loaded} tensors | skipped: {n_skipped}")

        if n_skipped > n_loaded:
            print(f"  WARNING: [{name}] skipped more than loaded — "
                  f"possible architecture mismatch.")

        model.eval()
        loaded_models.append(model)

    print("=" * 50)
    return loaded_models


# ==========================================
# INFERENCE
# ==========================================

def predict_one_logits(model, rgb, ir):
    """
    Returns raw logits (NO sigmoid) upsampled to input resolution.
    Logit-space averaging before sigmoid is more accurate than
    averaging probabilities.
    """
    h, w = rgb.shape[2], rgb.shape[3]
    with torch.no_grad():
        out    = model(rgb, ir)
        logits = out[0] if isinstance(out, tuple) else out
    if logits.shape[2:] != (h, w):
        logits = F.interpolate(logits, size=(h, w),
                               mode="bilinear", align_corners=False)
    return logits


def ensemble_predict(models, rgb, ir):
    """
    Weighted soft-voting in logit space.
    
    RGB-IR image pairs are independently processed by all three models.
    Each model produces per-pixel logits, which are upsampled to the
    original 640×640 resolution. Raw logits from each model are accumulated
    with per-model weights before a single sigmoid activation and threshold
    are applied.
    
    Formally, let L_k denote the logit map of model k and w_k its ensemble
    weight with sum(w_k) = 1. The ensemble prediction is:
        ŷ = (σ(sum(w_k * L_k)) > τ)
    
    where σ(·) is the sigmoid function and τ is a threshold calibrated on
    the validation set. Operating in logit space preserves the full
    confidence signal of each model.
    """
    keys    = ["m1", "m2", "m3"]
    weights = CONFIG["WEIGHTS"]

    h, w = rgb.shape[2], rgb.shape[3]
    weighted_logit_sum = torch.zeros(rgb.shape[0], 1, h, w, device=rgb.device)
    individual_votes   = []

    for model, key in zip(models, keys):
        logits = predict_one_logits(model, rgb, ir)

        assert logits.shape[2:] == (h, w), (
            f"[{key}] shape mismatch: {logits.shape[2:]} vs ({h}, {w})"
        )

        weighted_logit_sum += logits * weights[key]

        # individual binary vote (visualisation only)
        vote = (torch.sigmoid(logits) > CONFIG["THRESHOLDS"][key]).float()
        individual_votes.append(vote)

    # sigmoid applied ONCE on the weighted logit average
    avg_prob      = torch.sigmoid(weighted_logit_sum)
    ensemble_pred = (avg_prob > CONFIG["ENSEMBLE_THRESHOLD"]).float()

    return ensemble_pred, avg_prob, individual_votes


# ==========================================
# VISUALIZATION
# ==========================================

def save_vis(save_dir, rgb, mask, pred, avg_prob, individual_votes, fname):
    img_path = os.path.join(save_dir, f"ens_{fname}")

    r_np    = rgb[0].permute(1, 2, 0).cpu().numpy()
    r_np    = (r_np - r_np.min()) / (r_np.max() - r_np.min() + 1e-7)
    m_np    = mask[0].squeeze().cpu().numpy()
    p_np    = pred[0].squeeze().cpu().numpy()
    prob_np = avg_prob[0].squeeze().cpu().numpy()
    v_names = ["YOLOv12m", "HybridModel_nb2_", "EfficientNetFire"]
    votes_np = [v[0].squeeze().cpu().numpy() for v in individual_votes]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    axes[0].imshow(r_np);               axes[0].set_title("RGB");            axes[0].axis("off")
    axes[1].imshow(m_np, cmap="gray");  axes[1].set_title("Ground Truth");   axes[1].axis("off")
    axes[2].imshow(p_np, cmap="gray");  axes[2].set_title("Ensemble Pred");  axes[2].axis("off")
    im = axes[3].imshow(prob_np, cmap="hot", vmin=0, vmax=1)
    axes[3].set_title("Avg Probability"); axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    for idx, (vname, vnp) in enumerate(zip(v_names, votes_np)):
        axes[4 + idx].imshow(vnp, cmap="gray")
        axes[4 + idx].set_title(f"{vname} vote")
        axes[4 + idx].axis("off")

    axes[7].axis("off")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==========================================
# INDIVIDUAL MODEL EVALUATION
# Reproduces train_val.py metric calculation exactly.
# Run this first to confirm per-model numbers match reported results.
# ==========================================

def evaluate_individual_models():
    """
    Evaluates each model individually using the same per-image averaging
    as train_val.py. Use this to verify the pipeline reproduces reported
    numbers before trusting ensemble results.
    """
    device = CONFIG["DEVICE"]
    models_info = [
        ("YOLOv12m",    YOLOv12m(num_classes=1),           CONFIG["MODEL_PATHS"]["m1"]),
        ("HybridModel", HybridModel_nb2_(num_classes=1),    CONFIG["MODEL_PATHS"]["m2"]),
        ("EffNetFire",  EfficientNetFire_b2(num_classes=1), CONFIG["MODEL_PATHS"]["m3"]),
    ]

    ds = FireDataset(CONFIG["ROOT_DIR"], split="test", size=CONFIG["IMG_SIZE"])
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    print("\n" + "=" * 70)
    print("INDIVIDUAL MODEL EVALUATION  (per-image averaging, t=0.50)")
    print("=" * 70)
    print(f"{'Model':<18} {'mIoU':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Acc':>7}")
    print("-" * 58)

    for name, model, path in models_info:
        sd = load_checkpoint_state_dict(path, device)
        load_partial_state_dict(model, sd)
        model.to(device).eval()

        all_prec, all_rec, all_f1 = [], [], []
        all_iou_fire, all_iou_bg, all_miou, all_acc = [], [], [], []

        with torch.no_grad():
            for rgb, ir, mask, _, _ in dl:
                rgb, ir, mask = rgb.to(device), ir.to(device), mask.to(device)

                out    = model(rgb, ir)
                logits = out[0] if isinstance(out, tuple) else out
                if logits.shape[2:] != mask.shape[2:]:
                    logits = F.interpolate(logits, size=mask.shape[2:],
                                           mode="bilinear", align_corners=False)

                pred = (torch.sigmoid(logits) > 0.5).float()
                p, r, f1, iou_f, iou_b, miou, acc = calculate_metrics(pred, mask)

                all_prec.append(p);       all_rec.append(r)
                all_f1.append(f1);        all_iou_fire.append(iou_f)
                all_iou_bg.append(iou_b); all_miou.append(miou)
                all_acc.append(acc)

        print(f"{name:<18} {avg(all_miou)*100:>6.2f} {avg(all_f1)*100:>7.2f} "
              f"{avg(all_prec)*100:>7.2f} {avg(all_rec)*100:>7.2f} "
              f"{avg(all_acc)*100:>7.2f}")

    print("=" * 70)


# ==========================================
# THRESHOLD SWEEP  (val set)
# ==========================================

def find_best_threshold(models):
    device = CONFIG["DEVICE"]

    ds_val = FireDataset(CONFIG["ROOT_DIR"], split="val", size=CONFIG["IMG_SIZE"])
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0,
                        pin_memory=(device == "cuda"))

    # store per-image (prob_map, mask) pairs
    all_probs, all_masks = [], []
    for rgb, ir, mask, _, _ in tqdm(dl_val, desc="Finding best threshold"):
        rgb, ir = rgb.to(device), ir.to(device)
        _, avg_prob, _ = ensemble_predict(models, rgb, ir)
        all_probs.append(avg_prob.cpu())
        all_masks.append(mask.cpu())

    best_miou, best_t = 0.0, CONFIG["ENSEMBLE_THRESHOLD"]

    for t in np.arange(0.20, 0.70, 0.02):
        img_mious = []
        for prob, mask in zip(all_probs, all_masks):
            pred = (prob > t).float()
            _, _, _, _, _, miou, _ = calculate_metrics(pred, mask)
            img_mious.append(miou)

        miou_val = avg(img_mious)

        if miou_val > best_miou:
            best_miou = miou_val
            best_t    = round(float(t), 2)

    return best_t, best_miou


# ==========================================
# ENSEMBLE EVALUATION
# ==========================================

def evaluate_ensemble(models=None):
    """
    Evaluates the 3-model weighted logit ensemble on the test set.
    Uses per-image metric averaging to match train_val.py exactly.
    """
    save_dir = increment_path(CONFIG["SAVE_DIR_BASE"])
    os.makedirs(save_dir, exist_ok=True)

    device = CONFIG["DEVICE"]
    if models is None:
        models = load_models()

    ds = FireDataset(CONFIG["ROOT_DIR"], split="test", size=CONFIG["IMG_SIZE"])
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                    pin_memory=(device == "cuda"))

    # per-image metric lists — matches train_val.py
    all_prec, all_rec, all_f1 = [], [], []
    all_iou_fire, all_iou_bg, all_miou, all_acc = [], [], [], []
    saved = 0

    for rgb, ir, mask, fname, _ in tqdm(dl, desc="Evaluating"):
        rgb, ir, mask = rgb.to(device), ir.to(device), mask.to(device)

        pred_mask, avg_prob, individual_votes = ensemble_predict(models, rgb, ir)

        # per-image metrics
        p, r, f1, iou_f, iou_b, miou, acc = calculate_metrics(pred_mask, mask)
        all_prec.append(p);       all_rec.append(r)
        all_f1.append(f1);        all_iou_fire.append(iou_f)
        all_iou_bg.append(iou_b); all_miou.append(miou)
        all_acc.append(acc)

        if saved < CONFIG["SAVE_LIMIT"]:
            save_vis(save_dir, rgb, mask, pred_mask,
                     avg_prob, individual_votes, fname[0])
            saved += 1

    results_txt = (
        f"WEIGHTED SOFT-VOTING ENSEMBLE RESULTS\n"
        f"======================================\n"
        f"IoU (Fire):        {avg(all_iou_fire)*100:.2f}\n"
        f"IoU (Background):  {avg(all_iou_bg)*100:.2f}\n"
        f"mIoU:              {avg(all_miou)*100:.2f}\n"
        f"F1-Score:          {avg(all_f1)*100:.2f}\n"
        f"Precision:         {avg(all_prec)*100:.2f}\n"
        f"Recall:            {avg(all_rec)*100:.2f}\n"
        f"Overall Accuracy:  {avg(all_acc)*100:.2f}\n"
        f"======================================\n"
    )

    print("\n" + results_txt)

    results_path = os.path.join(save_dir, "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(results_txt)
    print(f"Results saved to: {results_path}")

    return avg(all_miou)


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    models = load_models()

    # Step 1 — verify individual model numbers match reported results
    evaluate_individual_models()

    # Step 2 — find best ensemble threshold on val set
    best_t, _ = find_best_threshold(models)
    CONFIG["ENSEMBLE_THRESHOLD"] = best_t

    # Step 3 — run final ensemble evaluation on test set
    evaluate_ensemble(models)