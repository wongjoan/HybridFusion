# HybridFusionNet: Multimodal Wildfire Segmentation using YOLOv12 and EfficientNet

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv12-blue?style=for-the-badge)](https://github.com/ultralytics/ultralytics)

PyTorch code for RGB-infrared (RGB-IR) wildfire segmentation. The project was developed during a research internship at Deakin University's Institute for Intelligent Systems Research and Innovation (IISRI). 

The framework segments active fire pixels from aligned RGB and IR image pairs. Fusing the two modalities targets common failure cases for RGB-only models: smoke occlusion, low light and cluttered backgrounds.

---

## Architecture Overview

Three dual-stream networks are trained separately, then optionally combined by a logit-space ensemble:

- **`DualModalYOLO`**: Parallel YOLOv12m backbones. Modalities are fused by element-wise addition at the feature pyramid (FPN) level, using `C3k2` and `A2C2f` attention blocks.
- **`EfficientNetFire`**: Dual EfficientNet-B2 backbones. Fusion uses cross-modal attention (RGB queries, IR keys/values), followed by CBAM spatial and channel recalibration.
- **`HybridModel`**: An asymmetric design pairing a YOLOv12n backbone (RGB) with an EfficientNet-B2 backbone (IR). A **SpatialGateFusion** module learns per-pixel, per-channel mixing weights, so the network can weight the IR stream more where smoke obscures the RGB view.

---

## Key Components

1. **Spatial Gate Fusion**: Produces a `[B, C, H, W]` gate map, so every pixel and channel learns its own RGB/IR mixing ratio instead of using one scalar weight.
2. **Edge-Aware Encoders**: Lightweight convolutional branches extract high-frequency boundaries from the raw RGB and IR inputs and supervise the fire perimeter.
3. **Logit-Space Ensemble**: Raw logits from the three models are combined as a weighted sum before a single sigmoid and threshold.
4. **Hybrid Loss**: Focal loss (hard-pixel mining on the top 70% highest-loss pixels), Dice loss and Tversky loss (penalizing false negatives) for strong foreground/background imbalance.

---

## Repository Structure

~~~
HybridFusionNet/
├── model/
│   ├── EfficientNetB2.py          # EfficientNetFire architecture & modules
│   ├── YOLOv12m.py                # DualModalYOLO architecture
│   └── YOLOv12n_EfficientNetB2.py # HybridModel architecture & SpatialGateFusion
├── train.py                       # Training script (loss, optimizer, loop)
├── val.py                         # Standalone validation metrics
├── ensemble.py                    # Logit-space ensemble & threshold sweep
├── requirements.txt
└── README.md
~~~

---

## Installation

~~~bash
git clone https://github.com/wongjoan/HybridFusion.git
cd HybridFusion
conda create -n wildfire_env python=3.9
conda activate wildfire_env
pip install -r requirements.txt
~~~

`requirements.txt`: torch>=2.0.0, torchvision>=0.15.0, ultralytics>=8.1.0, albumentations>=1.3.0, opencv-python, numpy, tqdm, matplotlib.

---

## Datasets

Experiments were run separately on two public benchmarks. Raw data is not redistributed here because of licensing.

| Dataset | View | Pairs used | Train | Validation |
|---|---|---|---|---|
| Corsican Fire Database (CFDB), Toulouse et al. (2017) | Ground | 100 (curated subset) | 80 | 20 |
| RGB-T Wildfire Dataset, Rui et al. (2023) | Ground and aerial (UAV) | 1,367 | 1,221 | 146 |

Both are available from their original authors (CFDB: http://cfdb.univ-corse.fr). The split is a single 80:20 hold-out per dataset. There is no separate test set.

`FireDataset` expects registered RGB/IR pairs with binary masks:

~~~
dataset_root/
├── train/
│   ├── rgb/     # e.g. 001_rgb.jpg
│   ├── ir/      # e.g. 001_ir.jpg (1-channel grayscale)
│   └── masks/   # e.g. 001_gt.png (binary 0/255)
└── val/
    ├── rgb/
    ├── ir/
    └── masks/
~~~

**Preprocessing:** RGB and IR are resized to 640x640 and scaled to [0, 1]. Geometric augmentations (horizontal flip p=0.5, vertical flip p=0.2, affine p=0.5 with scale 0.9-1.1, rotation +-10 degrees, translation +-5%, shear +-5 degrees) are applied jointly to RGB, IR and the mask. Photometric augmentations (brightness/contrast, HSV, Gaussian noise, CLAHE) are applied to the RGB stream only. Validation uses resizing and normalization only.

---

## Pretrained Weights

- **EfficientNet-B2**: Downloaded automatically from `torchvision` (`weights='DEFAULT'`). The 1-channel IR stem sums the 3-channel RGB filters.
- **YOLOv12**: Download `yolov12n-seg.pt` (for `HybridModel`) and `yolov12m-seg.pt` (for `DualModalYOLO`) from Ultralytics and update the weight paths in the scripts.

---

## Training

1. Edit the `CONFIG` dictionary in `train.py`:

~~~python
CONFIG = {
    "ROOT_DIR": "/path/to/dataset",
    "SAVE_DIR": "./runs/hybrid_model",
    "DEVICE": "cuda",
    "BATCH_SIZE": 2,
    "EPOCHS": 100,
}
~~~

2. Uncomment the model to train in `train()`:

~~~python
model = HybridModel_nb2_().to(CONFIG["DEVICE"])
# model = YOLOv12m(num_classes=1).to(CONFIG["DEVICE"])
# model = EfficientNetFire_b2(num_classes=1).to(CONFIG["DEVICE"])
~~~

3. Run `python train.py`. Weights are saved as `best.pt` and `last.pt`.

**Training setup (all models):** AdamW, initial learning rate 1e-3 (minimum 1e-6) with cosine annealing and a 3-epoch linear warm-up, weight decay 1e-4, batch size 2, up to 100 epochs, early stopping (patience 20) on validation mIoU, gradient clipping at 10.0, input 640x640. Trained on one NVIDIA RTX 5070 (12 GB).

**Loss:** `L = L_primary + 0.4 * L_aux`, where `L_primary = 1.0 * Focal + 0.5 * Dice + 0.5 * Tversky` (Focal gamma=2.0 with hard-pixel mining on the top 70%; Tversky alpha=0.3, beta=0.7). Auxiliary supervision is applied at the P4 and P5 heads.

---

## Evaluation and Ensemble

Single-model validation: `python val.py` (probability threshold 0.5).

Ensemble: set `CONFIG["MODEL_PATHS"]` in `ensemble.py` to your three `best.pt` files and run `python ensemble.py`. The default weights are `[0.2, 0.25, 0.55]`. The script evaluates each model, sweeps the binary threshold on the validation set (grid search over 0.20-0.65 in steps of 0.02; best value 0.46 on CFDB), and reports mIoU, F1, precision, recall and visualizations.

Because the ensemble threshold is chosen on the same validation images that are reported, ensemble numbers are optimistic compared with the single-model numbers, which use a fixed 0.5 threshold.

---

## Results

### Metric definitions

All metrics are computed per image and averaged over the validation set.

- **F1, Precision, Recall**: fire class, pixel-wise.
- **mIoU**: mean of fire-class IoU and background-class IoU. Because background pixels dominate, this value is higher than fire-class IoU alone. **F1 is the better metric for comparing models.**
- **Acc**: pixel accuracy.

The unimodal baseline mIoU values came from earlier runs and appear to use a different definition (they match fire-class IoU). They are marked "n/a" below until the baselines are re-evaluated with the same script. F1, precision, recall and accuracy are comparable across all rows.

### CFDB (20 validation pairs)

| Model | Modality | mIoU (%) | F1 (%) | Precision (%) | Recall (%) | Acc (%) |
|---|---|---|---|---|---|---|
| YOLOv12m-seg | RGB | n/a | 93.26 | 93.14 | 93.38 | 99.27 |
| YOLOv12m-seg | IR | n/a | 74.49 | 71.70 | 77.51 | 97.14 |
| EfficientNet-B0 | RGB | n/a | 92.64 | 94.20 | 91.40 | 99.27 |
| EfficientNet-B0 | IR | n/a | 89.51 | 90.95 | 88.20 | 98.64 |
| DualModalYOLO | RGB+IR | 91.68 | 91.43 | 91.13 | 92.19 | 98.77 |
| EfficientNetFire | RGB+IR | 91.98 | 91.83 | 90.39 | 93.31 | 99.11 |
| HybridModel | RGB+IR | 92.16 | 91.95 | 91.27 | 93.00 | 98.84 |
| Soft ensemble | RGB+IR | 92.56 | 92.39 | 92.15 | 92.91 | 98.90 |

On this small validation set, the RGB-only YOLOv12m-seg baseline remains competitive on F1 and accuracy.

### RGB-T Wildfire Dataset (146 validation pairs)

| Model | mIoU (%) | F1 (%) |
|---|---|---|
| Dual YOLOv12 | 85.70 | 81.81 |
| Dual EfficientNet | 87.87 | 86.36 |
| **HybridModel** | **89.56** | **88.49** |
| Soft ensemble | 87.40 | 84.06 |

HybridModel is the strongest model on RGB-T. The ensemble does not improve on it there, so the CFDB ensemble gain does not carry over to this dataset.

### Ablation: HybridModel components (RGB-T)

| Configuration | mIoU (%) | F1 (%) |
|---|---|---|
| Baseline (bare minimum) | 87.01 | 84.41 |
| w/o aux head | 86.75 | 84.10 |
| w/o spatial gate fusion | 86.83 | 84.10 |
| w/o CBAM | 86.88 | 84.20 |
| w/o edge encoder | 86.98 | 84.43 |
| **Full HybridModel** | **89.56** | **88.49** |

Each removal lands close to the baseline, so the components appear to help mainly in combination. Each configuration was trained once and seed variance is not reported.

### Prior work (reported numbers, not re-run)

Published results use different splits and metric implementations, so only F1 is listed and it should be read as indicative.

| Dataset | Method | F1 (%) |
|---|---|---|
| CFDB | Harkat et al., DeepLabv3+ | 87.86 |
| CFDB | Harkat et al. | 85.84 |
| RGB-T | Rui et al. | 93.10 |

---

## Limitations

- Small validation sets (20 pairs on CFDB, 146 on RGB-T) and a single hold-out split, with no separate test set. Differences of a few tenths of a point are within noise.
- Model selection (early stopping on validation mIoU) and the ensemble threshold use the same validation images that are reported.
- No repeated runs or confidence intervals.
- The ablation shows the full model against ablated variants, but not each component's isolated contribution.

---

## Citation

A citation entry will be added when the accompanying manuscript is available. Until then, please reference this repository:

~~~
Wong Jing Tao Joanne. HybridFusionNet: Multimodal Wildfire Segmentation using YOLOv12 and EfficientNet.
GitHub repository, https://github.com/wongjoan/HybridFusion, 2026.
~~~

Dataset citations: Toulouse et al. (2017) for CFDB; Rui et al. (2023) for the RGB-T Wildfire Dataset.

---

## License

MIT License. See `LICENSE`.

## Contact

Wong Jing Tao Joanne: wongjingtao1001@gmail.com
