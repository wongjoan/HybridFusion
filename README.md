# HybridFusionNet: Multimodal Wildfire Segmentation using YOLOv12 and EfficientNet

[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv12-blue?style=for-the-badge)](https://github.com/ultralytics/ultralytics)

Official PyTorch implementation of the paper **"Multimodal Wildfire Segmentation using YOLOv12 and EfficientNet"**. 

This repository contains the code for a highly robust, ensemble-based multimodal semantic segmentation framework designed for pixel-level wildfire mapping. By leveraging both Red-Green-Blue (RGB) and Infrared (IR) thermal imagery, the system mitigates common environmental challenges such as dense smoke occlusion, nighttime darkness, and complex background clutter.

---

## 🧠 Architecture Overview

HybridFusionNet does not rely on a single model. Instead, it aggregates predictions from three architecturally distinct dual-stream networks to minimize individual architectural biases:

* **`DualModalYOLO`**: Utilizes parallel YOLOv12m backbones. It fuses modalities via element-wise addition at the feature pyramid (FPN) level. It excels at capturing high-frequency spatial structures using `C3k2` and `A2C2f` attention blocks.
* **`EfficientNetFire`**: Employs dual EfficientNet-B2 backbones. Fusion is handled via **Cross-Modal Attention**, where RGB queries interact with IR key/value pairs, followed by CBAM spatial and channel recalibration.
* **`HybridModel` (Best Single Model)**: An asymmetric architecture pairing a YOLOv12n backbone (for RGB) with an EfficientNet-B2 backbone (for IR). It uses a novel **SpatialGateFusion** module to dynamically learn per-pixel, per-channel mixing weights, allowing the model to adaptively "trust" the IR stream when smoke occludes the RGB view.

---

## ✨ Key Innovations

1. **Spatial Gate Fusion**: Unlike scalar fusion weights, this module generates a `[B, C, H, W]` gate map. Every pixel and channel independently determines its optimal RGB-IR mixing ratio.
2. **Edge-Aware Encoders**: Dedicated lightweight convolutional branches extract high-frequency boundaries directly from raw RGB and IR inputs, explicitly supervising the fire perimeter to prevent boundary degradation in deep layers.
3. **Logit-Space Ensemble**: The ensemble aggregates raw logits (confidence scores) from the three models before applying a single sigmoid activation and threshold. This preserves the "strength" of model convictions better than probability-space averaging.
4. **Hybrid Loss Function**: Combines **Focal Loss** (with online hard example mining on the top 70% hardest pixels), **Dice Loss**, and **Tversky Loss** (heavily penalizing false negatives) to conquer extreme foreground-background class imbalance.

---

## 📁 Repository Structure

```text
HybridFusionNet/
├── model/
│   ├── EfficientNetB2.py          # EfficientNetFire architecture & modules
│   ├── YOLOv12m.py                # DualModalYOLO architecture
│   └── YOLOv12n_EfficientNetB2.py # HybridModel architecture & SpatialGateFusion
├── train.py                       # Main training script (Loss, Optimizer, Loop)
├── val.py                         # Standalone validation metric calculation
├── ensemble.py                    # Logit-space soft-voting ensemble & threshold sweep
├── requirements.txt               # Dependency list
└── README.md
```

---

## ⚙️ Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/wongjoan/HybridFusion.git
cd HybridFusion
```

**2. Create a virtual environment (Recommended):**
```bash
conda create -n wildfire_env python=3.9
conda activate wildfire_env
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

The `requirements.txt` contains:
```text
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.1.0
albumentations>=1.3.0
opencv-python
numpy
tqdm
matplotlib
```

---

## 📊 Dataset Preparation

The `FireDataset` class requires perfectly aligned/registered RGB and IR image pairs along with binary ground-truth masks. The folder structure must mirror the following:

```text
dataset_root/
├── train/
│   ├── rgb/           # e.g., 001_rgb.jpg
│   ├── ir/            # e.g., 001_ir.jpg (1-channel grayscale)
│   └── masks/         # e.g., 001_gt.png (binary 0/255)
└── test/ (or val/)
    ├── rgb/
    ├── ir/
    └── masks/
```

* **Augmentations**: Geometric transforms (flips, scaling, rotations) are applied jointly to RGB, IR, and the mask to preserve alignment. Photometric transforms (HSV, CLAHE, noise) are applied **only** to the RGB stream.

---

## 📥 Pretrained Weights

To achieve the reported metrics, the backbones must be initialized with pretrained weights.

1. **EfficientNet-B2**: Weights are automatically downloaded from `torchvision` (`weights='DEFAULT'`). The 1-channel IR stem adapts these weights by summing the 3-channel RGB filters.
2. **YOLOv12 Weights**: You must download the base YOLOv12 segmentation weights from Ultralytics and place them in your project folder.
   * Download `yolov12n-seg.pt` (for `HybridModel`)
   * Download `yolov12m-seg.pt` (for `DualModalYOLO`)
   
Update the paths inside the scripts where weights are loaded.

---

## 🚀 Training the Models

You must train each model individually before running the ensemble.

1. Open `train.py`.
2. Update the `CONFIG` dictionary:
   ```python
   CONFIG = {
       "ROOT_DIR": "/path/to/dataset",
       "SAVE_DIR": "./runs/hybrid_model",
       "DEVICE": "cuda",
       "BATCH_SIZE": 2,      
       "EPOCHS": 100,
       # ...
   }
   ```
3. Uncomment the model you wish to train in the `train()` function:
   ```python
   model = HybridModel_nb2_().to(CONFIG["DEVICE"])
   # model = YOLOv12m(num_classes=1).to(CONFIG["DEVICE"])
   # model = EfficientNetFire_b2(num_classes=1).to(CONFIG["DEVICE"])
   ```
4. Run training:
   ```bash
   python train.py
   ```

Weights will be saved as `best.pt` and `last.pt` in the configured `SAVE_DIR`.

---

## 🔬 Evaluation & Ensemble Inference

### Single Model Validation
To evaluate a single trained model on the test set:
```bash
python val.py
```

### Weighted Logit Ensemble
To run the full HybridFusionNet ensemble:
1. Update `CONFIG["MODEL_PATHS"]` in `ensemble.py` with paths to your three `best.pt` files.
2. (Optional) Adjust `CONFIG["WEIGHTS"]`. Default is `[0.2, 0.25, 0.55]`.
3. Run:
   ```bash
   python ensemble.py
   ```

**The ensemble script:**
1. Evaluates each model individually to verify correct loading
2. Finds the optimal binary threshold on the validation set
3. Runs final inference on the test set
4. Outputs `mIoU`, `F1`, `Precision`, `Recall`, and visualizations

---

## 📈 Quantitative Results

### Main Benchmark: Corsican Fire Database (CFDB)

| Model | Modality | mIoU (%) | F1-Score (%) | Precision (%) | Recall (%) | Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| DeepLabv3+ (Baseline) | RGB | 80.29 | 87.86 | - | - | 96.11 |
| YOLOv12m-seg | RGB Only | 87.37 | 93.26 | 93.14 | 93.38 | 99.27 |
| DualModalYOLO | RGB + IR | 91.68 | 91.43 | 91.13 | 92.19 | 98.77 |
| EfficientNetFire | RGB + IR | 91.98 | 91.83 | 90.39 | 93.31 | 99.11 |
| **HybridModel** | RGB + IR | 92.16 | 91.95 | 91.27 | 93.00 | 98.84 |
| **Soft Ensemble** | **RGB + IR** | **92.56** | **92.39** | 92.15 | 92.91 | 98.90 |

### Ablation Study: HybridModel Components

| Configuration | mIoU (%) | F1-Score (%) |
| :--- | :---: | :---: |
| Baseline | 87.01 | 84.41 |
| w/o Aux Head | 86.75 | 84.10 |
| w/o Spatial Gate Fusion | 86.83 | 84.10 |
| w/o CBAM | 86.88 | 84.20 |
| w/o Edge Encoder | 86.98 | 84.43 |
| **Full HybridModel** | **89.56** | **88.49** |

---

## 🗄️ Data Availability

The experiments rely on external wildfire benchmarks. Due to licensing policies, the raw datasets are not hosted here but are available upon request:

* **Corsican Fire Database (CFDB)**
  * Available upon request: http://cfdb.univ-corse.fr
  * Citation: Toulouse et al. (2017)

* **RGB-T Wildfire Dataset**
  * Available upon request from original authors
  * Citation: Rui et al. (2023)

---

## 📜 Citation

If you use this code in your research, please cite:

```bibtex
@article{hybridfusionnet2026,
  title={Multimodal Wildfire Segmentation using YOLOv12 and EfficientNet},
  author={Author, Name},
  journal={Journal Name},
  year={2026}
}
```

---

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open issues and submit pull requests.

---

## 📧 Contact

For inquiries, please contact: your-email@example.com
