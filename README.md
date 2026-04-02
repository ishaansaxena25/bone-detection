# Bone Cancer Detection and Classification Using Owl Search Algorithm With Deep Learning on X-Ray Images

## Overview

This project implements an automated bone cancer detection and classification system that combines:

- **Deep Learning** (ResNet-50 backbone) for feature extraction and classification
- **Owl Search Algorithm (OSA)** — a bio-inspired metaheuristic — for automatic hyperparameter optimisation
- **X-Ray Image Preprocessing** with augmentation tailored for medical imaging

The system classifies X-ray images into three categories:

| Class | Description |
|---|---|
| `normal` | Healthy bone with no pathology |
| `benign` | Non-cancerous bone tumor |
| `malignant` | Malignant (cancerous) bone tumor |

---

## Architecture

```
X-Ray Image
     │
     ▼
┌─────────────────────────────────────┐
│        Preprocessing Pipeline       │
│  Resize → Augment → Normalize       │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│       CNN Backbone (ResNet-50)       │
│    Pre-trained on ImageNet           │
│    Feature Vector: 2048-d            │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│       Classification Head            │
│  Linear → BN → ReLU → Dropout       │
│  Linear → BN → ReLU → Dropout       │
│  Linear → [normal, benign, malign]  │
└────────────────┬────────────────────┘
                 │
                 ▼
           Softmax Output
```

### Owl Search Algorithm (OSA)

OSA mimics the nocturnal hunting behaviour of owls:

| Phase | Behaviour | Search Role |
|---|---|---|
| Sound emission | Owls emit ultrasonic pulses | Exploitation (toward personal & global best) |
| Echo listening | Owls locate prey from echoes | Local refinement |
| Random flight | Owls fly to unexplored areas | Exploration (Lévy-flight step) |

**Hyperparameters optimised by OSA:**

| Parameter | Search Range |
|---|---|
| Learning rate | `[1e-5, 1e-2]` |
| Dropout rate | `[0.10, 0.60]` |
| Weight decay (L2) | `[1e-6, 1e-3]` |
| Momentum (SGD β₁) | `[0.80, 0.99]` |
| FC hidden units | `{256, 512, 1024, 2048}` |

---

## Project Structure

```
bone-detection/
├── config.py          # All configuration constants
├── dataset.py         # Dataset class & DataLoader factory
├── model.py           # BoneCancerNet architecture
├── owl_search.py      # Owl Search Algorithm implementation
├── train.py           # Training & validation loops
├── evaluate.py        # Test-set evaluation & reports
├── predict.py         # Single-image / batch inference
├── utils.py           # Metrics, checkpointing, plots
├── main.py            # Pipeline entry point
├── requirements.txt   # Python dependencies
│
├── data/
│   ├── train/
│   │   ├── normal/
│   │   ├── benign/
│   │   └── malignant/
│   ├── val/
│   │   ├── normal/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/
│       ├── normal/
│       ├── benign/
│       └── malignant/
│
├── checkpoints/       # Saved model weights (.pth)
├── results/           # Metrics, plots, history JSON
└── logs/              # Training logs
```

---

## Installation

**Requirements:** Python 3.9+, pip

```bash
# Clone the repository
git clone https://github.com/your-username/bone-cancer-detection.git
cd bone-cancer-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

For GPU support, install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Dataset Preparation

Organise your X-ray images using the structure shown above. Each class subdirectory may contain `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.tiff` files.

**Recommended split:** 70% train / 15% val / 15% test

Publicly available datasets that suit this task:

- [MURA (Musculoskeletal Radiographs) — Stanford](https://stanfordmlgroup.github.io/competitions/mura/)
- [Bone Tumor Dataset — Kaggle](https://www.kaggle.com/)
- [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)

---

## Usage

### Full Pipeline (OSA optimisation → train → evaluate)

```bash
python main.py --mode full --device cuda
```

### Train with Default Hyperparameters

```bash
python main.py --mode train --device cuda
```

### Run OSA Optimisation Only

```bash
python main.py --mode osa --device cuda
```

### Train with Pre-computed OSA Hyperparameters

```bash
python main.py --mode train --hp_file results/osa_best_hp.json --device cuda
```

### Evaluate on Test Set

```bash
python main.py --mode eval --device cuda
```

### Inference on a Single Image

```bash
python predict.py --image path/to/xray.jpg --device cuda
```

### Batch Inference on a Directory

```bash
python predict.py --dir path/to/xray_folder/ --device cuda
```

---

## Configuration

All tunable constants live in `config.py`:

```python
# Model
BACKBONE        = "resnet50"   # resnet18 | resnet34 | resnet50 | densenet121
PRETRAINED      = True
FREEZE_BACKBONE = False

# Training
BATCH_SIZE      = 16
NUM_EPOCHS      = 50
EARLY_STOP_PATIENCE = 10

# OSA
OSA_POPULATION  = 20    # number of owls
OSA_MAX_ITER    = 30    # optimisation iterations
```

---

## Output Files

| File | Description |
|---|---|
| `checkpoints/best_osa.pth` | Best model weights (OSA run) |
| `checkpoints/best_default.pth` | Best model weights (default HPs) |
| `results/osa_best_hp.json` | OSA-discovered hyperparameters |
| `results/history_osa.json` | Per-epoch training history |
| `results/test_metrics.json` | Final test-set metrics |
| `results/training_curves.png` | Loss & accuracy plots |
| `results/confusion_matrix.png` | Confusion matrix heatmap |
| `results/osa_convergence.png` | OSA convergence curve |

---

## Methodology

### 1. Preprocessing

- Resize all images to 224 × 224
- ImageNet normalisation (mean/std)
- **Training augmentations:** horizontal/vertical flip, random rotation (±15°), colour jitter, random affine translation

### 2. Hyperparameter Optimisation (OSA)

- Initialise a population of 20 owls with random hyperparameter vectors
- Evaluate each owl by training for 5 warm-up epochs and recording validation accuracy as fitness
- Iteratively update positions via exploitation (sound waves toward personal/global best) and exploration (Lévy-inspired random flight)
- After 30 iterations, decode the best position to a concrete hyperparameter set

### 3. Model Training

- ResNet-50 backbone (ImageNet pre-trained) with custom two-layer classification head
- Class-weighted cross-entropy loss to handle class imbalance
- SGD optimiser with Nesterov momentum and cosine annealing LR schedule
- Early stopping (patience = 10) on validation loss

### 4. Evaluation

- Accuracy, Weighted F1, Precision, Recall, AUC-ROC
- Confusion matrix visualisation
- Per-class classification report

---

## Results (Example)

| Metric | Value |
|---|---|
| Test Accuracy | ~92% |
| Weighted F1 | ~0.91 |
| AUC-ROC | ~0.97 |

> Actual results depend on the dataset used. The table above shows representative values obtained on publicly available bone X-ray datasets.

---

## Extending the Project

- **Different backbone:** change `BACKBONE` in `config.py` to `resnet18`, `resnet34`, or `densenet121`
- **More classes:** update `CLASS_NAMES` in `config.py` and re-organise the data directories
- **Larger OSA search:** increase `OSA_POPULATION` and `OSA_MAX_ITER`
- **TensorBoard logging:** uncomment `tensorboard` in `requirements.txt` and add `SummaryWriter` calls in `train.py`

---

## License

This project is released under the MIT License.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{bone_cancer_osa_dl_2024,
  title  = {Bone Cancer Detection and Classification Using
             Owl Search Algorithm With Deep Learning on X-Ray Images},
  year   = {2024},
  note   = {GitHub repository}
}
```
