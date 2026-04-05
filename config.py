"""
Configuration settings for Bone Cancer Detection and Classification
Using Owl Search Algorithm with Deep Learning on X-Ray Images
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
TRAIN_DIR       = os.path.join(DATA_DIR, "train")
VAL_DIR         = os.path.join(DATA_DIR, "val")
TEST_DIR        = os.path.join(DATA_DIR, "test")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
LOG_DIR         = os.path.join(BASE_DIR, "logs")

for d in [CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
# NOTE: benign is excluded — no training images available.
# Add images to data/train/benign/ and restore "benign" here to enable 3-class mode.
CLASS_NAMES = [
    "normal",
    "malignant",
]
NUM_CLASSES  = len(CLASS_NAMES)
IMAGE_SIZE   = (224, 224)   # (H, W)
CHANNELS     = 3

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE          = 32    # larger batch → more stable gradients
NUM_EPOCHS          = 40
NUM_WORKERS         = 0     # 0 = main process only (safe on Windows)
PIN_MEMORY          = False
EARLY_STOP_PATIENCE = 8

# ─── Owl Search Algorithm (OSA) Hyper-parameters ──────────────────────────────
OSA_POPULATION   = 20
OSA_MAX_ITER     = 30
OSA_DIM          = 5

# Search bounds: [lr, dropout, weight_decay, momentum, fc_units_idx]
OSA_LOWER_BOUNDS = [1e-5,  0.1,  1e-6, 0.80,  0]
OSA_UPPER_BOUNDS = [1e-2,  0.6,  1e-3, 0.99,  3]

# Discrete choices for fully-connected hidden units
FC_UNIT_CHOICES  = [256, 512, 1024, 2048]

# ─── Model ────────────────────────────────────────────────────────────────────
BACKBONE        = "resnet50"
PRETRAINED      = True
FREEZE_BACKBONE = False

# Default hyper-parameters (overridden by OSA result after optimisation)
DEFAULT_LR           = 3e-4   # Adam sweet-spot for fine-tuning ResNet
DEFAULT_BACKBONE_LR  = 3e-5   # 10× lower LR for pretrained backbone layers
DEFAULT_DROPOUT      = 0.4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MOMENTUM     = 0.9
DEFAULT_FC_UNITS     = 512
LABEL_SMOOTHING      = 0.1    # regularises overconfident predictions

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUGMENT_TRAIN = True
AUGMENT_PROB  = 0.5

# ─── Evaluation ───────────────────────────────────────────────────────────────
THRESHOLD = 0.5
