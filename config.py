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
CLASS_NAMES = [
    "normal",
    "benign",
    "malignant",
]
NUM_CLASSES  = len(CLASS_NAMES)
IMAGE_SIZE   = (224, 224)   # (H, W)
CHANNELS     = 3

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE   = 16
NUM_EPOCHS   = 50
NUM_WORKERS  = 4
PIN_MEMORY   = True
EARLY_STOP_PATIENCE = 10

# ─── Owl Search Algorithm (OSA) Hyper-parameters ──────────────────────────────
OSA_POPULATION   = 20       # number of owls (candidate solutions)
OSA_MAX_ITER     = 30       # optimisation iterations
OSA_DIM          = 5        # dimensionality = number of HPs being tuned

# Search bounds: [lr, dropout, weight_decay, momentum, fc_units_idx]
OSA_LOWER_BOUNDS = [1e-5,  0.1,  1e-6, 0.80,  0]
OSA_UPPER_BOUNDS = [1e-2,  0.6,  1e-3, 0.99,  3]

# Discrete choices for fully-connected hidden units
FC_UNIT_CHOICES  = [256, 512, 1024, 2048]

# ─── Model ────────────────────────────────────────────────────────────────────
BACKBONE        = "resnet50"   # resnet18 | resnet34 | resnet50 | densenet121
PRETRAINED      = True
FREEZE_BACKBONE = False        # set True to fine-tune only the head

# Default hyper-parameters (overridden by OSA result after optimisation)
DEFAULT_LR           = 1e-4
DEFAULT_DROPOUT      = 0.3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MOMENTUM     = 0.9
DEFAULT_FC_UNITS     = 512

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUGMENT_TRAIN = True
AUGMENT_PROB  = 0.5

# ─── Evaluation ───────────────────────────────────────────────────────────────
THRESHOLD = 0.5   # binary probability threshold (used for two-class sub-tasks)
