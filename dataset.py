"""
Dataset loading and preprocessing for bone cancer X-ray images.
Expects the following directory structure:

    data/
      train/
        normal/      *.jpg | *.png
        benign/      *.jpg | *.png
        malignant/   *.jpg | *.png
      val/  (same layout)
      test/ (same layout)
"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """Return torchvision transforms for a given split (train | val | test)."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train" and config.AUGMENT_TRAIN:
        return transforms.Compose([
            transforms.Resize((256, 256)),                              # slightly larger then crop
            transforms.RandomCrop(config.IMAGE_SIZE),                   # random crop = free augmentation
            transforms.RandomHorizontalFlip(p=config.AUGMENT_PROB),
            transforms.RandomVerticalFlip(p=config.AUGMENT_PROB * 0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.15, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                    scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.1),                          # X-rays are greyscale anyway
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),         # hides small patches → robustness
        ])

    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(config.IMAGE_SIZE),   # deterministic centre crop at test time
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ─── Dataset ──────────────────────────────────────────────────────────────────

class BoneCancerDataset(Dataset):
    """Custom Dataset for bone cancer X-ray classification."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, root: str, split: str = "train"):
        self.root      = root
        self.split     = split
        self.transform = get_transforms(split)
        self.samples   = []          # list of (image_path, label_idx)
        self.class_to_idx = {c: i for i, c in enumerate(config.CLASS_NAMES)}

        self._load_samples()

    def _load_samples(self):
        for cls_name in config.CLASS_NAMES:
            cls_dir = os.path.join(self.root, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"[WARN] Missing class directory: {cls_dir}")
                continue
            label = self.class_to_idx[cls_name]
            for fname in os.listdir(cls_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.EXTENSIONS:
                    self.samples.append((os.path.join(cls_dir, fname), label))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.root}. "
                "Ensure the directory structure matches the expected layout."
            )

    # ── Class counts for weighted sampling / loss ──────────────────────────
    def class_counts(self) -> list:
        counts = [0] * config.NUM_CLASSES
        for _, lbl in self.samples:
            counts[lbl] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        counts = self.class_counts()
        total  = sum(counts)
        weights = [total / (config.NUM_CLASSES * c) if c > 0 else 0.0
                   for c in counts]
        return torch.tensor(weights, dtype=torch.float32)

    # ── Dunder methods ────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloader(split: str, shuffle: bool = None) -> DataLoader:
    """Build and return a DataLoader for the requested split."""
    root_map = {
        "train": config.TRAIN_DIR,
        "val":   config.VAL_DIR,
        "test":  config.TEST_DIR,
    }
    if split not in root_map:
        raise ValueError(f"Unknown split '{split}'. Choose from {list(root_map)}")

    dataset = BoneCancerDataset(root=root_map[split], split=split)

    if shuffle is None:
        shuffle = (split == "train")

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=(split == "train"),
    )
    print(f"[{split.upper()}] {len(dataset)} images | {len(loader)} batches")
    return loader, dataset
