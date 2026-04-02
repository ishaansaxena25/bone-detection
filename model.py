"""
Deep Learning model for Bone Cancer Detection and Classification.

Architecture:
  - Pre-trained CNN backbone (ResNet-50 / DenseNet-121)
  - Custom fully-connected classification head
  - Hyper-parameters injected by the Owl Search Algorithm
"""

import torch
import torch.nn as nn
from torchvision import models

import config


# ─── Backbone factory ─────────────────────────────────────────────────────────

def _build_backbone(name: str, pretrained: bool):
    """Return (backbone_model, feature_dim) pair."""
    weights_arg = "IMAGENET1K_V1" if pretrained else None

    if name == "resnet18":
        backbone = models.resnet18(weights=weights_arg)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif name == "resnet34":
        backbone = models.resnet34(weights=weights_arg)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif name == "resnet50":
        backbone = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif name == "densenet121":
        backbone = models.densenet121(weights=weights_arg)
        feat_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()

    else:
        raise ValueError(f"Unsupported backbone: {name}")

    return backbone, feat_dim


# ─── Main Model ───────────────────────────────────────────────────────────────

class BoneCancerNet(nn.Module):
    """
    Bone Cancer Detection Network.

    Parameters
    ----------
    backbone     : CNN architecture name (see config.BACKBONE)
    pretrained   : load ImageNet weights
    num_classes  : number of output classes
    fc_units     : hidden units in the classification head
    dropout      : dropout probability
    freeze       : freeze backbone weights (feature extraction mode)
    """

    def __init__(
        self,
        backbone:    str  = config.BACKBONE,
        pretrained:  bool = config.PRETRAINED,
        num_classes: int  = config.NUM_CLASSES,
        fc_units:    int  = config.DEFAULT_FC_UNITS,
        dropout:     float = config.DEFAULT_DROPOUT,
        freeze:      bool = config.FREEZE_BACKBONE,
    ):
        super().__init__()
        self.backbone_name = backbone

        self.backbone, feat_dim = _build_backbone(backbone, pretrained)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_units, fc_units // 2),
            nn.BatchNorm1d(fc_units // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(fc_units // 2, num_classes),
        )

        self._init_head()

    def _init_head(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits   = self.classifier(features)
        return logits

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone feature embeddings (before classification head)."""
        with torch.no_grad():
            return self.backbone(x)


# ─── Convenience builders ─────────────────────────────────────────────────────

def build_model(
    fc_units:    int   = config.DEFAULT_FC_UNITS,
    dropout:     float = config.DEFAULT_DROPOUT,
    device:      str   = "cpu",
) -> BoneCancerNet:
    """Instantiate and move model to device."""
    model = BoneCancerNet(fc_units=fc_units, dropout=dropout)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config.BACKBONE} | trainable params: {n_params:,}")
    return model


def load_checkpoint(path: str, device: str = "cpu") -> BoneCancerNet:
    """Load a saved model checkpoint."""
    ckpt  = torch.load(path, map_location=device)
    model = BoneCancerNet(
        fc_units = ckpt.get("fc_units", config.DEFAULT_FC_UNITS),
        dropout  = ckpt.get("dropout",  config.DEFAULT_DROPOUT),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"Loaded checkpoint: {path}")
    return model
