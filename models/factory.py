"""Build student backbones as feature extractors; falls back to timm for unknown archs."""
from __future__ import annotations

import torch.nn as nn

from models import repvit as _repvit
from models.mobileone import mobileone

# RepViT factories
_REPVIT = {
    "repvit_m0_6": _repvit.repvit_m0_6,
    "repvit_m0_9": _repvit.repvit_m0_9,
    "repvit_m1_0": _repvit.repvit_m1_0,
    "repvit_m1_1": _repvit.repvit_m1_1,
    "repvit_m1_5": _repvit.repvit_m1_5,
    "repvit_m2_3": _repvit.repvit_m2_3,
}

# MobileOne variants live in mobileone.PARAMS (mu0..mu2, s0..s4).


def build_backbone(arch: str, num_classes: int = 0) -> nn.Module:
    """Build a student backbone; num_classes<=0 returns the pooled feature."""
    a = arch.lower()
    if a.startswith("mobileone_"):
        variant = a.split("_", 1)[1]  # mu0/mu1/mu2/s0..s4
        n = num_classes if num_classes and num_classes > 0 else 1000
        m = mobileone(variant=variant, num_classes=n)
        if not num_classes or num_classes <= 0:
            # pooled feature (leave the Apple class body untouched)
            m.linear = nn.Identity()
        return m
    if a in _REPVIT:
        return _REPVIT[a](num_classes=num_classes)
    import timm

    return timm.create_model(a, pretrained=False, num_classes=num_classes)
