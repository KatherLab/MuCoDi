# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Adapted from Facebook Research MoCo v3 under CC BY-NC 4.0.
# See NOTICE and LICENSES/CC-BY-NC-4.0.txt.

import random

from PIL import ImageFilter, ImageOps


class TwoCropsTransform:
    """Create two independently augmented views of one image."""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, image):
        return [self.base_transform1(image), self.base_transform2(image)]


class GaussianBlur:
    """Gaussian blur augmentation used by SimCLR and MoCo v3."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, image):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))


class Solarize:
    """Solarization augmentation used by BYOL and MoCo v3."""

    def __call__(self, image):
        return ImageOps.solarize(image)
