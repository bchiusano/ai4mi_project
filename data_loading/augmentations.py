"""Paired online augmentations for 2-D CT segmentation.

All random transformations are intentionally mild. Spatial transforms are
sampled once and applied to both the CT and class mask; intensity transforms
are applied only to the CT. Validation and test datasets should pass no joint
transform at all.

The image may have several channels (2.5D input: neighbouring slices stacked as
channels); every channel gets the same spatial and intensity transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as torch_functional
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional


AUGMENTATION_MODES = ("none", "geometric", "intensity", "combined")


def _uniform(low: float, high: float) -> float:
    return float(torch.empty(1).uniform_(low, high).item())


def _adjust_contrast(image: Tensor, factor: float) -> Tensor:
    """Blend with the mean intensity, like torchvision's adjust_contrast on a single channel.

    torchvision treats a 3-channel tensor as RGB and rejects other channel counts, but the
    channels of a 2.5D input are neighbouring CT slices, which share one mean.
    """
    mean = image.mean()
    return torch.clamp(mean + factor * (image - mean), 0, 1)


def _adjust_gamma(image: Tensor, gamma: float) -> Tensor:
    """Same as torchvision's adjust_gamma for float images, which also only accepts 1 or 3 channels."""
    return torch.clamp(image, 0, 1) ** gamma


@dataclass(frozen=True)
class SegmentationAugmentation:
    geometric: bool = True
    intensity: bool = True
    geometric_probability: float = 0.5
    intensity_probability: float = 0.5
    max_rotation_degrees: float = 5.0
    max_translation_fraction: float = 0.03
    scale_range: tuple[float, float] = (0.9, 1.1)
    brightness_delta: float = 0.02
    contrast_range: tuple[float, float] = (0.95, 1.05)
    gamma_range: tuple[float, float] = (0.95, 1.05)
    max_noise_sigma: float = 0.01

    def __post_init__(self) -> None:
        for name, probability in (
            ("geometric_probability", self.geometric_probability),
            ("intensity_probability", self.intensity_probability),
        ):
            if not 0 <= probability <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_rotation_degrees < 0 or self.max_translation_fraction < 0:
            raise ValueError("Geometric augmentation limits must be non-negative")
        if self.scale_range[0] <= 0 or self.scale_range[0] > self.scale_range[1]:
            raise ValueError("Invalid scale range")
        if self.brightness_delta < 0 or self.max_noise_sigma < 0:
            raise ValueError("Intensity augmentation limits must be non-negative")

    def _geometric(self, image: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        _, height, width = image.shape
        angle = _uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
        max_x = self.max_translation_fraction * width
        max_y = self.max_translation_fraction * height
        translate = [int(round(_uniform(-max_x, max_x))), int(round(_uniform(-max_y, max_y)))]
        scale = _uniform(*self.scale_range)

        image = vision_functional.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

        class_mask = target.argmax(dim=0, keepdim=True).to(torch.float32)
        class_mask = vision_functional.affine(
            class_mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        ).round().to(torch.int64)
        target = torch_functional.one_hot(
            class_mask.squeeze(0), num_classes=target.shape[0]
        ).permute(2, 0, 1).to(target.dtype)
        return image, target

    def _intensity(self, image: Tensor) -> Tensor:
        image = torch.clamp(image + _uniform(-self.brightness_delta, self.brightness_delta), 0, 1)
        image = _adjust_contrast(image, _uniform(*self.contrast_range))
        image = _adjust_gamma(image, _uniform(*self.gamma_range))
        sigma = _uniform(0.0, self.max_noise_sigma)
        if sigma > 0:
            image = image + torch.randn_like(image) * sigma
        return torch.clamp(image, 0, 1)

    def __call__(self, image: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if image.ndim != 3 or target.ndim != 3:
            raise ValueError("Expected image and one-hot target with shape (C, H, W)")
        if image.shape[1:] != target.shape[1:]:
            raise ValueError("Image and target spatial dimensions differ")

        if self.geometric and torch.rand(1).item() < self.geometric_probability:
            image, target = self._geometric(image, target)
        if self.intensity and torch.rand(1).item() < self.intensity_probability:
            image = self._intensity(image)

        if not torch.all(target.sum(dim=0) == 1):
            raise RuntimeError("Augmentation produced a non one-hot target")
        return image.contiguous(), target.contiguous()


def build_augmentation(mode: str) -> SegmentationAugmentation | None:
    if mode not in AUGMENTATION_MODES:
        raise ValueError(f"Unknown augmentation {mode!r}; choose from {AUGMENTATION_MODES}")
    if mode == "none":
        return None
    return SegmentationAugmentation(
        geometric=mode in {"geometric", "combined"},
        intensity=mode in {"intensity", "combined"},
    )
