#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union

import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset

from utils import patient_id_from_stem


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


def neighbour_indices(paths: list[Path], context: int) -> list[list[int]]:
    """For each slice, the indices of the 2 * context + 1 slices centered on it.

    Expects the paths sorted by patient, then slice number (as make_dataset does).
    Neighbours never cross a patient boundary: at the first and last slices of a
    volume, the edge slice is repeated.
    """
    patients: list[str] = [patient_id_from_stem(p.stem) for p in paths]

    # First and last index of each patient's slices
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    for i, patient in enumerate(patients):
        first.setdefault(patient, i)
        last[patient] = i

    return [[min(max(i + d, first[patient]), last[patient]) for d in range(-context, context + 1)]
            for i, patient in enumerate(patients)]


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False,
                 context: int = 0):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize
        self.context: int = context  # Number of neighbouring slices on each side (2.5D input)

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]
        self.neighbours = neighbour_indices([img for img, _ in self.files], context)

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor
        if self.context == 0:
            img = self.img_transform(Image.open(img_path))
        else:
            # Stack the neighbouring slices as channels, the current one in the middle
            img = torch.cat([self.img_transform(Image.open(self.files[j][0]))
                             for j in self.neighbours[index]], dim=0)

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict
