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

import json
import re
from pathlib import Path
from typing import Callable, Union

from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset


PATIENT_PATTERN = re.compile(r"^(Patient_\d+)_\d+$")


def patient_from_slice(path: Path) -> str:
    match = PATIENT_PATTERN.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unexpected SegTHOR slice filename: {path.name}")
    return match.group(1)


def patient_ids_for_subset(root: Path, subset: str) -> set[str] | None:
    """Return IDs from a pooled split manifest, or ``None`` for legacy folders."""

    split_path = root / "split.json"
    pooled_images = root / "img"
    if not split_path.is_file() or not pooled_images.is_dir():
        return None

    split = json.loads(split_path.read_text(encoding="utf-8"))
    return set(split[subset])


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    selected_patients = patient_ids_for_subset(root, subset)
    if selected_patients is None:
        img_path = root / subset / 'img'
        full_path = root / subset / 'gt'
    else:
        img_path = root / 'img'
        full_path = root / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    if selected_patients is not None:
        images = [path for path in images if patient_from_slice(path) in selected_patients]

    labels_by_name = {path.name: path for path in full_path.glob("*.png")}
    if labels_by_name:
        missing = [path.name for path in images if path.name not in labels_by_name]
        if missing:
            raise RuntimeError(f"Missing labels for {len(missing)} images; first: {missing[0]}")
        full_labels: list[Path | None] = [labels_by_name[path.name] for path in images]
    else:
        full_labels = [None] * len(images)

    if not images:
        raise RuntimeError(f"No images found for subset={subset!r}, root={root}")

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, debug=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]
        self.has_labels = all(gt_path is not None for _, gt_path in self.files)

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = self.img_transform(Image.open(img_path))

        data_dict = {"stems": img_path.stem}

        if self.has_labels:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        data_dict["images"] = img

        return data_dict
