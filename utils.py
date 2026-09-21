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
from functools import partial
from multiprocessing import Pool
from contextlib import AbstractContextManager
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import Tensor, einsum

tqdm_ = partial(tqdm, dynamic_ncols=True,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]')


class Dcm(AbstractContextManager):
    # Dummy Context manager
    def __exit__(self, *args, **kwargs):
        pass


# Functools
A = TypeVar("A")
B = TypeVar("B")


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    return Pool().starmap(fn, iter)


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
    #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

    assert sset(seg, list(range(K))), (uniq(seg), K)

    b, *img_shape = seg.shape

    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)

    return res


def probs2class(probs: Tensor) -> Tensor:
    b, _, *img_shape = probs.shape
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, K, *_ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


# Save the raw predictions
def save_images(segs: Tensor, names: Iterable[str], root: Path) -> None:
        for seg, name in zip(segs, names):
                save_path = (root / name).with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if len(seg.shape) == 2:
                        Image.fromarray(seg.detach().cpu().numpy().astype(np.uint8)).save(save_path)
                elif len(seg.shape) == 3:
                        np.save(str(save_path), seg.detach().cpu().numpy())
                else:
                        raise ValueError(seg.shape)


# Metrics
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def patient_id_from_stem(stem: str) -> str:
    """Return the patient portion of a sliced filename.

    SegTHOR slices are named like ``Patient_01_0123``.  Non-SegTHOR names are
    returned unchanged, which keeps the helper safe for the toy datasets.
    """
    patient_id, separator, slice_index = stem.rpartition("_")
    if separator and patient_id and slice_index.isdigit():
        return patient_id
    return stem


class PatientVolumeDice:
    """Accumulate exact per-patient 3D Dice statistics from 2D batches.

    Storing intersections and cardinalities is mathematically equivalent to
    stacking every slice into a 3D volume, while using far less memory.  A
    patient/class pair that is empty in both prediction and target is returned
    as NaN instead of receiving an artificial perfect score.
    """

    def __init__(self, classes: int):
        self.classes = classes
        self._intersection: dict[str, Tensor] = {}
        self._cardinality: dict[str, Tensor] = {}

    def update(self, preds: Tensor, target: Tensor, stems: Iterable[str]) -> None:
        assert preds.shape == target.shape
        assert one_hot(preds)
        assert one_hot(target)
        assert preds.shape[1] == self.classes

        stem_list = list(stems)
        assert len(stem_list) == preds.shape[0]

        spatial_dims = tuple(range(2, preds.ndim))
        pred_bool = preds.detach().bool()
        target_bool = target.detach().bool()
        intersection = (pred_bool & target_bool).sum(dim=spatial_dims, dtype=torch.int64).cpu()
        cardinality = (pred_bool.sum(dim=spatial_dims, dtype=torch.int64)
                       + target_bool.sum(dim=spatial_dims, dtype=torch.int64)).cpu()

        for sample_index, stem in enumerate(stem_list):
            patient_id = patient_id_from_stem(stem)
            if patient_id not in self._intersection:
                self._intersection[patient_id] = torch.zeros(self.classes, dtype=torch.int64)
                self._cardinality[patient_id] = torch.zeros(self.classes, dtype=torch.int64)
            self._intersection[patient_id] += intersection[sample_index]
            self._cardinality[patient_id] += cardinality[sample_index]

    def compute(self) -> tuple[list[str], Tensor]:
        if not self._intersection:
            raise RuntimeError("No samples were added to PatientVolumeDice")

        patient_ids = sorted(self._intersection)
        intersection = torch.stack([self._intersection[p] for p in patient_ids]).double()
        cardinality = torch.stack([self._cardinality[p] for p in patient_ids]).double()
        nan = torch.full_like(cardinality, float("nan"))
        dice = torch.where(cardinality > 0, 2 * intersection / cardinality, nan)
        return patient_ids, dice.float()


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a & b
    assert sset(res, [0, 1])

    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])

    res = a | b
    assert sset(res, [0, 1])

    return res
