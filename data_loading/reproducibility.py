"""Reusable seeding helpers for models and data loaders.

The dataset split seed is deliberately separate from a training run seed.
Keep the split seed fixed while varying initialization and data-loader seeds.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and deterministic backend settings."""

    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Max-unpool and a small number of CUDA operations may not have a strictly
    # deterministic implementation on every PyTorch/CUDA combination. Warn
    # rather than crashing so the same code remains usable by peer models.
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_data_loader_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the deterministic PyTorch worker seed."""

    del worker_id  # The unique seed is supplied by DataLoader via torch.initial_seed().
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def data_loader_generator(seed: int) -> torch.Generator:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return torch.Generator().manual_seed(seed)
