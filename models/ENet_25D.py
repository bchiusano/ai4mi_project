#!/usr/bin/env python3.10

import torch.nn as nn

from models.ENet import ENet


class ENet_25D(ENet):
        """ENet that accepts several stacked slices (2.5D) as input channels.

        The initial block of ENet concatenates the output of conv0 with a
        max-pooled copy of the input, and the first bottleneck expects exactly
        K channels. The original conv0 produces K - 1 channels, which only adds
        up for a single input channel. Here conv0 produces K - in_dim channels
        instead, so the concatenation has K channels for any in_dim.
        With in_dim=1 this is identical to ENet.
        """

        def __init__(self, in_dim: int, out_dim: int, **kwargs):
                super().__init__(in_dim, out_dim, **kwargs)
                K: int = kwargs["kernels"] if "kernels" in kwargs else 16  # n_kernels
                assert in_dim < K, f"{in_dim=} input channels need more than {K=} initial kernels"

                self.conv0 = nn.Conv2d(in_dim, K - in_dim, kernel_size=3, stride=2, padding=1)
