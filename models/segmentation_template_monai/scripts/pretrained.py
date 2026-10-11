"""Initialise SwinUNETR from the MONAI BTCV bundle (huggingface.co/MONAI/swin_unetr_btcv_segmentation).

The BTCV model predicts 14 abdominal classes. Everything except the final 1x1x1 output convolution is loaded as is;
the output layer is rebuilt for the 5 SegTHOR classes, copying the channels of the organs both datasets share.
"""

import torch

# SegTHOR class -> BTCV class: background, esophagus, aorta. Heart (2) and trachea (3) have no BTCV counterpart and
# keep their random initialisation.
SEGTHOR_FROM_BTCV = {0: 0, 1: 5, 4: 8}

OUT_KEYS = ("out.conv.conv.weight", "out.conv.conv.bias")


def load_btcv_weights(network: torch.nn.Module, path: str) -> torch.nn.Module:
    net = network.module if hasattr(network, "module") else network  # unwrap DistributedDataParallel
    state = torch.load(path, map_location="cpu", weights_only=True)
    out_weight, out_bias = (state.pop(k) for k in OUT_KEYS)

    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected or set(missing) != set(OUT_KEYS):
        raise RuntimeError(f"Pretrained weights do not match the network: missing={missing}, unexpected={unexpected}")

    out = net.out.conv.conv
    with torch.no_grad():
        for segthor_class, btcv_class in SEGTHOR_FROM_BTCV.items():
            out.weight[segthor_class] = out_weight[btcv_class]
            out.bias[segthor_class] = out_bias[btcv_class]

    print(f"Loaded pretrained BTCV weights from {path}, output channels initialised from BTCV: {SEGTHOR_FROM_BTCV}")
    return network
