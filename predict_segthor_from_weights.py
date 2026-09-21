#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ENet import ENet


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SegTHOR masks from saved ENet weights")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    image_paths = sorted(args.image_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {args.image_dir}")
    existing = list(args.output_dir.glob("*.png")) if args.output_dir.exists() else []
    if existing:
        raise RuntimeError(
            f"Output directory already contains {len(existing)} PNG files: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = ENet(1, 5, kernels=8, factor=2)
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.inference_mode():
        for start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[start:start + args.batch_size]
            arrays = [
                np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
                for path in batch_paths
            ]
            images = torch.from_numpy(np.stack(arrays)[:, None, :, :])
            predictions = model(images).argmax(dim=1).cpu().numpy().astype(np.uint8)
            for path, prediction in zip(batch_paths, predictions):
                Image.fromarray(prediction * 63).save(args.output_dir / path.name)
            print(
                f"Generated {min(start + args.batch_size, len(image_paths))}/{len(image_paths)}",
                end="\r",
                flush=True,
            )

    print(f"\nSaved {len(image_paths)} predictions to: {args.output_dir}")


if __name__ == "__main__":
    main()
