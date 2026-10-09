#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

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

import argparse
import json
import os
import random
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from functools import partial 

from data_loading.dataset import SliceDataset
from data_loading.augmentations import AUGMENTATION_MODES, build_augmentation
from data_loading.reproducibility import (
    data_loader_generator,
    seed_data_loader_worker,
    seed_everything,
)
from models.ShallowNet import shallowCNN
from models.ENet import ENet
from models.ENet_25D import ENet_25D
from models.UNet import UNet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   PatientVolumeDice,
                   patient_id_from_stem,
                   save_images)

from losses import (CrossEntropy, DiceLoss, CrossEntropyDiceLoss)

LOSSES: dict[str, Any] = {'ce': CrossEntropy, 'dice': DiceLoss, 'ce_dice': CrossEntropyDiceLoss}

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CORRECTED"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
# Full 40-patient training set; other folders (CV folds, fixed splits) can be passed with --data-dir
datasets_params["SEGTHOR_FULL"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

def img_transform(img):
        img = img.convert('L')
        img = np.array(img)[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def gt_transform(K, img):
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]

def compute_class_weights(spec: str, gt_paths: list[Path], K: int) -> list[float] | None:
        """Per-class weights for the cross-entropy, from a --class-weights value.

        'none': no weighting. 'inv' / 'sqrt_inv': inverse (square root) of the pixel
        frequency of each class in the given ground truths. Otherwise a comma-separated
        list of K numbers. Computed weights are normalized to average to 1.
        """
        if spec == 'none':
                return None

        if spec in ['inv', 'sqrt_inv']:
                counts = torch.zeros(K, dtype=torch.float64)
                for gt_path in gt_paths:
                        counts += gt_transform(K, Image.open(gt_path)).sum(dim=(1, 2))
                if (counts == 0).any():
                        raise ValueError(f"Some classes never appear in the ground truth: {counts.tolist()}")

                weights = counts.sum() / counts
                if spec == 'sqrt_inv':
                        weights = weights.sqrt()
                weights = weights / weights.mean()
                print(f">> Class pixel frequencies: {(counts / counts.sum()).tolist()}")
        else:
                weights = torch.tensor([float(w) for w in spec.split(',')])
                if len(weights) != K:
                        raise ValueError(f"--class-weights needs {K} values, got {len(weights)}: {spec}")

        print(f">> Cross-entropy class weights: {weights.tolist()}")
        return weights.tolist()


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
        """Write to a temporary file first, so a job killed while saving keeps the previous checkpoint."""
        tmp_path = path.with_suffix(".tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)


def extend_log(log: Tensor, epochs: int) -> Tensor:
        """Pad a per-epoch log with NaN epochs, when a run is resumed with more --epochs than it started with."""
        if len(log) >= epochs:
                return log[:epochs]
        padding = torch.full((epochs - len(log), *log.shape[1:]), float("nan"), dtype=log.dtype)
        return torch.cat([log, padding])


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    seed_everything(args.seed)

    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    in_dim: int = 2 * args.context + 1  # Current slice plus its neighbours on each side (2.5D)
    net_class = datasets_params[args.dataset]['net']
    in_dim: int = 2 * args.context + 1

    model_classes = {
        "enet": ENet,
        "unet": UNet,
    }

    net_class = (
        datasets_params[args.dataset]["net"]
        if args.model is None
        else model_classes[args.model]
    )

    if args.context > 0:
        if net_class is not ENet:
            raise ValueError(
                "--context > 0 (2.5D input) is only supported with ENet"
            )
        net_class = ENet_25D

    net = net_class(
        in_dim,
        K,
        kernels=kernels,
        factor=factor,
    )
    net.init_weights()
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = args.data_dir if args.data_dir else Path("data") / args.dataset

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             joint_transform=build_augmentation(args.augmentation),  # Training set only
                             debug=args.debug,
                             context=args.context)
    train_generator = data_loader_generator(args.seed)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              shuffle=True,
                              worker_init_fn=seed_data_loader_worker,
                              generator=train_generator)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug,
                           context=args.context)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False,
                            worker_init_fn=seed_data_loader_worker,
                            generator=data_loader_generator(args.seed + 1))

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    try:
        import wandb
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Training logging requires wandb; install the project requirements first."
        ) from error

    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.class_weights != 'none' and args.loss == 'dice':
        raise ValueError("--class-weights only applies to the cross-entropy, so it cannot be used with --loss dice")
    class_weights = compute_class_weights(args.class_weights,
                                          [gt_path for _, gt_path in train_loader.dataset.files],
                                          K)

    root_dir = args.data_dir if args.data_dir else Path("data") / args.dataset
    split_path = root_dir / "split.json"
    split_metadata = (
        json.loads(split_path.read_text(encoding="utf-8"))
        if split_path.is_file()
        else None
    )
    run_metadata = {
        **{
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "split_seed": split_metadata.get("seed") if split_metadata else None,
        "test_patients": split_metadata.get("test") if split_metadata else None,
        "ce_class_weights": class_weights,
    }
    checkpoint_path: Path = args.dest / "last.pt"
    checkpoint: dict[str, Any] | None = None
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"--resume needs a checkpoint from a previous run: {checkpoint_path}")
        # Our own file, which also holds the Python and NumPy random states, so not weights_only
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["early_stopped"]:
            raise RuntimeError(f"The run in {args.dest} stopped early at epoch {checkpoint['epoch']}; "
                               "there is nothing to resume")
        print(f">>> Resuming from {checkpoint_path}, after epoch {checkpoint['epoch']}")
        run_metadata["resumed_after_epoch"] = checkpoint["epoch"]

    (args.dest / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    wandb_run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "ai4mi-segthor"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=os.environ.get("WANDB_RUN_NAME", f"{args.dataset}-{args.loss}"),
        group=os.environ.get("WANDB_RUN_GROUP"),
        mode=os.environ.get("WANDB_MODE", "online"),
        config=run_metadata | {"classes": K},
        # Continue the same wandb run when resuming
        id=checkpoint["wandb_id"] if checkpoint else None,
        resume="allow" if checkpoint else None,
    )

    if args.mode == "full":
        idk = list(range(K))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        idk = [0, 1, 3, 4]  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    loss_fn = LOSSES[args.loss](idk=idk) if class_weights is None \
        else LOSSES[args.loss](idk=idk, class_weights=class_weights)

    scheduler: Any = None
    if args.lr_scheduler == 'plateau':
        # Lower the learning rate when the validation 3D dice (the early stopping metric) stalls
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                               factor=args.lr_factor,
                                                               patience=args.lr_patience,
                                                               min_lr=args.min_lr)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.full((args.epochs, len(train_loader)), float("nan"))
    log_dice_tra: Tensor = torch.full((args.epochs, len(train_loader.dataset), K), float("nan"))
    log_loss_val: Tensor = torch.full((args.epochs, len(val_loader)), float("nan"))
    log_dice_val: Tensor = torch.full((args.epochs, len(val_loader.dataset), K), float("nan"))

    val_patient_ids = sorted({patient_id_from_stem(img_path.stem)
                              for img_path, _ in val_loader.dataset.files})
    log_dice3d_val: Tensor = torch.full((args.epochs, len(val_patient_ids), K), float("nan"))
    log_lr: Tensor = torch.full((args.epochs,), float("nan"))  # Learning rate used in each epoch
    with open(args.dest / "dice3d_val_patients.txt", "w") as f:
        f.write("\n".join(val_patient_ids) + "\n")

    best_dice: float = float("-inf")
    epochs_without_improvement: int = 0
    start_epoch: int = 0

    if checkpoint:
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        best_dice = checkpoint["best_dice"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        start_epoch = checkpoint["epoch"] + 1
        logs = checkpoint["logs"]
        log_loss_tra, log_dice_tra, log_loss_val, log_dice_val, log_dice3d_val, log_lr = (
            extend_log(logs[name], args.epochs)
            for name in ["loss_tra", "dice_tra", "loss_val", "dice_val", "dice3d_val", "lr"])
        # Same random state as at the end of the saved epoch: same data order, dropout and augmentation
        random.setstate(checkpoint["rng"]["python"])
        np.random.set_state(checkpoint["rng"]["numpy"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        if torch.cuda.is_available() and checkpoint["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        train_loader.generator.set_state(checkpoint["rng"]["train_loader"])
        val_loader.generator.set_state(checkpoint["rng"]["val_loader"])
        if start_epoch >= args.epochs:
            print(f">>> Already trained {start_epoch} epochs; increase --epochs to train longer")

    for e in range(start_epoch, args.epochs):
        patient_dice = PatientVolumeDice(K)
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        patient_dice.update(pred_seg, gt, data['stems'])

                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"SliceDice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"SliceDice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        epoch_patient_ids, epoch_dice3d = patient_dice.compute()
        if epoch_patient_ids != val_patient_ids:
            raise RuntimeError(f"Validation patients changed: {epoch_patient_ids} != {val_patient_ids}")
        log_dice3d_val[e] = epoch_dice3d

        organ_scores = torch.nanmean(epoch_dice3d[:, 1:], dim=0)
        current_dice: float = torch.nanmean(organ_scores).item()
        if not np.isfinite(current_dice):
            raise RuntimeError("No finite foreground patient-level Dice values were computed")
        organ_message = " ".join(
            f"Dice-{k}={organ_scores[k - 1].item():05.3f}"
            if torch.isfinite(organ_scores[k - 1]) else f"Dice-{k}=nan"
            for k in range(1, K)
        )
        print(f">>> Patient-level 3D validation Dice at epoch {e}: {current_dice:05.3f} "
              + organ_message)

        epoch_metrics = {
            "epoch": e,
            "train/loss": torch.nanmean(log_loss_tra[e]).item(),
            "validation/loss": torch.nanmean(log_loss_val[e]).item(),
            "validation/slice_dice": torch.nanmean(log_dice_val[e, :, 1:]).item(),
            "validation/3d_dice": current_dice,
            "lr": optimizer.param_groups[0]['lr'],
        }
        epoch_metrics |= {
            f"validation/3d_dice_class_{k}": organ_scores[k - 1].item()
            for k in range(1, K)
            if torch.isfinite(organ_scores[k - 1])
        }
        wandb_run.log(epoch_metrics, step=e)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)
        np.save(args.dest / "dice3d_val.npy", log_dice3d_val)
        log_lr[e] = optimizer.param_groups[0]['lr']
        np.save(args.dest / "lr.npy", log_lr)

        if current_dice > best_dice:
            if np.isfinite(best_dice):
                message = (f">>> Improved patient-level 3D dice at epoch {e}: "
                           f"{best_dice:05.3f}->{current_dice:05.3f} DSC")
            else:
                message = f">>> Initial patient-level 3D dice at epoch {e}: {current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            previous_lr = optimizer.param_groups[0]['lr']
            scheduler.step(current_dice)
            if optimizer.param_groups[0]['lr'] < previous_lr:
                print(f">>> Lowered learning rate at epoch {e}: {previous_lr:.2e}->{optimizer.param_groups[0]['lr']:.2e}")

        early_stopped: bool = bool(args.patience and epochs_without_improvement >= args.patience)

        # Everything needed to continue after this epoch with --resume
        save_checkpoint(checkpoint_path, {
            "epoch": e,
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_dice": best_dice,
            "epochs_without_improvement": epochs_without_improvement,
            "early_stopped": early_stopped,
            "logs": {"loss_tra": log_loss_tra, "dice_tra": log_dice_tra, "loss_val": log_loss_val,
                     "dice_val": log_dice_val, "dice3d_val": log_dice3d_val, "lr": log_lr},
            "rng": {"python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "train_loader": train_loader.generator.get_state(),
                    "val_loader": val_loader.generator.get_state()},
            "wandb_id": wandb_run.id,
        })

        if early_stopped:
            message = (f">>> Early stopping at epoch {e}: no improvement of the patient-level 3D dice "
                       f"for {args.patience} epochs (best {best_dice:05.3f})")
            print(message)
            with open(args.dest / "early_stopping.txt", 'w') as f:
                f.write(message)
            wandb_run.summary["stopped_epoch"] = e
            break

    wandb_run.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--loss', default='ce_dice', choices=list(LOSSES.keys()),
                        help="Loss function to use for training.")
    parser.add_argument('--augmentation', default='none', choices=AUGMENTATION_MODES,
                        help="Online augmentation of the training slices (data_loading/augmentations.py): "
                             "'geometric' (rotation, translation, scaling), 'intensity' (brightness, contrast, "
                             "gamma, noise) or 'combined'. Validation is never augmented.")
    parser.add_argument('--seed', default=0, type=int,
                        help="Random seed for weight initialization, data shuffling, dropout and augmentation.")
    parser.add_argument('--lr', default=0.0005, type=float, help="Initial learning rate (Adam).")
    parser.add_argument('--lr-scheduler', default='none', choices=['none', 'plateau'],
                        help="'plateau': multiply the learning rate by --lr-factor when the validation 3D dice "
                             "has not improved for more than --lr-patience epochs.")
    parser.add_argument('--lr-patience', default=4, type=int,
                        help="Plateau scheduler: epochs without improvement before lowering the learning rate. "
                             "Keep it below --patience, so the rate is lowered before training stops.")
    parser.add_argument('--lr-factor', default=0.5, type=float, help="Plateau scheduler: reduction factor.")
    parser.add_argument('--min-lr', default=1e-6, type=float, help="Lower bound for the learning rate.")
    parser.add_argument('--patience', default=0, type=int,
                        help="Early stopping: stop after this many epochs without improvement of the "
                             "validation patient-level 3D dice (0 = disabled).")
    parser.add_argument('--data-dir', type=Path, default=None,
                        help="Folder with the train/ and val/ splits, e.g. one cross-validation fold. "
                             "Defaults to data/<dataset>.")
    parser.add_argument('--context', default=0, type=int,
                        help="2.5D input: number of neighbouring slices on each side of the current one "
                             "stacked as input channels (0 = plain 2D, 1 = 3 slices, ...). Uses ENet_25D.")
    parser.add_argument('--class-weights', default='none',
                        help="Class weights for the cross-entropy (also the CE part of ce_dice): "
                             "'none', 'inv' or 'sqrt_inv' (inverse / sqrt inverse pixel frequency on "
                             "the training set), or K comma-separated values, e.g. '0.5,1,1,1,2'.")
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--resume', action='store_true',
                        help="Continue the run in --dest from its last.pt checkpoint (saved after every epoch). "
                             "--epochs can be raised to train an earlier run for longer.")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    parser.add_argument("--model",
        choices=["enet", "unet"],
        default=None,
        help=(
            "Segmentation architecture. If omitted, the dataset's "
            "default architecture is used."
        ),
    )
    args = parser.parse_args()

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
