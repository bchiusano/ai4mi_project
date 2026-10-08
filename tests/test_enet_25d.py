import unittest
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from data_loading.dataset import SliceDataset, neighbour_indices
from losses import CrossEntropy, CrossEntropyDiceLoss
from main import compute_class_weights, gt_transform, img_transform
from models.ENet import ENet
from models.ENet_25D import ENet_25D
from utils import class2one_hot


class WeightedCrossEntropyTests(unittest.TestCase):
    def setUp(self):
        target_class = torch.tensor([[[0, 0], [0, 1]]], dtype=torch.int64)
        self.target = class2one_hot(target_class, K=2)
        # Confident and right on the background, unsure on the foreground pixel
        self.pred = torch.tensor([[[[0.9, 0.9], [0.9, 0.4]],
                                   [[0.1, 0.1], [0.1, 0.6]]]])

    def test_uniform_weights_match_unweighted(self):
        plain = CrossEntropy(idk=[0, 1])(self.pred, self.target)
        uniform = CrossEntropy(idk=[0, 1], class_weights=[3.0, 3.0])(self.pred, self.target)
        self.assertTrue(torch.allclose(plain, uniform))

    def test_weights_shift_loss_towards_weighted_class(self):
        plain = CrossEntropy(idk=[0, 1])(self.pred, self.target)
        weighted = CrossEntropy(idk=[0, 1], class_weights=[1.0, 10.0])(self.pred, self.target)
        # The foreground pixel has the highest error, so upweighting it raises the loss
        self.assertGreater(weighted.item(), plain.item())

    def test_weights_follow_class_index_with_partial_idk(self):
        target = class2one_hot(torch.tensor([[[0, 1], [2, 2]]]), K=3)
        pred = torch.full((1, 3, 2, 2), 1 / 3)
        loss = CrossEntropy(idk=[0, 2], class_weights=[1.0, 100.0, 1.0])(pred, target)
        # Class 1 is not supervised, so its weight must not matter
        self.assertTrue(torch.allclose(loss, CrossEntropy(idk=[0, 2])(pred, target)))

    def test_ce_dice_forwards_weights(self):
        loss = CrossEntropyDiceLoss(idk=[0, 1], class_weights=[1.0, 10.0])
        self.assertTrue(torch.equal(loss.ce.class_weights, torch.tensor([1.0, 10.0])))


class ClassWeightComputationTests(unittest.TestCase):
    def test_inverse_frequency_weights(self):
        with TemporaryDirectory() as tmp:
            gt = np.zeros((4, 4), dtype=np.uint8)
            gt[0, :2] = 63  # 2 pixels of class 1, 14 of background
            path = Path(tmp) / "Patient_01_0000.png"
            Image.fromarray(gt).save(path)
            gt[0, :2] = [126, 189]
            gt[1, 0] = 252
            path2 = Path(tmp) / "Patient_01_0001.png"
            Image.fromarray(gt).save(path2)

            weights = compute_class_weights('inv', [path, path2], K=5)
            sqrt_weights = compute_class_weights('sqrt_inv', [path, path2], K=5)

        self.assertAlmostEqual(np.mean(weights), 1.0)
        self.assertAlmostEqual(np.mean(sqrt_weights), 1.0)
        self.assertLess(weights[0], weights[1])
        # Square root flattens the weights
        self.assertGreater(min(sqrt_weights) / max(sqrt_weights), min(weights) / max(weights))

    def test_manual_and_none(self):
        self.assertIsNone(compute_class_weights('none', [], K=5))
        self.assertEqual(compute_class_weights('0.5,1,1,1,2', [], K=5), [0.5, 1, 1, 1, 2])
        with self.assertRaises(ValueError):
            compute_class_weights('1,2', [], K=5)


class SliceStackingTests(unittest.TestCase):
    def test_neighbours_stay_within_patient(self):
        paths = [Path(f"Patient_01_000{i}.png") for i in range(3)] \
            + [Path(f"Patient_02_000{i}.png") for i in range(2)]
        self.assertEqual(neighbour_indices(paths, 1),
                         [[0, 0, 1], [0, 1, 2], [1, 2, 2], [3, 3, 4], [3, 4, 4]])
        self.assertEqual(neighbour_indices(paths, 0), [[i] for i in range(5)])

    def test_dataset_stacks_slices_with_current_in_the_middle(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub in ["img", "gt"]:
                (root / "train" / sub).mkdir(parents=True)
            for i in range(3):
                Image.fromarray(np.full((4, 4), 10 * (i + 1), dtype=np.uint8)) \
                    .save(root / "train" / "img" / f"Patient_01_000{i}.png")
                Image.fromarray(np.zeros((4, 4), dtype=np.uint8)) \
                    .save(root / "train" / "gt" / f"Patient_01_000{i}.png")

            dataset = SliceDataset('train', root, img_transform=img_transform,
                                   gt_transform=partial(gt_transform, 5), context=1)
            img = dataset[0]["images"]

        self.assertEqual(tuple(img.shape), (3, 4, 4))
        values = [round(v * 255) for v in img[:, 0, 0].tolist()]
        self.assertEqual(values, [10, 10, 20])  # First slice repeated at the volume edge


class ENet25DTests(unittest.TestCase):
    def test_forward_shape_with_stacked_slices(self):
        for in_dim in [1, 3, 5]:
            with self.subTest(in_dim=in_dim):
                net = ENet_25D(in_dim, 5, kernels=8, factor=2)
                out = net(torch.rand(2, in_dim, 64, 64))
                self.assertEqual(tuple(out.shape), (2, 5, 64, 64))

    def test_single_channel_matches_enet(self):
        enet = ENet(1, 5, kernels=8, factor=2)
        enet_25d = ENet_25D(1, 5, kernels=8, factor=2)
        enet_25d.load_state_dict(enet.state_dict())


if __name__ == "__main__":
    unittest.main()
