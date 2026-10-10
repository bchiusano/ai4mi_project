import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from data_loading.augmentations import SegmentationAugmentation, build_augmentation
from data_loading.dataset import SliceDataset
from data_loading.reproducibility import seed_everything


class SegmentationAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.image = torch.zeros((1, 32, 32), dtype=torch.float32)
        self.image[:, 10:22, 12:20] = 0.75
        class_mask = torch.zeros((32, 32), dtype=torch.int64)
        class_mask[10:22, 12:20] = 2
        self.target = torch.nn.functional.one_hot(class_mask, num_classes=5).permute(2, 0, 1).float()

    def test_none_mode_disables_augmentation(self):
        self.assertIsNone(build_augmentation("none"))

    def test_default_strengths_match_cohort_based_limits(self):
        transform = SegmentationAugmentation()

        self.assertEqual(transform.max_rotation_degrees, 5.0)
        self.assertEqual(transform.max_translation_fraction, 0.03)
        self.assertEqual(transform.scale_range, (0.9, 1.1))
        self.assertEqual(transform.brightness_delta, 0.02)
        self.assertEqual(transform.contrast_range, (0.95, 1.05))
        self.assertEqual(transform.gamma_range, (0.95, 1.05))
        self.assertEqual(transform.max_noise_sigma, 0.01)

    def test_geometric_transform_preserves_valid_one_hot_mask(self):
        transform = SegmentationAugmentation(
            geometric=True,
            intensity=False,
            geometric_probability=1.0,
        )
        torch.manual_seed(3)
        image, target = transform(self.image, self.target)

        self.assertEqual(image.shape, self.image.shape)
        self.assertEqual(target.shape, self.target.shape)
        self.assertTrue(torch.all(target.sum(dim=0) == 1))
        self.assertTrue(set(torch.unique(target).tolist()).issubset({0.0, 1.0}))

    def test_intensity_transform_does_not_change_mask(self):
        transform = SegmentationAugmentation(
            geometric=False,
            intensity=True,
            intensity_probability=1.0,
        )
        torch.manual_seed(4)
        image, target = transform(self.image, self.target)

        self.assertTrue(torch.equal(target, self.target))
        self.assertFalse(torch.equal(image, self.image))
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_same_run_seed_reproduces_online_augmentation(self):
        transform = SegmentationAugmentation(
            geometric=True,
            intensity=True,
            geometric_probability=1.0,
            intensity_probability=1.0,
        )
        seed_everything(2)
        first_image, first_target = transform(self.image, self.target)
        seed_everything(2)
        second_image, second_target = transform(self.image, self.target)

        self.assertTrue(torch.equal(first_image, second_image))
        self.assertTrue(torch.equal(first_target, second_target))

    def test_25d_slices_get_the_same_geometric_transform(self):
        # Three identical neighbouring slices must stay identical: one transform for the whole stack
        image = self.image.repeat(3, 1, 1)
        transform = SegmentationAugmentation(geometric=True, intensity=False, geometric_probability=1.0)
        torch.manual_seed(5)
        augmented, target = transform(image, self.target)

        self.assertEqual(augmented.shape, image.shape)
        self.assertFalse(torch.equal(augmented, image))
        self.assertTrue(torch.equal(augmented[0], augmented[1]))
        self.assertTrue(torch.equal(augmented[0], augmented[2]))
        # The mask moves with the image: the organ stays where the bright region is
        # (edges are blurred by the bilinear image interpolation, so compare means)
        self.assertGreater(float(augmented[0][target[2] == 1].mean()), 0.6)
        self.assertLess(float(augmented[0][target[2] == 0].mean()), 0.1)

    def test_intensity_transform_supports_any_number_of_slices(self):
        transform = SegmentationAugmentation(geometric=False, intensity=True, intensity_probability=1.0)
        for channels in (1, 3, 5):
            image = torch.rand(channels, 16, 16)
            torch.manual_seed(6)
            augmented, _ = transform(image, self.target[:, :16, :16])

            self.assertEqual(augmented.shape, image.shape)
            self.assertGreaterEqual(float(augmented.min()), 0.0)
            self.assertLessEqual(float(augmented.max()), 1.0)

    def test_single_slice_contrast_and_gamma_match_torchvision(self):
        from torchvision.transforms import functional as vision_functional

        from data_loading.augmentations import _adjust_contrast, _adjust_gamma

        image = torch.rand(1, 16, 16)
        self.assertTrue(torch.allclose(_adjust_contrast(image, 1.05),
                                       vision_functional.adjust_contrast(image, 1.05), atol=1e-6))
        self.assertTrue(torch.allclose(_adjust_gamma(image, 0.95),
                                       vision_functional.adjust_gamma(image, 0.95), atol=1e-6))


class SliceDatasetJointTransformTests(unittest.TestCase):
    @staticmethod
    def _image_transform(image):
        return torch.from_numpy(np.array(image, copy=True)).float().unsqueeze(0) / 255

    @staticmethod
    def _target_transform(image):
        labels = torch.from_numpy(np.array(image, copy=True)).long()
        return torch.nn.functional.one_hot(labels, num_classes=5).permute(2, 0, 1).float()

    def _make_patient(self, root: Path, slices: int):
        (root / "train" / "img").mkdir(parents=True)
        (root / "train" / "gt").mkdir(parents=True)
        for i in range(slices):
            Image.fromarray(np.full((8, 8), 64 * (i + 1), dtype=np.uint8)).save(
                root / "train" / "img" / f"Patient_01_{i:04d}.png"
            )
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(
                root / "train" / "gt" / f"Patient_01_{i:04d}.png"
            )

    def test_slice_dataset_returns_jointly_transformed_image(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._make_patient(root, 1)

            dataset = SliceDataset(
                "train",
                root,
                img_transform=self._image_transform,
                gt_transform=self._target_transform,
                joint_transform=lambda image, target: (image + 0.25, target),
            )
            sample = dataset[0]

            expected = torch.full((1, 8, 8), 64 / 255 + 0.25)
            self.assertTrue(torch.allclose(sample["images"], expected))
            self.assertTrue(torch.all(sample["gts"].sum(dim=0) == 1))

    def test_25d_dataset_passes_the_whole_slice_stack_to_the_joint_transform(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._make_patient(root, 3)
            shapes = []

            def joint_transform(image, target):
                shapes.append(tuple(image.shape))
                return image + 0.1, target

            dataset = SliceDataset(
                "train",
                root,
                img_transform=self._image_transform,
                gt_transform=self._target_transform,
                joint_transform=joint_transform,
                context=1,
            )
            sample = dataset[1]

            self.assertEqual(shapes, [(3, 8, 8)])
            expected = torch.stack([torch.full((8, 8), 64 * (i + 1) / 255 + 0.1) for i in range(3)])
            self.assertTrue(torch.allclose(sample["images"], expected))


if __name__ == "__main__":
    unittest.main()
