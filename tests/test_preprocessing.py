import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from PIL import Image

from preprocessing.slice_segthor import (
    crop_shape_for_spacing,
    get_patient_split,
    get_splits,
    slice_patient,
    window_ct,
)


class IntensityPreprocessingTests(unittest.TestCase):
    def test_fixed_hu_window_clips_and_scales(self):
        values = np.array([-2000, -1000, 0, 1000, 5000], dtype=np.int16)
        result = window_ct(values, -1000, 1000)

        np.testing.assert_array_equal(result, np.array([0, 0, 128, 255, 255], dtype=np.uint8))

    def test_invalid_hu_window_is_rejected(self):
        with self.assertRaises(ValueError):
            window_ct(np.zeros((2, 2), dtype=np.int16), 100, 100)

class SpatialPreprocessingTests(unittest.TestCase):
    def test_crop_shape_uses_physical_spacing(self):
        crop = crop_shape_for_spacing(
            input_shape=(512, 512, 200),
            input_spacing=(0.9765625, 0.9765625, 2.5),
            output_shape=(256, 256),
            target_spacing=(1.5, 1.5),
        )

        self.assertEqual(crop, (393, 393))
        self.assertAlmostEqual(0.9765625 * crop[0] / 256, 1.5, places=2)

    def test_crop_larger_than_input_is_rejected(self):
        with self.assertRaises(ValueError):
            crop_shape_for_spacing(
                input_shape=(16, 16, 2),
                input_spacing=(1.0, 1.0, 1.0),
                output_shape=(32, 32),
                target_spacing=(1.0, 1.0),
            )

    def test_patient_processing_preserves_label_classes_and_metadata(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            patient_dir = root / "source" / "train" / "Patient_01"
            patient_dir.mkdir(parents=True)
            destination = root / "processed" / "train"

            ct = np.full((16, 16, 2), -1000, dtype=np.int16)
            ct[6:10, 6:10, :] = 0
            gt = np.zeros((16, 16, 2), dtype=np.uint8)
            gt[7:9, 7:9, 0] = 1
            gt[7:9, 7:9, 1] = 4
            affine = np.diag([-2.0, -2.0, 2.0, 1.0])
            nib.save(nib.Nifti1Image(ct, affine), patient_dir / "Patient_01.nii.gz")
            nib.save(nib.Nifti1Image(gt, affine), patient_dir / "GT.nii.gz")

            metadata = slice_patient(
                "Patient_01",
                destination,
                root / "source",
                shape=(8, 8),
                target_spacing=(1.0, 1.0),
                intensity_window=(-1000.0, 1000.0),
            )

            self.assertEqual(metadata["crop_shape"], [4, 4])
            self.assertEqual(metadata["effective_spacing_mm"], [1.0, 1.0, 2.0])
            self.assertEqual(metadata["orientation"], "LPS")
            label_files = sorted((destination / "gt").glob("*.png"))
            self.assertEqual(len(label_files), 2)
            encoded_values = set()
            for label_file in label_files:
                encoded_values.update(np.unique(np.asarray(Image.open(label_file))).astype(int))
            self.assertEqual(encoded_values, {0, 63, 252})

    def test_patient_processing_rejects_crop_that_removes_foreground(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            patient_dir = root / "source" / "train" / "Patient_01"
            patient_dir.mkdir(parents=True)
            ct = np.zeros((16, 16, 1), dtype=np.int16)
            gt = np.zeros((16, 16, 1), dtype=np.uint8)
            gt[0, 0, 0] = 1
            affine = np.diag([-2.0, -2.0, 2.0, 1.0])
            nib.save(nib.Nifti1Image(ct, affine), patient_dir / "Patient_01.nii.gz")
            nib.save(nib.Nifti1Image(gt, affine), patient_dir / "GT.nii.gz")

            with self.assertRaisesRegex(ValueError, "discard"):
                slice_patient(
                    "Patient_01",
                    root / "processed" / "train",
                    root / "source",
                    shape=(8, 8),
                    target_spacing=(1.0, 1.0),
                    intensity_window=(-1000.0, 1000.0),
                )


class SplitTests(unittest.TestCase):
    def test_split_is_patient_level_disjoint_and_deterministic(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train = root / "train"
            for patient_index in range(1, 41):
                (train / f"Patient_{patient_index:02d}").mkdir(parents=True)

            first = get_splits(root, validation_count=6, test_count=6, seed=0)
            second = get_splits(root, validation_count=6, test_count=6, seed=0)

            self.assertEqual(first, second)
            training, validation, test = first
            self.assertEqual((len(training), len(validation), len(test)), (28, 6, 6))
            self.assertFalse(set(training) & set(validation))
            self.assertFalse((set(training) | set(validation)) & set(test))
            self.assertEqual(len(set(training) | set(validation) | set(test)), 40)

    def test_fixed_split_has_expected_counts_and_cohort_balance(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train = root / "train"
            for patient_index in range(1, 41):
                (train / f"Patient_{patient_index:02d}").mkdir(parents=True)

            manifest = get_patient_split(root, validation_count=6, test_count=6, seed=0)
            training = set(manifest["train"])
            validation = set(manifest["val"])
            test = set(manifest["test"])

            self.assertEqual(len(training), 28)
            self.assertEqual(len(validation), 6)
            self.assertEqual(len(test), 6)
            self.assertFalse(training & validation)
            self.assertFalse((training | validation) & test)
            self.assertEqual(len(training | validation | test), 40)
            self.assertTrue(all(int(patient.split("_")[1]) >= 21 for patient in test))
            self.assertEqual(
                sum(int(patient.split("_")[1]) <= 20 for patient in validation),
                4,
            )
            self.assertEqual(
                sum(int(patient.split("_")[1]) >= 21 for patient in validation),
                2,
            )


if __name__ == "__main__":
    unittest.main()
