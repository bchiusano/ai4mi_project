import csv
import unittest
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import nibabel as nib
from PIL import Image

from evaluation.evaluate_saved_epochs import discover_epochs
from evaluation.evaluate_saved_predictions import (
    evaluate_hd95_directory,
    evaluate_prediction_directory,
)
from losses import CrossEntropy, CrossEntropyDiceLoss, DiceLoss
from utils import (
    PatientVolumeDice,
    class2one_hot,
    dice_coef,
    patient_id_from_stem,
    surface_hd95,
)
from postprocessing.connected_compontents import (
    keep_largest_component,
    select_improving_classes,
)


class LossTests(unittest.TestCase):
    def setUp(self):
        target_class = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.int64)
        self.target = class2one_hot(target_class, K=2)
        self.perfect = self.target.float()
        self.wrong = 1.0 - self.perfect

    def test_losses_prefer_perfect_prediction(self):
        for loss in (
            CrossEntropy(idk=[0, 1]),
            DiceLoss(idk=[0, 1]),
            CrossEntropyDiceLoss(idk=[0, 1]),
        ):
            with self.subTest(loss=loss.__class__.__name__):
                self.assertLess(loss(self.perfect, self.target).item(),
                                loss(self.wrong, self.target).item())

    def test_combined_loss_is_sum_of_components(self):
        ce = CrossEntropy(idk=[0, 1])
        dice = DiceLoss(idk=[0, 1])
        combined = CrossEntropyDiceLoss(idk=[0, 1])
        expected = ce(self.perfect, self.target) + dice(self.perfect, self.target)
        self.assertTrue(torch.allclose(combined(self.perfect, self.target), expected))


class PatientVolumeDiceTests(unittest.TestCase):
    def test_patient_id_parsing(self):
        self.assertEqual(patient_id_from_stem("Patient_19_0123"), "Patient_19")
        self.assertEqual(patient_id_from_stem("toy"), "toy")

    def test_volume_dice_does_not_reward_empty_slices(self):
        # Slice 0 contains class 1 in the target but the prediction misses it.
        target_0 = class2one_hot(torch.tensor([[[1, 1], [0, 0]]]), K=2)
        pred_0 = class2one_hot(torch.zeros((1, 2, 2), dtype=torch.int64), K=2)

        # Slice 1 is empty for class 1 in both target and prediction.  The old
        # slice-average metric awards a perfect class-1 Dice to this slice.
        target_1 = class2one_hot(torch.zeros((1, 2, 2), dtype=torch.int64), K=2)
        pred_1 = target_1.clone()

        slice_average = torch.cat((dice_coef(pred_0, target_0),
                                   dice_coef(pred_1, target_1)))[:, 1].mean()
        self.assertAlmostEqual(slice_average.item(), 0.5, places=6)

        metric = PatientVolumeDice(classes=2)
        metric.update(pred_0, target_0, ["Patient_01_0000"])
        metric.update(pred_1, target_1, ["Patient_01_0001"])
        patients, volume_dice = metric.compute()

        self.assertEqual(patients, ["Patient_01"])
        self.assertAlmostEqual(volume_dice[0, 1].item(), 0.0, places=6)

    def test_perfect_patient_volume_scores_one(self):
        labels = torch.tensor([
            [[0, 1], [1, 0]],
            [[1, 1], [0, 0]],
        ], dtype=torch.int64)
        one_hot = class2one_hot(labels, K=2)
        metric = PatientVolumeDice(classes=2)
        metric.update(one_hot, one_hot, ["Patient_01_0000", "Patient_01_0001"])
        _, volume_dice = metric.compute()
        self.assertTrue(torch.allclose(volume_dice, torch.ones_like(volume_dice)))

    def test_empty_patient_class_is_nan(self):
        labels = class2one_hot(torch.zeros((1, 2, 2), dtype=torch.int64), K=2)
        metric = PatientVolumeDice(classes=2)
        metric.update(labels, labels, ["Patient_01_0000"])
        _, volume_dice = metric.compute()
        self.assertTrue(torch.isnan(volume_dice[0, 1]))


class SurfaceHD95Tests(unittest.TestCase):
    def test_perfect_masks_have_zero_hd95(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        self.assertEqual(surface_hd95(mask, mask, (1.0, 1.0, 2.5)), 0.0)

    def test_hd95_uses_physical_spacing(self):
        target = np.zeros((3, 3, 3), dtype=bool)
        prediction = np.zeros_like(target)
        target[1, 1, 0] = True
        prediction[1, 1, 1] = True
        self.assertEqual(surface_hd95(prediction, target, (1.0, 1.0, 2.5)), 2.5)

    def test_empty_mask_semantics(self):
        empty = np.zeros((3, 3, 3), dtype=bool)
        present = empty.copy()
        present[1, 1, 1] = True
        self.assertTrue(np.isnan(surface_hd95(empty, empty)))
        self.assertTrue(np.isinf(surface_hd95(empty, present)))


class SavedPredictionEvaluationTests(unittest.TestCase):
    @staticmethod
    def save_mask(path: Path, labels: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((labels * 63).astype(np.uint8)).save(path)

    def test_saved_predictions_are_combined_as_patient_volumes(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pred_dir = root / "pred"
            gt_dir = root / "gt"

            target_with_organ = np.array([[1, 1], [0, 0]], dtype=np.uint8)
            empty = np.zeros((2, 2), dtype=np.uint8)
            self.save_mask(gt_dir / "Patient_01_0000.png", target_with_organ)
            self.save_mask(gt_dir / "Patient_01_0001.png", empty)
            self.save_mask(pred_dir / "Patient_01_0000.png", empty)
            self.save_mask(pred_dir / "Patient_01_0001.png", empty)

            patients, dice, _ = evaluate_prediction_directory(
                pred_dir, gt_dir, classes=2
            )
            self.assertEqual(patients, ["Patient_01"])
            self.assertAlmostEqual(dice[0, 1], 0.0, places=6)

    def test_saved_predictions_use_3d_spacing_for_hd95(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pred_dir = root / "pred"
            gt_dir = root / "gt"
            empty = np.zeros((3, 3), dtype=np.uint8)
            target_0 = empty.copy()
            target_0[1, 1] = 1
            pred_1 = empty.copy()
            pred_1[1, 1] = 1

            self.save_mask(gt_dir / "Patient_01_0000.png", target_0)
            self.save_mask(gt_dir / "Patient_01_0001.png", empty)
            self.save_mask(pred_dir / "Patient_01_0000.png", empty)
            self.save_mask(pred_dir / "Patient_01_0001.png", pred_1)

            patients, hd95, _ = evaluate_hd95_directory(
                pred_dir,
                gt_dir,
                {"Patient_01": (1.0, 1.0, 3.0)},
                classes=2,
            )
            self.assertEqual(patients, ["Patient_01"])
            self.assertEqual(hd95[0, 1], 3.0)

    def test_epoch_discovery_is_numeric_and_requires_prediction_split(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "iter010" / "val").mkdir(parents=True)
            (root / "iter002" / "val").mkdir(parents=True)
            (root / "iter003").mkdir()
            (root / "iteration004" / "val").mkdir(parents=True)
            self.assertEqual(
                discover_epochs(root, "val"),
                [(2, root / "iter002" / "val"), (10, root / "iter010" / "val")],
            )

    def test_all_epoch_cli_selects_best_volume_dice(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir = root / "run"
            gt_dir = root / "gt"
            output_dir = root / "output"
            target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
            empty = np.zeros((2, 2), dtype=np.uint8)
            name = "Patient_01_0000.png"
            self.save_mask(gt_dir / name, target)
            self.save_mask(run_dir / "iter000" / "val" / name, empty)
            self.save_mask(run_dir / "iter001" / "val" / name, target)

            subprocess.run(
                [
                    sys.executable,
                    "-m", "evaluation.evaluate_saved_epochs",
                    "--run-dir", str(run_dir),
                    "--gt-dir", str(gt_dir),
                    "--output-dir", str(output_dir),
                    "--prefix", "test",
                    "--classes", "2",
                    "--workers", "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = (output_dir / "test_best_epoch_3d_dice.txt").read_text()
            self.assertIn("Best patient-level 3D-Dice epoch: 1", summary)
            self.assertIn("Mean foreground patient-level 3D Dice: 1.000000", summary)


class ConnectedComponentsPostprocessingTests(unittest.TestCase):
    def test_largest_component_is_kept_in_3d_per_class(self):
        labels = np.zeros((2, 5, 5), dtype=np.uint8)
        labels[:, 1:3, 1:3] = 1
        labels[0, 4, 4] = 1
        labels[0, 0, 4] = 2

        cleaned = keep_largest_component(labels, 1)

        self.assertEqual(np.count_nonzero(cleaned == 1), 8)
        self.assertEqual(cleaned[0, 4, 4], 0)
        self.assertEqual(cleaned[0, 0, 4], 2)

    def test_largest_component_respects_connectivity(self):
        labels = np.zeros((2, 2, 2), dtype=np.uint8)
        labels[0, 0, 0] = 1
        labels[1, 1, 0] = 1

        cleaned_6 = keep_largest_component(labels, 1, connectivity=6)
        cleaned_18 = keep_largest_component(labels, 1, connectivity=18)

        self.assertEqual(np.count_nonzero(cleaned_6 == 1), 1)
        self.assertEqual(np.count_nonzero(cleaned_18 == 1), 2)

    def test_validation_selects_only_classes_with_better_mean_dice(self):
        prediction = np.zeros((1, 8, 8), dtype=np.uint8)
        target = np.zeros_like(prediction)
        prediction[0, 1:4, 1:4] = target[0, 1:4, 1:4] = 1
        prediction[0, 7, 7] = 1
        prediction[0, 1:3, 5:7] = target[0, 1:3, 5:7] = 2
        prediction[0, 6:8, 5:7] = target[0, 6:8, 5:7] = 2

        selected, metrics = select_improving_classes(
            {"Patient_01": prediction}, {"Patient_01": target}, classes=3
        )

        self.assertEqual(selected, [1])
        self.assertTrue(metrics["1"]["selected"])
        self.assertFalse(metrics["2"]["selected"])


class ComparisonPostprocessingTests(unittest.TestCase):
    def test_comparison_groups_png_slices_per_model_and_matches_patient_gt(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "heldout_predictions"
            prediction_a_dir = input_dir / "model_a"
            prediction_b_dir = input_dir / "model_b"
            prediction_c_dir = input_dir / "model_c"
            gt_dir = root / "heldout_ground_truth" / "Patient_01"
            prediction_a_dir.mkdir(parents=True)
            prediction_b_dir.mkdir(parents=True)
            prediction_c_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)

            target = np.zeros((6, 6, 3), dtype=np.uint8)
            target[1:3, 1:3, 0:2] = 1
            prediction_xyz = target.copy()
            prediction_xyz[4:6, 4:6, 2] = 1
            prediction_slices = np.transpose(prediction_xyz, (2, 0, 1))
            affine = np.diag([1.0, 1.5, 2.0, 1.0])
            nib.save(nib.Nifti1Image(target, affine), str(gt_dir / "GT.nii.gz"))
            # The adjacent source CT must not be selected as the label volume.
            nib.save(
                nib.Nifti1Image(np.zeros_like(target), affine),
                str(gt_dir / "Patient_01.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image(prediction_xyz, affine),
                str(prediction_c_dir / "Patient_01.nii.gz"),
            )
            for index, prediction_slice in enumerate(prediction_slices):
                Image.fromarray(prediction_slice * 63).save(
                    prediction_a_dir / f"Patient_01_{index:04d}.png"
                )
                Image.fromarray(target[:, :, index] * 63).save(
                    prediction_b_dir / f"Patient_01_{index:04d}.png"
                )
            output_dir = root / "comparison"
            script = Path(__file__).resolve().parents[1] / "postprocessing" / "comparison.py"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input-dir",
                    str(input_dir),
                    "--gt-dir",
                    str(root / "heldout_ground_truth"),
                    "--output-dir",
                    str(output_dir),
                    "--classes",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(
                f"Matching model_a/Patient_01 -> {gt_dir / 'GT.nii.gz'}",
                completed.stdout,
            )

            with (output_dir / "per_patient_per_class.csv").open(
                newline="", encoding="utf-8"
            ) as csv_file:
                rows = list(csv.DictReader(csv_file))
            by_model = {row["model"]: row for row in rows}
            self.assertLess(float(by_model["model_a"]["dice_before"]), 1.0)
            self.assertEqual(float(by_model["model_a"]["dice_after"]), 1.0)
            self.assertEqual(float(by_model["model_b"]["dice_before"]), 1.0)
            self.assertEqual(float(by_model["model_b"]["dice_after"]), 1.0)
            self.assertLess(float(by_model["model_c"]["dice_before"]), 1.0)
            self.assertEqual(float(by_model["model_c"]["dice_after"]), 1.0)

            with (output_dir / "per_patient_summary.csv").open(
                newline="", encoding="utf-8"
            ) as csv_file:
                summary_rows = list(csv.DictReader(csv_file))
            self.assertEqual(
                {row["ground_truth"] for row in summary_rows},
                {str(gt_dir / "GT.nii.gz")},
            )
            processed_path = (
                output_dir / "processed_predictions" / "model_a" / "Patient_01.nii.gz"
            )
            processed = nib.load(str(processed_path))
            np.testing.assert_array_equal(np.asanyarray(processed.dataobj), target)
            np.testing.assert_allclose(processed.affine, affine)
            self.assertTrue(
                (
                    output_dir
                    / "processed_predictions"
                    / "model_b"
                    / "Patient_01.nii.gz"
                ).is_file()
            )
            processed_nifti = nib.load(
                str(
                    output_dir
                    / "processed_predictions"
                    / "model_c"
                    / "Patient_01.nii.gz"
                )
            )
            np.testing.assert_array_equal(
                np.asanyarray(processed_nifti.dataobj), target
            )

    def test_test_postprocessing_exports_all_model_slices_without_metrics(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "heldout_predictions"
            model_a = input_dir / "model_a"
            model_b = input_dir / "model_b"
            model_c = input_dir / "model_c"
            source_dir = root / "source"
            patient_dir = source_dir / "Patient_02"
            model_a.mkdir(parents=True)
            model_b.mkdir(parents=True)
            model_c.mkdir(parents=True)
            patient_dir.mkdir(parents=True)

            source = np.zeros((4, 4, 2), dtype=np.int16)
            affine = np.diag([1.0, 2.0, 3.0, 1.0])
            nifti_prediction = np.zeros_like(source, dtype=np.uint8)
            nifti_prediction[1:3, 1:3, :] = 1
            nifti_prediction[3, 3, 1] = 1
            nib.save(
                nib.Nifti1Image(nifti_prediction, affine),
                str(model_c / "Patient_02.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image(source, affine),
                str(patient_dir / "Patient_02.nii.gz"),
            )
            for model_dir in (model_a, model_b):
                for index in range(2):
                    mask = np.zeros((4, 4), dtype=np.uint8)
                    mask[1:3, 1:3] = 1
                    Image.fromarray(mask * 63).save(
                        model_dir / f"Patient_02_{index:04d}.png"
                    )

            output_dir = root / "processed"
            script = (
                Path(__file__).resolve().parents[1]
                / "postprocessing"
                / "postprocess_test.py"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input-dir",
                    str(input_dir),
                    "--reference-dir",
                    str(source_dir),
                    "--output-dir",
                    str(output_dir),
                    "--classes",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            for model_name in ("model_a", "model_b", "model_c"):
                output_path = output_dir / model_name / "Patient_02.nii.gz"
                self.assertTrue(output_path.is_file())
                result = nib.load(str(output_path))
                self.assertEqual(result.shape, source.shape)
                np.testing.assert_allclose(result.affine, affine)
                if model_name == "model_c":
                    expected = np.zeros_like(nifti_prediction)
                    expected[1:3, 1:3, :] = 1
                    np.testing.assert_array_equal(
                        np.asanyarray(result.dataobj), expected
                    )
            self.assertFalse(any(output_dir.glob("*.csv")))


if __name__ == "__main__":
    unittest.main()
