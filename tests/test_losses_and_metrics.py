import unittest
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from evaluate_saved_epochs import discover_epochs
from evaluate_saved_predictions import evaluate_prediction_directory
from losses import CrossEntropy, CrossEntropyDiceLoss, DiceLoss
from utils import PatientVolumeDice, class2one_hot, dice_coef, patient_id_from_stem


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
                    str(Path(__file__).parents[1] / "evaluate_saved_epochs.py"),
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


if __name__ == "__main__":
    unittest.main()
