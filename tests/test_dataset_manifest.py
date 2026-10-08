import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from data_loading.dataset import make_dataset


class SplitAwareDatasetTests(unittest.TestCase):
    def test_manifest_filters_all_labelled_subsets(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "img").mkdir()
            (root / "gt").mkdir()
            patients = ["Patient_01", "Patient_02", "Patient_03"]
            for patient in patients:
                array = np.zeros((4, 4), dtype=np.uint8)
                Image.fromarray(array).save(root / "img" / f"{patient}_0000.png")
                Image.fromarray(array).save(root / "gt" / f"{patient}_0000.png")
            (root / "split.json").write_text(
                json.dumps(
                    {
                        "train": ["Patient_01"],
                        "val": ["Patient_02"],
                        "test": ["Patient_03"],
                    }
                ),
                encoding="utf-8",
            )

            train = make_dataset(root, "train")
            validation = make_dataset(root, "val")
            test = make_dataset(root, "test")

            self.assertEqual(train[0][0].stem, "Patient_01_0000")
            self.assertEqual(validation[0][0].stem, "Patient_02_0000")
            self.assertEqual(test[0][0].stem, "Patient_03_0000")
            self.assertIsNotNone(test[0][1])


if __name__ == "__main__":
    unittest.main()
