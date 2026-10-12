import unittest

from plotting.render_results_table import extrema_cells, numeric_value


class RenderResultsTableTests(unittest.TestCase):
    def test_numeric_value_accepts_formatted_metrics(self):
        self.assertEqual(numeric_value("12.45 mm"), 12.45)
        self.assertEqual(numeric_value("0.770 ± 0.053"), 0.770)
        self.assertEqual(numeric_value("+0.0015"), 0.0015)
        self.assertIsNone(numeric_value("—"))

    def test_best_and_worst_follow_metric_direction(self):
        header = ["Run", "Dice", "HD95"]
        rows = [
            ["A", "0.80", "10 mm"],
            ["B", "0.90", "12 mm"],
            ["C", "0.70", "8 mm"],
        ]
        best, worst = extrema_cells(header, rows, ["Dice"], ["HD95"])
        self.assertEqual(best, {(1, 1), (2, 2)})
        self.assertEqual(worst, {(2, 1), (1, 2)})

    def test_ties_are_all_highlighted(self):
        header = ["Run", "Dice"]
        rows = [["A", "0.9"], ["B", "0.9"], ["C", "0.8"]]
        best, worst = extrema_cells(header, rows, ["Dice"], [])
        self.assertEqual(best, {(0, 1), (1, 1)})
        self.assertEqual(worst, {(2, 1)})


if __name__ == "__main__":
    unittest.main()
