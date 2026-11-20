"""Verifica o limiar e os valores de referência da Tabela 2."""

import unittest

from comparar_modelos import summarize_predictions


class EvaluationTest(unittest.TestCase):
    def test_published_intervals_given_reported_counts(self):
        cases = (
            (9, 10, ((95.0, 57.7, 99.9), (100.0, 58.9, 100.0),
                     (90.0, 50.9, 99.9), (94.7, 54.6, 99.9))),
            (8, 10, ((90.0, 52.3, 99.1), (100.0, 53.1, 100.0),
                     (80.0, 40.1, 98.3), (88.9, 45.7, 99.1))),
        )
        for tp, tn, expected in cases:
            with self.subTest(tp=tp, tn=tn):
                scores = [1] * tp + [0] * (10 - tp) + [0] * tn + [1] * (10 - tn)
                report = summarize_predictions([1] * 10 + [0] * 10, scores)
                self.assertEqual(report["confusion"], {
                    "tp": tp, "fn": 10 - tp, "fp": 10 - tn, "tn": tn,
                })
                actual = tuple(
                    tuple(round(100 * value, 1) for value in (row["estimate"], *row["ci95"]))
                    for row in report["metrics"].values()
                )
                self.assertEqual(actual, expected)

    def test_threshold_strictness_and_predicted_class_confidence(self):
        report = summarize_predictions([1, 1, 0, 0], [0.8, 0.1, 0.5, 0.9])
        self.assertEqual(report["confusion"], {"tp": 1, "fn": 1, "fp": 1, "tn": 1})
        self.assertAlmostEqual(report["confidence"]["true_positives_mean"], 0.8)
        self.assertAlmostEqual(report["confidence"]["errors_mean"], 0.9)

    def test_undefined_precision_without_positive_predictions(self):
        report = summarize_predictions([1, 0], [0.5, 0])
        self.assertIsNone(report["metrics"]["precision"]["estimate"])

    def test_unbalanced_data_rejected(self):
        with self.assertRaisesRegex(ValueError, "número igual"):
            summarize_predictions([1, 0, 0], [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
