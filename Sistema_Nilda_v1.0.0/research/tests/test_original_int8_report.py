"""Regression check: only the archived MiniCare original INT8 is evaluated."""
import json
from pathlib import Path
import unittest

REPORT = Path(__file__).resolve().parents[1] / "results" / "evaluation.json"

class OriginalINT8ReportTest(unittest.TestCase):
    def test_only_original_archived_models(self):
        r=json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(r["results"]),{"newcare_h5","minicare_tflite_int8"})
        self.assertEqual(set(r["tflite_tensors"]),{"minicare_tflite_int8"})
        self.assertEqual(r["results"]["minicare_tflite_int8"]["confusion"],{"tp":8,"fn":2,"fp":0,"tn":10})
    def test_int8_metrics(self):
        r=json.loads(REPORT.read_text(encoding="utf-8"))["results"]["minicare_tflite_int8"]["metrics"]
        self.assertAlmostEqual(r["accuracy"]["estimate"],0.9)
        self.assertAlmostEqual(r["precision"]["estimate"],1.0)
        self.assertAlmostEqual(r["recall"]["estimate"],0.8)
        self.assertAlmostEqual(r["f1"]["estimate"],8/9)

if __name__=="__main__":unittest.main()
