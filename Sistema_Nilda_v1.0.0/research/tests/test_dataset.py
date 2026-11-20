"""Check the original synthetic sessions and separation of holdout data."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'Ready-main/newcare/model/dados'

def load(filename):
    raw = json.loads((DATA_DIR / filename).read_text(encoding='utf8'))
    return json.loads(raw) if isinstance(raw, str) else raw

class DatasetIntegrityTest(unittest.TestCase):
    def test_session_counts_and_labels(self):
        for filename, n, expected in (
            ('dataset.json', 200, {0:100, 1:100}),
            ('dataset-validacao.json', 20, {0:10, 1:10}),
        ):
            sessions={}
            for row in load(filename):
                sessions.setdefault(row['date'], set()).add(int(row['stress']))
            self.assertEqual(len(sessions), n)
            self.assertTrue(all(len(labels)==1 for labels in sessions.values()))
            self.assertEqual(dict(Counter(next(iter(l)) for l in sessions.values())),expected)

    def test_no_shared_session_ids(self):
        train={row['date'] for row in load('dataset.json')}
        holdout={row['date'] for row in load('dataset-validacao.json')}
        self.assertFalse(train & holdout)
if __name__ == '__main__': unittest.main()
