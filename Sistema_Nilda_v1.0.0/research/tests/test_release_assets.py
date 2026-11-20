
import unittest
from pathlib import Path
import json
import hashlib

ROOT=Path(__file__).resolve().parents[2]

class PublicPackageTests(unittest.TestCase):
    def test_only_original_minicare(self):
        assert (ROOT/'Ready-main/minicare/model/dados/modelCNN-minicare-int8.tflite').is_file()
        artifacts=[p.name for p in (ROOT/'Ready-main/minicare/model/dados').iterdir() if p.is_file()]
        self.assertEqual(artifacts,['modelCNN-minicare-int8.tflite'])

    def test_no_third_party_original_software(self):
        forbidden=('smartcare-machinelearning-master','modelCNN-1605022224.h5',
                   'modelCNN-minicare-fp32.tflite','modelCNN-minicare-1785364663.h5')
        for path in ROOT.rglob('*'):
            self.assertFalse(any(a in path.as_posix() for a in forbidden),path)

    def test_scholarly_links_all_present(self):
        text=(ROOT/'REFERENCIAS.md').read_text(encoding='utf8')
        for n in range(1,18):self.assertIn(f'{n}. **',text)
        self.assertIn('10.1007/s42979-022-01091-3',text)

if __name__=='__main__':unittest.main()
