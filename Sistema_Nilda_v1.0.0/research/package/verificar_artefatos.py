
"""Validate research artifact hashes and the absence of old third-party artifacts."""
from pathlib import Path
from hashlib import sha256
ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    'Ready-main/newcare/model/dados/modelCNN-multi-1784648730.h5':'92d595e2adcbf39c951509fcfa42f4971d47d866ed4f5c67261fe814df12c941',
    'Ready-main/minicare/model/dados/modelCNN-minicare-int8.tflite':'3644b03d92ee446514c2df052c8285d7107a1fd33c3551e49ac1b9933cbd821a',
    'Ready-main/newcare/model/dados/feature_scaler.save':'64688ba1c79f009ea9ffdaaf30c42c636ba2800e3ceabd5356a37fbbfdc85336',
    'Ready-main/newcare/model/dados/dataset.json':'25845a0d66022285e4b631d3101e574f60aa5996275c65222b759bfaf3473b71',
    'Ready-main/newcare/model/dados/dataset-validacao.json':'afac7512dfd3903bb87ec006da20937591b699a5eecee7377b7825ae1360bc75',
}

def main():
    for rel,expected in EXPECTED.items():
        actual=sha256((ROOT/rel).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError(f'Hash divergence for {rel}: {actual}')
        print('OK',rel)
    prohibited=('smartcare-machinelearning-master','modelCNN-1605022224.h5',
                'modelCNN-minicare-fp32.tflite','modelCNN-minicare-1785364663.h5')
    for path in ROOT.rglob('*'):
        if path.is_file() and any(item in path.as_posix() for item in prohibited):
            raise ValueError('Forbidden research artifact path: '+str(path))
    print('Research assets and source dataset hashes verified; no SmartCare original model found.')
if __name__ == '__main__':main()
