import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NEWCARE_DIR = str(Path(__file__).resolve().parents[1])
if NEWCARE_DIR not in sys.path:
    sys.path.insert(0, NEWCARE_DIR)

from newcare.trajectory import (
    tortuosity,
    fractal_dimension,
    back_and_forth_count,
    room_transitions,
    load_json_data,
)


SPLIT_SEED = 42
HOLDOUT_FRACTION = 0.25
DATA_DIR = Path(__file__).resolve().parent / "dados"


def split_training_data(values, labels):
    return train_test_split(
        values, labels, test_size=HOLDOUT_FRACTION, random_state=SPLIT_SEED,
        stratify=labels,
    )


def load_session_groups(path):
    data = load_json_data(path)
    df = pd.DataFrame(data)
    df["stress"] = df["stress"].astype(int)
    grouped = df.groupby("date", sort=True)
    if grouped["stress"].nunique().ne(1).any():
        raise ValueError(f"Inconsistent stress labels within a session in {path}")
    return grouped


def extract_features_from_json(path):
    features_list = []
    y_list = []
    for _, group in load_session_groups(path):
        points = group[["x", "y"]].values.astype(float)
        stress = group["stress"].iloc[0]
        t = tortuosity(points)
        d = fractal_dimension(points)
        bf = back_and_forth_count(points)
        rt = room_transitions(points)
        features_list.append([t, d, bf, rt])
        y_list.append(stress)
    return np.array(features_list), np.array(y_list)


def main():
    parser = argparse.ArgumentParser(description="Extrair e normalizar as quatro características.")
    parser.add_argument(
        "--output-dir", type=Path, default=DATA_DIR,
        help="Diretório para os arquivos gerados (padrão: dados ao lado deste script).",
    )
    output_dir = parser.parse_args().output_dir

    print("Extracting features from dataset.json (training + internal holdout)...")
    feats_all, labels_all = extract_features_from_json(DATA_DIR / "dataset.json")
    print(f"  Total samples: {len(feats_all)}")
    print(f"  Feature stats - T: mean={feats_all[:,0].mean():.3f}, std={feats_all[:,0].std():.3f}")
    print(f"  Feature stats - D: mean={feats_all[:,1].mean():.3f}, std={feats_all[:,1].std():.3f}")

    X_train, X_test, y_train, y_test = split_training_data(feats_all, labels_all)
    print(f"  Train: {len(X_train)}, Internal holdout: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nExtracting features from dataset-validacao.json (external evaluation)...")
    feats_val, labels_val = extract_features_from_json(DATA_DIR / "dataset-validacao.json")
    print(f"  External evaluation samples: {len(feats_val)}")
    X_val_scaled = scaler.transform(feats_val)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features_train.npy", X_train_scaled)
    np.save(output_dir / "features_test.npy", X_test_scaled)
    np.save(output_dir / "y_train_features.npy", y_train)
    np.save(output_dir / "y_test_features.npy", y_test)
    np.save(output_dir / "features_val.npy", X_val_scaled)
    np.save(output_dir / "y_val_features.npy", labels_val)
    joblib.dump(scaler, output_dir / "feature_scaler.save")
    print(f"Features and scaler saved to {output_dir}")


if __name__ == "__main__":
    main()
