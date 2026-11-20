"""Shared fit and reporting protocol for the NewCare and MiniCare CNNs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import set_random_seed
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from model.extract_features import DATA_DIR, SPLIT_SEED, load_session_groups, split_training_data
from newcare.trajectory import render_trajectory


def prepare_images(json_path, target_size):
    images, labels = [], []
    for _, group in load_session_groups(json_path):
        points = group[["x", "y"]].values.astype(int)
        images.append(resize(render_trajectory(points), target_size))
        labels.append(group["stress"].iloc[0])
    return np.asarray(images)[..., np.newaxis], np.asarray(labels)


def load_features(directory, split, labels):
    path = directory / f"features_{split}.npy"
    features = np.load(path, allow_pickle=False)
    feature_labels = np.load(directory / f"y_{split}_features.npy", allow_pickle=False)
    if not np.array_equal(feature_labels, labels):
        raise ValueError(f"Image and feature labels differ in {split}. Regenerate the feature files.")
    if features.shape != (len(labels), 4) or not np.isfinite(features).all():
        raise ValueError(f"Expected four finite features per session in {path}")
    return features


def run_training(build_model, model_name, model_prefix, target_size, default_output_dir):
    parser = argparse.ArgumentParser(description=f"Treinar o modelo {model_name}.")
    parser.add_argument("--feature-dir", type=Path, default=DATA_DIR, help="Diretório das características .npy.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Destino dos pesos e relatório.")
    parser.add_argument("--epochs", type=int, default=200, help="Número máximo de épocas (padrão: 200).")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    tf.get_logger().setLevel("ERROR")
    set_random_seed(SPLIT_SEED)
    tf.config.experimental.enable_op_determinism()

    images, labels = prepare_images(DATA_DIR / "dataset.json", target_size)
    images_train, images_holdout, labels_train, labels_holdout = split_training_data(images, labels)
    images_external, labels_external = prepare_images(DATA_DIR / "dataset-validacao.json", target_size)
    features_train = load_features(args.feature_dir, "train", labels_train)
    features_holdout = load_features(args.feature_dir, "test", labels_holdout)
    features_external = load_features(args.feature_dir, "val", labels_external)
    print(f"Train: {len(labels_train)}, Internal holdout: {len(labels_holdout)}, "
          f"External evaluation: {len(labels_external)}", flush=True)

    model = build_model()
    model.summary()
    early_stop = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    history = model.fit(
        [images_train, features_train], labels_train, epochs=args.epochs, batch_size=16,
        validation_data=([images_holdout, features_holdout], labels_holdout),
        callbacks=[early_stop, reduce_lr], verbose=2, shuffle=True,
    )

    external_loss, external_accuracy = model.evaluate(
        [images_external, features_external], labels_external, verbose=0,
    )
    external_scores = model.predict([images_external, features_external], verbose=0).ravel()
    external_predictions = external_scores > 0.5
    logs = {metric: [float(value) for value in values] for metric, values in history.history.items()}
    best_epoch = int(np.argmin(logs["val_loss"]))
    report = {
        "model": model_name,
        "seed": SPLIT_SEED,
        "deterministic_tensorflow_ops": True,
        "epochs_requested": args.epochs,
        "epochs_executed": len(logs["val_loss"]),
        "stopped_early": bool(early_stop.stopped_epoch),
        "n_train": len(labels_train),
        "n_internal_holdout": len(labels_holdout),
        "n_external_evaluation": len(labels_external),
        "parameters": model.count_params(),
        "best_internal_holdout_epoch": best_epoch + 1,
        "best_internal_holdout_loss": logs["val_loss"][best_epoch],
        "external_evaluation": {
            "loss": float(external_loss),
            "accuracy": float(external_accuracy),
            "precision": float(precision_score(labels_external, external_predictions, zero_division=0)),
            "recall": float(recall_score(labels_external, external_predictions, zero_division=0)),
            "f1": float(f1_score(labels_external, external_predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(labels_external, external_predictions, labels=[0, 1]).tolist(),
        },
        "history": logs,
    }
    report_json = json.dumps(report, indent=2, allow_nan=False) + "\n"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_basename = args.output_dir / f"modelCNN-{model_prefix}-{time.time_ns()}"
    model.save(model_basename.with_suffix(".keras"))
    model.save(model_basename.with_suffix(".h5"))
    model_basename.with_suffix(".json").write_text(report_json, encoding="utf-8")
    print(f"Best held-out loss: {report['best_internal_holdout_loss']:.6f} "
          f"(epoch {report['best_internal_holdout_epoch']} of {report['epochs_executed']})")
    print("External evaluation:", json.dumps(report["external_evaluation"]))
    print(f"Model and training report saved with prefix {model_basename}")
